import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import logging
import os
from datetime import datetime
import yfinance as yf
from typing import Dict, Optional, Tuple, Union
import pandas as pd

# Set up logging with absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'black_scholes.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

class CalculationError(Exception):
    """Custom exception for calculation errors"""
    pass

class BSImpliedVolatility:
    def __init__(self):
        self.N = norm.cdf
        self.N_prime = norm.pdf
        logger.info("Initialized BSImpliedVolatility calculator")
    
    def validate_inputs(self, S: float, K: float, T: float, r: float, sigma: Optional[float] = None) -> None:
        """
        Validate input parameters for BS calculations
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility (optional)
            
        Raises:
            ValidationError: If any input parameter is invalid
        """
        try:
            if S <= 0:
                raise ValidationError(f"Stock price must be positive, got {S}")
            if K <= 0:
                raise ValidationError(f"Strike price must be positive, got {K}")
            if T <= 0:
                raise ValidationError(f"Time to maturity must be positive, got {T}")
            if r < 0:
                raise ValidationError(f"Risk-free rate cannot be negative, got {r}")
            if sigma is not None and sigma <= 0:
                raise ValidationError(f"Volatility must be positive, got {sigma}")
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            raise
    
    def bs_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price with error handling
        """
        try:
            self.validate_inputs(S, K, T, r, sigma)
            
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            call_price = S*self.N(d1) - K*np.exp(-r*T)*self.N(d2)
            
            if not np.isfinite(call_price):
                raise CalculationError("Calculated price is not finite")
                
            return call_price
            
        except Exception as e:
            logger.error(f"Error in bs_call calculation: {str(e)}")
            raise CalculationError(f"Failed to calculate BS call price: {str(e)}")
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega with error handling"""
        try:
            self.validate_inputs(S, K, T, r, sigma)
            
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            vega = S * np.sqrt(T) * self.N_prime(d1)
            
            if not np.isfinite(vega):
                raise CalculationError("Calculated vega is not finite")
                
            return vega
            
        except Exception as e:
            logger.error(f"Error in vega calculation: {str(e)}")
            raise CalculationError(f"Failed to calculate vega: {str(e)}")
    
    def implied_vol_newton(self, C: float, S: float, K: float, T: float, r: float, 
                          sigma_init: float = 0.5, tol: float = 1e-5, 
                          max_iter: int = 100) -> Tuple[float, int]:
        """
        Calculate implied volatility using Newton-Raphson method with enhanced error handling
        
        Returns:
            Tuple[float, int]: (implied volatility, number of iterations)
        """
        try:
            self.validate_inputs(S, K, T, r)
            if C < 0:
                raise ValidationError(f"Option price must be non-negative, got {C}")
            
            sigma = sigma_init
            
            for i in range(max_iter):
                price = self.bs_call(S, K, T, r, sigma)
                diff = C - price
                
                if abs(diff) < tol:
                    logger.info(f"Newton method converged in {i+1} iterations")
                    return sigma, i+1
                
                v = self.vega(S, K, T, r, sigma)
                if abs(v) < 1e-10:
                    logger.warning("Vega near zero, potential numerical instability")
                    return sigma, i+1
                
                sigma = sigma + diff/v
                
                if sigma <= 0:
                    sigma = 0.0001
                    logger.warning("Volatility adjusted to minimum value")
            
            logger.warning(f"Newton method did not converge after {max_iter} iterations")
            return sigma, max_iter
            
        except Exception as e:
            logger.error(f"Error in Newton-Raphson calculation: {str(e)}")
            raise CalculationError(f"Failed to calculate implied volatility using Newton method: {str(e)}")

    def implied_vol_bisection(self, C: float, S: float, K: float, T: float, r: float,
                            sigma_min: float = 0.0001, sigma_max: float = 4, 
                            tol: float = 1e-5) -> Optional[float]:
        """Calculate implied volatility using Bisection method with enhanced error handling"""
        try:
            self.validate_inputs(S, K, T, r)
            if C < 0:
                raise ValidationError(f"Option price must be non-negative, got {C}")
            
            def objective(sigma):
                return self.bs_call(S, K, T, r, sigma) - C
            
            implied_vol = brentq(objective, sigma_min, sigma_max, xtol=tol)
            logger.info("Bisection method successfully converged")
            return implied_vol
            
        except ValueError as ve:
            logger.error(f"Bisection method failed to converge: {str(ve)}")
            return None
        except Exception as e:
            logger.error(f"Error in Bisection calculation: {str(e)}")
            raise CalculationError(f"Failed to calculate implied volatility using Bisection method: {str(e)}")

def get_option_data(symbol: str, expiry_date: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch option data from Yahoo Finance with enhanced error handling
    """
    try:
        logger.info(f"Fetching option data for {symbol}")
        ticker = yf.Ticker(symbol)
        
        if expiry_date is None:
            expiry_dates = ticker.options
            if not expiry_dates:
                logger.error(f"No option data available for {symbol}")
                return None
            expiry_date = expiry_dates[0]
        
        opt = ticker.option_chain(expiry_date)
        stock_price = ticker.history(period='1d')['Close'].iloc[-1]
        
        if opt is None or stock_price is None:
            logger.error(f"Failed to fetch complete data for {symbol}")
            return None
            
        logger.info(f"Successfully fetched option data for {symbol}")
        return {
            'calls': opt.calls,
            'puts': opt.puts,
            'stock_price': stock_price,
            'expiry_date': expiry_date
        }
        
    except Exception as e:
        logger.error(f"Error fetching option data: {str(e)}")
        return None