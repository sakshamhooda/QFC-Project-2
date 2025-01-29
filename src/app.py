import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from black_scholes import BSImpliedVolatility, get_option_data
import logging
import os
import plotly.graph_objects as go
from typing import Tuple, Optional

# Set up logging with absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_time_to_maturity(expiry_date: str) -> float:
    """Calculate time to maturity in years"""
    try:
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        return max((expiry - today).days / 365.0, 0.0001)
    except Exception as e:
        logger.error(f"Error calculating time to maturity: {str(e)}")
        st.error(f"Error calculating time to maturity: {str(e)}")
        return 0.0001

def plot_volatility_smile(strikes: np.ndarray, ivs: np.ndarray, stock_price: float) -> go.Figure:
    """Create volatility smile plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes,
        y=ivs,
        mode='lines+markers',
        name='Implied Volatility'
    ))
    
    # Add vertical line for current stock price
    fig.add_vline(x=stock_price, line_dash="dash", line_color="red",
                  annotation_text=f"Current Price: ${stock_price:.2f}")
    
    fig.update_layout(
        title='Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        hovermode='x'
    )
    return fig

def main():
    st.set_page_config(page_title="Implied Volatility Calculator", layout="wide")
    
    st.title("Black-Scholes Implied Volatility Calculator")
    
    # Initialize BS calculator
    bs_calculator = BSImpliedVolatility()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # Stock symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        
        try:
            # Fetch option data
            option_data = get_option_data(symbol)
            
            if option_data is None:
                st.error(f"No option data available for {symbol}")
                return
            
            # Get expiration dates
            ticker = yf.Ticker(symbol)
            expiry_dates = ticker.options
            
            selected_expiry = st.selectbox(
                "Expiration Date",
                options=expiry_dates,
                index=0
            )
            
            # Risk-free rate input
            risk_free_rate = st.number_input(
                "Risk-free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0
            ) / 100.0
            
            # Method selection
            method = st.radio(
                "Calculation Method",
                options=["Newton-Raphson", "Bisection"]
            )
            
        except Exception as e:
            logger.error(f"Error in sidebar setup: {str(e)}")
            st.error(f"Error loading option data: {str(e)}")
            return
    
    # Main content
    try:
        # Fetch updated option data for selected expiry
        option_data = get_option_data(symbol, selected_expiry)
        
        if option_data is None:
            st.error(f"Failed to fetch option data for {symbol}")
            return
        
        # Display current stock price
        stock_price = option_data['stock_price']
        st.metric("Current Stock Price", f"${stock_price:.2f}")
        
        # Calculate time to maturity
        T = calculate_time_to_maturity(selected_expiry)
        
        # Process call options
        calls_df = option_data['calls'].copy()
        
        # Calculate implied volatility for each strike
        ivs = []
        valid_strikes = []
        
        for _, row in calls_df.iterrows():
            try:
                if method == "Newton-Raphson":
                    iv, _ = bs_calculator.implied_vol_newton(
                        row['lastPrice'], stock_price, row['strike'],
                        T, risk_free_rate
                    )
                else:
                    iv = bs_calculator.implied_vol_bisection(
                        row['lastPrice'], stock_price, row['strike'],
                        T, risk_free_rate
                    )
                
                if iv is not None and 0 < iv < 4:  # Filter out extreme values
                    ivs.append(iv)
                    valid_strikes.append(row['strike'])
            
            except Exception as e:
                logger.warning(f"Skipping strike {row['strike']}: {str(e)}")
                continue
        
        # Create volatility smile plot
        if len(valid_strikes) > 0:
            fig = plot_volatility_smile(
                np.array(valid_strikes),
                np.array(ivs),
                stock_price
            )
            st.plotly_chart(fig, use_container_width=True)
        
            # Display results table
            results_df = pd.DataFrame({
                'Strike': valid_strikes,
                'Implied Volatility': ivs
            })
            results_df['Implied Volatility'] = results_df['Implied Volatility'].map('{:.2%}'.format)
            st.dataframe(results_df)
        else:
            st.warning("No valid implied volatility values calculated")
        
    except Exception as e:
        logger.error(f"Error in main calculation: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()