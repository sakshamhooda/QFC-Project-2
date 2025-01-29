# Black-Scholes Implied Volatility Calculator

This project implements a Black-Scholes implied volatility calculator with a web-based GUI using Streamlit. It features real-time option data fetching, multiple calculation methods, and interactive visualizations.
<span style="color: red;">You could readily access the application on internet using this link.</span>  
<span style="color: blue;"><a href="https://qfc-project-2-sakshamhooda-varunsaxena.streamlit.app/">Click here to access</a></span>


## Features

- Real-time option data fetching using Yahoo Finance API
- Two calculation methods: Newton-Raphson and Bisection
- Interactive GUI with Streamlit
- Volatility smile visualization
- Comprehensive error handling and logging
- Support for multiple stock symbols and expiration dates

## Project Structure

```
QFC-PROJECT-2/
├── src/
│   ├── black_scholes.py     # Core implementation of BS model
│   └── app.py               # Streamlit GUI application
├── logs/                    # Log files directory
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Setup Instructions

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory:
```bash
cd QFC-PROJECT-2
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. The application will open in your default web browser.

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
2. Select an expiration date from the available options
3. Adjust the risk-free rate if needed
4. Choose between Newton-Raphson and Bisection calculation methods
5. View the volatility smile plot and detailed results table

## Error Handling

- The application includes comprehensive error handling and logging
- Check the `logs` directory for detailed error logs
- The GUI will display user-friendly error messages when issues occur

## Notes

- Option data is fetched in real-time from Yahoo Finance
- Calculations might take a few seconds depending on the number of strike prices
- Some strike prices might be filtered out if they produce invalid implied volatilities
