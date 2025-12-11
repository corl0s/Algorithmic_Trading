# create_test_data.py
import pickle
import os
import yfinance as yf
import pandas as pd
import sys
import os

# Get the path to the root of the project (deep-rl-trading) 
# by going up two levels from the script's current location (src/rl)
# and add it to the system path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Now your original imports will work:
from src.stock.stock import Stock # Import your Stock class
# ... (rest of the script)

# --- Configuration ---
TEST_TICKERS = ["MSFT", "GOOGL"] # Choose one or more tickers for testing
TEST_PERIOD = "10y" # Get 2 years of data
TEST_TIMEFRAME = "1d"
# window must match the window parameter used in Agent init, which defaults to 364
TEST_WINDOW = 364 
TEST_DATA_DIR = os.path.join("data", "test")
# ---------------------

def create_and_pickle_stock_data(ticker, period, timeframe, window, save_dir):
    print(f"Loading data for {ticker}...")
    
    # Instantiate the Stock class, which handles yfinance download and observation space creation
    try:
        # Note: If yfinance is failing due to auto_adjust change,
        # you might need to update the load_stock function or yfinance library.
        stock_object = Stock(
            ticker=ticker, 
            period=period, 
            timeframe=timeframe, 
            window=window
        )
    except Exception as e:
        print(f"Error creating Stock object for {ticker}: {e}")
        return

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True) 

    # Pickle the Stock object
    save_path = os.path.join(save_dir, f"{ticker}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(stock_object, f)

    print(f"Successfully saved {ticker} Stock object to {save_path}")

if __name__ == '__main__':
    for ticker in TEST_TICKERS:
        create_and_pickle_stock_data(
            ticker, 
            TEST_PERIOD, 
            TEST_TIMEFRAME, 
            TEST_WINDOW, 
            TEST_DATA_DIR
        )