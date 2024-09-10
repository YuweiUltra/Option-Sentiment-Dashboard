import pandas as pd
from chart_crawler import process_ticker_chart, Parallel, delayed
from config import cpus
from utils import get_close_price, get_latest_trading_day
from tqdm import tqdm
import logging

# Get the latest trading date
latest_trading_date, _ = get_latest_trading_day()

# Configure logging
logging.basicConfig(
    filename=f'./log_files/run_{latest_trading_date}.log',  # Log file name
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Set the logging level
)

# Read SP500 tickers from CSV
df = pd.read_csv('./sp500_companies.csv')
sp500_tickers = list(df.Symbol.unique())

# Show progress while getting close price for SP500 tickers
for ticker in tqdm(sp500_tickers, desc="Processing SP500 close prices"):
    try:
        df = get_close_price(ticker, latest_trading_date)
    except Exception:
        logging.error(f"Cannot get price data for {ticker} on {latest_trading_date}")

# Parallel processing with progress bar for SP500 tickers
n_jobs = min(len(sp500_tickers), cpus)
Parallel(n_jobs=n_jobs)(
    delayed(process_ticker_chart)(ticker) for ticker in tqdm(sp500_tickers, desc="Processing ticker contracts")
)
