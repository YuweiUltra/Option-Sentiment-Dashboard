import pandas as pd
from crawler.chart_crawler import process_ticker_chart, Parallel, delayed
from config import cpus
from utils import get_latest_trading_day, get_tickers
from tqdm import tqdm

# Get the latest trading date
latest_trading_date, _ = get_latest_trading_day()

tickers = get_tickers()

# Parallel processing with progress bar for SP500 tickers
n_jobs = min(len(tickers), cpus)
Parallel(n_jobs=n_jobs)(
    delayed(process_ticker_chart)(ticker) for ticker in tqdm(tickers, desc="Processing ticker contracts")
)
