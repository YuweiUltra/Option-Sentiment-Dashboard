import pandas as pd
from chart_crawler import process_ticker_chart, Parallel, delayed
from config import cpus
from utils import get_close_price, get_latest_trading_day
from tqdm import tqdm

# Get the latest trading date
latest_trading_date, _ = get_latest_trading_day()

# Read SP500 tickers from CSV
df = pd.read_csv('./sp500_companies.csv')
sp500_tickers = list(df.Symbol.unique())

# Show progress while getting close price for SP500 tickers
for ticker in tqdm(sp500_tickers, desc="Processing SP500 close prices"):
    df = get_close_price(ticker, latest_trading_date)

# Parallel processing with progress bar for SP500 tickers
n_jobs = min(len(sp500_tickers), cpus)
Parallel(n_jobs=n_jobs)(
    delayed(process_ticker_chart)(ticker) for ticker in tqdm(sp500_tickers, desc="Processing ticker contracts")
)
