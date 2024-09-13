from config import base_url, tries, timeout, headless, account, password, LOG_DIR, RAW_DATA_DIR
from playwright.sync_api import Playwright, sync_playwright, expect
import os
import time
import json
import random
import shutil
import logging
from joblib import Parallel, delayed
from fake_useragent import UserAgent

from src.utils import get_latest_trading_day, get_close_price

today, _ = get_latest_trading_day()

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f'run_{today}.log'),  # Log file name
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Set the logging level
)


def close_modal_if_visible(page):
    heading = page.get_by_role("heading", name="Get more with OptionCharts")
    if heading.is_visible():
        close_button = page.locator("#requires_subscription_ad_modal_id").get_by_label("Close")
        while close_button.is_visible():
            close_button.click()
            page.reload()


def chart_crawler(playwright: Playwright, url, ticker) -> None:
    # File path where the data will be stored
    base_path = os.path.join(RAW_DATA_DIR, f"{today}/{ticker}")
    fps = [f"{base_path}/OpenInterest.csv", f"{base_path}/ImpliedVolatility.csv", f"{base_path}/GammaExposure.csv",
           f"{base_path}/Greeks.csv",
           f"{base_path}/Volume.csv", f"{base_path}/MaxPain.csv",
           f"{base_path}/overview_table1.json", f"{base_path}/overview_table2.json",
           f"{base_path}/NetGammaExposure.csv", ]

    # Check if the file already exists
    if all(os.path.exists(fp) for fp in fps):
        print(f"Data for {ticker} already exists. Skipping crawler.")
        return  # Skip the crawling process

    retries = tries  # Number of retries
    for attempt in range(retries):
        time.sleep(random.uniform(0.2, 0.5))
        try:
            ############## user agent ################
            ua = UserAgent()
            user_agent = ua.random

            # Set headers with the random User-Agent
            headers = {
                "User-Agent": user_agent,
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }

            ############## open url ################
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent=user_agent,
                extra_http_headers=headers,
                accept_downloads=True,
            )
            page = context.new_page()
            page.goto("https://optioncharts.io")
            page.get_by_role("link", name="Log in").click()
            page.get_by_label("Email address").click()
            page.get_by_label("Email address").fill(account)
            page.get_by_label("Password").click()
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Sign in").click()
            page.goto(url, timeout=timeout)

            ############## abstract overview info ################
            expect(page.get_by_role("strong")).to_contain_text(ticker)
            tables = page.query_selector_all('table')
            assert len(tables) == 4
            data_1 = []
            for table in tables[:3]:
                rows = table.query_selector_all('tr')
                for row in rows:
                    heads = row.query_selector_all('th')
                    key = [head.inner_text() for head in heads]
                    cells = row.query_selector_all('td')
                    value = [cell.inner_text() for cell in cells]
                    if key and value:
                        data_1.append({key[0]: value[0]})

            data_2 = []
            keys = ['Expiration', 'Volume Calls', 'Volume Puts', 'Volume Put-Call Ratio', 'Open Interest Calls',
                    'Open Interest Puts', 'Open Interest Put-Call Ratio', 'Implied Volatility', 'Max Pain',
                    'Max Pain vs Current Price']
            rows = tables[-1].query_selector_all('tr')
            for row in rows[2:]:
                cells = row.query_selector_all('td')
                value = [cell.inner_text() for cell in cells]
                if len(value) == len(keys):
                    row_dict = {keys[i]: value[i] for i in range(len(keys))}
                    data_2.append(row_dict)

            # Create directories if they don't exist
            os.makedirs(base_path, exist_ok=True)
            with open(fps[-3], 'w') as file:
                json.dump(data_1, file, indent=4)
            with open(fps[-2], 'w') as file:
                json.dump(data_2, file, indent=4)

            ############## go to charts url#############

            page.get_by_role("link", name="Option Charts").click()
            expect(page.locator("#expiration-dates-form-button-1")).to_contain_text("Expiration Dates")
            expect(page.get_by_role("strong")).to_contain_text(ticker)

            ############## extract data ################

            close_modal_if_visible(page)
            page.get_by_role("button", name="Expiration Dates").click()
            close_modal_if_visible(page)
            page.get_by_role("button", name="Select All").click()
            close_modal_if_visible(page)
            page.get_by_role("button", name="Done").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[0]), exist_ok=True)
            shutil.move(download_path, fps[0])
            print(f"Downloaded file saved at: {fps[0]}")

            ############## extract data ################
            page.get_by_role("link", name="Volatility Skew").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[1]), exist_ok=True)
            shutil.move(download_path, fps[1])
            print(f"Downloaded file saved at: {fps[1]}")

            ############## extract data ################
            page.get_by_role("link", name="Gamma Exposure (GEX)").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[2]), exist_ok=True)
            shutil.move(download_path, fps[2])
            print(f"Downloaded file saved at: {fps[2]}")

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").nth(1).click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[-1]), exist_ok=True)
            shutil.move(download_path, fps[-1])
            print(f"Downloaded file saved at: {fps[-1]}")

            ############## extract data ################
            page.get_by_role("link", name="Greeks").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[3]), exist_ok=True)
            shutil.move(download_path, fps[3])
            print(f"Downloaded file saved at: {fps[3]}")

            ############## extract data ################
            page.get_by_role("link", name="Volume").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[4]), exist_ok=True)
            shutil.move(download_path, fps[4])
            print(f"Downloaded file saved at: {fps[4]}")

            ############## extract data ################
            page.get_by_role("link", name="Max Pain").click()
            close_modal_if_visible(page)

            with page.expect_download() as download_info:
                page.locator("[id^='csv_download_']").first.click()
                close_modal_if_visible(page)
            download = download_info.value
            download_path = download.path()

            # Move the downloaded file to the desired location
            os.makedirs(os.path.dirname(fps[5]), exist_ok=True)
            shutil.move(download_path, fps[5])
            print(f"Downloaded file saved at: {fps[5]}")

            # Close context and browser
            context.close()
            browser.close()

            # Print success message
            logging.info(f"Successfully processed {ticker}")
            break  # Exit loop if successful
        except (AssertionError, Exception) as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(0.5)  # Wait before retrying
                continue
            else:
                logging.error(f"Failed to process {ticker} after {retries} attempts.")


def process_ticker_chart(ticker):
    try:
        get_close_price(ticker, today)
    except Exception:
        logging.error(f"Cannot get price data for {ticker} on {today}")

    url = base_url + f"/{ticker}"
    with sync_playwright() as playwright:
        chart_crawler(playwright, url, ticker)


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA', 'GOOG', 'AAPL', 'META', 'MSFT', 'TSLA', 'AMZN', 'TSM', 'MU', 'QQQ']

    n_jobs = min(len(tickers), 1)
    Parallel(n_jobs=n_jobs)(delayed(process_ticker_chart)(ticker) for ticker in tickers)
