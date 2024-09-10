from config import base_url, tries, timeout, headless, account, password
from playwright.sync_api import Playwright, sync_playwright, expect
import os
import time
import random
import shutil
import logging
from joblib import Parallel, delayed
from fake_useragent import UserAgent

from utils import get_latest_trading_day

today, _ = get_latest_trading_day()

# Configure logging
logging.basicConfig(
    filename=f'./log_files/run_{today}.log',  # Log file name
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
    base_path = f"./raw_data/{today}/{ticker}"
    fps = [f"{base_path}/OpenInterest.csv", f"{base_path}/ImpliedVolatility.csv", f"{base_path}/GammaExposure.csv",
           f"{base_path}/Greeks.csv",
           f"{base_path}/Volume.csv", f"{base_path}/MaxPain.csv",
           f"{base_path}/NetGammaExposure.csv"]

    # Check if the file already exists
    if all(os.path.exists(fp) for fp in fps):
        print(f"Data for {ticker} already exists. Skipping crawler.")
        return  # Skip the crawling process

    retries = tries  # Number of retries
    for attempt in range(retries):
        time.sleep(random.uniform(0.2, 0.5))
        try:
            ############## open url ################
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
    url = base_url + f"/{ticker}"
    with sync_playwright() as playwright:
        chart_crawler(playwright, url, ticker)


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA', 'GOOG', 'AAPL', 'META', 'MSFT', 'TSLA', 'AMZN', 'TSM', 'MU', 'QQQ']

    n_jobs = min(len(tickers), 1)
    Parallel(n_jobs=n_jobs)(delayed(process_ticker_chart)(ticker) for ticker in tickers)
