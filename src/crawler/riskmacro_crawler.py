import re
from playwright.sync_api import Playwright, sync_playwright, expect, TimeoutError
import numpy as np
import pandas as pd
import os
from datetime import datetime
from config import RISKMACRO_DATA_DIR


def run(playwright: Playwright) -> None:
    name_url = {
        "标普": 523,
        '沪深300': 518,
        "10yr美债": 527,
        "欧元": 529,
        "英镑": 533,
        "原油": 543,
        "活牛": 551,
        "大豆": 547,
        "玉米": 549,
        "30yr美债": 1234,
        "铜": 541,
        "澳元": 535,
        "纳斯达克": 525,
        "日元": 531,
        "天然气": 545,
        "黄金": 537,
        "白银": 539
    }

    today = datetime.today().strftime('%Y%m%d')
    data_folder = os.path.join(RISKMACRO_DATA_DIR, today)
    os.makedirs(data_folder, exist_ok=True)

    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://riskmacro.com/")

    # Login Process
    page.get_by_role("button", name="登录").click()
    page.get_by_label("登录用户名 用作登录：字母或数字的组合，最少6").fill("zkn2929")
    page.get_by_label("密码", exact=True).fill("18505892232")
    page.get_by_role("button", name="快速登录").click()
    page.get_by_role("link", name="指数 独家").click()

    for name, url in name_url.items():
        file_path = os.path.join(data_folder, f"{name}_{today}.csv")

        # Check if the result file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping scraping for '{name}'.")
            continue  # Skip to the next name

        print(f"Starting scraping for '{name}'...")

        try:
            page.goto(f"https://riskmacro.com/{url}.html")
            page.get_by_text("Delta-Neutral").click()
            page.get_by_text("Gamma-Impulse").nth(1).click()
            page.get_by_text("Gamma Range(Down)", exact=True).click()
            page.get_by_text("Gamma-Range(Up)").click()
            page.get_by_text("Vol(ATM)").nth(1).click()
            page.get_by_text("Vol(15C-15P)").nth(1).click()

            # --- Start of Conditional Clicking ---
            delta_adjust_locator = page.get_by_text("Delta调整后市价")

            if delta_adjust_locator.count() > 0:
                # Click on the second occurrence of "Delta调整后市价"
                delta_adjust_locator.nth(1).click()
                print("Clicked on 'Delta调整后市价'")
            else:
                # If "Delta调整后市价" does not exist, click on "Delta调整后市场价格"
                delta_adjust_market_locator = page.get_by_text("Delta调整后市场价格")
                if delta_adjust_market_locator.count() > 0:
                    delta_adjust_market_locator.nth(1).click()
                    print("Clicked on 'Delta调整后市场价格'")
                else:
                    # Handle the scenario where neither text exists
                    print("Neither 'Delta调整后市价' nor 'Delta调整后市场价格' was found on the page.")
                    # Optionally, skip to the next name or handle as needed
                    continue  # Here, we choose to skip to the next name
            # --- End of Conditional Clicking ---

            page.wait_for_selector("g.highcharts-series")

            gamma_skew = []
            price = []

            bars = page.locator(
                "g.highcharts-series.highcharts-series-5.highcharts-column-series.highcharts-color-3.highcharts-tracker.highcharts-dense-data > path.highcharts-point"
            )
            bar_count = bars.count()
            print(f"Total number of bars for {name}: {bar_count}")

            for i in range(bar_count):
                date_pattern = r'\d{4}/\d{2}/\d{2}'
                value_pattern = r'Gamma-Skew:\s*(-?\d+(?:\.\d+)?)'
                bar = bars.nth(i)
                if not bar.is_visible():
                    print(f"Bar {i + 1} is not visible, skipping.")
                    continue
                try:
                    bar.click()
                    page.wait_for_selector("g.highcharts-label.highcharts-tooltip", timeout=2000)
                    tooltip = page.locator("g.highcharts-label.highcharts-tooltip")
                    if tooltip.is_visible():
                        data_text = tooltip.text_content().strip()

                        date_match = re.search(date_pattern, data_text)
                        date = date_match.group(0) if date_match else None

                        value_match = re.search(value_pattern, data_text)
                        value = value_match.group(1) if value_match else None
                        if not (date is None or value is None):
                            gamma_skew.append({'date': date, 'gamma skew': float(value)})
                        # print(f"Data point {i + 1}: {data_text}")
                    else:
                        print(f"No data at bar {i + 1}")
                except Exception as e:
                    print(f"Error processing bar {i + 1}: {e}")

            page.get_by_text("Gamma-Skew").nth(1).click()
            chart_area = page.locator("g.highcharts-series-group")
            bounding_box = chart_area.bounding_box()

            if bounding_box:
                x_start = bounding_box['x']
                x_end = x_start + bounding_box['width']
                y_start = bounding_box['y']
                y_end = bounding_box['y'] + bounding_box['height']
                num_samples = bar_count * 3
                x_positions = [x_start + i * (x_end - x_start) / num_samples for i in range(num_samples + 1)]
                y_position = (y_end + y_start) / 2 if url == 525 else y_end * 0.8 + y_start * 0.2

                for i, x in enumerate(x_positions):
                    date_pattern = r'\d{4}/\d{2}/\d{2}'
                    value_pattern = r'Settlement Price:\s*([\d\s.]+)'
                    try:
                        # Move the mouse to (x, y_position + 10) and click
                        page.mouse.click(x, y_position)

                        # Optionally, wait for a brief moment after clicking
                        page.wait_for_timeout(100)  # 100 milliseconds

                        # Wait for the tooltip to appear
                        page.wait_for_selector("g.highcharts-label.highcharts-tooltip", timeout=100)
                        tooltip = page.locator("g.highcharts-label.highcharts-tooltip")
                        if tooltip.is_visible():
                            data_text = tooltip.text_content().strip()

                            date_match = re.search(date_pattern, data_text)
                            date = date_match.group(0) if date_match else None

                            value_match = re.search(value_pattern, data_text)
                            value = value_match.group(1) if value_match else None

                            # print(f"Data point {i + 1}: {data_text}")
                            if not (date is None or value is None):
                                value = value.replace(' ', '')
                                price.append({'date': date, 'price': float(value)})

                        else:
                            print(f"No data at position {i + 1}")
                    except TimeoutError:
                        print(f"Timeout while trying to interact at position {i + 1}")
                    except Exception as e:
                        print(f"Error at position {i + 1}: {e}")
            else:
                print("Unable to get chart bounding box.")

            gamma_skew_df = pd.DataFrame(gamma_skew)
            price_df = pd.DataFrame(price)
            price_df = price_df.drop_duplicates()
            df = pd.merge(left=gamma_skew_df, right=price_df, on='date', how='outer')
            df = df.sort_values('date')
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df.to_csv(file_path, index=True, encoding='utf-8-sig')
            print(f"Scraping for '{name}' completed and saved to {file_path}.")

        except Exception as e:
            print(f"An error occurred while scraping '{name}': {e}")

    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
