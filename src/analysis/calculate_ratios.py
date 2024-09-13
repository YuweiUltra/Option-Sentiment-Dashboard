import os
import pandas as pd
import numpy as np
from src.utils import get_listed_dates, get_tickers
from datetime import datetime
from config import  ANALYSIS_DATA_DIR,RAW_DATA_DIR


def calculate_ratios(rewrite=False):
    """
    Calculate Gamma Exposure Ratio and Open Interest Call-Put Ratio for each ticker on each date.

    Parameters:
    - rewrite (bool): If True, force recalculation and overwrite existing file.
                      If False, read existing file if it exists.

    Returns:
    - pd.DataFrame: DataFrame containing the calculated ratios.
    """
    # Get today's date in YYYYMMDD format for the filename
    today_str = datetime.today().strftime('%Y%m%d')
    output_dir = ANALYSIS_DATA_DIR
    output_filename = f"ratio_{today_str}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If the file exists and rewrite is False, read and return the existing file
    if not rewrite and os.path.exists(output_path):
        try:
            skew_df = pd.read_csv(output_path)
            print(f"Existing file found. Reading from {output_path}.")
            return skew_df
        except Exception as e:
            print(f"Error reading the existing file: {e}. Proceeding to recalculate.")

    # Get the list of dates and tickers from utils
    dates = get_listed_dates()
    tickers = get_tickers()
    tickers.sort()

    # Initialize a list to collect the results
    results = []

    # Iterate over each date
    for date in dates:
        # Iterate over each ticker
        for ticker in tickers:
            # Initialize default values as NaN
            gamma_exposure_ratio = np.nan
            open_interest_call_put_ratio = np.nan

            # Define the file paths
            open_interest_path = os.path.join(RAW_DATA_DIR, date, ticker, 'OpenInterest.csv')
            greeks_path = os.path.join(RAW_DATA_DIR, date, ticker, 'Greeks.csv')

            # Check if both files exist
            if not os.path.exists(open_interest_path) or not os.path.exists(greeks_path):
                # If any file is missing, append NaN for this ticker and date
                results.append({
                    'date': date,
                    'ticker': ticker,
                    'gamma_exposure_ratio': gamma_exposure_ratio,
                    'open_interest_call_put_ratio': open_interest_call_put_ratio
                })
                continue  # Move to the next ticker

            try:
                # Read OpenInterest.csv
                df_oi = pd.read_csv(open_interest_path)
                # Read Greeks.csv
                df_greeks = pd.read_csv(greeks_path)

                # Ensure necessary columns exist
                if not {'contract_symbol', 'option_type', 'open_interest'}.issubset(df_oi.columns):
                    raise ValueError("Missing columns in OpenInterest.csv")
                if not {'contract_symbol', 'gamma'}.issubset(df_greeks.columns):
                    raise ValueError("Missing columns in Greeks.csv")

                # Merge OpenInterest and Greeks on 'contract_symbol'
                df = pd.merge(df_oi, df_greeks[['contract_symbol', 'gamma']], on='contract_symbol', how='inner')

                # Calculate gamma_exposure
                df['gamma_exposure'] = df['open_interest'] * df['gamma']

                # Calculate Gamma Exposure Ratio
                # Group by strike and option_type
                df_grouped = df.groupby(['strike', 'option_type'])['gamma_exposure'].sum().reset_index()

                # Pivot to have CALL and PUT in separate columns
                df_pivot = df_grouped.pivot(index='strike', columns='option_type', values='gamma_exposure').fillna(0)

                # Calculate gamma_exposure_diff = CALL - PUT
                df_pivot['gamma_exposure_diff'] = df_pivot.get('CALL', 0) - df_pivot.get('PUT', 0)

                # Sum of gamma_exposure_diff
                sum_diff = df_pivot['gamma_exposure_diff'].sum()

                # Sum of absolute gamma_exposure_diff
                sum_abs_diff = df_pivot['gamma_exposure_diff'].abs().sum()

                # Calculate the ratio, handle division by zero
                if sum_abs_diff != 0:
                    gamma_exposure_ratio = round(sum_diff / sum_abs_diff, 2)
                else:
                    gamma_exposure_ratio = np.nan

                # Calculate Open Interest Call-Put Ratio
                sum_call_oi = df[df['option_type'] == 'CALL']['open_interest'].sum()
                sum_put_oi = df[df['option_type'] == 'PUT']['open_interest'].sum()

                if sum_put_oi != 0:
                    open_interest_call_put_ratio = round(sum_call_oi / sum_put_oi, 2)
                else:
                    open_interest_call_put_ratio = np.nan

            except Exception as e:
                # In case of any error, set ratios as NaN and optionally log the error
                print(f"Error processing date {date}, ticker {ticker}: {e}")
                gamma_exposure_ratio = np.nan
                open_interest_call_put_ratio = np.nan

            # Append the results
            results.append({
                'date': date,
                'ticker': ticker,
                'gamma_exposure_ratio': gamma_exposure_ratio,
                'open_interest_call_put_ratio': open_interest_call_put_ratio
            })

    # Create a DataFrame from the results
    skew_df = pd.DataFrame(results)

    try:
        # Save the DataFrame to the specified CSV file
        skew_df.to_csv(output_path, index=False)
        print(f"{output_path} has been successfully created.")
    except Exception as e:
        print(f"Error saving the file {output_path}: {e}")

    return skew_df


if __name__ == '__main__':
    # Example usage:
    # To calculate and overwrite existing file:
    # df = calculate_ratios(rewrite=True)

    # To read existing file if it exists, else calculate:
    df = calculate_ratios(rewrite=True)
    print(df.head())
