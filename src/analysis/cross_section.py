# src/analysis/cross_section.py

from dash import dcc, html, dash_table
import warnings
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from copy import deepcopy
from src.utils import (
    get_latest_trading_day,
    get_close_price,
    get_listed_dates,
    get_tickers
)
import statsmodels.api as sm
import plotly.express as px
from config import INDEX_DATA_DIR, RAW_DATA_DIR, ANALYSIS_DATA_DIR  # Ensure ANALYSIS_DATA_DIR is defined
# from calculate_ratios import calculate_ratios  # Uncomment if needed

warnings.filterwarnings("ignore")


def cal_gamma_skew(date, ticker):
    """
    Calculate the gamma skew and open interest for a given date and ticker.
    """
    try:
        price = get_close_price(ticker, date)
        lower_bound = price['Close'] * 0
        upper_bound = price['Close'] * 1000

        # Read CSV files
        df_oi = pd.read_csv(os.path.join(RAW_DATA_DIR, date, ticker, 'OpenInterest.csv'))
        df_greeks = pd.read_csv(os.path.join(RAW_DATA_DIR, date, ticker, 'Greeks.csv'))

        # Merge DataFrames
        df = pd.merge(df_oi, df_greeks[['contract_symbol', 'gamma']], on='contract_symbol')
        df['gamma_exposure'] = df['open_interest'] * df['gamma']
        df['date_str'] = df['contract_symbol'].str.extract(r'(\d{6})')
        df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d')
        df['expiration'] = df['date'].dt.strftime('%Y-%m-%d')

        # Filter expirations and strikes
        expirations = sorted(df['expiration'].unique().tolist())
        expirations_formatted = pd.to_datetime(expirations).strftime('%y%m%d').tolist()
        df_filtered = df[df['contract_symbol'].str.contains('|'.join(expirations_formatted))]
        df_filtered = df_filtered[
            (df_filtered['strike'] >= lower_bound) & (df_filtered['strike'] <= upper_bound)
        ]

        # Calculate skew if open interest threshold is met
        if df_filtered['open_interest'].sum() > 200000:
            df_grouped_diff = df_filtered.groupby('strike').apply(
                lambda x: x[x['option_type'] == 'CALL']['gamma_exposure'].sum() -
                          x[x['option_type'] == 'PUT']['gamma_exposure'].sum()
            ).reset_index(name='gamma_exposure_diff').dropna()

            skew = df_grouped_diff['gamma_exposure_diff'].sum() / df_grouped_diff['gamma_exposure_diff'].abs().sum()

            return skew, df_filtered['open_interest'].sum()
        else:
            return np.nan, np.nan
    except Exception as e:
        print(f"Error in cal_gamma_skew for {ticker} on {date}: {e}")
        return np.nan, np.nan


def prepare_skew_data(tickers, dates):
    """
    Prepare skew data for regression analysis.
    """
    skew_data = []
    dates = sorted(dates)
    for date in dates:
        for ticker in tickers:
            skew, open_interest = cal_gamma_skew(date, ticker)
            try:
                price = get_close_price(ticker, date, local=True)
            except:
                continue
            skew_data.append(
                {"date": date, "ticker": ticker, "open interest": open_interest, "skew": skew,
                 'return': price['return']}
            )
    skew_df = pd.DataFrame(skew_data)

    # Shift data for regression
    temp_df = deepcopy(skew_df)
    temp_df['return'] = temp_df.groupby('ticker')['return'].shift(-1)
    temp_df['current skew'] = temp_df.groupby('ticker')['skew'].shift(-1)
    temp_df['skew change'] = temp_df['current skew'] - temp_df['skew']
    temp_df.dropna(inplace=True)

    return temp_df


def perform_regression(temp_df):
    """
    Perform OLS regression on skew changes vs. returns.
    """
    X = temp_df[['skew change']]
    X = sm.add_constant(X)
    y = temp_df['return']

    model = sm.OLS(y, X).fit()

    temp_df['predicted_return'] = model.predict(X)
    temp_df['residual'] = np.abs(temp_df['return'] - temp_df['predicted_return'])

    return model, temp_df


def create_cross_section_figures(temp_df, model, top_n=5):
    """
    Create Plotly figures and prepare data for tables.
    """
    # Scatter Plot with Regression Line
    fig = px.scatter(
        temp_df,
        x='skew change',
        y='return',
        text='ticker',
        labels={'skew change': 'Skew Changes', 'return': 'Return'},
        hover_data={'ticker': True, 'skew change': True, 'return': True, 'residual': True},
        height=800
    )

    # Add Regression Line
    fig.add_traces(go.Scatter(
        x=temp_df['skew change'],
        y=temp_df['predicted_return'],
        mode='lines',
        name='Regression Line',
        line=dict(color='green', width=2)
    ))

    # Top N Residuals
    top_residuals = temp_df.nlargest(top_n, 'residual')
    fig.add_traces(go.Scatter(
        x=top_residuals['skew change'],
        y=top_residuals['return'],
        mode='markers',
        marker=dict(color='red', size=12),
        name=f'Top {top_n} Residuals',
        textposition='top center'
    ))

    # Regression Text Annotation
    regression_text = (
        f"Intercept: {model.params['const']:.4f}<br>"
        f"Slope: {model.params['skew change']:.4f}<br>"
        f"R-squared: {model.rsquared:.4f}<br>"
        f"P-value: {model.pvalues['skew change']:.4f}<br>"
        f"Data points: {len(temp_df)}"
    )

    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",  # Position to upper-left
        text=regression_text,
        showarrow=False,
        font=dict(size=12)
    )

    fig.update_traces(textposition='top center',
                      hovertemplate='<b>Current skew: %{customdata[0]}</b><br>Skew Change: %{x}<br>Return: %{y}<br>Residual: %{customdata[1]}<extra></extra>',
                      customdata=np.stack((temp_df['current skew'], temp_df['residual']), axis=-1))
    fig.update_layout(showlegend=True)

    # Prepare Tables
    temp_df=temp_df.sort_values(['skew change','current skew'])
    df_head = temp_df.head(10).to_dict('records')
    df_tail = temp_df.tail(10).to_dict('records')
    specific_tickers = ['NVDA', 'META', 'GOOG', 'GOOGL', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'MU', 'TSM']
    df_specific = temp_df[temp_df['ticker'].isin(specific_tickers)].sort_values(
        ['current skew', 'skew change', 'open interest'], ascending=False
    ).to_dict('records')

    return fig, df_head, df_tail, df_specific, model


def create_cross_section_content(date, dates, tickers, rewrite=False):
    """
    Create cross-section analysis content, save to HTML, and return Dash components.

    Parameters:
    - date (str): The selected date for analysis.
    - dates (list): List of available dates.
    - tickers (list): List of tickers.
    - rewrite (bool): Whether to regenerate the HTML file even if it exists.

    Returns:
    - html.Div or html.Iframe: Dash component containing the analysis.
    """
    # Define the HTML file path
    file_path = os.path.join(ANALYSIS_DATA_DIR, f'cross_section_{date}.html')

    # Check if the HTML file exists and rewrite is False
    if not rewrite and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                html_content = f.read()
            return html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '1200px', 'border': 'none'})
        except Exception as e:
            print(f"Error loading existing HTML file: {e}. Regenerating content.")
            # Proceed to regenerate if loading fails

    # Generate the analysis content
    selected_dates = [date]
    # Optionally, include the previous date for comparison
    try:
        current_index = dates.index(date)
        if current_index > 0:
            selected_dates.append(dates[current_index - 1])
    except ValueError:
        print(f"Selected date {date} not found in dates list.")

    temp_df = prepare_skew_data(tickers, selected_dates)
    if temp_df.empty:
        return html.Div("Insufficient data to perform cross-section analysis.")

    model, temp_df = perform_regression(temp_df)
    fig, df_head, df_tail, df_specific, model = create_cross_section_figures(temp_df, model)

    # Convert figure to HTML
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Convert tables to HTML using Pandas
    df_head_html = pd.DataFrame(df_head).to_html(index=False)
    df_tail_html = pd.DataFrame(df_tail).to_html(index=False)
    df_specific_html = pd.DataFrame(df_specific).to_html(index=False)

    # Combine all HTML content
    full_html = f"""
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h2>Cross-Section Analysis for {date}</h2>
            {fig_html}
            <h3>Top 10 Records</h3>
            {df_head_html}
            <h3>Bottom 10 Records</h3>
            {df_tail_html}
            <h3>Selected Tickers</h3>
            {df_specific_html}
        </body>
    </html>
    """

    # Save the HTML content to file
    try:
        with open(file_path, 'w') as f:
            f.write(full_html)
        print(f"Cross-section analysis saved to {file_path}")
    except Exception as e:
        print(f"Error saving HTML file: {e}")

    # Return the HTML content embedded in an Iframe
    return html.Iframe(srcDoc=full_html, style={'width': '100%', 'height': '1200px', 'border': 'none'})