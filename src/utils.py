import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
import yfinance as yf
import os
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pytz
from functools import lru_cache
from config import RAW_DATA_DIR, INDEX_DATA_DIR


def overview(dates, tickers):
    all_dicts = []
    if not isinstance(dates, list):
        dates = [dates]
    for date in dates:
        for ticker in tickers:
            try:
                with open(os.path.join(RAW_DATA_DIR, f"{date}/{ticker}/table1.json"), "r") as fp:
                    dicts = json.load(fp)
                    ticker_dict = {'ticker': ticker, 'date': date}
                    for d in dicts:
                        ticker_dict.update(d)
                    all_dicts.append(ticker_dict)
            except:
                pass
    df = pd.DataFrame(all_dicts).set_index(['date', 'ticker'])
    return df


def overview_by_expiration(dates, tickers):
    all_dicts = []
    if not isinstance(dates, list):
        dates = [dates]
    for date in dates:
        for ticker in tickers:
            try:
                with open(os.path.join(RAW_DATA_DIR, f"{date}/{ticker}/table2.json"), "r") as fp:
                    dicts = json.load(fp)
                    for d in dicts:
                        d['ticker'] = ticker
                        d['date'] = date
                    all_dicts.extend(dicts[:-1])
            except:
                pass
    df = pd.DataFrame(all_dicts).set_index(['date', 'ticker'])
    return df


def contract(dates, tickers):
    all_dicts = []
    if not isinstance(dates, list):
        dates = [dates]
    for date in dates:
        for ticker in tickers:
            try:
                with open(os.path.join(RAW_DATA_DIR, f"{date}/{ticker}/table3.json"), "r") as fp:
                    dicts = json.load(fp)
                    for d in dicts:
                        d['ticker'] = ticker
                        d['date'] = date
                    all_dicts.extend(dicts)
            except:
                pass
    df = pd.DataFrame(all_dicts).set_index(['date', 'ticker', 'maturity date'])

    df['Open Interest'] = df['Open Interest'].str.replace(',', '').astype(int)
    df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
    for col in df.columns:
        try:
            df[col] = df[col].str.replace(',', '')
        except:
            pass
        if col != 'type':
            if col == "Implied Volatility":
                df[col] = df[col].str.replace('%', '').astype(float) / 100
            df[col] = pd.to_numeric(df[col])
    return df


def calculate_days_to_maturity(df, current_date):
    current_date = pd.to_datetime(current_date)
    df['maturity date modified'] = pd.to_datetime(df['maturity date'].apply(lambda x: x.split(":")[0]))
    df['Days to Maturity'] = (df['maturity date modified'] - current_date).dt.days
    df.drop(columns=['maturity date modified'], inplace=True)
    return df


def get_latest_trading_day():
    eastern = pytz.timezone('US/Eastern')
    nyse = mcal.get_calendar('NYSE')
    current_time = datetime.now(eastern)

    # Get the last 30 days of valid trading days from NYSE
    schedule = nyse.valid_days((current_time - timedelta(days=30)).strftime('%Y-%m-%d'),
                               current_time.strftime('%Y-%m-%d'))

    last_trading_day = schedule[-1]

    market_close_time = datetime.combine(last_trading_day.date(), nyse.close_time, tzinfo=eastern)

    if current_time < market_close_time:
        latest_trading_day = schedule[-2]
        pre_latest_trading_day = schedule[-3]
    else:
        latest_trading_day = schedule[-1]
        pre_latest_trading_day = schedule[-2]

    return latest_trading_day.date().strftime('%Y-%m-%d'), pre_latest_trading_day.date().strftime('%Y-%m-%d')


def generate_surface_plots(df, ticker, z_axes):
    df_ticker = df[df['ticker'] == ticker]
    surface_plots = []

    for z_axis in z_axes:
        for option_type in ['call', 'put']:
            df_type = df_ticker[df_ticker['type'] == option_type]

            if df_type.empty:
                continue

            df_pivot = df_type.pivot(index='strike', columns='Days to Maturity', values=z_axis)
            df_pivot = df_pivot.interpolate(method='linear', axis=1).fillna(method='bfill').fillna(method='ffill')

            z = df_pivot.values.T
            x = df_pivot.index.values
            y = df_pivot.columns.values

            fig = go.Figure(data=[go.Surface(
                z=z,
                x=x,
                y=y,
                colorbar=dict(title=z_axis)
            )])

            fig.update_layout(
                title=f'{z_axis} Surface for {ticker} ({option_type.capitalize()})',
                scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Days to Maturity',
                    zaxis_title=z_axis,
                    xaxis=dict(showgrid=True, zeroline=True, showline=True, mirror=True, ticks='outside',
                               showticklabels=True, range=[x.min(), x.max()]),
                    yaxis=dict(showgrid=True, zeroline=True, showline=True, mirror=True, ticks='outside',
                               showticklabels=True, range=[y.min(), y.max()]),
                    zaxis=dict(showgrid=True, zeroline=True, showline=True, mirror=True, ticks='outside',
                               showticklabels=True),
                ),
                margin=dict(l=65, r=50, b=65, t=90),
                template="plotly_white",
                width=800,  # Adjust width to ensure plots fit side by side
                height=800,  # Adjust height to keep proportions
            )

            surface_plots.append(dcc.Graph(figure=fig))

    return surface_plots


def generate_expiration_plots(expiration_summary, ticker, columns):
    plots = []
    df_ticker = expiration_summary[expiration_summary['ticker'] == ticker]

    for col in columns:
        fig = px.line(
            df_ticker,
            x='maturity date',
            y=col,
            color='type',
            title=f'{col} for {ticker} by Maturity Date',
            labels={'maturity date': 'Maturity Date', col: col},
            color_discrete_map={'call': 'red', 'put': 'green'},  # Specify colors for call and put
        )

        fig.update_traces(marker=dict(size=10), line=dict(width=2))
        fig.update_layout(
            title=dict(x=0.5),
            xaxis_title="Maturity Date",
            yaxis_title=col,
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        plots.append(dcc.Graph(figure=fig))  # Return the figure wrapped in dcc.Graph

    return plots


def generate_strike_plots(df_strike, ticker, columns):
    plots = []
    df_ticker = df_strike[df_strike['ticker'] == ticker]

    for col in columns:
        if col == "Implied Volatility":
            df_ticker = df_ticker[df_ticker[col] != 0]

        maturity_dates = df_ticker['maturity date'].unique()
        for date in maturity_dates:
            df_ticker_date = df_ticker[df_ticker['maturity date'] == date]

            fig = px.line(
                df_ticker_date,
                x='strike',
                y=col,
                color='type',
                title=f'{col} for {ticker} by Strike with Maturity Date {date}',
                labels={'strike': 'Strike', col: col},
                color_discrete_map={'call': 'red', 'put': 'green'},  # Specify colors for call and put
            )

            fig.update_traces(marker=dict(size=10), line=dict(width=2))
            fig.update_layout(
                title=dict(x=0.5),
                xaxis_title="Strike",
                yaxis_title=col,
                template="plotly_white",
                font=dict(size=14),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=800,
                height=400,
            )
            plots.append(dcc.Graph(figure=fig))  # Return the figure wrapped in dcc.Graph

    return plots


def return_plots(close_prices):
    fig = px.bar(close_prices, x=close_prices.index, y='return', title='Stock Returns',
                 labels={'return': 'Return', 'ticker': 'Stock Ticker'})

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Value",
        legend_title="Metrics",
        title_x=0.5,
        template="plotly_white",
        font=dict(size=14),
        # width=1600,
        height=700,
        # xaxis=dict(
        #     rangeslider=dict(visible=True),  # Enable the range slider
        #     tickmode='linear',  # Ensure all tickers are shown
        # )
    )
    return dcc.Graph(figure=fig)


def generate_call_put_ratio_plot(expiration_summary=None, df_melted=None, add_one=True):
    if df_melted is None:
        sum_daily = expiration_summary.groupby(['ticker', 'type']).sum()

        def calculate_call_put_ratio(group):
            call_values = group[group.index.get_level_values('type') == 'call'].sum()
            put_values = group[group.index.get_level_values('type') == 'put'].sum()
            call_put_ratio = call_values / put_values.replace(0, float('nan'))
            return call_put_ratio

        call_put_ratios = sum_daily.groupby('ticker').apply(calculate_call_put_ratio)
        call_put_ratios = call_put_ratios.abs()
        call_put_ratios = call_put_ratios.reset_index()
        call_put_ratios = call_put_ratios[['ticker', 'Gamma']]
        # ['ticker', 'Volume', 'Open Interest', 'Delta', 'Gamma', 'Avg Gamma', 'Avg IV']]
        df_melted = call_put_ratios.melt(id_vars="ticker", var_name="Metric", value_name="Value")

    fig = px.bar(df_melted, x='ticker', y='Value', color='Metric', barmode='group',
                 title="CALL-PUT RATIO: Metrics Comparison Across Tickers",
                 labels={'Value': 'Value', 'ticker': 'Ticker'},
                 color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
                                          '#B6E880'])
    if add_one:
        shapes = [
            dict(
                type="line",
                x0=-0.5,
                y0=1,
                x1=len(df_melted['ticker'].unique()) - 0.5,
                y1=1,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
        ]
    else:
        shapes = []
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Value",
        legend_title="Metrics",
        title_x=0.5,
        template="plotly_white",
        font=dict(size=14),
        shapes=shapes,
        # width=1600,
        height=700,
    )
    return dcc.Graph(figure=fig), df_melted


def get_close_price(ticker, date, rewrite=False, local=False):
    dir_path = os.path.join(RAW_DATA_DIR, f"{date}/{ticker}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    json_file_path = os.path.join(dir_path, "close_price.json")

    if os.path.exists(json_file_path) and not rewrite:
        with open(json_file_path, 'r') as json_file:
            close_price_data = json.load(json_file)
        return close_price_data
    else:
        if not local:
            date_time = datetime.strptime(date, '%Y-%m-%d')
            data = yf.download(ticker, start=date_time - timedelta(days=30))
            data['return'] = data["Adj Close"].pct_change()
            close_price = data.loc[date, :]

            close_price_data = close_price.to_dict()
            with open(json_file_path, 'w') as json_file:
                json.dump(close_price_data, json_file)

            return close_price_data
        else:
            raise Exception('no data found')


@lru_cache(maxsize=10)
def prepare_data(date, tickers_tuple):
    tickers = list(tickers_tuple)

    df1, df2, df3 = overview(date, tickers), overview_by_expiration(date, tickers), contract(date, tickers)
    df3[['Gamma*Open Interest', 'Delta*Open Interest', 'Implied Volatility*Open Interest']] = df3[
        ['Gamma', 'Delta', 'Implied Volatility']].multiply(df3['Open Interest'], axis=0)

    grouped = df3.groupby(['ticker', 'date', 'maturity date', 'type'])
    summary = grouped.agg({
        'Volume': 'sum',
        'Gamma*Open Interest': 'sum',
        'Delta*Open Interest': 'sum',
        'Open Interest': 'sum',
        'Implied Volatility*Open Interest': 'sum'}).reset_index()

    summary['Avg IV'] = summary['Implied Volatility*Open Interest'] / summary['Open Interest']
    summary['Avg Gamma'] = summary['Gamma*Open Interest'] / summary['Open Interest']
    summary['Avg Delta'] = summary['Delta*Open Interest'] / summary['Open Interest']

    summary = summary.rename(columns={
        'Gamma*Open Interest': 'Gamma',
        'Delta*Open Interest': 'Delta'})

    expiration_summary = summary[
        ['ticker', 'date', 'maturity date', 'type', 'Volume', 'Open Interest', 'Gamma', 'Delta', 'Avg IV', 'Avg Gamma',
         'Avg Delta']]

    # Process surface and strike data efficiently by avoiding deepcopy and redundant reset_index calls
    df_surface = df3.loc[date].reset_index()
    df_surface = calculate_days_to_maturity(df_surface, date)
    df_strike = df3.loc[date].reset_index()

    return df_strike, df_surface, expiration_summary, df1, df2, df3


def get_listed_dates():
    dates = []
    for dir_name in os.listdir(RAW_DATA_DIR):
        full_path = os.path.join(RAW_DATA_DIR, dir_name)
        if os.path.isdir(full_path):
            dates.append(dir_name)
    dates.sort()
    return dates


# Filter the data based on the ITM/OTM selection and strike thresholds
def filter_itm_otm(df, itm_or_otm, left_threshold, right_threshold):
    if itm_or_otm is None:
        return df  # No filtering applied if ITM/OTM is None
    if left_threshold is not None and right_threshold is not None:
        if itm_or_otm == 'itm':  # In the Money
            return df[(df['strike'] >= left_threshold) & (df['strike'] <= right_threshold)]
        elif itm_or_otm == 'otm':  # Out of the Money
            return df[(df['strike'] < left_threshold) | (df['strike'] > right_threshold)]
    return df  # Return unfiltered if thresholds are not set


# Filter the expiration summary based on ITM/OTM and strike thresholds
def filter_expiration_summary(df_filtered):
    expiration_summary = pd.DataFrame()

    expiration_summary['Volume'] = df_filtered.groupby(by=['ticker', 'date', 'maturity date', 'type'])['Volume'].sum()
    expiration_summary['Gamma'] = df_filtered.groupby(by=['ticker', 'date', 'maturity date', 'type'])[
        'Gamma*Open Interest'].sum()
    expiration_summary['Delta'] = df_filtered.groupby(by=['ticker', 'date', 'maturity date', 'type'])[
        'Delta*Open Interest'].sum()
    expiration_summary['Open Interest'] = df_filtered.groupby(by=['ticker', 'date', 'maturity date', 'type'])[
        'Open Interest'].sum()
    expiration_summary['Avg IV'] = df_filtered.groupby(by=['ticker', 'date', 'maturity date', 'type'])[
                                       'Implied Volatility*Open Interest'].sum() / expiration_summary['Open Interest']
    expiration_summary['Avg Gamma'] = expiration_summary['Gamma'] / expiration_summary['Open Interest']
    expiration_summary['Avg Delta'] = expiration_summary['Delta'] / expiration_summary['Open Interest']
    expiration_summary.reset_index(['maturity date', 'type', 'ticker'], inplace=True)
    return expiration_summary


def filter_data_by_thresholds(ticker, date, df_surface, df_strike, expiration_summary, df3, itm_or_otm,
                              left_threshold_original,
                              right_threshold_original):
    if itm_or_otm is not None:
        price = get_close_price(ticker, date)
        left_threshold = price['Adj Close'] * (1 - left_threshold_original)
        right_threshold = price['Adj Close'] * (1 + right_threshold_original)

        filtered_df_surface = filter_itm_otm(df_surface[df_surface['ticker'] == ticker], itm_or_otm, left_threshold,
                                             right_threshold)
        filtered_df_strike = filter_itm_otm(df_strike[df_strike['ticker'] == ticker], itm_or_otm, left_threshold,
                                            right_threshold)

        df3_reset = df3.reset_index()
        filtered_df = filter_itm_otm(df3_reset[df3_reset['ticker'] == ticker], itm_or_otm, left_threshold,
                                     right_threshold)
        filtered_expiration_summary = filter_expiration_summary(filtered_df)
    else:
        filtered_df_surface = df_surface
        filtered_df_strike = df_strike
        filtered_expiration_summary = expiration_summary

    return filtered_df_surface, filtered_df_strike, filtered_expiration_summary


def filter_combined_data_by_thresholds(df3, date, itm_or_otm, left_threshold_original, right_threshold_original):
    df3_reset = df3.reset_index()
    combined_filtered_df = []
    unique_tickers = df3_reset['ticker'].unique()

    for ticker in unique_tickers:
        # Get the close price for the ticker on the given date
        close_price = get_close_price(ticker, date)
        adjusted_left_threshold = close_price['Adj Close'] * (1 - left_threshold_original)
        adjusted_right_threshold = close_price['Adj Close'] * (1 + right_threshold_original)

        # Filter the data for the current ticker based on the ITM/OTM and thresholds
        ticker_df = df3_reset[df3_reset['ticker'] == ticker]
        filtered_df = filter_itm_otm(ticker_df, itm_or_otm, adjusted_left_threshold, adjusted_right_threshold)
        combined_filtered_df.append(filtered_df)

    # Concatenate all filtered data
    combined_filtered_df = pd.concat(combined_filtered_df)

    # Generate expiration summary for the filtered data
    filtered_expiration_summary = filter_expiration_summary(combined_filtered_df)

    return combined_filtered_df, filtered_expiration_summary


def get_tickers():
    df = pd.read_csv(os.path.join(INDEX_DATA_DIR, f"sp500_companies.csv"))
    tickers = list(df.Symbol.unique())
    tickers.extend(['TSM', 'PLTR', 'QQQ', 'SPY'])

    tickers = list(set(tickers))
    return tickers
