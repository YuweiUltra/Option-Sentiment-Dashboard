from dash import dcc, html, dash_table
import warnings
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from copy import deepcopy
from src.utils import (get_latest_trading_day, get_close_price, get_listed_dates, get_tickers)
import statsmodels.api as sm
import plotly.express as px
from config import INDEX_DATA_DIR, RAW_DATA_DIR

warnings.filterwarnings("ignore")


def cal_gamma_skew(date, ticker):
    try:
        price = get_close_price(ticker, date)
        lower_bound = price['Close'] * 0.5
        upper_bound = price['Close'] * 1.5

        df_oi = pd.read_csv(os.path.join(RAW_DATA_DIR, date, ticker, 'OpenInterest.csv'))
        df_greeks = pd.read_csv(os.path.join(RAW_DATA_DIR, date, ticker, 'Greeks.csv'))

        df = pd.merge(df_oi, df_greeks[['contract_symbol', 'gamma']], on='contract_symbol')
        df['gamma_exposure'] = df['open_interest'] * df['gamma']
        df['date_str'] = df['contract_symbol'].str.extract(r'(\d{6})')
        df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d')
        df['expiration'] = df['date'].dt.strftime('%Y-%m-%d')

        expirations = sorted(df['expiration'].unique().tolist())
        expirations_formatted = pd.to_datetime(expirations).strftime('%y%m%d').tolist()
        df_filtered = df[df['contract_symbol'].str.contains('|'.join(expirations_formatted))]
        df_filtered = df_filtered[(df_filtered['strike'] >= lower_bound) & (df_filtered['strike'] <= upper_bound)]

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

    temp_df = deepcopy(skew_df)
    temp_df['return'] = temp_df.groupby('ticker')['return'].shift(-1)
    temp_df['current skew'] = temp_df.groupby('ticker')['skew'].shift(-1)
    temp_df['skew change'] = temp_df['current skew'] - temp_df['skew']
    temp_df.dropna(inplace=True)

    return temp_df


def perform_regression(temp_df):
    X = temp_df[['skew change']]
    X = sm.add_constant(X)
    y = temp_df['return']

    model = sm.OLS(y, X).fit()

    intercept = model.params['const']
    slope = model.params['skew change']
    r_squared = model.rsquared
    p_value = model.pvalues['skew change']

    temp_df['predicted_return'] = model.predict(X)
    temp_df['residual'] = np.abs(temp_df['return'] - temp_df['predicted_return'])

    return model, temp_df


def create_cross_section_figures(temp_df, model, temp_df_sorted, top_n=5):
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

    fig.add_traces(go.Scatter(
        x=temp_df['skew change'],
        y=temp_df['predicted_return'],
        mode='lines',
        name='Regression Line', ))

    # Top N Residuals
    top_residuals = temp_df.nlargest(top_n, 'residual')
    fig.add_traces(go.Scatter(
        x=top_residuals['skew change'],
        y=top_residuals['return'],
        mode='markers+text',
        marker=dict(color='red', size=12),
        name=f'Top {top_n} Residuals',
        text=top_residuals['ticker'],
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

    # Tables
    df_head = temp_df.head(10).to_dict('records')
    df_tail = temp_df.tail(10).to_dict('records')
    specific_tickers = ['NVDA', 'META', 'GOOG', 'GOOGL', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'MU', 'TSM']
    df_specific = temp_df[temp_df['ticker'].isin(specific_tickers)].sort_values(
        ['current skew', 'skew change', 'open interest'], ascending=False
    ).to_dict('records')

    return fig, df_head, df_tail, df_specific, model


def create_cross_section_content(date, dates, tickers):
    selected_dates = [date, dates[dates.index(date) - 1]]
    temp_df = prepare_skew_data(tickers, selected_dates)
    model, temp_df = perform_regression(temp_df)
    fig, df_head, df_tail, df_specific, model = create_cross_section_figures(temp_df, model, temp_df)

    # Generate DataTables
    table_head = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in temp_df.head(10).columns],
        data=df_head,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '12px',
            'backgroundColor': 'white',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(200, 200, 200)',
            'color': 'black',
            'fontWeight': 'bold',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '14px'
        },
        page_size=10
    )

    table_tail = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in temp_df.tail(10).columns],
        data=df_tail,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '12px',
            'backgroundColor': 'white',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(200, 200, 200)',
            'color': 'black',
            'fontWeight': 'bold',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '14px'
        },
        page_size=10
    )

    table_specific = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in temp_df[temp_df['ticker'].isin(
            ['NVDA', 'META', 'GOOG', 'GOOGL', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'MU', 'TSM'])].columns],
        data=df_specific,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '12px',
            'backgroundColor': 'white',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(200, 200, 200)',
            'color': 'black',
            'fontWeight': 'bold',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '14px'
        },
        page_size=10
    )

    return html.Div([
        dcc.Graph(figure=fig),
        html.H3("Top 10 Records"),
        table_head,
        html.H3("Bottom 10 Records"),
        table_tail,
        html.H3("Selected Tickers"),
        table_specific
    ])
