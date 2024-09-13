import os.path
from dash import dcc, html, Input, Output, dash_table, Dash
import warnings
import json
import pandas as pd
import plotly.graph_objs as go
from src.utils import (get_latest_trading_day, get_close_price, get_listed_dates, get_tickers)
from src.analysis.calculate_ratios import calculate_ratios
from config import RAW_DATA_DIR

warnings.filterwarnings("ignore")

##############################################################################################################
# Data preparation
dates = get_listed_dates()
last_trading_date, pre_last_trading_date = get_latest_trading_day()
tickers = get_tickers()
tickers.sort()
ratios = calculate_ratios(rewrite=False)

##############################################################################################################
# Dash app setup with suppress_callback_exceptions=True
app = Dash(__name__,
           external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css'])
server = app.server

# Layout
app.layout = html.Div(
    children=[
        html.H1("Options Viewer Dashboard", style={'textAlign': 'center'}),  # White text color
        # Container to align dropdowns and inputs in a single line
        html.Div(
            children=[
                dcc.Dropdown(
                    id='date-dropdown',
                    options=[{'label': date, 'value': date} for date in dates],
                    value=dates[-1],  # Default value
                    clearable=False,
                    style={'width': '80%', 'margin-right': '5px'}
                ),
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                    value='AAPL',
                    clearable=False,
                    style={'width': '80%', 'margin-right': '5px'}  # Adjust width and spacing
                ),
                dcc.Dropdown(
                    id='expiration-dropdown',
                    options=[],
                    clearable=False,
                    multi=True,
                    style={'width': '80%', 'margin-right': '5px'}  # Adjust width and spacing
                ),
                dcc.Dropdown(
                    id='plot-selection-dropdown',
                    options=[
                        {'label': "Overview", 'value': 'overview'},
                        {'label': "Open Interest", 'value': 'open_interest'},
                        {'label': "Gamma Exposure", 'value': 'gamma_exposure'}
                    ],
                    value="overview",
                    clearable=False,
                    style={'width': '80%', 'margin-right': '5px'}
                )
            ],
            style={'display': 'flex', 'align-items': 'center'}  # Flexbox to align dropdowns in a row
        ),
        html.Div(id='plot-content'),
    ]
)


# Dynamically update the options in plot-selection-dropdown based on the plot category selected
@app.callback(
    Output('expiration-dropdown', 'options'),
    [Input('ticker-dropdown', 'value'),
     Input('date-dropdown', 'value')]
)
def update_expiration_dropdown_options(ticker, date):
    if ticker is None or date is None:
        return None
    else:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/OpenInterest.csv'))
        df['date_str'] = df['contract_symbol'].str.extract(r'(\d{6})')
        df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d')
        df = df.sort_values(by='date')
        return df['date'].dt.strftime('%Y-%m-%d').unique().tolist()


@app.callback(
    Output('ticker-dropdown', 'options'),
    Input('ticker-dropdown', 'search_value'),
)
def update_dropdown_options(search_value):
    if not search_value:
        # If no search term, display all options
        return [{'label': ticker, 'value': ticker} for ticker in tickers]

    search_lower = search_value.lower()

    # First, tickers that start with the search term
    starts_with = [ticker for ticker in tickers if ticker.lower().startswith(search_lower)]

    # Then, tickers that contain the search term elsewhere
    contains = [ticker for ticker in tickers if
                search_lower in ticker.lower() and not ticker.lower().startswith(search_lower)]

    # Combine the lists, giving priority to 'starts_with'
    sorted_tickers = starts_with + contains

    return [{'label': ticker, 'value': ticker} for ticker in sorted_tickers]


# Callback to render the appropriate plot content based on the selected plot type and plot category
@app.callback(
    Output('plot-content', 'children'),
    [Input('date-dropdown', 'value'),
     Input('ticker-dropdown', 'value'),
     Input('expiration-dropdown', 'value'),
     Input('plot-selection-dropdown', 'value')]
)
def render_content(date, ticker, expirations, selected_plot):
    try:
        price = get_close_price(ticker, date)
    except:
        return html.Div("Price data are not found.")

    lower_bound = price['Close'] * 0.7
    upper_bound = price['Close'] * 1.3

    if selected_plot == "overview":
        try:
            with open(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/overview_table1.json'), 'r') as file:
                overview_data1 = json.load(file)
            with open(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/overview_table2.json'), 'r') as file:
                overview_data2 = json.load(file)

            overview_data1 = [{"Metric": key, "Value": value} for item in overview_data1 for key, value in item.items()]

            ratios_filtered = ratios[ratios['ticker'] == ticker][
                ['date', 'gamma_exposure_ratio', 'open_interest_call_put_ratio']]

            fig_gamma_exposure_ratio = go.Figure(data=[
                go.Bar(
                    x=ratios_filtered['date'],
                    y=ratios_filtered['gamma_exposure_ratio'],
                    marker_color='rgba(75, 192, 192, 0.6)',
                    width=0.4
                )
            ])
            fig_gamma_exposure_ratio.update_layout(
                title='Gamma Exposure Ratio Over Time',
                xaxis_title='Date',
                yaxis_title='Gamma Exposure Ratio',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=ratios_filtered['date'].tolist(),
                    showgrid=False
                ),
                bargap=0.1
            )

            fig_open_interest_ratio = go.Figure(data=[
                go.Bar(
                    x=ratios_filtered['date'],
                    y=ratios_filtered['open_interest_call_put_ratio'],
                    marker_color='rgba(153, 102, 255, 0.6)',
                    width=0.4
                )
            ])
            fig_open_interest_ratio.update_layout(
                title='Open Interest Call-Put Ratio Over Time',
                xaxis_title='Date',
                yaxis_title='Open Interest Call-Put Ratio',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=ratios_filtered['date'].tolist(),
                    showgrid=False
                ),
                bargap=0.1
            )

            return html.Div([
                dcc.Graph(figure=fig_gamma_exposure_ratio),
                dcc.Graph(figure=fig_open_interest_ratio),
                dash_table.DataTable(
                    id='volatility-table',
                    columns=[
                        {"name": "Metric", "id": "Metric"},
                        {"name": "Value", "id": "Value"}
                    ],
                    data=overview_data1,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'fontFamily': 'Courier New, monospace',
                        'fontSize': '14px',
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
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(245, 245, 245)'
                        }
                    ]
                ),
                dash_table.DataTable(
                    id='options-table',
                    columns=[{"name": col, "id": col} for col in overview_data2[0].keys()],
                    data=overview_data2,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'fontFamily': 'Courier New, monospace',
                        'fontSize': '14px',
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
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(245, 245, 245)'
                        }
                    ],
                    page_size=10
                )
            ])
        except:
            return html.Div("Overview data are not found.")

    if selected_plot == "gamma_exposure":
        try:
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/OpenInterest.csv'))
            df_gamma = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/Greeks.csv'))

            df = pd.merge(df, df_gamma[['contract_symbol', 'gamma']], on='contract_symbol')
            df['gamma_exposure'] = df['open_interest'] * df['gamma']

            df['date_str'] = df['contract_symbol'].str.extract(r'(\d{6})')
            df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d')
            df['expiration'] = df['date'].dt.strftime('%Y-%m-%d')
            expirations_formatted = pd.to_datetime(expirations).strftime('%y%m%d').tolist()
            df_filtered = df[df['contract_symbol'].str.contains('|'.join(expirations_formatted))]
            df_filtered = df_filtered[(df_filtered['strike'] >= lower_bound) & (df_filtered['strike'] <= upper_bound)]

            df_call = df_filtered[df_filtered['option_type'] == 'CALL']
            df_put = df_filtered[df_filtered['option_type'] == 'PUT']

            sum_put = df_put.groupby('strike')['gamma_exposure'].sum(numeric_only=True).reset_index().dropna()
            sum_call = df_call.groupby('strike')['gamma_exposure'].sum(numeric_only=True).reset_index().dropna()

            max_y = max(sum_put['gamma_exposure'].max(), sum_call['gamma_exposure'].max())
            min_y = min(sum_put['gamma_exposure'].min(), sum_call['gamma_exposure'].min())

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sum_put['strike'],
                y=sum_put['gamma_exposure'],
                name='PUT',
                marker_color='rgba(255, 99, 132, 0.7)',
                hoverinfo='x+y'
            ))

            fig.add_trace(go.Bar(
                x=sum_call['strike'],
                y=sum_call['gamma_exposure'],
                name='CALL',
                marker_color='rgba(54, 162, 235, 0.7)',
                hoverinfo='x+y'
            ))

            fig.add_shape(
                type='line',
                x0=price['Close'],
                x1=price['Close'],
                y0=min_y,
                y1=max_y,
                line=dict(
                    color='rgba(128, 0, 128, 0.8)',
                    width=3,
                    dash='dash'
                ),
                name='Close Price'
            )

            fig.update_layout(
                barmode='stack',
                title=f"Gamma Exposure by Strike Price {ticker} (Total: {round(sum_call['gamma_exposure'].sum(numeric_only=True) + sum_put['gamma_exposure'].sum(numeric_only=True))} "
                      f"Call: {round(sum_call['gamma_exposure'].sum(numeric_only=True))} Put:{round(sum_put['gamma_exposure'].sum(numeric_only=True))} "
                      f"Call-Put-Ratio: {sum_call['gamma_exposure'].sum(numeric_only=True) / sum_put['gamma_exposure'].sum(numeric_only=True):.2f} ) ",
                title_x=0.5,
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                xaxis_title='Strike Price',
                yaxis_title='Gamma Exposure',
                legend_title='Option Type',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                font=dict(family="Courier New, monospace", size=14, color="black")
            )

            df_call_summary = df[df['option_type'] == 'CALL'].groupby('expiration').sum(numeric_only=True).nlargest(5,
                                                                                                                    'gamma_exposure').reset_index()
            df_put_summary = df[df['option_type'] == 'PUT'].groupby('expiration').sum(numeric_only=True).nlargest(5,
                                                                                                                  'gamma_exposure').reset_index()

            fig_call_summary = go.Figure()
            fig_call_summary.add_trace(go.Bar(
                y=[1, 2, 3, 4, 5],
                x=df_call_summary['gamma_exposure'],
                name='Top 5 CALL Expirations',
                orientation='h',
                marker_color='rgba(54, 162, 235, 0.7)',
                text=df_call_summary['expiration'],
                textposition='auto',
                hoverinfo='x+y'
            ))
            fig_call_summary.update_layout(
                title='Top 5 CALL Gamma Exposure Expirations',
                title_x=0.5,
                xaxis_title='Total Gamma Exposure',
                yaxis_title='Ranking',
                yaxis=dict(
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=df_call_summary['expiration'],
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                )
            )

            fig_put_summary = go.Figure()
            fig_put_summary.add_trace(go.Bar(
                y=[1, 2, 3, 4, 5],
                x=df_put_summary['gamma_exposure'],
                name='Top 5 PUT Expirations',
                orientation='h',
                marker_color='rgba(255, 99, 132, 0.7)',
                text=df_put_summary['expiration'],
                textposition='auto',
                hoverinfo='x+y'
            ))
            fig_put_summary.update_layout(
                title='Top 5 PUT Gamma Exposure Expirations',
                title_x=0.5,
                xaxis_title='Total Gamma Exposure',
                yaxis_title='Ranking',
                yaxis=dict(
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=df_put_summary['expiration'],
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                )
            )

            df_grouped_diff = df_filtered.groupby('strike').apply(
                lambda x: x[x['option_type'] == 'CALL']['gamma_exposure'].sum(numeric_only=True) -
                          x[x['option_type'] == 'PUT']['gamma_exposure'].sum(numeric_only=True)
            ).reset_index(name='gamma_exposure_diff').dropna()

            fig_gamma_diff = go.Figure()

            fig_gamma_diff.add_trace(go.Bar(
                x=df_grouped_diff['strike'],
                y=df_grouped_diff['gamma_exposure_diff'],
                name='CALL minus PUT Gamma Exposure',
                marker_color='rgba(128, 0, 255, 0.7)',
                hoverinfo='x+y'
            ))

            fig_gamma_diff.add_shape(
                type='line',
                x0=price['Close'],
                x1=price['Close'],
                y0=df_grouped_diff['gamma_exposure_diff'].min(),
                y1=df_grouped_diff['gamma_exposure_diff'].max(),
                line=dict(
                    color='rgba(128, 0, 128, 0.8)',
                    width=3,
                    dash='dash'
                ),
                name='Close Price'
            )

            fig_gamma_diff.update_layout(
                title=f"Gamma Exposure Difference (CALL minus PUT) {ticker} (Total: {round(df_grouped_diff['gamma_exposure_diff'].sum(numeric_only=True))} "
                      f"Exposure ratio: {df_grouped_diff['gamma_exposure_diff'].sum(numeric_only=True) / df_grouped_diff['gamma_exposure_diff'].abs().sum(numeric_only=True):.2f} )",
                title_x=0.5,
                xaxis_title='Strike Price',
                yaxis_title='Gamma Exposure Difference',
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                )
            )

            return html.Div([
                dcc.Graph(figure=fig),
                dcc.Graph(figure=fig_gamma_diff),
                dcc.Graph(figure=fig_call_summary),
                dcc.Graph(figure=fig_put_summary)
            ])
        except:
            return html.Div("Gamma Exposure data are not found.")

    if selected_plot == "open_interest":
        try:
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, f'{date}/{ticker}/OpenInterest.csv'))
            df['date_str'] = df['contract_symbol'].str.extract(r'(\d{6})')
            df['date'] = pd.to_datetime(df['date_str'], format='%y%m%d')
            df['expiration'] = df['date'].dt.strftime('%Y-%m-%d')
            expirations_formatted = pd.to_datetime(expirations).strftime('%y%m%d').tolist()
            df_filtered = df[df['contract_symbol'].str.contains('|'.join(expirations_formatted))]
            df_filtered = df_filtered[(df_filtered['strike'] >= lower_bound) & (df_filtered['strike'] <= upper_bound)]

            df_call = df_filtered[df_filtered['option_type'] == 'CALL']
            df_put = df_filtered[df_filtered['option_type'] == 'PUT']

            sum_put = df_put.groupby('strike')['open_interest'].sum(numeric_only=True).reset_index().dropna()
            sum_call = df_call.groupby('strike')['open_interest'].sum(numeric_only=True).reset_index().dropna()

            max_y = max(sum_put['open_interest'].max(), sum_call['open_interest'].max())
            min_y = min(sum_put['open_interest'].min(), sum_call['open_interest'].min())

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sum_put['strike'],
                y=sum_put['open_interest'],
                name='PUT',
                marker_color='rgba(255, 99, 132, 0.7)',
                hoverinfo='x+y'
            ))

            fig.add_trace(go.Bar(
                x=sum_call['strike'],
                y=sum_call['open_interest'],
                name='CALL',
                marker_color='rgba(54, 162, 235, 0.7)',
                hoverinfo='x+y'
            ))

            fig.add_shape(
                type='line',
                x0=price['Close'],
                x1=price['Close'],
                y0=min_y,
                y1=max_y,
                line=dict(
                    color='rgba(128, 0, 128, 0.8)',
                    width=3,
                    dash='dash'
                ),
                name='Close Price'
            )

            fig.update_layout(
                barmode='stack',
                title=f'Open Interest by Strike Price {ticker} (Total: {round(df_filtered["open_interest"].sum(numeric_only=True))} '
                      f'Call: {round(df_filtered["open_interest"][df_filtered["option_type"] == "CALL"].sum(numeric_only=True))} '
                      f'PUT: {round(df_filtered["open_interest"][df_filtered["option_type"] == "PUT"].sum(numeric_only=True))} '
                      f'Call-Put-Ratio: {df_filtered["open_interest"][df_filtered["option_type"] == "CALL"].sum(numeric_only=True) / df_filtered["open_interest"][df_filtered["option_type"] == "PUT"].sum(numeric_only=True):.2f}) ',
                title_x=0.5,
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                xaxis_title='Strike Price',
                yaxis_title='Open Interest',
                legend_title='Option Type',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                font=dict(family="Courier New, monospace", size=14, color="black")
            )

            df_call_summary = df[df['option_type'] == 'CALL'].groupby('expiration').sum(numeric_only=True).nlargest(5,
                                                                                                                    'open_interest').reset_index()
            df_put_summary = df[df['option_type'] == 'PUT'].groupby('expiration').sum(numeric_only=True).nlargest(5,
                                                                                                                  'open_interest').reset_index()

            fig_call_summary = go.Figure()
            fig_call_summary.add_trace(go.Bar(
                y=[1, 2, 3, 4, 5],
                x=df_call_summary['open_interest'],
                name='Top 5 CALL Expirations',
                orientation='h',
                marker_color='rgba(54, 162, 235, 0.7)',
                text=df_call_summary['expiration'],
                textposition='auto',
                hoverinfo='x+y'
            ))
            fig_call_summary.update_layout(
                title='Top 5 CALL Open Interest Expirations',
                title_x=0.5,
                xaxis_title='Total Open Interest',
                yaxis_title='Ranking',
                yaxis=dict(
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=df_call_summary['expiration'],
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                )
            )

            fig_put_summary = go.Figure()
            fig_put_summary.add_trace(go.Bar(
                y=[1, 2, 3, 4, 5],
                x=df_put_summary['open_interest'],
                name='Top 5 PUT Expirations',
                orientation='h',
                marker_color='rgba(255, 99, 132, 0.7)',
                text=df_put_summary['expiration'],
                textposition='auto',
                hoverinfo='x+y'
            ))
            fig_put_summary.update_layout(
                title='Top 5 PUT Open Interest Expirations',
                title_x=0.5,
                xaxis_title='Total Open Interest',
                yaxis_title='Ranking',
                yaxis=dict(
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=df_put_summary['expiration'],
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    gridcolor='lightgrey'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_font=dict(family="Courier New, monospace", size=20, color="black"),
                font=dict(family="Courier New, monospace", size=14, color="black"),
                xaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    showgrid=False
                )
            )

            return html.Div([
                dcc.Graph(figure=fig),
                dcc.Graph(figure=fig_call_summary),
                dcc.Graph(figure=fig_put_summary)
            ])
        except:
            return html.Div("Open Interest data are not found")

    return html.Div("Please select a plot type.")


if __name__ == '__main__':
    fixed_port = 8050
    app.run_server(debug=True, port=fixed_port, host='0.0.0.0')
