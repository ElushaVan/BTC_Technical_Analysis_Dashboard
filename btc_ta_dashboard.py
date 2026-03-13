import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

@st.cache_data
def load_btc_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')  # ubah ke 'ms' kalau perlu
    df = df.set_index('Datetime')
    df = df.sort_index()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def resample_to_timeframe(btc_df, timeframe_str):
    rule_map = {'1min': '1T', '5min': '5T', '15min': '15T',
                '1H': '1H', '4H': '4H', '1D': '1D', '1W': '1W'}
    rule = rule_map.get(timeframe_str, '5T')
    resampled = btc_df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled

def compute_indicators(df):
    df_out = df.copy()
    df_out['sma_20'] = SMAIndicator(df_out['Close'], window=20).sma_indicator()
    df_out['sma_50'] = SMAIndicator(df_out['Close'], window=50).sma_indicator()
    df_out['rsi_14'] = RSIIndicator(df_out['Close'], window=14).rsi()
    
    macd = MACD(df_out['Close'])
    df_out['macd_line'] = macd.macd()
    df_out['macd_signal'] = macd.macd_signal()
    df_out['macd_histogram'] = macd.macd_diff()
    
    bb = BollingerBands(df_out['Close'], window=20, window_dev=2)
    df_out['bb_upper'] = bb.bollinger_hband()
    df_out['bb_middle'] = bb.bollinger_mavg()
    df_out['bb_lower'] = bb.bollinger_lband()
    
    return df_out

def plot_ta_chart(ta_df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Candlestick + Indicators', 'Volume', 'RSI 14', 'MACD'),
                        row_heights=[0.50, 0.15, 0.15, 0.20])

    fig.add_trace(go.Candlestick(x=ta_df.index, open=ta_df['Open'], high=ta_df['High'],
                                 low=ta_df['Low'], close=ta_df['Close'], name='BTC/USD',
                                 increasing_line_color='green', decreasing_line_color='red'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['sma_50'], name='SMA 50', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['bb_upper'], name='BB Upper', line=dict(color='grey', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['bb_lower'], name='BB Lower', line=dict(color='grey', dash='dash')), row=1, col=1)

    fig.add_trace(go.Bar(x=ta_df.index, y=ta_df['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['rsi_14'], name='RSI 14', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['macd_line'], name='MACD', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=ta_df.index, y=ta_df['macd_signal'], name='Signal', line=dict(color='orange')), row=4, col=1)
    fig.add_trace(go.Bar(x=ta_df.index, y=ta_df['macd_histogram'], name='Histogram'), row=4, col=1)

    fig.update_layout(title='Bitcoin Technical Analysis Dashboard', height=900,
                      hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Bagian utama app ---
st.title("Bitcoin Technical Analysis Dashboard")

uploaded = st.file_uploader("Upload file", type="csv")

if uploaded is not None:
    with st.spinner("Processing dataset..."):
        btc_df = load_btc_data(uploaded)
    
    st.success(f"Dataset loaded! {len(btc_df):,} baris.")
    
    start_date, end_date = st.date_input(
        "Date Range",
        [btc_df.index.min().date(), btc_df.index.max().date()],
        min_value=btc_df.index.min().date(),
        max_value=btc_df.index.max().date()
    )

    filtered_df = btc_df.loc[start_date:end_date]

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
    else:
        timeframe = st.selectbox(
            "Select Timeframe",
            ['1min', '5min', '15min', '1H', '4H', '1D', '1W'],
            index=1
        )

        resampled_df = resample_to_timeframe(filtered_df, timeframe)

        if resampled_df.empty:
            st.warning("No data available after resampling.")
        else:
            ta_df = compute_indicators(resampled_df)
            fig = plot_ta_chart(ta_df)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file first.")