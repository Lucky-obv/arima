import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import datetime as dt

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide",
    page_icon="📈"
)

# ------------------------------
# Title
# ------------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#4A90E2;">
        📈 Animated Stock Price Forecast Dashboard
    </h1>
    <p style="text-align:center; font-size:18px;">
        Enter any stock ticker to view historical data and ARIMA forecast with beautiful animations.
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# INPUT PANEL
# ------------------------------
col1, col2, col3 = st.columns([2,2,1])

with col1:
    ticker = st.text_input("Enter Stock Symbol:", value="1299.HK")

with col2:
    start_date = st.date_input("Start Date", dt.date(2024,1,1))

with col3:
    forecast_days = st.number_input("Forecast Days", min_value=5, max_value=60, value=10)

end_date = dt.date.today()

# ------------------------------
# STATIONARITY CHECK
# ------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    p_value = result[1]
    if p_value < 0.05:
        return "✔️ The series is *Stationary* (ADF p-value < 0.05)"
    else:
        return "❌ The series is *Not Stationary* (ADF p-value >= 0.05)"

# ------------------------------
# MAIN BUTTON
# ------------------------------
if st.button("Generate Animated Forecast 🚀", type="primary"):

    with st.spinner("Fetching data... please wait ⏳"):

        # download stock
        df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ No data found. Check stock symbol.")
    else:

        df = df.reset_index()

        # Stationarity
        st.subheader("📌 Stationarity Test (ADF)")
        st.info(check_stationarity(df["Close"]))

        # Build ARIMA model
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(df["Close"], order=(5,0,0))
            model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)

        # Create forecast dates
        forecast_dates = pd.date_range(
            start=df["Date"].iloc[-1],
            periods=forecast_days + 1,
            freq="B"
        )[1:]

        # ------------------------------
        # PLOTLY ANIMATED CHART
        # ------------------------------

        fig = go.Figure()

        # Actual price line
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Actual Price",
                line=dict(color="#00A8E8", width=3),
            )
        )

        # Forecast line (animated)
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#FF4C61", width=3, dash="dash"),
                marker=dict(size=6),
            )
        )

        # Layout styling
        fig.update_layout(
            title=f"{ticker} — Actual vs Forecasted Prices",
            title_x=0.5,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Close Price",
            hovermode="x",
            legend=dict(
                title="Legend",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor="rgba(10,10,30,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=80, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data
        st.subheader("📄 Data Used")
        st.dataframe(df)
