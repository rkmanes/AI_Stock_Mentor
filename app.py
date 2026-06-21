import streamlit as st
import ta
import joblib
import yfinance as yf
import pandas as pd

NIFTY50 = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "AXISBANK.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
    "TITAN.NS",
    "NESTLEIND.NS",
    "BAJAJFINSV.NS",
    "WIPRO.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TECHM.NS",
    "HCLTECH.NS",
    "ONGC.NS",
    "M&M.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "GRASIM.NS",
    "INDUSINDBK.NS",
    "BAJAJ-AUTO.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "BPCL.NS",
    "EICHERMOT.NS",
    "DRREDDY.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "HEROMOTOCO.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "SHRIRAMFIN.NS",
    "ADANIENT.NS",
    "TRENT.NS",
    "BEL.NS",
    "JIOFIN.NS"
]

st.title("AI Stock Mentor")
st.write(
    """
    This application uses Machine Learning and Technical Indicators
    to identify stocks that may gain more than 3% in the next
    10 trading days.
    """
)

model = joblib.load(
    "models/random_forest_nifty50_v1.pkl"
)

st.success("Model Loaded Successfully!")

st.info(
    "Random Forest Model Accuracy: 77.88%"
)

stock_name = st.selectbox(
    "Enter NSE Stock Symbol",
    NIFTY50,
    
)
 
if st.button("Analyze Stock"):

    st.write(
        f"Downloading data for {stock_name}..."
    )

    data = yf.download(
        stock_name,
        period="1y",
        progress=False
    )
    data.columns = data.columns.get_level_values(0)

    if data.empty:

        st.error(
            "Stock not found!"
        )

    else:

        st.success(
            "Latest trading data downloaded successfully!"
        )

        close_prices = data["Close"].squeeze()

        data["RSI"] = ta.momentum.RSIIndicator(
            close=close_prices
        ).rsi()

        data["MA20"] = close_prices.rolling(20).mean()

        data["MA50"] = close_prices.rolling(50).mean()

        macd = ta.trend.MACD(
            close=close_prices
        )

        data["MACD"] = macd.macd()

        data["MACD_SIGNAL"] = macd.macd_signal()

        data["VOLUME"] = data["Volume"]
        

        data = data.dropna()
        
        st.subheader(
            "1 Year Price Chart"
        )

        st.line_chart(
            data["Close"]
        )   

        latest = data.iloc[-1]
        st.subheader("Current Market Data")

        st.write(
            f"Current Price: ₹{latest['Close']:.2f}"
        )

        # Display latest indicator values

        indicator_df = pd.DataFrame({
            "Indicator": [
                "RSI",
                "MA20",
                "MA50",
                "MACD",
                "MACD Signal",
                "Volume"
            ],
            "Value": [
                round(latest["RSI"], 2),
                round(latest["MA20"], 2),
                round(latest["MA50"], 2),
                round(latest["MACD"], 2),
                round(latest["MACD_SIGNAL"], 2),
                int(latest["VOLUME"])
            ]
        })

        st.dataframe(
            indicator_df,
            use_container_width=True
        )

        if latest["RSI"] > 70:

            st.warning(
                "RSI indicates overbought conditions."
            )

        elif latest["RSI"] < 30:

            st.success(
                "RSI indicates oversold conditions."
            )

        else:

            st.info(
                "RSI is in neutral range."
            )

        if latest["MA20"] > latest["MA50"]:

            st.success(
                "Bullish Trend: MA20 is above MA50"
            )

        else:

            st.warning(
                "Bearish Trend: MA20 is below MA50"
            )


        X_pred = pd.DataFrame(
            [[
                latest["RSI"],
                latest["MA20"],
                latest["MA50"],
                latest["MACD"],
                latest["MACD_SIGNAL"],
                latest["VOLUME"]
            ]],
            columns=[
                "RSI",
                "MA20",
                "MA50",
                "MACD",
                "MACD_SIGNAL",
                "VOLUME"
            ]
        )

        st.subheader(
            "AI Prediction"
        )

        prediction = model.predict(
            X_pred
        )[0]
        confidence = (
            model.predict_proba(X_pred)[0].max()
            * 100
        )
        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        if confidence > 80:

            st.success(
                "High Confidence Prediction"
            )

        elif confidence > 60:

            st.info(
                "Moderate Confidence Prediction"
            )

        else:

            st.warning(
                "Low Confidence Prediction"
            )

        if prediction == 1:

            st.success(
                "🟢 BUY CANDIDATE"
            )

            st.write(
                "The model predicts a potential gain greater than 3% within the next 10 trading days."
            )

        else:

            st.warning(
                "🟡 HOLD / AVOID"
            )

            st.write(
                "The model does not predict a gain greater than 3% within the next 10 trading days."
            )