import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

from nifty50_list import NIFTY50

all_data = []

for stock in NIFTY50:
    # print(f"Downloading {stock}...")

    data = yf.download(
        stock,
        start="2020-01-01",
        end="2026-01-01",
        progress=False
    )

    if data.empty:
        continue

    close_prices = data["Close"].squeeze()

    # Features
    data["RSI"] = ta.momentum.RSIIndicator(
        close=close_prices
    ).rsi()

    data["MA20"] = close_prices.rolling(20).mean()

    data["MA50"] = close_prices.rolling(50).mean()

    macd = ta.trend.MACD(close=close_prices)

    data["MACD"] = macd.macd()

    data["MACD_SIGNAL"] = macd.macd_signal()

    data["VOLUME"] = data["Volume"]

    # Target
    data["Future_Close"] = close_prices.shift(-10)

    data["Return_10D"] = (
        (data["Future_Close"] - close_prices)
        / close_prices
    ) * 100

    data["TARGET"] = (
        data["Return_10D"] > 3
    ).astype(int)

    data["STOCK"] = stock

    data = data.dropna()

    all_data.append(data)

combined_data = pd.concat(
    all_data,
    ignore_index=False
)

X = combined_data[
    [
        "RSI",
        "MA20",
        "MA50",
        "MACD",
        "MACD_SIGNAL",
        "VOLUME"
    ]
]

y = combined_data["TARGET"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
) 

model = RandomForestClassifier(
    n_estimators=55,
    random_state=42,
    max_depth=22
)

model.fit(
    X_train,
    y_train
)
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test
)
xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)



joblib.dump(
    model,
    "models/random_forest_nifty50_v2.pkl"
)

train_accuracy = model.score(
    X_train,
    y_train
)

print(
    "Train Accuracy:",
    round(train_accuracy, 4)
)

print(
    "Test Accuracy:",
    round(accuracy, 4)
)

joblib.dump(
    list(X.columns),
    "models/features.pkl"
)