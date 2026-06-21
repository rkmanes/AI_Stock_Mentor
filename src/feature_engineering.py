import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

data = yf.download(
    "TCS.NS",
    start="2020-01-01",
    end="2025-01-01"
)

close_prices = data["Close"].squeeze()

data["RSI"] = ta.momentum.RSIIndicator(
    close=close_prices
).rsi()

data["MA20"] = close_prices.rolling(20).mean()

data["MA50"] = close_prices.rolling(50).mean() 

macd = ta.trend.MACD(close=close_prices)

data["MACD"] = macd.macd()
data["MACD_SIGNAL"] = macd.macd_signal()

data["Future_Close"] = close_prices.shift(-10)

data["Return_10D"] = (
    (data["Future_Close"] - close_prices)
    / close_prices
) * 100

data["TARGET"] = (
    data["Return_10D"] > 3
).astype(int)
 
# print(
#     data[
#         [
#             "RSI",
#             "MA20",
#             "MA50",
#             "MACD",
#             "Return_10D",
#             "TARGET"
#         ]
#     ].head(100)
# )
data = data.dropna()

X = data[
    [
        "RSI",
        "MA20",
        "MA50",
        "MACD",
        "MACD_SIGNAL"
    ]
]

y = data["TARGET"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
) 

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
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

print("Accuracy:", accuracy)


print(
    classification_report(
        y_test,
        predictions
    )
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

print(
    "XGBoost Accuracy:",
    xgb_accuracy
)
print(
    classification_report(
        y_test,
        xgb_predictions
    )
)

feature_importance = xgb_model.feature_importances_
for feature, importance in zip(
    X.columns,
    feature_importance
):
    print(
        feature,
        ":",
        round(importance, 4)
    )

joblib.dump(
    xgb_model,
    "models/xgb_model.pkl"
)
