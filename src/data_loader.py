import yfinance as yf

ticker = "TCS.NS"

data = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-01-01"
)

print(data.head())

print("\nShape:", data.shape)