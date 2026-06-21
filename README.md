# AI Stock Mentor

## Overview

AI Stock Mentor is a Machine Learning based stock analysis application that predicts whether a NIFTY 50 stock has the potential to gain more than 3% within the next 10 trading days.

The project combines technical analysis indicators with machine learning algorithms to help users make data-driven investment decisions.

---

## Features

* Analysis of NIFTY 50 stocks
* Live stock data retrieval using Yahoo Finance
* Machine Learning based stock prediction
* Confidence score for each prediction
* One-year stock price chart
* RSI (Relative Strength Index) analysis
* Moving Average trend detection
* Technical indicator dashboard
* Interactive Streamlit web application

---

## Technical Indicators Used

The model uses the following indicators as input features:

1. RSI (Relative Strength Index)
2. MA20 (20-Day Moving Average)
3. MA50 (50-Day Moving Average)
4. MACD
5. MACD Signal
6. Volume

---

## Machine Learning Models and the results

### Random Forest Classifier

* Accuracy: 75.67%
* Final deployed model

### XGBoost Classifier

* Accuracy: 70.38%

Random Forest achieved better performance and was selected as the final model for deployment.

---

## Dataset

* Source: Yahoo Finance
* Stocks: NIFTY 50 Companies
* Time Period: January 2020 – January 2026
* Total Records: Approximately 69,000

---

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-Learn
* XGBoost
* Yahoo Finance (yfinance)
* TA (Technical Analysis Library)

---

## Project Workflow

1. Download historical stock data
2. Calculate technical indicators
3. Generate target labels based on future returns
4. Train machine learning models
5. Evaluate model performance
6. Save trained model
7. Deploy prediction interface using Streamlit

---

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* News sentiment analysis
* Portfolio management features
* Additional machine learning models
* Real-time stock alerts
* Enhanced stock search and filtering

---

## Author

Developed as a Machine Learning project for stock market prediction using technical indicators and ensemble learning methods.

## Note

The trained Random Forest model file is not included in the repository due to GitHub file size limitations. Users can regenerate the model using the training scripts provided in the project.
