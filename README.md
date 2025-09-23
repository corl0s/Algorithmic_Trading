# Project Proposal

## Team Members
Vishnuram Ayyavu Vijayakumar

Hilton Paul Tony Raj

Vijay Krishna Sundaran Saravanan


**Algorithmic Trading Strategy for Options and Stock Data**

## Goal:
We aim to predict profitable trading opportunities by combining stock price movements and options market signals such as implied volatility and trends. The goal is to design an algorithmic trading strategy that makes dynamic buy, sell, or hedge decisions. Success is measured by achieving higher risk-adjusted returns (e.g., Sharpe ratio) than a simple buy-and-hold strategy.


## Data Collection process plan
We can collect historical stock price data (**OHLCV**) from free sources like Yahoo Finance (`yfinance` library in Python) or Alpha Vantage (free API for stocks). For options, only current option chain snapshots (strikes, expiry, IV, open interest, volume) are freely available from Yahoo Finance. These datasets can be aligned by timestamp so that options market signals from day *t* can be used to predict stock movements and trading opportunities on day *t+1*. Finally, the combined dataset is cleaned, adjusted for splits/dividends, and structured for modeling.


## Modeling Approach

The algorithmic trading system will combine **predictive modeling** and **classical financial theory** to generate trading signals. Historical stock and options data will be preprocessed and enriched with technical and options-based features. **Neural networks** (Deep architectures, LSTMs, Transformers) will predict future prices and volatility, while classical models such as **Black–Scholes** will provide theoretical prices and implied volatility benchmarks. Trading signals will be derived from the predicted price movements, adjusted for thresholds and risk limits, and executed through **paper trading**.

## Visualizing the data

For multivariate time series, heatmaps, lag plots and scatter matrix plots can be efficient to observe the change of stock prices over time. To visualize forecasting, we plan to use line plots and cumulative return curves and analyze the performance.

## Test Plan

The algorithm will be evaluated via **paper trading** using the **Alpaca API**. After training on historical stock and options data, it will execute trades in a simulated live market environment, allowing performance assessment (**profit/loss, Sharpe ratio, drawdown**) without risking real capital.
