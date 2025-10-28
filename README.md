<!-- # Project Proposal

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

The algorithm will be evaluated via **paper trading** using the **Alpaca API**. After training on historical stock and options data, it will execute trades in a simulated live market environment, allowing performance assessment (**profit/loss, Sharpe ratio, drawdown**) without risking real capital. -->


## Video Presentation - https://youtu.be/5qRYx9y6_DM

# **Historical Stock Data Visualization and Transformer-Based Price Prediction**

## **1. Objective**
We analyze stock market data using various visualization techniques and model historical stock data using a Transformer-based deep learning model. This helps understand trends, volatility, relationships between assets, and predictive performance of modern sequence models on financial data.

---

## **2. Data Overview**

**Dataset Source:** Yahoo Finance  
**Duration:** 2013-01-01 to 2023-01-01  
**Tickers Used:** `AAPL`, `MSFT`, `GOOG`, `AMZN`, `AAME`

The dataset contains **2,518 trading days**, with six columns per stock:
- Open  
- High  
- Low  
- Close  
- Adjusted Close  
- Volume

There were **no missing or duplicate entries**, confirming a clean dataset for visualization and modeling.

---

## **3. Data Visualizations**

For below visualization we choose AAPL historical stock data. 

### **3.1 Line Chart of Closing Prices**

![png](Data_Visualisation_files/Data_Visualisation_4_0.png)


**Purpose:**  
To visualize long-term trends and patterns in Apple’s stock price.

**Description:**  
A line plot of AAPL’s closing price over time shows the steady upward growth of Apple’s stock over the 10-year period, with occasional corrections and short-term volatility.

**Interpretation:**  
The consistent growth reflects Apple’s strong business performance and market dominance. Sudden dips often correspond to broader market events or earnings reports.

---

### **3.2 Moving Averages (50-day and 200-day)**

![png](Data_Visualisation_files/Data_Visualisation_6_0.png)

**Purpose:**  
To smooth short-term fluctuations and reveal longer-term trends, common in technical analysis.

**Description:**  
The 50-day (short-term) and 200-day (long-term) moving averages were plotted along with the closing price.  
- When the 50-day MA crosses above the 200-day MA (Golden Cross), it suggests a potential uptrend.  
- When it crosses below (Death Cross), it may indicate a potential downtrend.

**Interpretation:**  
The visualization highlights the momentum and trend reversals in Apple’s stock, helping identify periods of bullish or bearish sentiment.

---

### **3.3 Daily Returns Distribution**

![png](Data_Visualisation_files/Data_Visualisation_8_0.png)
    
**Purpose:**  
To assess volatility and risk by examining how much the stock’s price changes daily.

**Description:**  
A histogram of daily returns shows most returns clustered around zero, with a few large positive or negative outliers.

**Interpretation:**  
The shape indicates that while daily price changes are typically small, occasional large fluctuations occur — typical of financial time series with “fat tails”.

---

### **3.4 Rolling Volatility (30-Day Window)**
    
![png](Data_Visualisation_files/Data_Visualisation_10_0.png)

**Purpose:**  
To analyze time-varying volatility over time.

**Description:**  
A rolling standard deviation of daily returns (30-day window) was plotted to visualize changing volatility levels.

**Interpretation:**  
Spikes in rolling volatility correspond to periods of uncertainty — such as market-wide downturns or economic news events. This helps risk managers anticipate turbulent market phases.

---

### **3.5 Correlation Heatmap**

![png](Data_Visualisation_files/Data_Visualisation_12_1.png)
    
**Purpose:**  
To visualize relationships among multiple tech stocks (AAPL, MSFT, GOOG, AMZN, AAME).

**Description:**  
A heatmap was created using Pearson correlation coefficients of percentage changes in closing prices.

**Interpretation:**  
- AAPL, MSFT, GOOG, and AMZN show strong positive correlations (≈0.8–0.9), indicating they tend to move together — typical of tech sector stocks.  
- AAME, a smaller firm, has weaker correlations, implying different market drivers.

---

### **3.6 Multi-Stock Comparison**

![png](Data_Visualisation_files/Data_Visualisation_13_1.png)

**Purpose:**  
To compare the performance of several stocks over time.

**Description:**  
A line chart plotted the adjusted closing prices for all five tickers from 2013–2023.

**Interpretation:**  
Tech giants (AAPL, MSFT, GOOG, AMZN) exhibited similar growth trajectories, while AAME remained relatively stagnant. This visualizes sector dominance and scale disparities.

---

### **3.7 Normalized Price Comparison**

![png](Data_Visualisation_files/Data_Visualisation_15_0.png)


**Purpose:**  
To enable comparison across stocks with different price scales.

**Description:**  
Prices were normalized using z-score normalization:  

$z = \frac{(x - \text{mean})}{\text{std}}$

Normalized plots show how each stock’s relative performance deviates from its mean.

**Interpretation:**  
This allows fairer comparison of trends — e.g., how volatile each stock is relative to itself, not its absolute price.

---


## **4. Understanding the Transformer Architecture for Stock Prediction**

### **4.1 Overview of the Transformer**
The **Transformer** architecture was originally introduced in the paper *"Attention is All You Need"* (Vaswani et al., 2017) for machine translation. Unlike recurrent models such as RNNs or LSTMs, Transformers **do not rely on sequential recurrence**. Instead, they use a mechanism called **self-attention** to model dependencies between elements in a sequence — regardless of their distance apart.

This makes Transformers extremely powerful for **time-series forecasting**, especially when long-range dependencies exist like in the case of stock prices.

---

### **4.2 Key Components**
#### **1. Input Embedding**
Each time step (e.g., a day’s stock features like Open, High, Low, Close, Volume) is projected into a high-dimensional vector space.  
This converts raw numerical data into dense embeddings that capture feature interactions.

#### **2. Positional Encoding**
Because Transformers process all time steps in parallel (not sequentially like RNNs), they lose the concept of “order.”  
To address this, **positional encodings** are added to embeddings to inject information about the position of each day in the sequence (e.g., day 1, day 2, etc.).

Mathematically:

$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$


$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

These periodic encodings allow the model to infer temporal relationships.

#### **3. Self-Attention Mechanism**
The heart of the Transformer is **self-attention**, which computes how strongly each time step should attend to every other step.  
This is defined as:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$


- **Q (Query):** Represents the current time step we’re focusing on  
- **K (Key):** Represents other time steps in the sequence  
- **V (Value):** The information we want to aggregate  

The attention output allows the model to dynamically weigh which previous days are most relevant for predicting the next stock price.

#### **4. Multi-Head Attention**
Instead of one attention mechanism, multiple heads run in parallel, allowing the model to learn different types of temporal and feature relationships simultaneously — e.g., one head might focus on recent volatility, while another tracks long-term growth patterns.

#### **5. Feed-Forward and Normalization Layers**
After attention, a small neural network processes the aggregated information for each time step, followed by **layer normalization** and **residual connections** to stabilize training and avoid vanishing gradients.

#### **6. Output Layer**
Finally, a linear layer maps the Transformer’s hidden representation to a single output — in this case, the predicted normalized **closing price**.

---

### **6.3 Why the Transformer is Suitable for Stock Market Prediction**

| **Reason** | **Explanation** |
|-------------|-----------------|
| **Captures Long-Term Dependencies** | Stock prices often depend on trends spanning weeks or months. Unlike RNNs, Transformers can model relationships across long time horizons efficiently. |
| **Handles Multivariate Inputs** | The model can process multiple correlated features (Open, High, Low, Volume, etc.) and learn how they collectively influence the target. |
| **Parallel Computation** | Self-attention allows simultaneous processing of all time steps, making it faster and more scalable than sequential models like LSTMs. |
| **Flexible Attention Mechanism** | The model can dynamically focus on the most relevant time steps — for instance, ignoring minor fluctuations and attending to significant market shifts. |
| **Generalization Ability** | With enough data, Transformers can generalize across different stocks or regimes, especially when pre-trained on large datasets. |

---

### **6.4 Challenges in Applying Transformers to Stock Markets**
While powerful, Transformers are **not a silver bullet** for financial forecasting:

1. **High Noise & Non-Stationarity:**  
   Financial time series are noisy and regime-dependent. Sudden market events can break learned patterns, making predictions unreliable.

2. **Data Scarcity:**  
   Compared to NLP datasets (millions of samples), stock market data has limited historical observations (~2,500 trading days per decade).

3. **Overfitting Risk:**  
   Transformers are parameter-heavy. Without strong regularization or large datasets, they may overfit to short-term noise.

4. **Causality & Lag:**  
   Transformers capture correlations but not causation. Market movements are influenced by external factors (news, economy) beyond price history.

5. **Interpretability:**  
   Attention weights can offer some interpretability, but the reasoning behind model decisions in volatile markets remains opaque.

---

### **6.5 Summary**
Transformers are a **promising tool** for modeling sequential dependencies and patterns in stock data due to their ability to attend over long historical contexts and handle multivariate inputs. However, success in stock prediction requires:
- Careful feature engineering (technical indicators, sentiment, macro data)
- Adaptive learning strategies
- Regularization and hybrid architectures (e.g., **CNN + Transformer**, **LSTM + Attention**)

When designed thoughtfully, Transformers can serve as the **foundation for advanced financial forecasting systems**, providing insights into market trends rather than merely point predictions.




## **4. Transformer-Based Stock Price Prediction**


### **4.1 Data Preprocessing**
- Used **AAME** stock for modeling.  
- Normalized all columns (`Low`, `Open`, `High`, `Close`, `Volume`, `Adjusted Close`) to zero mean and unit variance for stable training.  

![png](Data_Visualisation_files/Data_Visualisation_19_0.png)

- Split into:
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%  
- Created sequences of 10 time steps for input to the model.

---

### **4.2 Model Architecture**
**Model Type:** Transformer Encoder  
**Components:**
- **Embedding Layer:** Projects input features to a higher-dimensional hidden space.  
- **Transformer Encoder:** Captures temporal dependencies and relationships using self-attention.  
- **Fully Connected Layer:** Outputs the predicted closing price.

**Hyperparameters:**
- Context Window size: 10/30 (Two experiments)
- Hidden size: 64  
- Layers: 2  
- Attention heads: 4  
- Learning rate: 0.001  
- Epochs: 80  
- Loss: Mean Squared Error (MSE)

---

### **4.3 Training and Evaluation**

<!-- The experiment has been performed for two context windows, 10 and 30 

Training and validation losses decreased consistently, showing the model successfully learned patterns from the data without severe overfitting.

![png](Data_Visualisation_files/Data_Visualisation_24_0.png)

for context windows 10

**Test Set Metrics:**

| Metric | Value |
|:-------|-------:|
| MAE | 0.1275 |
| MSE | 0.0322 |
| RMSE | 0.1795 |
| MAPE | 27.32% |
| Directional Accuracy | 46.74% |


for context windows 30
Test Set Metrics:
MAE:  0.1230
MSE:  0.0265
RMSE: 0.1627
MAPE: 26.5362%
Directional Accuracy: 48.56% -->


### **4.3 Experimental Results with Different Context Windows**

The experiment was performed using **two context window sizes** — 10 and 30 — to analyze how the model’s receptive field (amount of past data it considers) affects predictive performance.

Training and validation losses decreased consistently for both settings, showing that the Transformer successfully learned temporal patterns from the stock data without severe overfitting.

---

#### **Results for Context Window = 10**

![Training and Validation Loss for window size 10](Data_Visualisation_files/Data_Visualisation_24_0.png)

**Test Set Metrics:**

| **Metric** | **Value** |
|:------------|----------:|
| **MAE** | 0.1275 |
| **MSE** | 0.0322 |
| **RMSE** | 0.1795 |
| **MAPE** | 27.32% |
| **Directional Accuracy** | 46.74% |

---

#### **Results for Context Window = 30**
![Training and Validation Loss for window size 30](Data_Visualisation_files/train_val_30.png)

**Test Set Metrics:**

| **Metric** | **Value** |
|:------------|----------:|
| **MAE** | 0.1230 |
| **MSE** | 0.0265 |
| **RMSE** | 0.1627 |
| **MAPE** | 26.54% |
| **Directional Accuracy** | 48.56% |

---

**Analysis**

- Increasing the **context window from 10 to 30** improved all evaluation metrics slightly.  
- The lower **MAE**, **MSE**, and **RMSE** values indicate better predictive accuracy.  
- The improvement in **Directional Accuracy** (from 46.74% → 48.56%) shows that the model became better at predicting the direction of price changes.  
- A longer context window allows the Transformer to leverage **more temporal context**, helping it capture **long-term dependencies and trends** in stock movement.  
- However, the gain is moderate, reflecting the **stochastic and non-stationary nature** of stock prices.


<!-- **Interpretation:**  
The Transformer achieved moderate accuracy. The high MAPE and ~47% directional accuracy indicate limited predictive power—expected given the noise and non-stationarity of stock prices. -->

---

<!-- ### **4.4 Visual Results**
1. **Predicted vs Actual Prices (Normalized):**  
   Shows model predictions closely following actual values in trend but with slight phase lag.

    ![png](Data_Visualisation_files/Data_Visualisation_24_2.png)


2. **Cumulative Return Curve:**  
   Simulates a simple long-only strategy using predicted directions. The cumulative profit curve indicates that model-driven trading yields only marginal gains, reflecting difficulty in directional forecasting.

   ![png](Data_Visualisation_files/Data_Visualisation_24_4.png) -->

### **4.4 Visual Results**

The Transformer model’s performance was evaluated visually for **two context window sizes** — 10 and 30 — to compare how expanding the historical context affects prediction quality and simulated trading performance.

---

#### **A. Context Window = 10**

1. **Predicted vs Actual Prices (Normalized):**  
   The model’s predictions closely follow the overall trend of the true stock prices, though with slight lag and minor deviations in short-term fluctuations.  
   This indicates the Transformer effectively captures general temporal patterns but struggles with short-term volatility.

   ![Predicted vs Actual (Window 10)](Data_Visualisation_files/Data_Visualisation_24_2.png)

2. **Cumulative Return Curve:**  
   A simulated **long-only strategy** was applied using the predicted directions.  
   The cumulative return curve shows marginal gains, suggesting the model captures directional movement slightly better than random, but not strongly enough for significant profit.

   ![Cumulative Return (Window 10)](Data_Visualisation_files/Data_Visualisation_24_4.png)

---

#### **B. Context Window = 30**

1. **Predicted vs Actual Prices (Normalized):**  
   With a longer context window, predictions align more closely with the true values, showing smoother and more stable trend tracking.  
   The extended sequence length allows the Transformer to better understand long-term dependencies in stock movement.

   ![Predicted vs Actual (Window 30)](Data_Visualisation_files/pred_30.png)

2. **Cumulative Return Curve:**  
   The trading simulation for the 30-day window displays a slightly improved cumulative return.  
   This suggests that incorporating a longer temporal context helps the model identify trend reversals and sustained price movements more effectively.

   ![Cumulative Return (Window 30)](Data_Visualisation_files/cumulative_prof_30.png)

---

### **Analysis**
- **Visual Trend Tracking:** Increasing the context window enhances the model’s ability to align with long-term price trends and reduces noise in predictions.  
- **Trading Simulation:** Both setups produce modest gains, but the 30-day window yields smoother and higher cumulative returns, indicating better directional consistency.  
- **Interpretation:** Stock market prediction remains inherently uncertain, yet these visual results demonstrate that **context length plays a vital role** in how well the Transformer captures temporal dependencies and volatility structures.



---

<!-- ## **5. Insights & Conclusion**
- **Visualization Phase:** Provided strong exploratory understanding of stock dynamics—trend, volatility, and inter-stock relationships.  
- **Modeling Phase:** Demonstrated that while Transformers can learn temporal patterns, pure price-based prediction remains challenging due to market noise and non-stationarity.  
- **Next Steps:**  
  - Incorporate technical indicators (RSI, MACD) and macroeconomic signals.  
  - Use hybrid models combining CNNs + Transformers.  
  - Test adaptive learning rates and attention mechanisms for regime shifts.

--- -->


### **5 Results for AAPL Stock**

**Test Set Metrics:**

| **Metric** | **Value** |
|:------------|----------:|
| **MAE** | 1.6292 |
| **MSE** | 2.7182 |
| **RMSE** | 1.6487 |
| **MAPE** | 86.07% |
| **Directional Accuracy** | 44.57% |

---

#### **Visual Results**

1. **Predicted vs Actual Close Price (Normalized)**  

![AAPL True vs Predicted](Data_Visualisation_files/pred_apple.png)

   The predictions remain nearly flat while the true prices fluctuate sharply, indicating that the model failed to capture temporal dynamics.  

   
2. **Simulated Cumulative Profit (Normalized)**  
      ![AAPL True vs Predicted](Data_Visualisation_files/cum_apple.png)

   The cumulative return curve trends downward, showing poor directional forecasting and overall loss in simulated trading.

---

### **Analysis**
The weak performance on AAPL likely stems from:

- **High volatility and external influence:** Price movements depend on news and global events, which pure price data can’t capture.  
- **Limited features:** Only OHLC and Volume data were used, offering little predictive signal.  

In short, while Transformers can model long-term dependencies, they struggle with **highly efficient, noisy markets like AAPL** without richer contextual features or adaptive mechanisms.




# Option Pricing Methods: Black-Scholes and Binomial Model

For options data, we first begin with an analytical approach using two fundamental option pricing methods widely used in quantitative finance: the **Black-Scholes Model** and the **Binomial Options Pricing Model**. These methods are essential for valuing options and understanding the dynamics of financial derivatives. As an initial analysis and observation, we currently only propose these two analytical method solution for Options data and further will shift towards deep neural networks.

---

## Data Processing

For both the analytical methods, we are only using data taken from Yahoo Finance. We have used the 'yfinance' library to extract the data. The stock that we are concerned with is "AAPL" as it has high consistency in option prices across time.

Traditionally, yfinance consists of historical stock pricing data. However, we can exercise a method named 'yf.Ticker' to configure towards options prices and extract it based on a specific exercising date in the future. Currently, yfinance has options data until '2028-01-21'.

The extracted data has the following data fields "contractSymbol", "strike", "bid", "ask", "impliedVolatility", "price", "remaining". The data does not need require much pre-processing. It is cleaned and normalized values. However, to understand this dataset, we had to conduct exploratory data analysis. We have plotted basic information about the data for a short period of time and observed trends.

A common visualization that is conducted to understand options data is "Implied volatility vs Strike price". This is done to observe the variations in the implied volatility when compared to the strike price, plotted over specific dates. 



![Alt text](scholes_yoptions_files/scholes_yoptions_7_0.png)

![Alt text](scholes_yoptions_files/scholes_yoptions_9_1.png)


## 1. Black-Scholes Model

### Overview

The Black-Scholes model is a mathematical model used for pricing European-style options. Primarily there are two types of Options, European option (which can only be exercised at date of maturity) and American option (which can be exercised anytime before or on the date of maturity). It provides a closed-form analytical formula to calculate the theoretical price of call and put options based on certain assumptions.

---

### Key Assumptions

- **Lognormal Stock Price Movement:** The underlying asset price follows a geometric Brownian motion with constant volatility and a constant drift rate.
- **No Dividends:** The underlying asset pays no dividends or distributions during the life of the option.
- **Frictionless Markets:** No transaction costs or taxes; trading is continuous.
- **Constant Risk-Free Rate:** The risk-free interest rate is known and remains constant throughout the option's life.
- **European Options:** The option can only be exercised at maturity.
- **No Arbitrage:** Markets operate efficiently without arbitrage opportunities.
- **Continuous Trading:** The underlying asset can be bought/sold in any quantity at any time.

---

### Mathematical Formulation

The Black-Scholes equation is a partial differential equation (PDE) derived using stochastic calculus and Ito's lemma that governs the price \( V(S,t) \) of an option as a function of underlying asset price $S$  and time $t$:


$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$


Where:

- \( V(S,t) \) = option price at underlying price \( S \) and time \( t \)  
- $\sigma$ = volatility of the underlying asset  
-  $r$  = risk-free interest rate

---

### Closed-Form Solution for European Call Option

Using boundary conditions and the terminal payoff, the analytical solution for a European call option price at time  $t=0$ (current time) is:


$C = S_0 N(d_1) - K e^{-rT} N(d_2)$


Similarly, for a put option:


$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$


where 

$
\begin{aligned}
d_1 &= \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}} \\
d_2 &= d_1 - \sigma \sqrt{T}
\end{aligned}
$

Definitions:

| Symbol | Meaning                                      |
|---------|----------------------------------------------|
| $S_0$ | Current price of the underlying asset        |
| $K$   | Strike price of the option                     |
| $T$   | Time to maturity (in years)                   |
| $r$   | Constant risk-free interest rate (annualized) |
| $\sigma$ | Volatility of the underlying asset (annualized standard deviation) |
| $N(\cdot)$ | Cumulative distribution function (CDF) of the standard normal distribution |

---





### Interpretation

- $N(d_1)$ can be viewed as the *risk-adjusted probability* that the option will finish in the money, adjusted for dividends and interest.
- $N(d_2)$ is related to the probability adjusted for the strike price discounting.
- The formula ensures that the option price incorporates time value, intrinsic value, and volatility.

---
### Implementation

We have chosen the following time to be our expiry - "expiry = "2026-06-18". Thus, we can take the total time period until the date of expiry from current We have formulted the black-scholes equation through python and have written a custom function:

```

def black_scholes_call(S, X, T, r, sigma):
    if sigma == 0 or T == 0:
        return 0
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0)
                  - X * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

```

We then pass this function through the given dataset and obtain the predicted call price as a data field. We record this observation as a separate column and extract as a csv file. Thus, we have a predicted call option price for the given dataset that is calculated in accordance to black-scholes model.


![Alt text](scholes_yoptions_files/scholes_yoptions_6_0.png "Predicted Call vs Strike Price")

Finally to evaluate our model, we find the $R^2$ score between the predicted Black Scholes call price and the Yahoo Last price. The $R^2$ score was found to be 0.941, which is good.

![Alt text](scholes_yoptions_files/scholes_yoptions_8_0.png "Predicted Call vs Strike Price")



---


### Greeks (Sensitivities)

The Black-Scholes model also allows computation of the option Greeks, which are the sensitivities of the option price to various parameters:

| Greek | Financial Meaning                             | Mathematical Interpretation                      |
|-------|----------------------------------------------|-------------------------------------------------|
| Delta ($\Delta$) | Sensitivity of option price to underlying price changes | $\frac{\partial C}{\partial S}$               |
| Gamma ($\Gamma$) | Rate of change of Delta w.r.t underlying price          | $\frac{\partial^2 C}{\partial S^2}$            |
| Theta ($\Theta$) | Sensitivity to time decay                                   | $\frac{\partial C}{\partial t}$                |
| Vega | Sensitivity to volatility changes                 | $\frac{\partial C}{\partial \sigma}$           |
| Rho | Sensitivity to interest rate changes                 | $\frac{\partial C}{\partial r}$                 |

---

### Limitations

- Only applicable to European options (no early exercise).
- Assumes constant volatility and interest rates, which may not hold in practice.
- Assumes log-normal price distribution.
- Ignores dividends (though extensions exist).

---






## 2. Binomial Options Pricing Model

### Overview

The Binomial Options Pricing Model, introduced by Cox, Ross, and Rubinstein (1979), is a discrete-time approach for option valuation. It constructs a binomial price tree to model the possible paths an underlying asset price can take over the life of an option, enabling valuation of both European and American options.

---

### Advantages

- Suitable for **American options** where early exercise is possible.
- Models the price of the underlying asset as a recombining binomial tree, with up and down moves in each discrete time step.
- Flexible enough to handle dividends, varying volatility, and other features that the Black-Scholes model cannot.
- Can approximate the Black-Scholes price as the number of time steps increases.

---

### Model Construction

#### Price Tree

- Divide the option's life $T$ into $n$ discrete time intervals of length:


$\Delta t = \frac{T}{n}$


- At each step:

$
\begin{cases}
\text{Price moves up by a factor } u = e^{\sigma \sqrt{\Delta t}} \\
\text{Price moves down by a factor } d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}
\end{cases}
$

This ensures the tree recombines.

#### Risk-Neutral Probability

The risk-neutral probability $p$ of an up-move is calculated as:

$
p = \frac{e^{r \Delta t} - d}{u - d}
$

Here:

- $r$ = risk-free interest rate  
- $p$ is not an actual probability but a tool that adjusts expected payoffs so their discounted expected value equals the current option price (no arbitrage condition).

---

### Valuation Process: Backward Induction

1. **Calculate Terminal Payoffs**

At maturity (final nodes), the option payoff is:

- Call option payoff: $\max(S_n - K, 0)$
- Put option payoff: $\max(K - S_n, 0)$

Where $S_n = S_0 \times u^j \times d^{n-j}$, with j up-moves out of n.

2. **Step Back Through the Tree**

For each preceding node i, calculate option value as the discounted expected value under risk-neutral probabilities:

$
C_{i,j} = e^{-r \Delta t} \left[p C_{i+1,j+1} + (1-p) C_{i+1,j}\right]
$

3. **Adjust for Early Exercise (American Options)**

If pricing American options, at each node compare the *intrinsic value* (exercise immediately) with the *continuation value* (holding the option):

$
C_{i,j} = \max \left( \text{intrinsic value}, \ e^{-r \Delta t}[p C_{i+1,j+1} + (1-p) C_{i+1,j}] \right)
$

---

### Implementation

We built a function that translates the the Binomial Option pricer to process throught the dataset. The function is as follows:

```
def binomial_call_price(S, K, T, r, sigma, n=100):
    """
    Compute the European call option price using the Cox-Ross-Rubinstein model.
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Terminal prices
    ST = S * d**np.arange(n, -1, -1) * u**np.arange(0, n+1)
    payoff = np.maximum(ST - K, 0)
    
    # Backward induction
    for i in range(n-1, -1, -1):
        payoff = np.exp(-r * dt) * (p * payoff[1:] + (1 - p) * payoff[:-1])
    
    return payoff[0]
```

For simpler comparison, we have used the European call options data. We finally extract the whole data as a csv file including the predicted Call price.


---


### Advantages and Use Cases

- Straightforward and intuitive approach.
- Handles American exercise features easily.
- Can incorporate dividends and complex conditions.
- Converges to Black-Scholes price as number of steps $n \to \infty$.

---

### Limitations

- Computationally intensive for very large n, though efficient algorithms exist.
- Less elegant analytical solution compared to Black-Scholes.
- More numerical than closed-form.

---

## Summary Comparison

| Feature                      | Black-Scholes Model               | Binomial Options Pricing Model     |
|------------------------------|---------------------------------|-----------------------------------|
| Approach                     | Continuous-time, analytical PDE | Discrete-time, numerical tree     |
| Solution Type                | Closed-form formula              | Numerical iterative procedure     |
| Option Types                 | European only                   | European and American             |
| Dividends Treatment          | No (extensions exist)            | Easily included                   |
| Complexity                  | Requires understanding of stochastic calculus | Conceptually simpler, intuitive  |
| Computational Efficiency     | Very efficient                   | Computationally heavier for large steps |
| Adaptability                | Less flexible                    | Highly flexible                   |

---

# Hybrid GARCH + LSTM Model for Stock Market Prediction

<!-- **Author:** Vijay Krishna Sundaran Saravanan  
**BU ID:** U63589838  
**Developed Using:** Python (PyTorch, ARCH, TA-Lib) in VS Code Jupyter Notebook   -->

---

## Project Overview

This project builds a **hybrid AI-based trading framework** that combines statistical volatility modeling (**GARCH**) with deep learning (**LSTM**) to predict stock market movements and generate algorithmic trading signals.  

The model focuses on **Apple (AAPL)** stock from the NASDAQ dataset (1980–2022).  
It forecasts daily returns and assesses market volatility, enabling **risk-aware trading decisions**.

---

## Dataset

**Source:** [Kaggle – Stock Market Data (Paul Timothy Mooney)](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)  
**Folder Used:** NASDAQ → `AAPL.csv`  
**Date Range:** 1980-12-12 → 2022-12-12  
**Rows:** 10,590  
**Columns:** Date, Open, High, Low, Close, Volume, Adjusted Close  

---

## Project Pipeline

### 1. Data Preprocessing

- Converted the `Date` column to datetime format.  
- Checked for missing or duplicate values.  
- Sorted by date and ensured data continuity.  
- Displayed dataset information, shape, and descriptive statistics.  
- Visualized AAPL’s historical behavior.

#### Visualizations:
- **AAPL Closing Price Over Time**

![alt text](<images/aapl closing price.png>)

- **Trading Volume Over Time**

![alt text](<images/trading volume.png>)

- **Close Price Distribution**

![alt text](<images/close price distribution.png>)

---

### 2. Feature Engineering

Created new financial indicators and technical signals to enrich the dataset:  
- **Log Returns** – daily percentage change of adjusted close.  
- **RSI (Relative Strength Index)** – momentum indicator for overbought/oversold conditions.  
- **MACD & Signal Line** – identifies trend direction and strength.  
- **EMA 20 & EMA 50** – short- and long-term exponential moving averages.  
- **Rolling Volatility (20-day)** – captures short-term volatility changes.  

#### Visualizations:

- **Log Returns Over Time**

![alt text](<images/aapl log returns.png>)

- **RSI (Overbought / Oversold Levels)**

![alt text](<images/aapl rsi.png>)

- **MACD & Signal Line**

![alt text](<images/aapl macd.png>)

- **EMA Indicators (20 & 50)**

![alt text](<images/aapl ema.png>)

- **Rolling Volatility (20-day)**

![alt text](<images/aapl rolling.png>)

- **Feature Correlation Heatmap**

![alt text](<images/aapl heatmap.png>)

---

### 3. GARCH(1,1) Modeling

A **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** model was applied to log returns using the `arch` package.

#### Purpose:
To model **time-varying volatility**, representing how market risk evolves over time.

#### Key Parameters:
| Parameter | Description |
|------------|--------------|
| ω (omega) | Long-term average variance |
| α (alpha) | Reaction to recent volatility shocks |
| β (beta) | Persistence of volatility over time |

Two new columns were created:
- `GARCH_Volatility` (daily standard deviation)  
- `GARCH_AnnVol` (annualized volatility)  

#### Visualization:
- **GARCH Conditional Volatility vs Rolling Volatility**

![alt text](<images/aaple garch.png>)

---

### 4. LSTM Deep Learning Model 

The **LSTM** (Long Short-Term Memory) model learns sequential patterns from time-series data to predict the **next-day log return**.

**Input:** 60-day rolling window of 9 selected features (including GARCH outputs)

#### Model Architecture:
- 2 LSTM layers (128 hidden units each)  
- Fully connected layers (ReLU activation, Dropout=0.2)  
- Output: single value (predicted next-day return)  

| Component | Configuration |
|------------|---------------|
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 25 |
| Batch Size | 32 |
| Parameters | 207,425 |
| Device | CPU |

---

### 5. Evaluation

After training, model predictions were compared to actual test data.

#### Evaluation Metrics:
| Metric | Value |
|--------:|-------:|
| MSE | 0.000339 |
| RMSE | 0.0184 |
| R² | -0.0119 |
| Correlation | 0.013 |

#### Visualization:
- **Model Predictions vs Actual Returns**

![alt text](<images\model prediction.png>)

---

### 6. Trading Simulation

A simple trading logic was applied:

| Condition | Action |
|------------|---------|
| Predicted Return > 0 | BUY |
| Predicted Return < 0 | SELL |
| Else | HOLD |

<!-- Users can input any date (1981–2022) to test predicted vs actual movement. -->

<!-- #### Example:

**Date:** 2022-11-30  
**Predicted Change:** -0.11%  
**Actual Change:** +0.19%  
**Suggested Action:** SELL   -->

---

## Visual Summary

1. **AAPL Closing Price Over Time** – long-term growth visualization.  
2. **RSI, MACD, EMA Indicators** – signal momentum and market trends.  
3. **Rolling & GARCH Volatility** – reflect market risk dynamics.  
4. **LSTM Predictions vs Actual Returns** – demonstrate model learning.  

![alt text](<images/aapl vs.png>)

<!-- ---

## Technologies Used

**Language:** Python  

**Libraries:**
- `pandas`, `numpy`, `matplotlib`, `seaborn`  
- `ta` (technical analysis indicators)  
- `arch` (volatility modeling)  
- `torch` (PyTorch deep learning)  
- `scikit-learn` (scaling and metrics)  
- `plotly` (interactive visualization)

--- -->

## Concept Summary

| Component | Role |
|------------|------|
| **GARCH(1,1)** | Models dynamic volatility and market risk |
| **LSTM** | Learns time-dependent patterns from price data |
| **Hybrid Model** | Combines GARCH’s volatility awareness with LSTM’s trend prediction |

This hybrid design bridges **financial econometrics** and **deep learning**, allowing the system to better adapt to real-world market behavior.



---

## Summary

- Built a complete **hybrid GARCH + LSTM** pipeline integrating both statistical and neural models.  
- Implemented every stage — from **data preprocessing** to **trading signal generation**.  
- Demonstrated a realistic simulation of automated trading logic.  



# Future Work
- **Automated Trading System:**  
  Extend this project into a live **paper trading bot** using the **Alpaca API**, enabling real-time execution of trades based on Transformer model predictions.

- **Performance Optimization:**  
  Implement continuous model retraining and adaptive learning to adjust to evolving market conditions and improve trading decision accuracy.

- **End-to-End Trading Pipeline:**  
  Develop a full pipeline integrating **data ingestion**, **signal generation**, **risk management**, and **portfolio evaluation**, ultimately transitioning from paper trading to a deployable automated trading strategy.


