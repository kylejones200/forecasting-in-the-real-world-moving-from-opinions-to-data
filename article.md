---
author: "Kyle Jones"
date_published: "June 17, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/forecasting-in-the-real-world-moving-from-opinions-to-data-2ebc24c98bbc"
---

# Forecasting in the Real World: Moving From Opinions to Data Why simple methods persist, and what it takes to adopt advanced
forecasting in practice

### **Forecasting in the Real World: Moving From Opinions to Data** 

#### Why simple methods persist, and what it takes to adopt advanced forecasting in practice
Forecasting is the backbone of decision-making in industries that rely on planning, budgeting, and operational efficiency. Companies forecast to anticipate demand, allocate resources, set revenue targets, and manage supply chains. But how familiar are practitioners with forecasting methods? Which methods do they use most often? And how has that changed over time?

In one of the earliest empirical studies on forecasting usage, Mentzer and Cox (1984) surveyed industry professionals to assess their familiarity with forecasting techniques. The results are revealing. Subjective methods such as jury of executive opinion and sales force composite were highly familiar, with over 75 percent of respondents reporting strong familiarity. Quantitative methods such as moving averages and straight-line projections were similarly well-known, with more than 80 percent reporting they were very familiar.


<figcaption>Data from Mentzer and Cox (1984)</figcaption>


But there was a steep drop when it came to more sophisticated techniques. Life cycle analysis, classical decomposition, and Box-Jenkins (ARIMA) models were familiar to fewer than half the respondents. Only 26 percent reported strong familiarity with Box-Jenkins models, and 65 percent were completely unfamiliar.

These results paint a clear picture: in 1984, most forecasting in practice leaned on simple, easy-to-explain methods. Complex statistical modeling had not yet gained widespread acceptance.

### What Has Changed Since Then?
Forty years later, the gap between theory and practice has narrowed, but not vanished. Many businesses still rely on basic methods. Moving averages and straight-line projections remain dominant, especially in sales forecasting. However, several major changes have reshaped the landscape:

1.  [**Rise of Predictive Analytics**: Companies now use historical data alongside real-time signals to improve accuracy. This includes web traffic, economic indicators, and customer behavior.]
2.  [**Adoption of Machine Learning**: Methods like gradient boosting, random forests, and LSTM neural networks are being used to forecast demand, sales, and maintenance needs. These methods can uncover nonlinear patterns that classical methods miss.]
3.  [**Automated Forecasting Platforms**: Tools like Amazon Forecast, Azure ML, and open-source libraries like Prophet, Darts, and Nixtla make it easier to deploy complex models without deep statistical training.]
4.  [**Cloud Computing and APIs**: The shift to cloud platforms has made real-time, large-scale forecasting more accessible. Forecasts can now be generated across millions of SKUs or locations in seconds.]

### Subjective Forecasting Still Persists
Despite technological progress, subjective methods have not disappeared. Jury of executive opinion remains widespread in strategic planning, especially where data is sparse or the environment is volatile. Sales force composites are still used in retail and B2B sectors where frontline teams have direct market insight.

Customer expectations are now collected through digital channels --- surveys, reviews, and sentiment analysis. This subjective input feeds into hybrid models that blend human intuition with statistical inference.

### Case Studies in Practice
### Retail
In the retail sector, forecasting determines how much inventory to order, where to store it, and how to price it. A global fashion retailer might forecast demand for a winter jacket using past sales, current weather patterns, and social media mentions. They might use time series models for baseline trends and a decision tree model to adjust for promotions.

### Energy
Utilities use forecasting to predict energy consumption and generation needs. Here, Box-Jenkins models have seen a resurgence, particularly in combination with weather forecasts. Wind power companies use Monte Carlo simulations to estimate the probability distribution of output under different wind conditions.

### Manufacturing
Forecasting in manufacturing is used to plan production schedules, order raw materials, and anticipate equipment maintenance. Moving average methods are used for short-term production planning, while machine learning models help forecast machinery failures in predictive maintenance programs.

### Finance
In financial services, forecasts influence credit scoring, revenue projections, and risk modeling. Trend-line analysis and regression models remain widely used, but are increasingly supplemented with neural networks that model temporal dependencies in stock prices or customer behavior.

### Method Fatigue and Overfitting
One reason complex models haven't completely taken over is that they often require significant tuning, and can be sensitive to assumptions. Overfitting is a real risk in low-data environments. For many businesses, a simple average or exponential smoothing model works well enough --- and is easier to explain to stakeholders.

### Training and Tooling Gaps
Familiarity still shapes what gets used. While modern forecasting tools exist, many teams lack training in time series modeling, let alone deep learning. As a result, regression and exponential smoothing still form the core of most forecasting systems.

This reflects a truth that hasn't changed since 1984: people tend to use what they understand.

### The Role of Monte Carlo Simulation
Monte Carlo simulation has become more common as businesses embrace uncertainty. Instead of producing a single forecast, Monte Carlo methods generate a range of possible futures by simulating random draws from a distribution. This is especially useful in capital budgeting, portfolio management, and strategic risk planning.

For example, an oil company might model future cash flows under different scenarios for crude oil prices, each one informed by probabilistic price forecasts.

### Choosing the Right Method
There is no one-size-fits-all. Choosing the right forecasting method depends on:

- The amount and quality of historical data
- The predictability of the environment
- The technical capacity of the organization
- The cost of being wrong

In some cases, combining methods yields the best results. This is called ensemble forecasting, and it mirrors the logic of investment diversification. If one model performs poorly, others might offset the loss.

### Forecasting as Organizational Learning
Forecasts are not just numbers. They are expressions of what an organization believes about the future. As models are tested and updated, companies learn. Forecasting becomes a feedback loop --- one that can drive continuous improvement.

But this only happens when forecasts are taken seriously, tracked, and refined over time. Forecasting will never be perfect. But the goal is not perfection --- it is decision support. As data infrastructure improves and statistical literacy increases, companies can move beyond gut feeling and trend lines to more robust, defensible forecasts.

The future of forecasting is not defined by tools alone. It is defined by how well people use them.

#### Streamlit app
This simple app lets you adjust the forecasting method and see the change to different accuracy metrics.

```python
# forecasting_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Simulate time series data
np.random.seed(42)
n_periods = 100
time = np.arange(n_periods)
trend = 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(scale=5, size=n_periods)
data = trend + seasonality + noise
ts = pd.Series(data, index=pd.date_range("2020-01-01", periods=n_periods, freq="ME"))

# Split into train and test
train, test = ts[:-12], ts[-12:]

# Forecasting methods
def forecast_moving_average(train, test):
    forecast = [train[-12:].mean()] * len(test)
    return forecast

def forecast_simple_exp_smoothing(train, test):
    model = SimpleExpSmoothing(train).fit()
    forecast = model.forecast(len(test))
    return forecast

def forecast_holt(train, test):
    model = Holt(train).fit()
    forecast = model.forecast(len(test))
    return forecast

def forecast_hwes(train, test):
    model = HWES(train, seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(len(test))
    return forecast

def forecast_arima(train, test):
    model = ARIMA(train, order=(1, 1, 1)).fit()
    forecast = model.forecast(len(test))
    return forecast

def theils_u(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return num / denom


methods = {
    "Moving Average": forecast_moving_average,
    "Simple Exponential Smoothing": forecast_simple_exp_smoothing,
    "Holt's Linear Trend": forecast_holt,
    "Holt-Winters": forecast_hwes,
    "ARIMA": forecast_arima
}

# Streamlit app
st.title("Forecasting Method Comparison App")

method = st.selectbox("Choose a forecasting method:", list(methods.keys()))
forecast_func = methods[method]
forecast = forecast_func(train, test)

# Evaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100
theil_u_val = theils_u(test.values, forecast)

st.subheader("Forecast Evaluation Metrics")
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**MAPE**: {mape:.2f}%")
st.write(f"**Theil's U**: {theil_u_val:.3f}")

st.subheader("Forecast Plot")
fig, ax = plt.subplots(figsize=(10, 5))
train.plot(ax=ax, label="Train")
test.plot(ax=ax, label="Test")
pd.Series(forecast, index=test.index).plot(ax=ax, label="Forecast", linestyle="--")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

xticks = pd.date_range(start=train.index[0], end=test.index[-1], freq='YS-JAN')
ax.set_xticks(xticks)
ax.set_xticklabels([dt.strftime('%Y') for dt in xticks], fontsize=10)

# Tighten layout and add legend
ax.legend()
plt.tight_layout()
st.pyplot(fig)
```

To run this locally use \`\`\`streamlit run forecasting_app.py\`\`\`
