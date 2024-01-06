import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
import streamlit as st
import matplotlib.pyplot as plt
from pandas_datareader import data as web


st.set_page_config(page_title="Portfolio Optimization", page_icon="Portfolio Optimization")
st.markdown("# Portfolio Optimization Demo")


#add title and image
st.write("""
Portfolio Optimization Demo
         
         """)



st.sidebar.header = ('User Input')

#Function to get user input

def get_input():
    start_date = st.sidebar.text_input("Start Date:", "2020-01-02")
    end_date = st.sidebar.text_input("End Date:", "2024-01-01")
    tickers = st.sidebar.text_input("Enter multiple ticker symbols (seperated by commas)", "AMZN, GOOG, META")

    tickers = tickers.split(',')
    return start_date, end_date, tickers



end_date = datetime.today()

start_date = end_date - timedelta(days = 5 * 365)
print(start_date)
adj_close_df = pd.DataFrame()
print(adj_close_df)
start_date, end_date, tickers = get_input()

for ticker in tickers:
    data = yf.download(ticker, start = start_date, end = end_date)
    adj_close_df[ticker] = data['Adj Close']

#For graph with stock prices
    
df_graph = pd.DataFrame()
for ticker in tickers:
    df_graph[ticker] = yf.download(ticker, start = start_date, end = end_date)['Adj Close']
fig2, ax2 = plt.subplots()
for c in df_graph.columns.values:
    ax2.plot(df_graph[c], label=c)

ax2.set_title('Portfolio stock prices')
ax2.set_xlabel('Date', fontsize=9)
ax2.set_ylabel('Adj. Price USD ($)', fontsize=18)
ax2.legend(df_graph.columns.values, loc='upper left')

# Display the plot in Streamlit app using st.pyplot(fig)
st.pyplot(fig2)



log_returns = np.log(adj_close_df/adj_close_df.shift(1))

log_returns = log_returns.dropna()


cov_matrix = log_returns.cov()*252
print(cov_matrix)

def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return (weights, log_returns):
    if log_returns.empty:
        return 0
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)



fred = Fred(api_key='bd01fcf7d3f596933c52b8c55fd83abe')
ten_year_treasury_rate = fred.get_series_latest_release('GS10') /100

risk_free_rate = ten_year_treasury_rate.iloc[-1]
print(risk_free_rate)

def neg_sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)


constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(tickers))]

initial_weights = np.array([1/len(tickers)]*len(tickers))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args = (log_returns, cov_matrix, risk_free_rate), method = 'SLSQP', constraints = constraints, bounds= bounds)

optimal_weights = optimized_results.x



print("Optimal weights:")
explode = [0.1] * len(optimal_weights)

fig1, ax1 = plt.subplots()
ax1.pie(optimal_weights, explode= explode, labels= tickers, autopct= '%1.1f%%', shadow = True, startangle=90)

ax1.axis('equal')
st.pyplot(fig1)



optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)




st.subheader("Portfolio Statistics:")
st.write(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
st.write(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")



