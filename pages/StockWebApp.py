#description: stock market dashboard to show charts and data on stocks

#import libraries

import streamlit as st
import pandas as pd
from PIL import Image
import yfinance as yf
import requests
import xgboost as xgb
import matplotlib.pyplot as plt


st.set_page_config(page_title="Stock", page_icon="Stock Visualization")
st.markdown("# Plotting Demo")



#add title and image
st.write("""
#Stock market web application
         Visually show data on a stock.
         
         """)



st.sidebar.header = ('User Input')

#Function to get user input

def get_input():
    start_date = st.sidebar.text_input("Start Date:", "2020-01-02")
    end_date = st.sidebar.text_input("End Date:", "2024-01-01")
    stock_symbol = st.sidebar.text_input("Stock Symbol:", "AMZN")
    return start_date, end_date, stock_symbol


#Create function to get the company name

def get_company_name(symbol):
    url = f"https://query2.finance.yahoo.com/v6/finance/quoteSummary/{symbol}?modules=financialData&modules=quoteType&modules=defaultKeyStatistics&modules=assetProfile&modules=summaryDetail&ssl=true"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        company_name = data['quoteSummary']['result'][0]['assetProfile']['longName']
        return company_name
    else:
        return None


    
#function to get proper company data and time frame.


def get_data(stock_symbol, start, end):

    stock_information = yf.Ticker(stock_symbol)
    df = stock_information.history(start = start, end = end).reset_index()
    return df


start, end, symbol = get_input()
df = get_data(symbol, start, end)









st.header = (symbol + "Close Price\n")
st.line_chart(df['Close'])


st.header = (symbol + "Volumne\n")
st.line_chart(df['Volume'])

st.header = ("Data statistics")
st.write(df.describe())


