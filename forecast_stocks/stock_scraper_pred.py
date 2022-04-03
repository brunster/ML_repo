######
# Stock Web App 
#
# Author: Bryan Bruno
#
# allows users to select companies on the S&P500
# and forecast their stock value using FB Prophet
# all data retrival and forecasts are performed live
######

#import libraries
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import date
from plotly import graph_objs as pgo
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


#declaring date and period parameters
START_DATE = "2016-01-01"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
P_START = 1
P_END = 5
sec_tickers = []


#uses yahoo finance api to load all stock info
def load_data(ticker):
    data = yf.download(ticker,
                   START_DATE, CURRENT_DATE,
                   auto_adjust = True)
    data.reset_index(inplace = True)
    data.dropna(axis = 1, inplace = True)
    return data

#graphs all historic closing data
def plot_historic():
    fig = pgo.Figure()
    fig.add_trace(pgo.Scatter(x = data["Date"], y = data["Close"]))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

# joins securities and tickers into single list
def company_tickers():
    for t in range(len(tickers)):
        sec_tick = " - ".join([tickers[t], sec[t]])
        sec_tickers.append(sec_tick)    
    return sec_tickers

# indexes dict for more detailed pulldown menu search
def menu_choices(option):
    i = list(choices.keys()).index(option)
    return sec_tickers[i]

#setting page title
st.title("S&P 500 Stock Forecast App")

#scraping wikipedia for stock tickers and securities
tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Symbol.to_list()
sec = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Security.to_list()

#enriching tickers and securities
sec_tickers = company_tickers()
choices = dict(zip(tickers, sec_tickers))

#linking the ticker menu to the specific ticker lookup
ticker_select = st.selectbox("Select ",
                             options = list(choices.keys()),
                             format_func = menu_choices) 

#live update of data download
load_state = st.text("Downloading Stock Data...")
data = load_data(ticker_select)
load_state.text("Stock Data Download Complete!")

#displays historic closing values
st.subheader("Historic Positions")
plot_historic()

#lists last 7 days of trading history
st.subheader("Last 7 Days Trading Positions")
st.write(data.tail(7))

st.markdown("""
#### This App forecasts the selected stock value using Prophet. The default number of years to forecast is 1. You can choose up to 5 using the slider!
""")

#user may select the number of years they'd like to forecast
st.subheader("Choose Number of Years to Predict")
years = st.slider("", P_START, P_END)
days_total = years * 365

#creating a train set with current data selected
#prophet requires the 'ds' and 'y' labels for forecasting
train = data[["Date", "Close"]]
train = train.rename(columns = {"Date" : "ds", "Close" : "y"})

#requiring weekly and annual seasonality
#forecasts are on close - daily excluded
model = Prophet(daily_seasonality = False,
                weekly_seasonality = True,
                yearly_seasonality = True)

#training model
model.fit(train)
#creating forecast dataframe
future = model.make_future_dataframe(periods = days_total)
#forecasting
y_pred = model.predict(future)

#plotting forecasted results
#providing the final 60 day
#forecast and weights available to view
st.subheader("Closing Forecast")
fig = plot_plotly(model, y_pred)
st.plotly_chart(fig)
st.subheader("Last 60 Day Forecast")
st.write(y_pred.tail(30))