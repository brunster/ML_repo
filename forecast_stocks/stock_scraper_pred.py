# Import packages
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import date
from plotly import graph_objs as pgo
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


START_DATE = "2015-01-01"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")

def load_data(ticker):
    data = yf.download(ticker,
                   START_DATE, CURRENT_DATE,
                   auto_adjust = True, )
    data.reset_index(inplace = True)
    data.dropna(axis = 1, inplace = True)
    return data

def plot_historic():
    fig = pgo.Figure()
    fig.add_trace(pgo.Scatter(x = data["Date"], y = data["Close"]))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

st.title("Stock Prediction Web App S&P500")

tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Symbol.to_list()
sec = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Security.to_list()

sec_tickers = []
for t in range(len(tickers)):
    sec_tick = " - ".join([sec[t], tickers[t]])
    sec_tickers.append(sec_tick)

choices = dict(zip(sec, tickers))
ticker_select = st.selectbox("Select ", options = sec_tickers, format_func = lambda x: tickers[x])

load_state = st.text("Downloading Stock Data...")
data = load_data(ticker_select)
load_state.text("Stock Data Download Complete!")

st.subheader("Historic Positions")
plot_historic()

st.subheader("Choose Number of Years to Predict")
years = st.slider("", 1, 5)
days_total = years * 365

st.subheader("Most Recent Closing Positions")
st.write(data.tail(7))

train = data[["Date", "Close"]]
train = train.rename(columns = {"Date" : "ds", "Close" : "y"})

model = Prophet(daily_seasonality = False,
                weekly_seasonality = True,
                yearly_seasonality = True)

model.fit(train)
future = model.make_future_dataframe(periods = days_total, )
y_pred = model.predict(future)

st.subheader("Closing Forecast")
fig = plot_plotly(model, y_pred)
st.plotly_chart(fig)
st.write(y_pred.tail())
