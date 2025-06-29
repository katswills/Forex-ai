import MetaTrader5 as mt5
import pandas as pd
import ta
import joblib
from datetime import datetime
from xgboost import XGBClassifier
import streamlit as st
import plotly.graph_objects as go

# Initialize MetaTrader 5
if not mt5.initialize():
    st.error("MT5 initialization failed")
    exit()

# Get historical data
symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Feature engineering
df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
df['rsi'] = ta.momentum.rsi(df['close'], window=14)
df['macd'] = ta.trend.macd(df['close']).macd()
df.dropna(inplace=True)

# Simple target for training
df['target'] = (df['close'].shift(-5) > df['close']).astype(int)

# Train model
features = ['ema_50', 'ema_200', 'rsi', 'macd']
model = XGBClassifier()
model.fit(df[features], df['target'])

# Predict
df['prediction'] = model.predict(df[features])

# Streamlit UI
st.title("Forex AI Trader - EURUSD")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
fig.add_trace(go.Scatter(x=df[df['prediction']==1]['time'], y=df[df['prediction']==1]['close'], mode='markers', marker=dict(color='green', size=8), name='Buy Signal'))
fig.add_trace(go.Scatter(x=df[df['prediction']==0]['time'], y=df[df['prediction']==0]['close'], mode='markers', marker=dict(color='red', size=8), name='Sell Signal'))
st.plotly_chart(fig)