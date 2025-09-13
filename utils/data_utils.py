import ccxt
import pandas as pd
import numpy as np
from config import START_DATE, END_DATE

def load_crypto_data(symbol="BTC/USDT", start_date=START_DATE, end_date= END_DATE, timeframe="1d"):
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date + "T00:00:00Z")
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=2000)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        # next since = timestamp cuối + 1ms
        since = ohlcv[-1][0] + 1
        if end_date and pd.to_datetime(since, unit="ms") > pd.to_datetime(end_date):
            break
    df = pd.DataFrame(all_ohlcv, columns=["Timestamp","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Date", inplace=True)
    if end_date:
        df = df.loc[:end_date]
    return df


def add_features(df):
    df['H-L'] = df['High'] - df['Low']
    df['O-C'] = df['Open'] - df['Close']
    for ma in [7, 14, 21]:
        df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()
    df['SD_7'] = df['Close'].rolling(window=7).std()
    df['SD_21'] = df['Close'].rolling(window=21).std()
    df.dropna(inplace=True)
    return df

def create_sequences(scaled_x, scaled_y, pre_day):
    x, y = [], []
    for i in range(pre_day, len(scaled_x)):
        x.append(scaled_x[i-pre_day:i])
        y.append(scaled_y[i])
    return np.array(x), np.array(y)
