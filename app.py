import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
import time

st.set_page_config(page_title="TA Confluence – India (NSE Intraday + Position)", layout="wide")

# --- Helpers ---
def normalize_symbol(user_sym: str, exchange_suffix: str) -> str:
    s = (user_sym or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s[:-3]
    return f"{s}{exchange_suffix}"

def nse_base(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").replace(".BO", "")

# --- NSE API ---
def _nse_session():
    s = requests.Session()
    ua = {
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "accept": "application/json,text/plain,*/*",
        "accept-language": "en-US,en;q=0.9",
    }
    s.get("https://www.nseindia.com", headers=ua, timeout=15)
    time.sleep(1.0)
    return s, ua

def fetch_nse_intraday(symbol: str, interval: str = "5min") -> pd.DataFrame:
    base = nse_base(symbol)
    index_code = f"{base}EQN"
    s, ua = _nse_session()
    url = f"https://www.nseindia.com/api/chart-databyindex?index={index_code}"
    headers = {**ua, "referer": f"https://www.nseindia.com/get-quotes/equity?symbol={base}"}
    r = s.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"NSE HTTP {r.status_code}")
    data = r.json() or {}
    candles = data.get("candles")
    if not candles:
        raise RuntimeError("No intraday candles from NSE")
    rows = []
    for c in candles:
        if len(c) < 6: continue
        ts = pd.to_datetime(int(c[0]), unit="ms")
        rows.append({"Datetime": ts, "Open": float(c[1]), "High": float(c[2]),
                     "Low": float(c[3]), "Close": float(c[4]), "Volume": float(c[5])})
    df = pd.DataFrame(rows).sort_values("Datetime").set_index("Datetime")
    rule = {"1min": "1T", "5min": "5T", "15min": "15T"}.get(interval.lower(), "5T")
    if rule != "1T":
        df = df.resample(rule).agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    return df

# --- Yahoo Fallback ---
def fetch_yf_intraday(symbol: str, interval: str = "5min", lookback_days: int = 7) -> pd.DataFrame:
    tf_map = {"1min": "1m", "5min": "5m", "15min": "15m"}
    tf = tf_map.get(interval.lower(), "5m")
    max_days = 7 if tf == "1m" else 60
    t = yf.Ticker(symbol)
    try:
        df = t.history(period=f"{min(lookback_days, max_days)}d", interval=tf, auto_adjust=True)
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                                "Close": "Close", "Volume": "Volume"}).dropna()
        return df
    except Exception:
        return pd.DataFrame()

# --- Indicators ---
def add_indicators(df):
    df["EMA9"] = ta.ema(df["Close"], 9)
    df["EMA21"] = ta.ema(df["Close"], 21)
    df["SMA50"] = ta.sma(df["Close"], 50)
    df["SMA200"] = ta.sma(df["Close"], 200)
    df["RSI14"] = ta.rsi(df["Close"], 14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
    return df

def suggest_trade(df, threshold):
    last = df.iloc[-1]
    score = 0
    if last["EMA9"] > last["EMA21"]: score += 1
    if last["SMA50"] > last["SMA200"]: score += 1
    if last["RSI14"] > 60: score += 1
    return {"score": score, "valid": score >= threshold,
            "entry": last["Close"],
            "stop": last["Close"] - 1.5*last["ATR14"],
            "target": last["Close"] + 3*last["ATR14"]}

# --- UI ---
st.title("Technical Analysis – Free Intraday (NSE → Yahoo)")

mode = st.radio("Mode", ["intraday", "position"], horizontal=True)
colA, colB = st.columns([2,1])
with colA:
    user_symbol = st.text_input("Symbol", "RELIANCE")
with colB:
    exchange = st.selectbox("Exchange", ["NSE (.NS)", "BSE (.BO)"])

suffix = ".NS" if exchange.startswith("NSE") else ".BO"
symbol = normalize_symbol(user_symbol, suffix)

if mode == "intraday":
    ival = st.selectbox("Interval", ["1min","5min","15min"], index=1)
    if st.button("Analyze Intraday"):
        try:
            df, src = None, None
            try:
                df = fetch_nse_intraday(symbol, interval=ival)
                src = "NSE"
            except Exception as e:
                st.info(f"NSE failed: {e}. Trying Yahoo…")
                df = fetch_yf_intraday(symbol, interval=ival)
                src = "Yahoo"
            if df is None or df.empty:
                st.error("No intraday data found.")
                st.stop()
            df = add_indicators(df)
            idea = suggest_trade(df, threshold=2)
            st.write(f"Source: {src}")
            st.line_chart(df["Close"])
            st.write(idea)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    if st.button("Analyze Position"):
        df = yf.download(symbol, period="5y", interval="1d", auto_adjust=True)
        st.line_chart(df["Close"])
