import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="TA Confluence – India (Intraday & Position)", layout="wide")

# ------------------------
# Helpers
# ------------------------

def nse_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    # Default to NSE
    return f"{s}.NS"

def fetch_yf_daily(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df.dropna()
    else:
        df = pd.DataFrame()
    return df

def fetch_alpha_intraday(symbol: str, api_key: str, interval: str = "5min", outputsize: str = "full") -> pd.DataFrame:
    """Fetch intraday data from Alpha Vantage (free)."""
    base = symbol.upper()
    if base.endswith(".NS"):
        use = base
    elif base.endswith(".BO"):
        use = base.replace(".BO", ".BSE")
    else:
        use = base + ".NS"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": use,
        "interval": interval,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": api_key
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Alpha Vantage HTTP {r.status_code}")
    data = r.json()
    key = None
    for k in data.keys():
        if "Time Series" in k:
            key = k
            break
    if key is None:
        note = data.get("Note") or data.get("Information") or data
        raise RuntimeError(f"Alpha Vantage error/limit: {note}")
    ts = data[key]
    rows = []
    for t, ohlcv in ts.items():
        rows.append({
            "Datetime": pd.to_datetime(t),
            "Open": float(ohlcv.get("1. open", 0.0)),
            "High": float(ohlcv.get("2. high", 0.0)),
            "Low": float(ohlcv.get("3. low", 0.0)),
            "Close": float(ohlcv.get("4. close", 0.0)),
            "Volume": float(ohlcv.get("5. volume", 0.0)),
        })
    df = pd.DataFrame(rows).sort_values("Datetime").set_index("Datetime")
    return df

def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA9"] = ta.ema(out["Close"], 9)
    out["EMA21"] = ta.ema(out["Close"], 21)
    out["EMA50"] = ta.ema(out["Close"], 50)
    out["SMA50"] = ta.sma(out["Close"], 50)
    out["SMA200"] = ta.sma(out["Close"], 200)
    macd = ta.macd(out["Close"])
    if macd is not None and not macd.empty:
        out["MACD"] = macd.iloc[:,0]
        out["MACDsig"] = macd.iloc[:,1]
        out["MACDhist"] = macd.iloc[:,2]
    else:
        out["MACD"] = np.nan
        out["MACDsig"] = np.nan
        out["MACDhist"] = np.nan
    out["RSI14"] = ta.rsi(out["Close"], 14)
    stoch = ta.stoch(out["High"], out["Low"], out["Close"], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        out["STOCHk"] = stoch.iloc[:,0]
        out["STOCHd"] = stoch.iloc[:,1]
    else:
        out["STOCHk"] = np.nan
        out["STOCHd"] = np.nan
    bb = ta.bbands(out["Close"], 20, 2)
    if bb is not None and not bb.empty:
        out["BB_up"] = bb.iloc[:,0]
        out["BB_mid"] = bb.iloc[:,1]
        out["BB_low"] = bb.iloc[:,2]
    else:
        out["BB_up"] = out["BB_mid"] = out["BB_low"] = np.nan
    out["ATR14"] = ta.atr(out["High"], out["Low"], out["Close"], 14)
    out["OBV"] = ta.obv(out["Close"], out["Volume"])
    try:
        out["ADL"] = ta.ad(out["High"], out["Low"], out["Close"], out["Volume"])
    except Exception:
        out["ADL"] = np.nan
    return out

def vwap_series(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df or df["Volume"].isna().all():
        return pd.Series(index=df.index, dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = df.index.tz_localize(None).date if isinstance(df.index[0], pd.Timestamp) else df.index.date
    day_series = pd.Series(day, index=df.index)
    vwap = pd.Series(index=df.index, dtype=float)
    for d, g in df.groupby(day_series):
        pv = (tp.loc[g.index] * df.loc[g.index, "Volume"]).cumsum()
        vv = df.loc[g.index, "Volume"].cumsum().replace(0, np.nan)
        vwap.loc[g.index] = pv / vv
    return vwap

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = ta.atr(df["High"], df["Low"], df["Close"], period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    final_upperband = pd.Series(index=df.index, dtype=float)
    final_lowerband = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    for i in range(len(df)):
        if i == 0:
            final_upperband.iloc[i] = upperband.iloc[i]
            final_lowerband.iloc[i] = lowerband.iloc[i]
            trend.iloc[i] = 1
        else:
            final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i-1]) if df["Close"].iloc[i-1] > final_upperband.iloc[i-1] else upperband.iloc[i]
            final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i-1]) if df["Close"].iloc[i-1] < final_lowerband.iloc[i-1] else lowerband.iloc[i]
            trend.iloc[i] = 1 if df["Close"].iloc[i] > final_upperband.iloc[i-1] else (-1 if df["Close"].iloc[i] < final_lowerband.iloc[i-1] else trend.iloc[i-1])
    st_line = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        st_line.iloc[i] = final_lowerband.iloc[i] if trend.iloc[i] == 1 else final_upperband.iloc[i]
    return st_line

def compute_scores(df: pd.DataFrame, mode: str = "intraday") -> pd.DataFrame:
    out = df.copy()
    out = add_common_indicators(out)
    if mode == "intraday":
        out["VWAP"] = vwap_series(out)
        out["Supertrend"] = supertrend(out, period=10, multiplier=3.0)

    def s_trend(i):
        s = 0
        c = out["Close"].iloc[i]
        ema9 = out["EMA9"].iloc[i]
        ema21 = out["EMA21"].iloc[i]
        ema50 = out["EMA50"].iloc[i]
        sma50 = out["SMA50"].iloc[i]
        sma200 = out["SMA200"].iloc[i]
        if mode == "intraday":
            if not np.isnan(ema9) and not np.isnan(ema21): s += 1 if ema9 > ema21 else -1
            if not np.isnan(ema21) and not np.isnan(ema50): s += 1 if ema21 > ema50 else -1
            if not np.isnan(c) and not np.isnan(ema21): s += 1 if c > ema21 else -1
        else:
            if not np.isnan(ema21) and not np.isnan(ema50): s += 1 if ema21 > ema50 else -1
            if not np.isnan(sma50) and not np.isnan(sma200): s += 1 if sma50 > sma200 else -1
            if not np.isnan(c) and not np.isnan(sma50): s += 1 if c > sma50 else -1
        return s

    def s_momentum(i):
        s = 0
        macd = out["MACD"].iloc[i]; sig = out["MACDsig"].iloc[i]
        rsi = out["RSI14"].iloc[i]
        st_k = out["STOCHk"].iloc[i]; st_d = out["STOCHd"].iloc[i]
        if not np.isnan(macd) and not np.isnan(sig):
            s += 1 if macd > sig else -1
        if not np.isnan(rsi):
            if rsi > 60: s += 1
            elif rsi < 40: s -= 1
        if not np.isnan(st_k) and not np.isnan(st_d):
            s += 1 if st_k > st_d else -1
        return s

    def s_volatility(i):
        s = 0
        c = out["Close"].iloc[i]; mid = out["BB_mid"].iloc[i]
        if not np.isnan(c) and not np.isnan(mid):
            s += 1 if c > mid else -1
        return s

    def s_volume(i):
        s = 0
        cmf_like = out["ADL"].iloc[i]
        if not np.isnan(cmf_like):
            s += 1 if cmf_like > 0 else -1
        if i > 0 and not np.isnan(out["OBV"].iloc[i]) and not np.isnan(out["OBV"].iloc[i-1]):
            s += 1 if out["OBV"].iloc[i] > out["OBV"].iloc[i-1] else -1
        if mode == "intraday":
            v = out["VWAP"].iloc[i]
            if not np.isnan(v):
                s += 1 if out["Close"].iloc[i] >= v else -1
        return s

    scores = []
    for i in range(len(out)):
        stv = s_trend(i)
        smo = s_momentum(i)
        svo = s_volatility(i)
        svm = s_volume(i)
        total = stv + smo + svo + svm
        scores.append((stv, smo, svo, svm, total))
    out[["S_trend","S_mom","S_vol","S_volm","S_total"]] = pd.DataFrame(scores, index=out.index)
    return out

def suggest_trade(df: pd.DataFrame, mode: str, threshold: int):
    last = df.iloc[-1]
    atr = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else None
    close = float(last["Close"])
    valid = last["S_total"] >= threshold
    entry = close if valid else None
    if entry and atr:
        sl_mult = 1.5 if mode == "intraday" else 2.0
        sl = entry - sl_mult * atr
        tp = entry + 2 * (entry - sl)
    else:
        sl, tp = None, None
    horizon = "15–60 minutes" if mode == "intraday" else "5–15 trading days"
    return {
        "valid": bool(valid),
        "entry": entry,
        "stop": sl,
        "target": tp,
        "horizon": horizon,
        "score": int(last["S_total"])
    }

def backtest(df: pd.DataFrame, mode: str, threshold: int):
    trades = 0
    wins = 0
    pnls = []
    in_trade = False
    entry = sl = tp = None
    for i in range(1, len(df)):
        c = float(df["Close"].iloc[i])
        s = float(df["S_total"].iloc[i])
        atr_i = float(df["ATR14"].iloc[i]) if not np.isnan(df["ATR14"].iloc[i]) else None
        if not in_trade:
            if s >= threshold and float(df["S_total"].iloc[i-1]) < threshold:
                if atr_i is None: 
                    continue
                sl_mult = 1.5 if mode == "intraday" else 2.0
                entry = c
                sl = entry - sl_mult * atr_i
                tp = entry + 2 * (entry - sl)
                in_trade = True
        else:
            exit_now = False
            if atr_i is None:
                exit_now = True
            if c <= sl or c >= tp:
                exit_now = True
            if s <= 0:
                exit_now = True
            if exit_now:
                trades += 1
                pnl = (c - entry) / entry
                pnls.append(pnl)
                if pnl > 0: wins += 1
                in_trade = False
                entry = sl = tp = None
    if trades == 0:
        return {"trades": 0, "win_rate": None, "avg_return": None, "best": None, "worst": None}
    pn = np.array(pnls)
    return {
        "trades": trades,
        "win_rate": float((pn > 0).mean()),
        "avg_return": float(pn.mean()),
        "best": float(pn.max()),
        "worst": float(pn.min())
    }

# ------------------------
# UI
# ------------------------

st.title("Technical Analysis Confluence – Indian Stocks (Intraday & Position)")
st.caption("Free-data build: Yahoo Finance (daily) + Alpha Vantage (intraday). No system is 100% accurate—this app reports confluence + historical stats.")

mode = st.radio("Mode", ["intraday", "position"], horizontal=True, index=0)

colA, colB, colC = st.columns([1.4,1,1])
with colA:
    user_symbol = st.text_input("Company or symbol (e.g., RELIANCE, TCS, HDFCBANK, SBIN)", "RELIANCE")
with colB:
    exchange = st.selectbox("Exchange", ["NSE (.NS)", "BSE (.BO)"], index=0)
with colC:
    threshold = st.slider("Signal threshold", min_value=1, max_value=8, value=3, help="Minimum S_total for a new long entry.")

suffix = ".NS" if exchange.startswith("NSE") else ".BO"
symbol = nse_symbol(user_symbol).replace(".NS", suffix).replace(".BO", suffix)

if mode == "intraday":
    st.subheader("Intraday data source: Alpha Vantage (free)")
    default_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    api_key = st.text_input("Alpha Vantage API Key", value=default_key, type="password", help="Get a free key at alphavantage.co (or set ALPHAVANTAGE_API_KEY in Streamlit Cloud Secrets)")
    ival = st.selectbox("Interval", ["1min","5min","15min"], index=1)
    lookback_days = st.number_input("Lookback days (for display)", min_value=1, max_value=120, value=15, step=1)
    if st.button("Analyze intraday", type="primary"):
        if not api_key:
            st.error("Please enter your Alpha Vantage API key.")
            st.stop()
        try:
            df = fetch_alpha_intraday(symbol, api_key, interval=ival, outputsize="full")
        except Exception as e:
            st.error(f"Failed to fetch intraday data: {e}")
            st.stop()
        if len(df) > 0:
            df = df[df.index >= (df.index.max() - pd.Timedelta(days=int(lookback_days)))]
        if df.empty:
            st.warning("No intraday data found for this symbol/interval.")
            st.stop()
        df = compute_scores(df, mode="intraday")
        idea = suggest_trade(df, mode="intraday", threshold=int(threshold))
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"### {symbol} – Intraday price with Bollinger Bands")
            st.line_chart(df[["Close","BB_up","BB_mid","BB_low"]].dropna())
            st.markdown("### Confluence score (S_total)")
            st.line_chart(df[["S_total"]].dropna())
        with col2:
            st.markdown("### Latest read")
            st.metric("S_total", f"{idea['score']}")
            st.metric("Close", f"{df['Close'].iloc[-1]:.2f}")
            st.metric("ATR(14)", f"{df['ATR14'].iloc[-1]:.2f}")
            if idea["valid"]:
                st.success("**Long idea**")
                st.write(f"- **Entry** ≈ {idea['entry']:.2f}")
                st.write(f"- **Stop-loss** ≈ {idea['stop']:.2f} (≈ 1.5×ATR)")
                st.write(f"- **Target** ≈ {idea['target']:.2f} (≈ 2R)")
                st.write(f"- **Time window:** {idea['horizon']}")
            else:
                st.info("No fresh long signal at the chosen threshold. Consider waiting or lowering the threshold.")
        st.markdown("### Quick backtest (recent period)")
        bt = backtest(df, mode="intraday", threshold=int(threshold))
        if bt["trades"] == 0:
            st.write("No completed trades in sample window.")
        else:
            st.write(f"- Trades: **{bt['trades']}**")
            st.write(f"- Win-rate: **{bt['win_rate']:.0%}**")
            st.write(f"- Avg return / trade: **{bt['avg_return']:.2%}**")
            st.write(f"- Best / Worst: **{bt['best']:.2%} / {bt['worst']:.2%}**")

else:
    st.subheader("Position data source: Yahoo Finance (free daily data)")
    period = st.selectbox("History period", ["2y","5y","10y","max"], index=1)
    if st.button("Analyze position", type="primary"):
        df = fetch_yf_daily(symbol, period=period, interval="1d")
        if df.empty:
            st.error("No daily data found. Try switching exchange suffix (.NS/.BO) or another symbol.")
            st.stop()
        df = compute_scores(df, mode="position")
        idea = suggest_trade(df, mode="position", threshold=int(threshold))
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"### {symbol} – Daily price with Bollinger Bands")
            st.line_chart(df[["Close","BB_up","BB_mid","BB_low"]].dropna())
            st.markdown("### Confluence score (S_total)")
            st.line_chart(df[["S_total"]].dropna())
        with col2:
            st.markdown("### Latest read")
            st.metric("S_total", f"{idea['score']}")
            st.metric("Close", f"{df['Close'].iloc[-1]:.2f}")
            st.metric("ATR(14)", f"{df['ATR14'].iloc[-1]:.2f}")
            if idea["valid"]:
                st.success("**Long idea**")
                st.write(f"- **Entry** ≈ {idea['entry']:.2f}")
                st.write(f"- **Stop-loss** ≈ {idea['stop']:.2f} (≈ 2×ATR)")
                st.write(f"- **Target** ≈ {idea['target']:.2f} (≈ 2R)")
                st.write(f"- **Time window:** {idea['horizon']}")
            else:
                st.info("No fresh long signal at the chosen threshold. Consider waiting or lowering the threshold.")
        st.markdown("### Quick backtest (entire fetched period)")
        bt = backtest(df, mode="position", threshold=int(threshold))
        if bt["trades"] == 0:
            st.write("No completed trades in sample window.")
        else:
            st.write(f"- Trades: **{bt['trades']}**")
            st.write(f"- Win-rate: **{bt['win_rate']:.0%}**")
            st.write(f"- Avg return / trade: **{bt['avg_return']:.2%}**")
            st.write(f"- Best / Worst: **{bt['best']:.2%} / {bt['worst']:.2%}**")