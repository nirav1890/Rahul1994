import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
import time
from datetime import timedelta

st.set_page_config(page_title="TA Confluence – India (NSE Intraday + Position)", layout="wide")

# =========================
# Symbol helpers
# =========================
def normalize_symbol(user_sym: str, exchange_suffix: str) -> str:
    """
    Return a symbol with .NS or .BO based on the selected exchange.
    """
    s = (user_sym or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s[:-3]
    return f"{s}{exchange_suffix}"

def nse_base(symbol: str) -> str:
    """
    Strip suffix to get raw NSE base symbol (e.g., RELIANCE from RELIANCE.NS).
    """
    return symbol.upper().replace(".NS", "").replace(".BO", "")

# =========================
# Daily data (position) via Yahoo Finance
# =========================
def fetch_yf_daily(symbol: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}).dropna()
    return df

# =========================
# Intraday – primary: NSE (unofficial), fallback: Yahoo Finance
# =========================
def _nse_session():
    s = requests.Session()
    ua = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
    }
    # Warm up for cookies; short delay helps avoid 401
    s.get("https://www.nseindia.com", headers=ua, timeout=15)
    time.sleep(1.0)
    return s, ua

def fetch_nse_intraday(symbol: str, interval: str = "5min") -> pd.DataFrame:
    """
    Fetch intraday data from NSE 'chart-databyindex' endpoint and aggregate to OHLCV.
    interval: '1min' | '5min' | '15min'
    symbol: accepts RELIANCE, RELIANCE.NS, RELIANCE.BO (always fetched from NSE equity)
    """
    base = nse_base(symbol)
    index_code = f"{base}EQN"  # equity cash segment code for NSE chart endpoint

    s, ua = _nse_session()
    url = f"https://www.nseindia.com/api/chart-databyindex?index={index_code}"
    headers = {
        **ua,
        "referer": f"https://www.nseindia.com/get-quotes/equity?symbol={base}",
        "accept": "application/json, text/plain, */*",
    }

    r = s.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        # propagate status for the caller to decide fallback
        raise RuntimeError(f"NSE HTTP {r.status_code}")

    data = r.json() or {}
    candles = data.get("candles")
    graph = data.get("grapthData") or data.get("graphData")

    if candles:
        # [[ts_ms, open, high, low, close, volume], ...]
        rows = []
        for c in candles:
            if len(c) < 6:
                continue
            ts = pd.to_datetime(int(c[0]), unit="ms")
            rows.append({
                "Datetime": ts,
                "Open": float(c[1]),
                "High": float(c[2]),
                "Low": float(c[3]),
                "Close": float(c[4]),
                "Volume": float(c[5])
            })
        df = pd.DataFrame(rows).sort_values("Datetime").set_index("Datetime")
    elif graph:
        # [[ts_ms, price, vol?], ...] -> build 1-min OHLCV, then resample
        rows = []
        for g in graph:
            if len(g) < 2:
                continue
            ts = pd.to_datetime(int(g[0]), unit="ms")
            price = float(g[1])
            vol = float(g[2]) if len(g) > 2 and g[2] is not None else 0.0
            rows.append({"Datetime": ts, "Price": price, "Volume": vol})
        tick = pd.DataFrame(rows).sort_values("Datetime").set_index("Datetime")
        if tick.empty:
            raise RuntimeError("NSE returned empty intraday series")
        o = tick["Price"].resample("1T").first()
        h = tick["Price"].resample("1T").max()
        l = tick["Price"].resample("1T").min()
        c = tick["Price"].resample("1T").last()
        v = tick["Volume"].resample("1T").sum()
        df = pd.concat([o,h,l,c,v], axis=1)
        df.columns = ["Open","High","Low","Close","Volume"]
        df = df.dropna(how="any")
    else:
        raise RuntimeError(f"NSE returned no candles (keys: {list(data.keys())})")

    # Aggregate to requested interval
    interval = (interval or "5min").lower()
    rule = {"1min":"1T","5min":"5T","15min":"15T"}.get(interval, "5T")
    if rule != "1T":
        o = df["Open"].resample(rule, label="right", closed="right").first()
        h = df["High"].resample(rule, label="right", closed="right").max()
        l = df["Low"].resample(rule, label="right", closed="right").min()
        c = df["Close"].resample(rule, label="right", closed="right").last()
        v = df["Volume"].resample(rule, label="right", closed="right").sum()
        df = pd.concat([o,h,l,c,v], axis=1).dropna(how="any")

    return df

def fetch_yf_intraday(symbol_with_suffix: str, interval: str = "5min", lookback_days: int = 7) -> pd.DataFrame:
    """
    Yahoo intraday fallback. interval: '1min'|'5min'|'15min' mapped to '1m'|'5m'|'15m'.
    Uses a valid period based on interval so Yahoo returns data.
    """
    tf_map = {"1min": "1m", "5min": "5m", "15min": "15m"}
    tf = tf_map.get(interval.lower(), "5m")

    # Yahoo limits history by interval. Choose a valid period window.
    if tf == "1m":
        min_days, max_days = 7, 7
    else:  # 5m / 15m
        min_days, max_days = 7, 60
    period_days = min(max_days, max(lookback_days, min_days))

    df = yf.download(symbol_with_suffix, period=f"{int(period_days)}d", interval=tf, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}).dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def fetch_intraday_with_fallback(symbol_with_suffix: str, interval: str = "5min", lookback_days: int = 5) -> (pd.DataFrame, str):
    """
    Try NSE first (forces .NS for NSE query). On 401/403/empty -> fallback to Yahoo intraday
    with the user's chosen suffix (.NS or .BO). Returns (df, source_str).
    """
    # Force NSE suffix for primary fetch
    nse_try = normalize_symbol(nse_base(symbol_with_suffix), ".NS")
    try:
        df_nse = fetch_nse_intraday(nse_try, interval=interval)
        if not df_nse.empty:
            return df_nse, "NSE"
    except Exception as e:
        # Fallback for common block statuses or empty; ignore and proceed to Yahoo
        if "NSE HTTP 401" in str(e) or "NSE HTTP 403" in str(e) or "no candles" in str(e).lower():
            pass
        else:
            pass  # still try Yahoo

    # Yahoo fallback with the EXACT suffix user selected
    df_yf = fetch_yf_intraday(symbol_with_suffix, interval=interval, lookback_days=max(lookback_days, 7))
    if not df_yf.empty:
        return df_yf, "Yahoo"

    # Last resort: try NSE-suffixed symbol on Yahoo (e.g., RELIANCE.NS)
    df_yf2 = fetch_yf_intraday(nse_try, interval=interval, lookback_days=max(lookback_days, 7))
    if not df_yf2.empty:
        return df_yf2, "Yahoo(.NS)"

    raise RuntimeError("No intraday data from NSE or Yahoo for this symbol/interval.")

# =========================
# Indicators & scoring
# =========================
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
        out["MACD"] = out["MACDsig"] = out["MACDhist"] = np.nan

    out["RSI14"] = ta.rsi(out["Close"], 14)
    stoch = ta.stoch(out["High"], out["Low"], out["Close"], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        out["STOCHk"] = stoch.iloc[:,0]
        out["STOCHd"] = stoch.iloc[:,1]
    else:
        out["STOCHk"] = out["STOCHd"] = np.nan

    bb = ta.bbands(out["Close"], 20, 2)
    if bb is not None and not bb.empty:
        out["BB_up"], out["BB_mid"], out["BB_low"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
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
    day = df.index.tz_localize(None).date
    day_ser = pd.Series(day, index=df.index)
    vwap = pd.Series(index=df.index, dtype=float)
    for d, g in df.groupby(day_ser):
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
            final_upperband.iloc[i] = (
                min(upperband.iloc[i], final_upperband.iloc[i-1])
                if df["Close"].iloc[i-1] > final_upperband.iloc[i-1] else upperband.iloc[i]
            )
            final_lowerband.iloc[i] = (
                max(lowerband.iloc[i], final_lowerband.iloc[i-1])
                if df["Close"].iloc[i-1] < final_lowerband.iloc[i-1] else lowerband.iloc[i]
            )
            trend.iloc[i] = (
                1 if df["Close"].iloc[i] > final_upperband.iloc[i-1]
                else (-1 if df["Close"].iloc[i] < final_lowerband.iloc[i-1] else trend.iloc[i-1])
            )
    st_line = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        st_line.iloc[i] = final_lowerband.iloc[i] if trend.iloc[i] == 1 else final_upperband.iloc[i]
    return st_line

def compute_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = add_common_indicators(df.copy())
    if mode == "intraday":
        out["VWAP"] = vwap_series(out)
        out["Supertrend"] = supertrend(out, period=10, multiplier=3.0)

    def s_trend(i):
        s = 0
        c = out["Close"].iloc[i]
        ema9, ema21, ema50 = out["EMA9"].iloc[i], out["EMA21"].iloc[i], out["EMA50"].iloc[i]
        sma50, sma200 = out["SMA50"].iloc[i], out["SMA200"].iloc[i]
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
        macd, sig = out["MACD"].iloc[i], out["MACDsig"].iloc[i]
        rsi = out["RSI14"].iloc[i]
        st_k, st_d = out["STOCHk"].iloc[i], out["STOCHd"].iloc[i]
        if not np.isnan(macd) and not np.isnan(sig): s += 1 if macd > sig else -1
        if not np.isnan(rsi):
            if rsi > 60: s += 1
            elif rsi < 40: s -= 1
        if not np.isnan(st_k) and not np.isnan(st_d): s += 1 if st_k > st_d else -1
        return s

    def s_volatility(i):
        s = 0
        c, mid = out["Close"].iloc[i], out["BB_mid"].iloc[i]
        if not np.isnan(c) and not np.isnan(mid): s += 1 if c > mid else -1
        return s

    def s_volume(i):
        s = 0
        adl = out["ADL"].iloc[i]
        if not np.isnan(adl): s += 1 if adl > 0 else -1
        if i > 0 and not np.isnan(out["OBV"].iloc[i]) and not np.isnan(out["OBV"].iloc[i-1]):
            s += 1 if out["OBV"].iloc[i] > out["OBV"].iloc[i-1] else -1
        if mode == "intraday":
            v = out.get("VWAP", pd.Series(dtype=float)).iloc[i] if "VWAP" in out else np.nan
            if not np.isnan(v): s += 1 if out["Close"].iloc[i] >= v else -1
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
        tp = entry + 2 * (entry - sl)  # 2R
    else:
        sl, tp = None, None
    horizon = "15–60 minutes" if mode == "intraday" else "5–15 trading days"
    return {"valid": bool(valid), "entry": entry, "stop": sl, "target": tp, "horizon": horizon, "score": int(last["S_total"])}

def backtest(df: pd.DataFrame, mode: str, threshold: int):
    trades, wins, pnls = 0, 0, []
    in_trade, entry, sl, tp = False, None, None, None
    for i in range(1, len(df)):
        c = float(df["Close"].iloc[i])
        s = float(df["S_total"].iloc[i])
        atr_i = float(df["ATR14"].iloc[i]) if not np.isnan(df["ATR14"].iloc[i]) else None
        if not in_trade:
            if s >= threshold and float(df["S_total"].iloc[i-1]) < threshold and atr_i is not None:
                sl_mult = 1.5 if mode == "intraday" else 2.0
                entry = c
                sl = entry - sl_mult * atr_i
                tp = entry + 2 * (entry - sl)
                in_trade = True
        else:
            exit_now = (atr_i is None) or (c <= sl) or (c >= tp) or (s <= 0)
            if exit_now:
                trades += 1
                pnl = (c - entry) / entry
                pnls.append(pnl)
                if pnl > 0: wins += 1
                in_trade, entry, sl, tp = False, None, None, None
    if trades == 0:
        return {"trades": 0, "win_rate": None, "avg_return": None, "best": None, "worst": None}
    pn = np.array(pnls)
    return {"trades": trades, "win_rate": float((pn > 0).mean()), "avg_return": float(pn.mean()), "best": float(pn.max()), "worst": float(pn.min())}

# =========================
# UI
# =========================
st.title("Technical Analysis Confluence – Indian Stocks")
st.caption("Intraday tries NSE first and falls back to Yahoo if blocked. Position uses Yahoo daily data. No system is 100% accurate; this reports confluence + quick backtest.")

mode = st.radio("Mode", ["intraday", "position"], horizontal=True, index=0)

colA, colB, colC = st.columns([1.4,1,1])
with colA:
    user_symbol = st.text_input("Company or symbol (e.g., RELIANCE, TCS, HDFCBANK, SBIN)", "RELIANCE")
with colB:
    exchange = st.selectbox("Exchange", ["NSE (.NS)", "BSE (.BO)"], index=0)
with colC:
    threshold = st.slider("Signal threshold", min_value=1, max_value=8, value=3, help="Minimum S_total for a new long entry.")

suffix = ".NS" if exchange.startswith("NSE") else ".BO"
symbol = normalize_symbol(user_symbol, suffix)

if mode == "intraday":
    st.subheader("Intraday data source: NSE → Yahoo fallback")
    if suffix == ".BO":
        st.info("Intraday fetch prefers NSE. We’ll query NSE first, then Yahoo with your BSE symbol if needed.")

    ival = st.selectbox("Interval", ["1min","5min","15min"], index=1)
    lookback_days = st.number_input("Lookback days (display only)", min_value=1, max_value=60, value=5, step=1)

    if st.button("Analyze intraday", type="primary"):
        try:
            df_raw, src = fetch_intraday_with_fallback(symbol, interval=ival, lookback_days=int(lookback_days))
        except Exception as e:
            st.error(f"Failed to fetch intraday data: {e}")
            st.stop()

        # Trim display window
        if not df_raw.empty:
            cutoff = df_raw.index.max() - pd.Timedelta(days=int(lookback_days))
            df_raw = df_raw[df_raw.index >= cutoff]

        if df_raw.empty:
            st.warning("No intraday data returned for this symbol/interval.")
            st.stop()

        df = compute_scores(df_raw, mode="intraday")
        idea = suggest_trade(df, mode="intraday", threshold=int(threshold))

        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"### {symbol} – Intraday price (source: {src}) with Bollinger Bands")
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
    st.subheader("Position data source: Yahoo Finance (daily)")
    period = st.selectbox("History period", ["2y","5y","10y","max"], index=1)
    if st.button("Analyze position", type="primary"):
        df_raw = fetch_yf_daily(symbol, period=period)
        if df_raw.empty:
            st.error("No daily data found. Try switching exchange suffix (.NS/.BO) or another symbol.")
            st.stop()

        df = compute_scores(df_raw, mode="position")
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

        st.markdown("### Quick backtest (full period)")
        bt = backtest(df, mode="position", threshold=int(threshold))
        if bt["trades"] == 0:
            st.write("No completed trades in sample window.")
        else:
            st.write(f"- Trades: **{bt['trades']}**")
            st.write(f"- Win-rate: **{bt['win_rate']:.0%}**")
            st.write(f"- Avg return / trade: **{bt['avg_return']:.2%}**")
            st.write(f"- Best / Worst: **{bt['best']:.2%} / {bt['worst']:.2%}**")
