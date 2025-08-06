# TA Confluence – Indian Stocks (Intraday & Position)

A Streamlit app that analyzes **NSE/BSE** stocks using a **confluence** of proven technical indicators and produces a **trade idea** with **entry**, **stop-loss**, **target**, and a **time window**. Works with **free data**:
- **Intraday** from **Alpha Vantage** (free API key required)
- **Daily/Position** from **Yahoo Finance**

> No strategy is 100% accurate. The app reports a confluence score and a quick backtest so you can judge reliability.

## Features
- Modes: **intraday** and **position**
- Indicators: EMA/SMA, MACD, RSI, Stochastic, Bollinger Bands, ATR, OBV, A/D, plus VWAP & Supertrend for intraday
- **Confluence Score (S_total)** = Trend + Momentum + Volatility + Volume
- **Signal threshold** selector
- **Trade plan**: Entry, SL (ATR-based), Target (2R), Time horizon
- **Quick backtest** of recent signals

## Install & Run (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → New app → pick your repo/branch → `app.py`.
3. In **Settings → Secrets**, add:
```
ALPHAVANTAGE_API_KEY="your_key_here"
```
4. Deploy. Open the public URL on your phone.

## Notes
- Symbols: `RELIANCE` defaults to `RELIANCE.NS`; pick NSE/BSE in the UI.
- Alpha Vantage free plan is rate-limited. If you see a limit note, wait and retry.
- Backtest is simple and for orientation only.