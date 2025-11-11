# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Earnings Impact MVP", layout="wide")
st.title("Earnings / Events Impact Predictor — MVP")

# --- config / keys from Streamlit secrets ---
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "")
AV_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")

if not FINNHUB_KEY or not AV_KEY:
    st.warning("Add FINNHUB_KEY and ALPHA_VANTAGE_KEY to Streamlit secrets (see README below).")

col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker (US)", value="AAPL").upper()
with col2:
    years_back = st.slider("History (years)", 1, 5, 3)

# helper: fetch earnings calendar from Finnhub
def fetch_earnings(symbol, years):
    end = dt.date.today()
    start = end - dt.timedelta(days=365*years)
    url = f"https://finnhub.io/api/v1/calendar/earnings?from={start}&to={end}&symbol={symbol}&token={FINNHUB_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Finnhub error: {r.status_code}")
        return []
    data = r.json().get("earningsCalendar") or r.json().get("earnings") or r.json()
    # normalize: expect list of items with 'date' or 'reportDate'
    items = []
    for it in data:
        # possible keys: 'date', 'reportDate', 'epsEstimate', 'actual'
        d = it.get("date") or it.get("reportDate")
        if not d:
            continue
        try:
            items.append({
                "date": pd.to_datetime(d).date(),
                "symbol": symbol,
                "epsEstimate": it.get("epsEstimate"),
                "epsActual": it.get("epsActual"),
            })
        except Exception:
            continue
    return pd.DataFrame(items)

# helper: fetch daily OHLC from Alpha Vantage
def fetch_ohlc_av(symbol):
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED", "symbol":symbol, "outputsize":"full", "apikey":AV_KEY}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.error(f"AlphaVantage error: {r.status_code}")
        return pd.DataFrame()
    j = r.json()
    ts_key = next((k for k in j.keys() if "Time Series" in k), None)
    if ts_key is None:
        st.error("AlphaVantage response missing time series (rate limit or bad symbol).")
        return pd.DataFrame()
    ts = j[ts_key]
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
        "1. open":"open", "2. high":"high", "3. low":"low", "4. close":"close", "5. adjusted close":"adj_close", "6. volume":"volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)
    return df

# compute reaction windows
def compute_reactions(earn_df, price_df, windows=[-7,-3,-1,1,3,7]):
    rows = []
    for _, r in earn_df.iterrows():
        edate = pd.to_datetime(r["date"])
        if edate not in price_df.index:
            # find nearest business day (next trading day)
            if (edate + pd.Timedelta(days=1)) in price_df.index:
                ed = edate + pd.Timedelta(days=1)
            else:
                ed = price_df.index[price_df.index.get_loc(edate, method="nearest")]
        else:
            ed = edate
        base_price = price_df.loc[ed, "adj_close"]
        if np.isnan(base_price):
            continue
        out = {"date": ed.date(), "epsEstimate": r.get("epsEstimate"), "epsActual": r.get("epsActual")}
        for w in windows:
            target = ed + pd.Timedelta(days=w)
            # if target not in index, get nearest
            if target not in price_df.index:
                try:
                    target = price_df.index[price_df.index.get_indexer([target], method="nearest")[0]]
                except Exception:
                    continue
            p = price_df.loc[target, "adj_close"]
            out[f"ret_{w}d"] = (p - base_price) / base_price
        rows.append(out)
    return pd.DataFrame(rows)

if st.button("Run analysis"):
    with st.spinner("Fetching earnings and prices..."):
        earn_df = fetch_earnings(ticker, years_back)
        if earn_df.empty:
            st.info("No earnings found for ticker/time range.")
        else:
            price_df = fetch_ohlc_av(ticker)
            if price_df.empty:
                st.info("No price data returned.")
            else:
                react_df = compute_reactions(earn_df, price_df)
                if react_df.empty:
                    st.info("No reaction rows computed.")
                else:
                    st.subheader("Reaction table (sample)")
                    st.dataframe(react_df.head(20))
                    # heatmap: avg return by window
                    windows = [c for c in react_df.columns if c.startswith("ret_")]
                    avg = react_df[windows].mean().rename(index=lambda x: x.replace("ret_",""))
                    avg = avg.sort_index(key=lambda s: [int(i.replace("d","")) for i in s.index])
                    fig, ax = plt.subplots(figsize=(8,2))
                    sns.heatmap(avg.to_frame().T, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
                    ax.set_xlabel("Window (days)")
                    st.pyplot(fig)
                    st.success("Done — interpret historical tendencies, not advice.")
