import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

st.set_page_config(page_title="Earnings Impact Predictor", layout="wide")
st.title("Earnings / Events Impact Predictor — Yahoo Finance MVP")

# --- User inputs ---
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker (US)", value="AAPL").upper()
with col2:
    years_back = st.slider("History (years)", 1, 5, 3)

# --- Fetch earnings calendar ---
@st.cache_data(ttl=3600)
def fetch_earnings(symbol, years):
    stock = yf.Ticker(symbol)
    # fetch earnings history
    hist = stock.earnings_dates
    if hist.empty:
        return pd.DataFrame()
    start_date = dt.date.today() - dt.timedelta(days=365*years)
    hist = hist[hist.index.date >= start_date]
    hist.reset_index(inplace=True)
    hist.rename(columns={'Earnings Date':'date'}, inplace=True)
    return hist[['date','EPS Estimate','Reported EPS']]

# --- Fetch OHLC data ---
@st.cache_data(ttl=3600)
def fetch_prices(symbol):
    end = dt.date.today()
    start = end - dt.timedelta(days=365*5)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    
    # Ensure column names
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']  # fallback if adjusted close missing
    
    df = df[['Open','High','Low','Close','Adj Close','Volume']]
    return df


# --- Compute returns around earnings ---
def compute_reactions(earn_df, price_df, windows=[-7,-3,-1,1,3,7]):
    rows = []
    for _, r in earn_df.iterrows():
        edate = pd.to_datetime(r['date'])
        if edate not in price_df.index:
            # get nearest trading day
            try:
                ed = price_df.index[price_df.index.get_indexer([edate], method="nearest")[0]]
            except:
                continue
        else:
            ed = edate
        base_price = price_df.loc[ed, 'Adj Close']
        if np.isnan(base_price):
            continue
        out = {"date": ed.date(), "EPS Estimate": r.get("EPS Estimate"), "Reported EPS": r.get("Reported EPS")}
        for w in windows:
            target = ed + pd.Timedelta(days=w)
            if target not in price_df.index:
                try:
                    target = price_df.index[price_df.index.get_indexer([target], method="nearest")[0]]
                except:
                    continue
            p = price_df.loc[target, 'Adj Close']
            out[f'ret_{w}d'] = (p - base_price)/base_price
        rows.append(out)
    return pd.DataFrame(rows)

# --- Main ---
if st.button("Run analysis"):
    with st.spinner("Fetching earnings and price data..."):
        earn_df = fetch_earnings(ticker, years_back)
        if earn_df.empty:
            st.info("No earnings data found.")
        else:
            price_df = fetch_prices(ticker)
            if price_df.empty:
                st.info("No price data found.")
            else:
                react_df = compute_reactions(earn_df, price_df)
                if react_df.empty:
                    st.info("Could not compute reactions.")
                else:
                    st.subheader("Historical Reaction Table (sample)")
                    st.dataframe(react_df.head(20))
                    
                    # Heatmap of average returns
                    windows = [c for c in react_df.columns if c.startswith("ret_")]
                    avg = react_df[windows].mean().rename(index=lambda x: x.replace("ret_",""))
                    avg = avg.sort_index(key=lambda s: [int(i.replace("d","")) for i in s.index])
                    
                    fig, ax = plt.subplots(figsize=(8,2))
                    sns.heatmap(avg.to_frame().T, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
                    ax.set_xlabel("Window (days around earnings)")
                    st.pyplot(fig)
                    st.success("Done — interpret as historical tendencies, not financial advice.")
