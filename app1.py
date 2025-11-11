import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

st.title("📅 Earnings & Financial Events Calendar")

# ---- Finnhub API key ----
api_key = st.text_input("Enter your Finnhub API Key", type="password")

if api_key:
    # ---- Select date ----
    selected_date = st.date_input("Select a date", datetime.today())
    date_str = selected_date.strftime('%Y-%m-%d')

    # ---- Optional: Filter by ticker ----
    ticker = st.text_input("Optional: Enter a ticker symbol to filter earnings (e.g., AAPL)").upper()

    st.subheader(f"Events on {date_str}")

    # ---- Fetch Earnings Calendar ----
    earnings_url = f"https://finnhub.io/api/v1/calendar/earnings?from={date_str}&to={date_str}&token={api_key}"
    try:
        earnings_data = requests.get(earnings_url).json()
        earnings_list = earnings_data.get("earningsCalendar", [])

        if earnings_list:
            df_earnings = pd.DataFrame(earnings_list)

            # Filter by ticker if provided
            if ticker:
                df_earnings = df_earnings[df_earnings['symbol'] == ticker]
            
            if not df_earnings.empty:
                st.write("### 📈 Earnings Announcements")
                st.dataframe(df_earnings[['symbol', 'date', 'hour', 'epsEstimate', 'epsActual']])
            else:
                st.info(f"No earnings found for {ticker} on this date.")
        else:
            st.info("No earnings announcements for this date.")
    except Exception as e:
        st.error(f"Error fetching earnings: {e}")

    # ---- Fetch Economic Events ----
    events_url = f"https://finnhub.io/api/v1/calendar/economic-events?from={date_str}&to={date_str}&token={api_key}"
    try:
        events_data = requests.get(events_url).json()
        economic_list = events_data.get("economicCalendar", [])
        if economic_list:
            df_events = pd.DataFrame(economic_list)
            st.write("### 📊 Financial / Economic Events")
            st.dataframe(df_events[['country', 'event', 'actual', 'forecast', 'previous']])
        else:
            st.info("No financial events for this date.")
    except Exception as e:
        st.error(f"Error fetching events: {e}")

else:
    st.warning("Please enter your Finnhub API key to see events.")
