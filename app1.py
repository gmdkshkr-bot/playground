import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.title("📅 Earnings & Financial Events Calendar")

# ---- User API key ----
api_key = st.text_input("Enter your Finnhub API Key", type="password")

if api_key:

    # ---- Select date ----
    selected_date = st.date_input("Select a date", datetime.today())
    date_str = selected_date.strftime('%Y-%m-%d')

    st.subheader(f"Events on {date_str}")

    # ---- Fetch Earnings Calendar ----
    earnings_url = f"https://finnhub.io/api/v1/calendar/earnings?from={date_str}&to={date_str}&token={api_key}"
    try:
        earnings_data = requests.get(earnings_url).json()
        if earnings_data.get("earningsCalendar"):
            df_earnings = pd.DataFrame(earnings_data["earningsCalendar"])
            st.write("### 📈 Earnings Announcements")
            st.dataframe(df_earnings[['symbol', 'date', 'hour', 'epsEstimate', 'epsActual']])
        else:
            st.info("No earnings announcements for this date.")
    except Exception as e:
        st.error(f"Error fetching earnings: {e}")

    # ---- Fetch Economic Events ----
    events_url = f"https://finnhub.io/api/v1/calendar/economic-events?from={date_str}&to={date_str}&token={api_key}"
    try:
        events_data = requests.get(events_url).json()
        if events_data.get("economicCalendar"):
            df_events = pd.DataFrame(events_data["economicCalendar"])
            st.write("### 📊 Financial / Economic Events")
            st.dataframe(df_events[['country', 'event', 'actual', 'forecast', 'previous']])
        else:
            st.info("No financial events for this date.")
    except Exception as e:
        st.error(f"Error fetching events: {e}")
else:
    st.warning("Please enter your Finnhub API key to see events.")
