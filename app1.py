# app.py
import streamlit as st
import pandas as pd
import finnhub
from datetime import datetime, timedelta

# Finnhub API setup
finnhub_client = finnhub.Client(api_key="YOUR_API_KEY_HERE")

st.title("📅 Earnings & Financial Events Calendar")

# Select date
selected_date = st.date_input("Select a date", datetime.today())

# Convert to UNIX timestamp for API
start_ts = int(datetime.combine(selected_date, datetime.min.time()).timestamp())
end_ts = int(datetime.combine(selected_date, datetime.max.time()).timestamp())

st.subheader(f"Events on {selected_date.strftime('%Y-%m-%d')}")

# Fetch earnings calendar
try:
    earnings = finnhub_client.earnings_calendar(
        start=datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d'),
        end=datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d')
    )
    if earnings['earningsCalendar']:
        df_earnings = pd.DataFrame(earnings['earningsCalendar'])
        st.write("### Earnings Announcements")
        st.dataframe(df_earnings[['symbol', 'date', 'hour', 'epsEstimate', 'epsActual']])
    else:
        st.info("No earnings announcements for this date.")
except Exception as e:
    st.error(f"Error fetching earnings: {e}")

# Optional: Fetch major economic events (using Finnhub Economic Calendar)
try:
    events = finnhub_client.economic_calendar(
        from_=(selected_date).strftime('%Y-%m-%d'),
        to=(selected_date).strftime('%Y-%m-%d')
    )
    if events:
        st.write("### Financial / Economic Events")
        df_events = pd.DataFrame(events)
        st.dataframe(df_events[['country', 'event', 'actual', 'forecast', 'previous']])
    else:
        st.info("No financial events for this date.")
except Exception as e:
    st.error(f"Error fetching events: {e}")
