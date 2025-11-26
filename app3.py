import streamlit as st
import json
import pandas as pd
from PIL import Image
import io
import datetime 
import numpy as np
import plotly.express as px
import requests
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold 

# ----------------------------------------------------------------------
# 📌 0. Currency Conversion Setup & Globals
# ----------------------------------------------------------------------

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    EXCHANGE_API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"] 
except KeyError:
    st.error("❌ Please set 'GEMINI_API_KEY' and 'EXCHANGE_RATE_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)

# 💡 수정된 부분: 이제 exchange_rates는 {Currency Code: 1 Foreign Unit = X KRW}의 형태로 저장됩니다.
@st.cache_data
def get_exchange_rates():
    """
    Fetches real-time exchange rates using ExchangeRate-API (USD Base).
    Returns a dictionary: {currency_code: KRW_equivalent}
    """
    
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/USD"
    # KRW의 기준은 1 KRW = X KRW (1)
    exchange_rates = {'KRW': 1.0} 
    # Fallback Rates는 1 단위 외화당 KRW 값입니다.
    FALLBACK_RATES = {'KRW': 1.0, 'USD': 1350.00, 'EUR': 1450.00, 'JPY': 9.20}

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        conversion_rates = data.get('conversion_rates', {})
        
        # 1. KRW Rate (USD -> KRW)
        krw_per_usd = conversion_rates.get('KRW', 0)
        
        if krw_per_usd == 0 or data.get('result') != 'success':
              raise ValueError("API returned incomplete or failed data.")

        # 2. Store USD rate: 1 USD = krw_per_usd KRW
        exchange_rates['USD'] = krw_per_usd 
        
        # 3. Calculate EUR rate: 1 EUR = (KRW/USD) / (EUR/USD) = KRW/EUR
        eur_rate_vs_usd = conversion_rates.get('EUR', 0)
        if eur_rate_vs_usd > 0:
            # krw_per_usd는 (KRW/USD)를 의미하며, eur_rate_vs_usd는 (EUR/USD)를 의미합니다.
            # 1 EUR = krw_per_usd / eur_rate_vs_usd KRW 입니다.
            exchange_rates['EUR'] = krw_per_usd / eur_rate_vs_usd
        
        # 4. Calculate JPY rate: 1 JPY = (KRW/USD) / (JPY/USD) = KRW/JPY
        jpy_rate_vs_usd = conversion_rates.get('JPY', 0)
        if jpy_rate_vs_usd > 0:
            # 1 JPY = krw_per_usd / jpy_rate_vs_usd KRW 입니다.
            exchange_rates['JPY'] = krw_per_usd / jpy_rate_vs_usd
            
        st.sidebar.success(f"✅ Real-time rates loaded. (1 USD = {krw_per_usd:,.2f} KRW)")

        return exchange_rates

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Error. Using fallback rates. ({e})")
        return FALLBACK_RATES
        
    except Exception as e:
        st.warning(f"⚠️ Exchange Rate Processing Error. Using fallback rates. ({e})")
        return FALLBACK_RATES


# 💡 수정된 부분: rates 딕셔너리에서 1 외화당 KRW 값을 가져와 곱합니다.
def convert_to_krw(amount: float, currency: str, rates: dict) -> float:
    """ Converts a foreign currency amount to KRW using stored rates (1 Foreign Unit = X KRW). """
    currency_upper = currency.upper().strip()
    
    # rates.get(currency_upper)는 '1 외화당 KRW' 값입니다. 
    # 만약 해당 통화가 없으면 기본값으로 KRW의 환율인 1.0을 사용합니다.
    rate = rates.get(currency_upper, rates.get('KRW', 1.0))
    
    # 금액 * 환율(1외화당 KRW) = 최종 KRW 금액
    return amount * rate

# Global Categories (Internal classification names remain Korean for consistency with AI analysis prompt)
ALL_CATEGORIES = [
    "외식", "식재료", "카페/음료", "주류", 
    "생필품", "의료/건강", "교육/서적", "통신", "공과금",
    "대중교통", "유류비", "택시", "주차비", 
    "영화/공연", "여행", "취미", "게임", 
    "경조사", "이체/수수료", "비상금", "미분류"
]

def get_category_guide():
    guide = ""
    categories = {
        "Food": ["외식 (Dining Out)", "식재료 (Groceries)", "카페/음료 (Coffee/Beverages)", "주류 (Alcohol)"],
        "Household": ["생필품 (Necessities)", "의료/건강 (Medical/Health)", "교육/서적 (Education/Books)", "통신 (Communication)", "공과금 (Utilities)"],
        "Transport": ["대중교통 (Public Transport)", "유류비 (Fuel)", "택시 (Taxi)", "주차비 (Parking)"],
        "Culture": ["영화/공연 (Movies/Shows)", "여행 (Travel)", "취미 (Hobby)", "게임 (Games)"],
        "Other": ["경조사 (Events)", "이체/수수료 (Transfer/Fees)", "비상금 (Emergency Fund)", "미분류 (Unclassified)"],
    }
    for main, subs in categories.items():
        guide += f"- **{main}**: {', '.join(subs)}\n"
    return guide


# ----------------------------------------------------------------------
# 📌 2. Initialize Session State & Page Configuration
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    st.session_state.all_receipts_summary = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


st.set_page_config(
    page_title="Smart Receipt Analyzer & Tracker 🧾",
    layout="wide"
)


# ----------------------------------------------------------------------
# 📌 3. Sidebar and Main Title (Translated)
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("About This App")
    st.markdown("---")
    
    st.subheader("How to Use")
    st.markdown("""
    This application helps you manage your household ledger easily by using AI.
    1. **Upload / Manual Input:** Enter spending data via receipt image or manual form.
    2. **Auto-Convert:** Foreign currencies are automatically converted to **KRW** using real-time rates.
    3. **Analyze & Accumulate:** Results are added to the cumulative record.
    4. **Review & Chat:** Check the integrated report, spending charts, and get personalized financial advice.
    """)
    
    st.markdown("---")
    if st.session_state.all_receipts_items:
        st.info(f"Currently tracking {len(st.session_state.all_receipts_items)} receipts.")
        
st.title("🧾 AI Household Ledger: Receipt Analysis & Cumulative Tracking")
st.markdown("---")


# 📢 Fetch rates once at app startup
EXCHANGE_RATES = get_exchange_rates()


# --- 1. Gemini Analysis Function (Translated Prompt) ---
def analyze_receipt_with_gemini(_image: Image.Image):
    """
    Calls the Gemini model to extract data and categorize items from a receipt image.
    """
    
    prompt_template = """
    You are an expert in receipt analysis and ledger recording.
    Analyze the following items from the receipt image and **you must extract them in JSON format**.
    
    **CRITICAL INSTRUCTION:** The response must only contain the **JSON code block wrapped in backticks (```json)**. Do not include any explanations, greetings, or additional text outside the JSON code block.
    
    1. store_name: Store Name (text)
    2. date: Date (YYYY-MM-DD format)
    3. total_amount: Total Amount Paid (numbers only, no commas)
    4. currency_unit: Official currency code shown on the receipt (e.g., KRW, USD, EUR).
    5. items: List of purchased items. Each item must include:
        - name: Item Name (text)
        - price: Unit Price (numbers only, no commas)
        - quantity: Quantity (numbers only)
        - category: The most appropriate **Sub-Category** for this item, which must be **automatically classified** by you.
    
    **Classification Guide (Choose ONE sub-category for 'category' field):**
    - Food: **외식, 식재료, 카페/음료, 주류** (Dining Out, Groceries, Coffee/Beverages, Alcohol)
    - Household: **생필품, 의료/건강, 교육/서적, 통신, 공과금** (Necessities, Medical/Health, Education/Books, Communication, Utilities)
    - Transport: **대중교통, 유류비, 택시, 주차비** (Public Transport, Fuel, Taxi, Parking)
    - Culture: **영화/공연, 여행, 취미, 게임** (Movies/Shows, Travel, Hobby, Games)
    - Other: **경조사, 이체/수수료, 비상금, 미분류** (Events, Transfer/Fees, Emergency Fund, Unclassified)
        
    JSON Schema:
    ```json
    {
      "store_name": "...",
      "date": "...",
      "total_amount": ...,
      "currency_unit": "...",  
      "items": [
        {"name": "...", "price": ..., "quantity": ..., "category": "..."}
      ]
    }
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template, _image],
            config=genai.types.GenerateContentConfig(
                safety_settings=[
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                ]
            )
        )
        return response.text
    
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return None

# --- 2. AI Analysis Report Generation Function ---
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float, currency_unit: str, detailed_items_text: str):
    """
    Generates an AI analysis report based on aggregated spending data and detailed items.
    """
    summary_text = summary_df.to_string(index=False)
    
    prompt_template = f"""
    You are an AI ledger analyst providing professional financial advice.
    The user's **all accumulated spending** amounts to {total_amount:,.0f} {currency_unit}.
    
    Below is the category breakdown of all accumulated spending (Unit: {currency_unit}):
    --- Spending Summary Data ---
    {summary_text}
    ---
    
    **CRITICAL DETAILED DATA:** Below are the individual item names, their categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores if patterns are observed).
    --- Detailed Items Data (AI Category, Item Name, Total Spend) ---
    {detailed_items_text}
    ---

    Follow these instructions and provide an analysis report in a friendly and professional tone:
    1. Summarize the main characteristic of this total spending (e.g., the largest spending category and its driving factor based on individual items).
    2. Provide 2-3 sentences of helpful and friendly advice or commentary for the user. Try to mention a specific item or category-related pattern observed in the Detailed Items Data.
    3. The response must only contain the analysis content, starting directly with the summary, without any greetings or additional explanations.
    4. **CRITICAL:** When mentioning the total spending amount in the analysis, **you must include the currency unit** (e.g., "Total spending of 1,500,000 KRW").
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template],
        )
        return response.text
        
    except Exception as e:
        return "Failed to generate analysis report."


# ----------------------------------------------------------------------
# 📌 4. Streamlit UI: Tab Setup (Translated)
# ----------------------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Analysis & Tracking", "💬 Financial Expert Chat"])


# ======================================================================
#     		 	TAB 1: ANALYSIS & TRACKING
# ======================================================================
with tab1:
    
    # --- File Uploader and Analysis ---
    st.subheader("📸 Upload Receipt Image (AI Analysis)")
    uploaded_file = st.file_uploader(
        "Upload one receipt image (jpg, png) at a time. (Data will accumulate in the current session)", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=False 
    )


    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        is_already_analyzed = any(s.get('id') == file_id for s in st.session_state.all_receipts_summary)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Uploaded Receipt")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True) 

        with col2:
            st.subheader("📊 Analysis and Recording")
            
            if is_already_analyzed:
                st.warning("⚠️ This receipt has already been analyzed and added to the record. Please upload a different file.")
                analyze_button = st.button("✨ Start Receipt Analysis", disabled=True)
            else:
                analyze_button = st.button("✨ Start Receipt Analysis")


            if analyze_button and not is_already_analyzed:
                
                st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
                with st.spinner('AI is meticulously reading the receipt...'):
                    
                    json_data_text = analyze_receipt_with_gemini(image)

                    if json_data_text:
                        try:
                            if json_data_text.strip().startswith("```json"):
                                json_data_text = json_data_text.strip().lstrip("```json").rstrip("```").strip()
                            
                            receipt_data = json.loads(json_data_text)
                            
                            if not isinstance(receipt_data.get('total_amount'), (int, float)):
                                 receipt_data['total_amount'] = np.nan 

                            # --- Main Information Display ---
                            st.success("✅ Analysis Complete! Check the ledger data below.")
                            
                            currency_unit = receipt_data.get('currency_unit', '').strip()
                            display_unit = currency_unit if currency_unit else 'KRW'
                            total_amount = receipt_data.get('total_amount', 0)
                            
                            st.markdown(f"**🏠 Store Name:** {receipt_data.get('store_name', 'N/A')}")
                            st.markdown(f"**📅 Date:** {receipt_data.get('date', 'N/A')}")
                            st.subheader(f"💰 Total Amount Paid: {total_amount:,.0f} {display_unit}")
                            
                            # 💡 Display Applied Exchange Rate for AI Analysis
                            if display_unit != 'KRW':
                                applied_rate = EXCHANGE_RATES.get(display_unit, 1.0)
                                st.info(f"**📢 Applied Exchange Rate:** 1 {display_unit} = {applied_rate:,.4f} KRW (Rate fetched from API/Fallback)")
                                
                            st.markdown("---")


                            if 'items' in receipt_data and receipt_data['items']:
                                items_df = pd.DataFrame(receipt_data['items'])
                                
                                items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                                items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                                items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                                items_df['Total Spend'] = items_df['Unit Price'] * items_df['Quantity']
                                
                                st.subheader("🛒 Detailed Item Breakdown (Category Editable)")
                                
                                edited_df = st.data_editor(
                                    items_df,
                                    column_config={
                                        "AI Category": st.column_config.SelectboxColumn(
                                            "Final Category",
                                            help="Select the correct sub-category for this item.",
                                            width="medium",
                                            options=ALL_CATEGORIES,
                                            required=True,
                                        )
                                    },
                                    disabled=['Item Name', 'Unit Price', 'Quantity', 'Total Spend'],
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                # 📢 Currency Conversion for Accumulation (AI Analysis)
                                edited_df['Currency'] = display_unit
                                # 💡 Total Spend는 이미 외화 기준으로 계산되어 있습니다.
                                edited_df['Total Spend Numeric'] = pd.to_numeric(edited_df['Total Spend'], errors='coerce').fillna(0)
                                edited_df['KRW Total Spend'] = edited_df.apply(
                                    lambda row: convert_to_krw(row['Total Spend Numeric'], row['Currency'], EXCHANGE_RATES), axis=1
                                )
                                edited_df = edited_df.drop(columns=['Total Spend Numeric'])


                                # ** Accumulate Data: Store the edited DataFrame **
                                st.session_state.all_receipts_items.append(edited_df)
                                st.session_state.all_receipts_summary.append({
                                    'id': file_id, 
                                    'filename': uploaded_file.name,
                                    'Store': receipt_data.get('store_name', 'N/A'),
                                    'Total': edited_df['KRW Total Spend'].sum(), # Store KRW Total
                                    'Currency': 'KRW', # Standardize summary currency to KRW
                                    'Date': receipt_data.get('date', 'N/A'),
                                    'Original_Total': total_amount, 
                                    'Original_Currency': display_unit 
                                })

                                st.success(f"🎉 Data from {uploaded_file.name} successfully added (Converted to KRW)!")

                            else:
                                st.warning("Item list could not be found in the analysis result.")

                        except json.JSONDecodeError:
                            st.error("❌ Gemini analysis result is not a valid JSON format. (JSON parsing error)")
                        except Exception as e:
                            st.error(f"Unexpected error occurred during data processing: {e}")
                    else:
                        st.error("Analysis failed to complete. Please try again.")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- Manual Expense Input (Translated) ---
    # ----------------------------------------------------------------------
    st.subheader("📝 Manual Expense Input (No Receipt)")
    
    st.info("""
    **✅ Input Guide**
    Record your expense details easily.
    **💡 Category Scheme (Sub-Category)**
    """ + get_category_guide()
    )

    with st.form("manual_expense_form", clear_on_submit=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            manual_date = st.date_input("📅 Expense Date", value=datetime.date.today())
            manual_description = st.text_input("📝 Expense Item (Description)", placeholder="e.g., Lunch, Groceries")
            
        with col_m2:
            manual_store = st.text_input("🏠 Store/Merchant Name", placeholder="e.g., Local Diner, Starbucks")
            manual_amount = st.number_input("💰 Expense Amount (Numbers Only)", min_value=0.0, step=100.0, format="%.2f")
            
        with col_m3:
            manual_category = st.selectbox("📌 Category (Sub-Category)", options=ALL_CATEGORIES, index=ALL_CATEGORIES.index('미분류'))
            manual_currency = st.selectbox("Currency Unit", options=['KRW', 'USD', 'EUR', 'JPY'], index=0)
            
        submitted = st.form_submit_button("✅ Add to Ledger")

        if submitted:
            if manual_description and manual_amount > 0 and manual_category:
                
                # 📢 Currency Conversion for Manual Input
                krw_total = convert_to_krw(manual_amount, manual_currency, EXCHANGE_RATES)
                applied_rate = EXCHANGE_RATES.get(manual_currency, 1.0)

                # 1. Prepare Item DataFrame 
                manual_df = pd.DataFrame([{
                    'Item Name': manual_description,
                    'Unit Price': manual_amount, 
                    'Quantity': 1,
                    'AI Category': manual_category,
                    'Total Spend': manual_amount,
                    'Currency': manual_currency,
                    'KRW Total Spend': krw_total 
                }])
                
                # 2. Prepare Summary Data
                manual_summary = {
                    'id': f"manual-{pd.Timestamp.now().timestamp()}", 
                    'filename': 'Manual Entry',
                    'Store': manual_store if manual_store else 'Manual Entry',
                    'Total': krw_total, 
                    'Currency': 'KRW', 
                    'Date': manual_date.strftime('%Y-%m-%d'),
                    'Original_Total': manual_amount, 
                    'Original_Currency': manual_currency 
                }
                
                # 3. Accumulate Data
                st.session_state.all_receipts_items.append(manual_df)
                st.session_state.all_receipts_summary.append(manual_summary)
                
                # 💡 Modified Success Message
                if manual_currency != 'KRW':
                    rate_info = f" (Applied Rate: 1 {manual_currency} = {applied_rate:,.4f} KRW)"
                else:
                    rate_info = ""
                    
                st.success(f"🎉 {manual_date.strftime('%Y-%m-%d')} expense recorded ({manual_description}: {manual_amount:,.2f} {manual_currency} -> **{krw_total:,.0f} KRW**){rate_info}. Added to ledger.")
                st.rerun()
            else:
                st.error("❌ 'Expense Item', 'Expense Amount', and 'Category' are required fields. Amount must be greater than 0.")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- 5. Cumulative Data Analysis Section (ALL ANALYSIS IS KRW BASED) ---
    # ----------------------------------------------------------------------

    if st.session_state.all_receipts_items:
        st.markdown("---")
        st.title("📚 Cumulative Spending Analysis Report")
        
        # 1. Create a single DataFrame from all accumulated items
        all_items_df_numeric = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        # Defensive coding: KRW Total Spend must exist for analysis
        if 'KRW Total Spend' not in all_items_df_numeric.columns:
             st.warning("Old data structure detected. Recalculating KRW totals...")
             all_items_df_numeric['KRW Total Spend'] = all_items_df_numeric.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )

        display_currency_label = 'KRW'


        # A. Display Accumulated Receipts Summary Table (Translated/Modified)
        st.subheader(f"Total {len(st.session_state.all_receipts_summary)} Receipts Logged (Summary)")
        summary_df = pd.DataFrame(st.session_state.all_receipts_summary)
        
        # Ensure compatibility with older sessions that lack Original_ columns
        if 'Original_Total' not in summary_df.columns:
            summary_df['Original_Total'] = summary_df['Total'] 
        if 'Original_Currency' not in summary_df.columns:
            summary_df['Original_Currency'] = 'KRW' 
            
        # Conditional formatting for Amount Paid
        def format_amount_paid(row):
            krw_amount = f"{row['Total']:,.0f} KRW"
            
            if row['Original_Currency'] != 'KRW':
                original_amount = f"{row['Original_Total']:,.2f} {row['Original_Currency']}"
                return f"{original_amount} / {krw_amount}"
            
            return krw_amount
        
        summary_df['Amount Paid'] = summary_df.apply(format_amount_paid, axis=1)

        
        summary_df = summary_df.drop(columns=['id'])
        summary_df = summary_df[['Date', 'Store', 'Amount Paid', 'filename']] 
        summary_df.columns = ['Date', 'Store', 'Amount Paid', 'Source'] 

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🛒 Integrated Detail Items") 
        
        all_items_df_display = all_items_df_numeric.copy()
        
        all_items_df_display['Original Total'] = all_items_df_display.apply(
            lambda row: f"{row['Total Spend']:,.2f} {row['Currency']}", axis=1
        )
        all_items_df_display['KRW Equivalent'] = all_items_df_display['KRW Total Spend'].apply(
            lambda x: f"{x:,.0f} KRW"
        )
        
        st.dataframe(
            all_items_df_display[['Item Name', 'Original Total', 'KRW Equivalent', 'AI Category']], 
            use_container_width=True, 
            hide_index=True
        )

        # 2. Aggregate spending by category and visualize (KRW based)
        category_summary = all_items_df_numeric.groupby('AI Category')['KRW Total Spend'].sum().reset_index()
        category_summary.columns = ['Category', 'Amount']
        
        # --- Display Summary Table ---
        st.subheader("💰 Spending Summary by Category")
        category_summary_display = category_summary.copy()
        category_summary_display['Amount'] = category_summary_display['Amount'].apply(lambda x: f"{x:,.0f} {display_currency_label}")
        st.dataframe(category_summary_display, use_container_width=True, hide_index=True)

        # --- Visualization (Charts use KRW Amount) ---
        col_chart, col_pie = st.columns(2)
        
        with col_chart:
            st.subheader(f"Bar Chart Visualization (Unit: {display_currency_label})")
            st.bar_chart(category_summary.set_index('Category'))
            
        with col_pie:
            st.subheader(f"Pie Chart Visualization (Unit: {display_currency_label})")
            chart_data = category_summary[category_summary['Amount'] > 0] 
            
            if not chart_data.empty:
                fig = px.pie(
                    chart_data, values='Amount', names='Category', 
                    title=f'Spending Distribution by Category (Unit: {display_currency_label})', hole=.3, 
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No spending data found to generate the pie chart.")

        # --- Spending Trend Over Time Chart (KRW based) ---
        st.markdown("---")
        st.subheader("📈 Spending Trend Over Time")
        
        summary_df_raw = pd.DataFrame(st.session_state.all_receipts_summary)
        
        if not summary_df_raw.empty:
            
            summary_df_raw['Date'] = pd.to_datetime(summary_df_raw['Date'], errors='coerce')
            summary_df_raw['Total'] = pd.to_numeric(summary_df_raw['Total'], errors='coerce') 
            
            daily_spending = summary_df_raw.dropna(subset=['Date', 'Total'])
            daily_spending = daily_spending.groupby('Date')['Total'].sum().reset_index()
            daily_spending.columns = ['Date', 'Daily Total Spend']
            
            if not daily_spending.empty:
                fig_trend = px.line(
                    daily_spending, x='Date', y='Daily Total Spend',
                    title=f'Daily Spending Trend (Unit: {display_currency_label})',
                    labels={'Daily Total Spend': f'Total Spend ({display_currency_label})', 'Date': 'Date'},
                    markers=True
                )
                fig_trend.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.warning("Date data is not available or not properly formatted to show the trend chart.")
        
        # 3. Generate AI Analysis Report
        st.markdown("---")
        st.subheader("🤖 AI Expert's Analysis Summary")
        
        total_spent = category_summary['Amount'].sum()
        detailed_items_for_ai = all_items_df_numeric[['AI Category', 'Item Name', 'KRW Total Spend']]
        items_text = detailed_items_for_ai.to_string(index=False)
        
        ai_report = generate_ai_analysis(
            summary_df=category_summary,
            store_name="Multiple Stores",
            total_amount=total_spent,
            currency_unit=display_currency_label, 
            detailed_items_text=items_text
        )
        
        st.info(ai_report)
        
        # 4. Reset and Download Buttons
        st.markdown("---")
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False, encoding='utf-8-sig')

        csv = convert_df_to_csv(all_items_df_numeric) 
        st.download_button(
            label="⬇️ Download Full Cumulative Ledger Data (CSV)",
            data=csv,
            file_name=f"all_receipts_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

        if st.button("🧹 Reset Record", help="Clears all accumulated receipt analysis records in the app."):
            st.session_state.all_receipts_items = []
            st.session_state.all_receipts_summary = []
            st.session_state.chat_history = [] 
            st.rerun() 

# ======================================================================
#     		 	TAB 2: FINANCIAL EXPERT CHAT
# ======================================================================
with tab2:
    st.header("💬 Financial Expert Chat")
    
    if not st.session_state.all_receipts_items:
        st.warning("Please analyze at least one receipt in the 'Analysis & Tracking' tab before starting a consultation.")
    else:
        # Chat uses KRW-based analysis data
        all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        # Defensive check for KRW Total Spend column
        if 'KRW Total Spend' not in all_items_df.columns:
             all_items_df['KRW Total Spend'] = all_items_df.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )
        
        category_summary = all_items_df.groupby('AI Category')['KRW Total Spend'].sum().reset_index()
        total_spent = category_summary['KRW Total Spend'].sum()
        summary_text = category_summary.to_string(index=False)
        display_currency_label_chat = 'KRW'
        
        # Prepare detailed item data for the chatbot's system instruction
        detailed_items_for_chat = all_items_df[['AI Category', 'Item Name', 'KRW Total Spend']]
        items_text_for_chat = detailed_items_for_chat.to_string(index=False)
        
        # MODIFIED SYSTEM INSTRUCTION
        system_instruction = f"""
        You are a supportive, friendly, and highly knowledgeable Financial Expert. Your role is to provide personalized advice on saving money, budgeting, and making smarter consumption choices.
        
        The user's cumulative spending data for the current session is as follows (All converted to KRW):
        - Total Accumulated Spending: {total_spent:,.0f} {display_currency_label_chat}
        - Category Breakdown (Category, Amount, all in {display_currency_label_chat}):
        {summary_text}
        
        **CRITICAL DETAILED DATA:** Below are the individual item names, their categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores if patterns are observed).
        --- Detailed Items Data (AI Category, Item Name, KRW Total Spend) ---
        {items_text_for_chat}
        ---

        Base all your advice and responses on this data. When asked for advice, refer directly to their spending patterns (e.g., "I see 'Food' is your largest expense..." or refer to specific items). Keep your tone professional yet encouraging. **Always include the currency unit (KRW) when referring to monetary amounts.**
        """

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process user input
        if prompt := st.chat_input("Ask for financial advice or review your spending..."):
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Expert is thinking..."):
                    try:
                        contents = [
                            {"role": "user", "parts": [{"text": msg["content"]}]} 
                            for msg in st.session_state.chat_history
                        ]
                        
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=contents,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Chatbot API call failed: {e}")
