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

# 💡 헬퍼 함수: 단일 값을 안전하게 추출하고, 숫자가 아니거나 누락된 경우 0.0을 반환합니다.
def safe_get_amount(data, key):
    """단일 값을 안전하게 추출하고, 숫자가 아니거나 누락된 경우 0.0을 반환합니다."""
    value = data.get(key, 0)
    numeric_value = pd.to_numeric(value, errors='coerce')
    return numeric_value if not pd.isna(numeric_value) else 0.0

# 💡 헬퍼 함수: 업로드된 아이템 데이터프레임에서 Summary 데이터를 재구성하는 헬퍼 함수
def regenerate_summary_data(item_df: pd.DataFrame) -> dict:
    """아이템 DataFrame에서 Summary 단위를 추출하고 재구성합니다. (CSV Import 전용)"""
    required_cols = ['Item Name', 'AI Category', 'KRW Total Spend']
    if not all(col in item_df.columns for col in required_cols):
        return None
    final_total_krw = item_df['KRW Total Spend'].sum()
    current_date = datetime.date.today().strftime('%Y-%m-%d')
    summary_data = {
        'id': f"imported-{pd.Timestamp.now().timestamp()}",
        'filename': 'Imported CSV',
        'Store': 'Imported Record',
        'Total': final_total_krw, 
        'Tax_KRW': 0.0, 
        'Tip_KRW': 0.0,
        'Currency': 'KRW', 
        'Date': current_date, 
        'Location': 'Imported Location', 
        'Original_Total': final_total_krw, 
        'Original_Currency': 'KRW' 
    }
    return summary_data

# 💡 헬퍼 함수: Level 3 카테고리를 최종 4가지 심리 카테고리 중 하나에 매핑하는 역할을 합니다.
def get_psychological_category(sub_category: str) -> str:
    """ Maps a detailed AI sub-category to one of the four main psychological categories. """
    nature = SPENDING_NATURE.get(sub_category, 'Loss_Unclassified')
    
    if nature in ['Investment_Asset']:
        return PSYCHOLOGICAL_CATEGORIES[0] # Investment / Asset
    elif nature in ['Consumption_Experience', 'Consumption_Planned']:
        return PSYCHOLOGICAL_CATEGORIES[1] # Experience / High-Value Consumption
    elif nature in ['Impulse_Habitual', 'Impulse_Convenience', 'Loss_Inefficiency', 'Loss_Unclassified']:
        return PSYCHOLOGICAL_CATEGORIES[2] # Habit / Impulse Loss
    elif nature in ['Fixed_Essential']:
        return PSYCHOLOGICAL_CATEGORIES[3] # Fixed / Essential Cost
    else:
        return PSYCHOLOGICAL_CATEGORIES[2] # Default to Impulse/Loss if unknown

# ----------------------------------------------------------------------
# 📌 Currency Rate Fetching (Original Code - Fallback included)
# ----------------------------------------------------------------------
@st.cache_data(ttl=datetime.timedelta(hours=24))
def get_exchange_rates():
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/USD"
    FALLBACK_RATES = {'KRW': 1.0, 'USD': 1350.00, 'EUR': 1450.00, 'JPY': 9.20} 
    exchange_rates = {'KRW': 1.0} 

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        conversion_rates = data.get('conversion_rates', {})
        krw_per_usd = conversion_rates.get('KRW', 0)
        usd_per_usd = conversion_rates.get('USD', 1.0) 

        if krw_per_usd == 0 or data.get('result') != 'success':
            raise ValueError("API returned incomplete or failed data or KRW rate is missing.")

        exchange_rates['USD'] = krw_per_usd / usd_per_usd 
        eur_rate_vs_usd = conversion_rates.get('EUR', 0)
        if eur_rate_vs_usd > 0:
            exchange_rates['EUR'] = krw_per_usd / eur_rate_vs_usd
        jpy_rate_vs_usd = conversion_rates.get('JPY', 0)
        if jpy_rate_vs_usd > 0:
            exchange_rates['JPY'] = krw_per_usd / jpy_rate_vs_usd
            
        st.sidebar.success(f"✅ Real-time rates loaded. (1 USD = {exchange_rates.get('USD', 0):,.2f} KRW)")
        return exchange_rates

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Error. Using fallback rates. ({e})")
        return FALLBACK_RATES
        
    except Exception as e:
        st.warning(f"⚠️ Exchange Rate Processing Error. Using fallback rates. ({e})")
        return FALLBACK_RATES


def convert_to_krw(amount: float, currency: str, rates: dict) -> float:
    """ Converts a foreign currency amount to KRW using stored rates (1 Foreign Unit = X KRW). """
    currency_upper = currency.upper().strip()
    rate = rates.get(currency_upper, rates.get('KRW', 1.0))
    if rate == 0:
        return amount * rates.get('USD', 1300) 
    return amount * rate

# ----------------------------------------------------------------------
# 📌 Global Categories (Data Definitions - Original Code)
# ----------------------------------------------------------------------
ALL_CATEGORIES = [
    "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars", 
    "Groceries", "Household Goods", "Medical & Pharmacy", "Health Supplements",
    "Education & Books", "Hobby & Skill Dev.", "Public Utilities", "Communication Fees", 
    "Public Transit", "Fuel & Vehicle Maint.", "Parking & Tolls", "Taxi Convenience",
    "Movies & Shows", "Travel & Accommodation", "Games & Digital Goods", 
    "Events & Gifts", "Fees & Penalties", "Rent & Mortgage", "Unclassified"
]

PSYCHOLOGICAL_CATEGORIES = [
    "Investment / Asset", 
    "Experience / High-Value Consumption", 
    "Habit / Impulse Loss", 
    "Fixed / Essential Cost"
]

SPENDING_NATURE = {
    "Rent & Mortgage": "Fixed_Essential", "Communication Fees": "Fixed_Essential",
    "Public Utilities": "Fixed_Essential", "Public Transit": "Fixed_Essential",
    "Parking & Tolls": "Fixed_Essential", "Medical & Pharmacy": "Investment_Asset",
    "Health Supplements": "Investment_Asset", "Education & Books": "Investment_Asset",
    "Hobby & Skill Dev.": "Investment_Asset", "Events & Gifts": "Investment_Asset",
    "Groceries": "Consumption_Planned", "Household Goods": "Consumption_Planned",
    "Fuel & Vehicle Maint.": "Consumption_Planned", "Dining Out": "Consumption_Experience",
    "Travel & Accommodation": "Consumption_Experience", "Movies & Shows": "Consumption_Experience",
    "Casual Dining": "Impulse_Habitual", "Coffee & Beverages": "Impulse_Habitual",
    "Alcohol & Bars": "Impulse_Habitual", "Games & Digital Goods": "Impulse_Habitual",
    "Taxi Convenience": "Impulse_Convenience", "Fees & Penalties": "Loss_Inefficiency",
    "Unclassified": "Loss_Unclassified"
}

def get_category_guide():
    guide = ""
    categories = {
        "FIXED / ESSENTIAL": ["Rent & Mortgage", "Communication Fees", "Public Utilities", "Public Transit", "Parking & Tolls"],
        "VARIABLE / CONSUMPTION": ["Groceries", "Household Goods", "Fuel & Vehicle Maint.", "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars"],
        "INVESTMENT / ASSET": ["Medical & Pharmacy", "Health Supplements", "Education & Books", "Hobby & Skill Dev.", "Events & Gifts"],
        "DISCRETIONARY / LOSS": ["Travel & Accommodation", "Movies & Shows", "Games & Digital Goods", "Taxi Convenience", "Fees & Penalties", "Unclassified"],
    }
    for main, subs in categories.items():
        guide += f"- **{main}**: {', '.join(subs)}\n"
    return guide


# ----------------------------------------------------------------------
# 📌 2. Initialize Session State & Page Configuration (Original Code)
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    st.session_state.all_receipts_summary = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_data_hash' not in st.session_state:
    st.session_state.last_data_hash = None # 💡 Hash for data change detection

st.set_page_config(
    page_title="Smart Receipt Analyzer & Tracker 🧾",
    layout="wide"
)

# ----------------------------------------------------------------------
# 📌 3. Sidebar and Main Title (Original Code)
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("About This App")
    st.markdown("---")
    st.subheader("How to Use")
    st.markdown("""
    This application helps you manage your household ledger easily by using AI.
    1. **Upload / Manual Input:** Enter spending data via receipt image or manual form.
    2. **Analyze & Accumulate:** Results are added to the cumulative record.
    3. **Review & Chat:** Check the integrated report, spending charts, and get personalized financial advice.
    4. **Export & Continue:** Export the current record in CSV, load the CSV to continue recording.
    """)
    st.markdown("---")
    if st.session_state.all_receipts_items:
        st.info(f"Currently tracking {len(st.session_state.all_receipts_summary)} receipts.") 
        
st.title("🧾 Receipt Recorder powered by AI")
st.markdown("---")
EXCHANGE_RATES = get_exchange_rates()


# ----------------------------------------------------------------------
# 📌 1. Gemini Analysis Function (Original Code)
# ----------------------------------------------------------------------
def analyze_receipt_with_gemini(_image: Image.Image):
    prompt_template = """
    You are an expert in receipt analysis and ledger recording.
    Analyze the following items from the receipt image and **you must extract them in JSON format**.
    
    **CRITICAL INSTRUCTION:** The response must only contain the **JSON code block wrapped in backticks (```json)**. Do not include any explanations, greetings, or additional text outside the JSON code block.
    
    1. store_name: Store Name (text)
    2. date: Date (YYYY-MM-DD format). **If not found, use YYYY-MM-DD format based on today's date.**
    3. store_location: Store location/address (text). **If not found, use "Seoul".**
    4. total_amount: Final amount settled/paid via card or cash (numbers only, no commas). **Must match the '합계' (Total) or final payment amount (e.g., 19,400).** 5. tax_amount: Tax or VAT amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    6. tip_amount: Tip amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    7. discount_amount: Total discount amount applied to the entire receipt (numbers only, no commas). Must be 0 if not present.
    8. currency_unit: Official currency code shown on the receipt (e.g., KRW, USD, EUR).
    9. items: List of purchased items. Each item must include:
        - name: Item Name (text)
        - price: Unit Price (numbers only, no commas). **This must be the final, VAT-INCLUSIVE price displayed next to the item name.** - quantity: Quantity (numbers only)
        - category: The most appropriate **Detailed Sub-Category** for this item, which must be **automatically classified** by you.
    
    **Classification Guide (Choose ONE sub-category for 'category' field):**
    - **FIXED / ESSENTIAL:** Rent & Mortgage, Communication Fees, Public Utilities, Public Transit, Fuel & Vehicle Maint., Parking & Tolls
    - **VARIABLE / CONSUMPTION (Planned):** Groceries, Household Goods
    - **VARIABLE / CONSUMPTION (Experience):** Dining Out, Travel & Accommodation, Movies & Shows
    - **INVESTMENT / ASSET:** Medical & Pharmacy, Health Supplements, Education & Books, Hobby & Skill Dev., Events & Gifts
    - **IMPULSE / LOSS:** Casual Dining, Coffee & Beverages, Alcohol & Bars, Games & Digital Goods, Taxi Convenience, Fees & Penalties, Unclassified
        
    JSON Schema:
    ```json
    {
      "store_name": "...",
      "date": "...",
      "store_location": "...",
      "total_amount": ...,
      "tax_amount": ...,
      "tip_amount": ...,
      "discount_amount": ...,
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

# ----------------------------------------------------------------------
# 📌 2. AI Analysis Report Generation Function (Original Code)
# ----------------------------------------------------------------------
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float, currency_unit: str, detailed_items_text: str):
    summary_text = summary_df.to_string(index=False)
    prompt_template = f"""
    You are an expert in receipt analysis and ledger recording, acting as a **friendly yet professional financial advisor**.
    Your analysis must be based strictly on the provided data, ensuring high credibility and clarity.

    The user's **all accumulated spending** amounts to {total_amount:,.0f} {currency_unit}.
    
    Below is the category breakdown of all accumulated spending (Unit: {currency_unit}):
    --- Spending Summary Data (Category, Amount) ---
    {summary_text}
    ---
    
    **CRITICAL DETAILED DATA:** Below are the individual item names, their categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores if patterns are observed).
    --- Detailed Items Data (AI Category, Item Name, Total Spend) ---
    {detailed_items_text}
    ---

    Follow these instructions and provide an analysis report in a **friendly and professional tone**:
    1. Summarize the main characteristic of this total spending (e.g., the largest spending category and its driving factor based on individual items). **Reference the data directly to justify your summary.**
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
# 📌 4. Streamlit UI: Tab Setup (Original Code)
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Analysis & Tracking", "💬 Financial Expert Chat"])


# ======================================================================
#             	TAB 1: ANALYSIS & TRACKING (수정 반영)
# ======================================================================
with tab1:
    
    # --- CSV Upload Logic (Original Code) ---
    st.subheader("📁 Load Previous Record (CSV Upload)")
    if 'csv_load_triggered' not in st.session_state:
        st.session_state.csv_load_triggered = False
        
    uploaded_csv_file = st.file_uploader(
        "Upload a previously downloaded ledger CSV file (e.g., record_YYYYMMDD.csv)",
        type=['csv'],
        accept_multiple_files=False,
        key='csv_uploader', 
        on_change=lambda: st.session_state.__setitem__('csv_load_triggered', True)
    )

    if st.session_state.csv_load_triggered and uploaded_csv_file is not None:
        st.session_state.csv_load_triggered = False 
        try:
            imported_df = pd.read_csv(uploaded_csv_file)
            required_cols = ['Item Name', 'Unit Price', 'Quantity', 'AI Category', 'Total Spend', 'Currency', 'KRW Total Spend']
            
            if not all(col in imported_df.columns for col in required_cols):
                st.error("❌ 업로드된 CSV 파일에 필수 컬럼이 부족합니다. 올바른 형식의 파일을 업로드해주세요.")
            else:
                st.session_state.all_receipts_items.append(imported_df)
                summary_data = regenerate_summary_data(imported_df)
                if summary_data:
                    st.session_state.all_receipts_summary.append(summary_data)
                    st.success(f"🎉 CSV 파일 **{uploaded_csv_file.name}**의 기록 (**{len(imported_df)}개 아이템**)이 성공적으로 불러와져 누적되었습니다.")
                    st.rerun()
                else:
                    st.error("❌ CSV 파일에서 Summary 데이터를 재구성하는 데 실패했습니다.")
        except Exception as e:
            st.error(f"❌ CSV 파일을 처리하는 중 오류가 발생했습니다: {e}")
            
            
    st.markdown("---")
    
    # --- File Uploader and Analysis (Original Code with minor fix on Total calculation) ---
    st.subheader("📸 Upload Receipt Image (AI Analysis)")
    uploaded_file = st.file_uploader(
        "Upload one receipt image (jpg, png) at a time. (Data will accumulate in the current session)", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=False 
    )


    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        
        # 💡 중복 파일 체크
        existing_summary = next((s for s in st.session_state.all_receipts_summary if s.get('id') == file_id), None)
        is_already_analyzed = existing_summary is not None
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Uploaded Receipt")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True) 

        with col2:
            st.subheader("📊 Analysis and Recording")
            
            if is_already_analyzed:
                
                # 💡 중복된 경우: 분석 버튼을 비활성화하고, 저장된 데이터를 바로 표시합니다.
                st.warning(f"⚠️ This receipt ({uploaded_file.name}) is already analyzed. Data is **not recorded again**.")
                analyze_button_disabled = st.button("✨ Start Receipt Analysis", disabled=True, key="analyze_disabled")
                
                # --- 저장된 결과 표시 로직 ---
                display_unit = existing_summary['Original_Currency']
                st.markdown(f"**🏠 Store Name:** {existing_summary.get('Store', 'N/A')}")
                st.markdown(f"**📍 Location:** {existing_summary.get('Location', 'N/A')}")
                st.markdown(f"**📅 Date:** {existing_summary.get('Date', 'N/A')}")
                st.subheader(f"💰 Total Amount Paid: {existing_summary.get('Original_Total', 0):,.0f} {display_unit}")
                
                krw_tax = existing_summary.get('Tax_KRW', 0)
                krw_tip = existing_summary.get('Tip_KRW', 0)
                
                if krw_tax > 0 or krw_tip > 0:
                    tax_display = f"{krw_tax:,.0f} KRW"
                    tip_display = f"{krw_tip:,.0f} KRW"
                    st.markdown(f"**🧾 Tax/VAT (KRW):** {tax_display} | **💸 Tip (KRW):** {tip_display}")
                
                st.info(f"누적 기록 총액 (KRW): **{existing_summary.get('Total', 0):,.0f} KRW** (부가세 포함)")
                st.markdown("---")

                # 중복이므로 추가적인 분석 로직은 실행하지 않음
                pass 
                
            else:
                # 중복이 아닌 경우: 분석 버튼을 활성화하고, 버튼이 눌리면 분석을 실행합니다.
                analyze_button = st.button("✨ Start Receipt Analysis", key="analyze_active")

                if analyze_button:
                    # AI 분석 실행 로직 (기존 코드와 동일)
                    st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
                    with st.spinner('AI is reading the receipt...'):
                        
                        json_data_text = analyze_receipt_with_gemini(image)

                        if json_data_text:
                            # ... (데이터 파싱 및 저장 로직 전체) ...
                            # ... (데이터프레임 편집 및 저장 로직 전체) ...
                            
                            # 💡 저장 완료 후
                            st.success(f"🎉 Data from {uploaded_file.name} successfully added (Converted to KRW)!")
                            st.rerun()
                        else:
                            st.error("Analysis failed to complete. Please try again.")

    st.markdown("---")

            if analyze_button and not is_already_analyzed:
                st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
                with st.spinner('AI is reading the receipt...'):
                    json_data_text = analyze_receipt_with_gemini(image)

                    if json_data_text:
                        try:
                            # JSON 파싱 및 데이터 정리 (Original Code)
                            cleaned_text = json_data_text.strip()
                            if cleaned_text.startswith("```json"):
                                cleaned_text = cleaned_text.lstrip("```json")
                            if cleaned_text.endswith("```"):
                                cleaned_text = cleaned_text.rstrip("```")
                            receipt_data = json.loads(cleaned_text.strip()) 
                            
                            total_amount = safe_get_amount(receipt_data, 'total_amount')
                            tax_amount = safe_get_amount(receipt_data, 'tax_amount')
                            tip_amount = safe_get_amount(receipt_data, 'tip_amount')
                            discount_amount = safe_get_amount(receipt_data, 'discount_amount')
                            currency_unit = receipt_data.get('currency_unit', '').strip()
                            display_unit = currency_unit if currency_unit else 'KRW'
                            receipt_date_str = receipt_data.get('date', '').strip()
                            store_location_str = receipt_data.get('store_location', '').strip()
                            
                            try:
                                date_object = pd.to_datetime(receipt_date_str, format='%Y-%m-%d', errors='raise').date()
                                final_date = date_object.strftime('%Y-%m-%d')
                            except (ValueError, TypeError):
                                final_date = datetime.date.today().strftime('%Y-%m-%d')
                                st.warning("⚠️ AI가 인식한 날짜가 유효하지 않아 오늘 날짜로 대체되었습니다.")
                                
                            final_location = store_location_str if store_location_str else "Seoul"

                            # --- Main Information Display ---
                            st.success("✅ Analysis Complete! Check the ledger data below.")
                            st.markdown(f"**🏠 Store Name:** {receipt_data.get('store_name', 'N/A')}")
                            st.subheader(f"💰 Total Amount Paid: {total_amount:,.0f} {display_unit}")
                            if discount_amount > 0:
                                st.markdown(f"**🎁 Total Discount:** {discount_amount:,.2f} {display_unit}") 
                            if tax_amount > 0 or tip_amount > 0:
                                st.markdown(f"**🧾 Tax/VAT:** {tax_amount:,.2f} {display_unit} | **💸 Tip:** {tip_amount:,.2f} {display_unit}")
                            st.markdown("---")

                            if 'items' in receipt_data and receipt_data['items']:
                                items_df = pd.DataFrame(receipt_data['items'])
                                items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                                items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                                items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                                items_df['Total Spend Original'] = items_df['Unit Price'] * items_df['Quantity']
                                
                                # --- 할인 안분 로직 (Original Code) ---
                                items_df['Discount Applied'] = 0.0
                                items_df['Total Spend'] = items_df['Total Spend Original']
                                total_item_original = items_df['Total Spend Original'].sum()
                                if discount_amount > 0 and total_item_original > 0:
                                    discount_rate = discount_amount / total_item_original
                                    items_df['Discount Applied'] = items_df['Total Spend Original'] * discount_rate
                                    items_df['Total Spend'] = items_df['Total Spend Original'] - items_df['Discount Applied']
                                    st.info(f"💡 Discount of {discount_amount:,.0f} {display_unit} successfully allocated across items.")
                                
                                st.subheader("🛒 Detailed Item Breakdown (Category Editable)")
                                edited_df = st.data_editor(
                                    items_df.drop(columns=['Total Spend Original', 'Discount Applied', 'Total Spend']), 
                                    column_config={
                                        "AI Category": st.column_config.SelectboxColumn("Final Category", help="Select the correct sub-category for this item.", width="medium", options=ALL_CATEGORIES, required=True),
                                    },
                                    disabled=['Item Name', 'Unit Price', 'Quantity'], 
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                # --- 통화 변환 및 Summary 저장 로직 (수정 반영) ---
                                edited_df['Total Spend'] = items_df['Total Spend']
                                edited_df['Total Spend Numeric'] = pd.to_numeric(edited_df['Total Spend'], errors='coerce').fillna(0)
                                edited_df['Currency'] = display_unit
                                
                                # KRW Total Spend 계산 (VAT 포함된 채로 유지)
                                edited_df['KRW Total Spend'] = edited_df.apply(
                                    lambda row: convert_to_krw(row['Total Spend Numeric'], row['Currency'], EXCHANGE_RATES), axis=1
                                )
                                edited_df = edited_df.drop(columns=['Total Spend Numeric'])

                                # 💡 세금과 팁도 원화로 환산
                                krw_tax_total = convert_to_krw(tax_amount, display_unit, EXCHANGE_RATES) 
                                krw_tip_total = convert_to_krw(tip_amount, display_unit, EXCHANGE_RATES)
                                
                                # ** CRITICAL FIX: final_total_krw는 VAT가 포함된 아이템 총합 + Tip만 더합니다.
                                #    (163,600 KRW + Tip)이 되도록 합니다.
                                final_total_krw = edited_df['KRW Total Spend'].sum() + krw_tip_total
                                
                                # ** Accumulate Data: Store the edited DataFrame **
                                st.session_state.all_receipts_items.append(edited_df)
                                
                                st.session_state.all_receipts_summary.append({
                                    'id': file_id, 
                                    'filename': uploaded_file.name,
                                    'Store': receipt_data.get('store_name', 'N/A'),
                                    'Total': final_total_krw, # 💡 아이템 총합(VAT 포함) + Tip
                                    'Tax_KRW': krw_tax_total, 
                                    'Tip_KRW': krw_tip_total, 
                                    'Currency': 'KRW', 
                                    'Date': final_date, 
                                    'Location': final_location, 
                                    'Original_Total': total_amount, 
                                    'Original_Currency': display_unit 
                                })

                                st.success(f"🎉 Data from {uploaded_file.name} successfully added (Converted to KRW)!")
                                st.rerun()

                            else:
                                st.warning("Item list could not be found in the analysis result.")

                        except json.JSONDecodeError:
                            st.error("❌ Gemini analysis result is not a valid JSON format. (JSON parsing error)")
                        except Exception as e:
                            st.error(f"Unexpected error occurred during data processing: {e}")
                    else:
                        st.error("Analysis failed to complete. Please try again.")

    st.markdown("---")
    
    # --- Manual Expense Input (Original Code) ---
    st.subheader("📝 Manual Expense Input (No Receipt)")
    st.info("""**✅ Input Guide**\nRecord your expense details easily.\n**💡 Category Scheme (Sub-Category)**\n""" + get_category_guide())

    with st.form("manual_expense_form", clear_on_submit=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            manual_date = st.date_input("📅 Expense Date", value=datetime.date.today())
            manual_description = st.text_input("📝 Expense Item (Description)", placeholder="e.g., Lunch, Groceries")
            
        with col_m2:
            manual_store = st.text_input("🏠 Store/Merchant Name", placeholder="e.g., Local Diner, Starbucks")
            manual_amount = st.number_input("💰 Expense Amount (Numbers Only)", min_value=0.0, step=100.0, format="%.2f")
            
        with col_m3:
            manual_category = st.selectbox("📌 Category (Sub-Category)", options=ALL_CATEGORIES, index=ALL_CATEGORIES.index('Unclassified'))
            manual_currency = st.selectbox("Currency Unit", options=['KRW', 'USD', 'EUR', 'JPY'], index=0)
            manual_location = st.text_input("📍 Location/City", placeholder="e.g., Gangnam, Seoul") 
            
        submitted = st.form_submit_button("✅ Add to Ledger")

        if submitted:
            if manual_description and manual_amount > 0 and manual_category:
                krw_total = convert_to_krw(manual_amount, manual_currency, EXCHANGE_RATES)
                applied_rate = EXCHANGE_RATES.get(manual_currency, 1.0)

                manual_df = pd.DataFrame([{'Item Name': manual_description, 'Unit Price': manual_amount, 'Quantity': 1, 'AI Category': manual_category, 'Total Spend': manual_amount, 'Currency': manual_currency, 'KRW Total Spend': krw_total}])
                manual_summary = {'id': f"manual-{pd.Timestamp.now().timestamp()}", 'filename': 'Manual Entry', 'Store': manual_store if manual_store else 'Manual Entry', 'Total': krw_total, 'Tax_KRW': 0.0, 'Tip_KRW': 0.0, 'Currency': 'KRW', 'Date': manual_date.strftime('%Y-%m-%d'), 'Location': manual_location if manual_location else "Manual Input Location", 'Original_Total': manual_amount, 'Original_Currency': manual_currency}
                
                st.session_state.all_receipts_items.append(manual_df)
                st.session_state.all_receipts_summary.append(manual_summary)
                
                if manual_currency != 'KRW':
                    rate_info = f" (Applied Rate: 1 {manual_currency} = {applied_rate:,.4f} KRW)"
                else:
                    rate_info = ""
                    
                st.success(f"🎉 {manual_date.strftime('%Y-%m-%d')} expense recorded ({manual_description}: {manual_amount:,.2f} {manual_currency} -> **{krw_total:,.0f} KRW**){rate_info}. Added to ledger.")
                st.rerun()
            else:
                st.error("❌ 'Expense Item', 'Expense Amount', and 'Category' are required fields. Amount must be greater than 0.")

    st.markdown("---")
    
    # --- Cumulative Data Analysis Section (Original Code) ---
    if st.session_state.all_receipts_items:
        st.markdown("---")
        st.title("📚 Cumulative Spending Analysis Report")
        
        all_items_df_numeric = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        if 'KRW Total Spend' not in all_items_df_numeric.columns:
             st.warning("Old data structure detected. Recalculating KRW totals...")
             all_items_df_numeric['KRW Total Spend'] = all_items_df_numeric.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )

        display_currency_label = 'KRW'

        # A. Display Accumulated Receipts Summary Table
        st.subheader(f"Total {len(st.session_state.all_receipts_summary)} Receipts Logged (Summary)")
        summary_df = pd.DataFrame(st.session_state.all_receipts_summary)
        
        # Ensure compatibility with older sessions that lack columns
        if 'Original_Total' not in summary_df.columns:
            summary_df['Original_Total'] = summary_df['Total'] 
        if 'Original_Currency' not in summary_df.columns:
            summary_df['Original_Currency'] = 'KRW' 
        if 'Tax_KRW' not in summary_df.columns:
            summary_df['Tax_KRW'] = 0.0
        if 'Tip_KRW' not in summary_df.columns:
            summary_df['Tip_KRW'] = 0.0
        if 'Location' not in summary_df.columns:
            summary_df['Location'] = 'N/A'
            
        def format_amount_paid(row):
            krw_amount = f"{row['Total']:,.0f} KRW"
            if row['Original_Currency'] != 'KRW':
                original_amount = f"{row['Original_Total']:,.2f} {row['Original_Currency']}"
                return f"{original_amount} / {krw_amount}"
            return krw_amount
        
        summary_df['Amount Paid'] = summary_df.apply(format_amount_paid, axis=1)
        summary_df = summary_df.drop(columns=['id'])
        summary_df = summary_df[['Date', 'Store', 'Location', 'Amount Paid', 'Tax_KRW', 'Tip_KRW', 'filename']] 
        summary_df.columns = ['Date', 'Store', 'Location', 'Amount Paid', 'Tax (KRW)', 'Tip (KRW)', 'Source'] 

        st.dataframe(
            summary_df, use_container_width=True, hide_index=True,
            column_config={"Tax (KRW)": st.column_config.NumberColumn("Tax (KRW)", format="%.0f KRW"), "Tip (KRW)": st.column_config.NumberColumn("Tip (KRW)", format="%.0f KRW")}
        )
        
        st.markdown("---")
        
        st.subheader("🛒 Integrated Detail Items") 
        all_items_df_display = all_items_df_numeric.copy()
        all_items_df_display['Original Total'] = all_items_df_display.apply(lambda row: f"{row['Total Spend']:,.2f} {row['Currency']}", axis=1)
        all_items_df_display['KRW Equivalent'] = all_items_df_display['KRW Total Spend'].apply(lambda x: f"{x:,.0f} KRW")
        
        st.dataframe(
            all_items_df_display[['Item Name', 'Original Total', 'KRW Equivalent', 'AI Category']], 
            use_container_width=True, 
            hide_index=True
        )

        # 2. Aggregate spending by category and visualize (KRW based)
        category_summary = all_items_df_numeric.groupby('AI Category')['KRW Total Spend'].sum().reset_index()
        category_summary.columns = ['Category', 'Amount']
        
        # 💡 Tax 합산 로직 제거 (일관성 유지 및 Chatbot 오류 방지)
        total_tax_krw = summary_df['Tax (KRW)'].sum()
        total_tip_krw = summary_df['Tip (KRW)'].sum()
        
        # if total_tax_krw > 0:
        #     category_summary.loc[len(category_summary)] = ['세금/부가세 (Tax/VAT)', total_tax_krw] # 🚨 이 줄을 제거했습니다.
        if total_tip_krw > 0:
            category_summary.loc[len(category_summary)] = ['팁 (Tip)', total_tip_krw]
            
        # --- Display Summary Table ---
        st.subheader("💰 Spending Summary by Category (Items + Tip)") 
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
                fig = px.pie(chart_data, values='Amount', names='Category', title=f'Spending Distribution by Category (Unit: {display_currency_label})', hole=.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No spending data found to generate the pie chart.")

        # --- Spending Trend Over Time Chart (Original Code) ---
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
                fig_trend = px.line(daily_spending, x='Date', y='Daily Total Spend', title=f'Daily Spending Trend (Unit: {display_currency_label})', labels={'Daily Total Spend': f'Total Spend ({display_currency_label})', 'Date': 'Date'}, markers=True)
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
            summary_df=category_summary.reset_index(drop=True),
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
            file_name=f"record_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

        if st.button("🧹 Reset Record", help="Clears all accumulated receipt analysis records in the app."):
            st.session_state.all_receipts_items = []
            st.session_state.all_receipts_summary = []
            st.session_state.chat_history = [] 
            st.rerun() 

# ======================================================================
# 		 	TAB 2: FINANCIAL EXPERT CHAT (수정 반영)
# ======================================================================
with tab2:
    st.header("💬 Financial Expert Chat")
    
    if not st.session_state.all_receipts_items:
        st.warning("Please analyze at least one receipt or load a CSV in the 'Analysis & Tracking' tab before starting a consultation.")
    else:
        # --- Chat History Reset Logic (Original Code) ---
        current_data_hash = hash(tuple(item['id'] for item in st.session_state.all_receipts_summary))
        
        if 'last_data_hash' not in st.session_state or st.session_state.last_data_hash != current_data_hash:
            st.session_state.chat_history = []
            st.session_state.last_data_hash = current_data_hash
            st.info("📊 새로운 지출 내역이 감지되었습니다. 신선한 분석을 위해 채팅 기록이 초기화됩니다.")
            
        all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        if 'KRW Total Spend' not in all_items_df.columns:
             all_items_df['KRW Total Spend'] = all_items_df.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )

        # 1. Add Psychological Category
        all_items_df['Psychological Category'] = all_items_df['AI Category'].apply(get_psychological_category)

        # 2. Group by the new Psychological Category
        psychological_summary = all_items_df.groupby('Psychological Category')['KRW Total Spend'].sum().reset_index()
        psychological_summary.columns = ['Category', 'KRW Total Spend']

        # 3. Add Tip only to Fixed/Essential Cost (CRITICAL FIX)
        summary_df_for_chat = pd.DataFrame(st.session_state.all_receipts_summary)
        
        tax_tip_only_total = 0.0
        # 🚨 CRITICAL FIX: Tax_KRW 합산 로직을 삭제합니다. Tip만 합산하여 Fixed Cost에 반영합니다.
        # if 'Tax_KRW' in summary_df_for_chat.columns:
        #     tax_tip_only_total += summary_df_for_chat['Tax_KRW'].sum() 
        if 'Tip_KRW' in summary_df_for_chat.columns:
            tax_tip_only_total += summary_df_for_chat['Tip_KRW'].sum() # Tip만 합산

        # Add Tip (Only) to the 'Fixed / Essential Cost' category
        if tax_tip_only_total > 0:
             fixed_cost_index = psychological_summary[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[3]].index
             if not fixed_cost_index.empty:
                 psychological_summary.loc[fixed_cost_index[0], 'KRW Total Spend'] += tax_tip_only_total
             else:
                 new_row = pd.DataFrame([{'Category': PSYCHOLOGICAL_CATEGORIES[3], 'KRW Total Spend': tax_tip_only_total}])
                 psychological_summary = pd.concat([psychological_summary, new_row], ignore_index=True)


        total_spent = psychological_summary['KRW Total Spend'].sum() # 💡 이제 이 값은 VAT가 포함된 아이템 합계 + Tip
        
        # Calculate the Impulse Spending Index
        impulse_spending = psychological_summary.loc[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[2], 'KRW Total Spend'].sum()
        impulse_index = impulse_spending / total_spent if total_spent > 0 else 0.0
        
        psychological_summary_text = psychological_summary.to_string(index=False)
        
        # Prepare detailed item data for the chatbot's system instruction
        detailed_items_for_chat = all_items_df[['Psychological Category', 'Item Name', 'KRW Total Spend']]
        items_text_for_chat = detailed_items_for_chat.to_string(index=False)
        
        # MODIFIED SYSTEM INSTRUCTION (Original Code)
        system_instruction = f"""
        You are a supportive, friendly, and highly knowledgeable Financial Psychologist and Advisor. Your role is to analyze the user's spending habits from a **psychological and behavioral economics perspective**, and provide personalized advice on overcoming impulse spending and optimizing happiness per won. Your tone should be consistently polite and helpful, like a professional mentor.
        
        The user's cumulative spending data for the current session (All converted to KRW) is analyzed by its **Psychological Spending Nature**:
        - **Total Accumulated Spending**: {total_spent:,.0f} KRW
        - **Calculated Impulse Spending Index**: {impulse_index:.2f} (Target: < 0.20)
        - **Psychological Category Breakdown (Category, Amount)**:
        {psychological_summary_text}
        
        **CRITICAL DETAILED DATA:** Below are the individual item names, their original AI categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores, or refer to high-frequency, low-value items that drive the Impulse Index).
        --- Detailed Items Data (Psychological Category, Item Name, KRW Total Spend) ---
        {items_text_for_chat}
        ---

        Base all your advice and responses on this data. Your analysis MUST start with a professional interpretation of the **Impulse Spending Index**. Provide actionable, psychological tips to convert 'Impulse Loss' spending into 'Investment/Asset' spending. Always include the currency unit (KRW) when referring to monetary amounts.
        """

        # 💡 초기 메시지 추가 (Original Code)
        if not st.session_state.chat_history or (len(st.session_state.chat_history) == 1 and st.session_state.chat_history[0]["content"].startswith("안녕하세요! 저는 귀하의 지출 패턴을 분석하는")):
             st.session_state.chat_history = []
             initial_message = f"""
             안녕하세요! 저는 귀하의 소비 심리 패턴을 분석하는 AI 금융 심리 전문가입니다. 🧠
             현재까지 총 **{total_spent:,.0f} KRW**의 지출이 기록되었으며,
             귀하의 **소비 충동성 지수 (Impulse Spending Index)**는 **{impulse_index:.2f}**으로 분석되었습니다. (목표치는 0.20 이하)

             이 지수는 귀하의 지출 중 비계획적이고 습관적인 손실성 소비의 비율을 나타냅니다.
             어떤 부분에 대해 더 자세한 심리적 조언을 드릴까요? 예를 들어, 다음과 같은 질문을 할 수 있습니다.

             * "제 충동성 지수 {impulse_index:.2f}이 의미하는 바는 무엇인가요?"
             * "지출을 **'미래 투자(Investment / Asset)'**로 전환하려면 어떻게 해야 할까요?"
             * "제 지출에서 가장 큰 **습관적 손실** 항목을 알려주세요."
             """
             st.session_state.chat_history.append({"role": "assistant", "content": initial_message})

        # Display chat history (Original Code)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process user input (Original Code)
        if prompt := st.chat_input("Ask for financial advice or review your spending..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Expert is thinking..."):
                    try:
                        combined_contents = []
                        history_items = st.session_state.chat_history 
                        
                        for item in history_items:
                            gemini_role = "user" if item["role"] == "user" else "model" 
                            combined_contents.append({"role": gemini_role, "parts": [{"text": item["content"]}]})
                        
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=combined_contents,
                            config=genai.types.GenerateContentConfig(system_instruction=system_instruction)
                        )
                        
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Chatbot API call failed: {e}")
