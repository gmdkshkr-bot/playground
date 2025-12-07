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
import time 
from fpdf import FPDF # 📢 PDF 라이브러리 임포트 (fpdf2 설치 필요)

# ----------------------------------------------------------------------
# 📌 0. Currency Conversion Setup & Globals
# ----------------------------------------------------------------------

try:
    # 🚨 NOTE: These keys must be set in Streamlit Secrets.
    API_KEY = st.secrets["GEMINI_API_KEY"]
    EXCHANGE_RATE_API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"] 
    # 📢 Kakao API Key Load
    KAKAO_REST_API_KEY = st.secrets["KAKAO_REST_API_KEY"]
except KeyError:
    st.error("❌ Please set 'GEMINI_API_KEY', 'EXCHANGE_RATE_API_KEY', and 'KAKAO_REST_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)

# --- 📢 [UPDATED] Geocoding Helper Function (Kakao API Optimized) ---
@st.cache_data(ttl=datetime.timedelta(hours=48))
def geocode_address(address: str) -> tuple[float, float]:
    """
    Uses Kakao Local API to convert address to latitude and longitude. (Kakao Maps API)
    """
    if not address or address == "Manual Input Location" or address == "Imported Location":
        # Returns fallback coordinates (Seoul center)
        return 37.5665, 126.9780
    
    # 📢 Kakao Local API Call Setup
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and data.get('documents'):
            document = data['documents'][0]
            # Kakao API returns longitude (x) first, then latitude (y)
            lat = float(document.get('y', 0))
            lon = float(document.get('x', 0))
            
            if lat != 0 and lon != 0:
                return lat, lon

    except requests.exceptions.RequestException as e:
        pass
    except Exception as e:
        pass

    # Fallback coordinates (Seoul center)
    return 37.5665, 126.9780


# 💡 Helper function: Safely extracts a single amount value
def safe_get_amount(data, key):
    """Safely extracts a single value and returns 0.0 if non-numeric or missing."""
    value = data.get(key, 0)
    numeric_value = pd.to_numeric(value, errors='coerce')
    return numeric_value if not pd.isna(numeric_value) else 0.0

# 💡 Helper function: Regenerates Summary data for imported CSVs
def regenerate_summary_data(item_df: pd.DataFrame) -> dict:
    """Regenerates summary data from the item DataFrame for CSV import."""
    
    required_cols = ['Item Name', 'AI Category', 'KRW Total Spend']
    if not all(col in item_df.columns for col in required_cols):
        return None

    final_total_krw = item_df['KRW Total Spend'].sum()
    current_date = datetime.date.today().strftime('%Y-%m-%d')
    
    lat, lon = geocode_address("Imported Location")
    
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
        'Original_Currency': 'KRW',
        'latitude': lat,
        'longitude': lon
    }
    return summary_data

# 💡 Helper function: Maps sub-category to psychological category
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


@st.cache_data(ttl=datetime.timedelta(hours=24))
def get_exchange_rates():
    """
    Fetches real-time exchange rates using ExchangeRate-API (USD Base).
    Returns a dictionary: {currency_code: 1 Foreign Unit = X KRW}
    """
    
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/USD"
    # Fallback Rates: 1 Foreign Unit = X KRW
    FALLBACK_RATES = {'KRW': 1.0, 'USD': 1350.00, 'EUR': 1450.00, 'JPY': 9.20} 
    exchange_rates = {'KRW': 1.0} 

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        conversion_rates = data.get('conversion_rates', {})
        
        # 1. KRW Rate (USD -> KRW) extraction
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

# Global Categories (Internal classification names remain Korean for consistency with AI analysis prompt)
ALL_CATEGORIES = [
    "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars", 
    "Groceries", 
    "Household Essentials", "Beauty & Cosmetics", "Clothing & Fashion", # 📢 Detailed Categories
    "Medical & Pharmacy", "Health Supplements",
    "Education & Books", "Hobby & Skill Dev.", "Public Utilities", "Communication Fees", 
    "Public Transit", "Fuel & Vehicle Maint.", "Parking & Tolls", "Taxi Convenience",
    "Movies & Shows", "Travel & Accommodation", "Games & Digital Goods", 
    "Events & Gifts", "Fees & Penalties", "Rent & Mortgage", "Unclassified"
]

# The four main categories for the final analysis report.
PSYCHOLOGICAL_CATEGORIES = [
    "Investment / Asset", 
    "Experience / High-Value Consumption", 
    "Habit / Impulse Loss", 
    "Fixed / Essential Cost"
]

# --- New Global Variable for Psychological Analysis ---
# Maps the detailed sub-category to its primary psychological spending nature.
SPENDING_NATURE = {
    # FIXED / ESSENTIAL 
    "Rent & Mortgage": "Fixed_Essential",
    "Communication Fees": "Fixed_Essential",
    "Public Utilities": "Fixed_Essential",
    "Public Transit": "Fixed_Essential",
    "Parking & Tolls": "Fixed_Essential",
    
    # INVESTMENT / ASSET
    "Medical & Pharmacy": "Investment_Asset",
    "Health Supplements": "Investment_Asset",
    "Education & Books": "Investment_Asset",
    "Hobby & Skill Dev.": "Investment_Asset",
    "Events & Gifts": "Investment_Asset", 
    
    # PLANNED CONSUMPTION / VARIABLE 
    "Groceries": "Consumption_Planned",
    "Household Essentials": "Consumption_Planned", 
    "Fuel & Vehicle Maint.": "Consumption_Planned", 
    
    # EXPERIENCE / DISCRETIONARY 
    "Dining Out": "Consumption_Experience",
    "Travel & Accommodation": "Consumption_Experience",
    "Movies & Shows": "Consumption_Experience",
    "Beauty & Cosmetics": "Consumption_Experience",
    "Clothing & Fashion": "Consumption_Experience",
    
    # IMPULSE / LOSS 
    "Casual Dining": "Impulse_Habitual", 
    "Coffee & Beverages": "Impulse_Habitual",
    "Alcohol & Bars": "Impulse_Habitual",
    "Games & Digital Goods": "Impulse_Habitual",
    "Taxi Convenience": "Impulse_Convenience", 
    "Fees & Penalties": "Loss_Inefficiency",
    "Unclassified": "Loss_Unclassified"
}

# The four main categories for the final analysis report.
PSYCHOLOGICAL_CATEGORIES = [
    "Investment / Asset", 
    "Experience / High-Value Consumption", 
    "Habit / Impulse Loss", 
    "Fixed / Essential Cost"
]


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


def get_category_guide():
    # 💡 Updated category guide in English
    guide = ""
    categories = {
        "FIXED / ESSENTIAL": ["Rent & Mortgage", "Communication Fees", "Public Utilities", "Public Transit", "Parking & Tolls"],
        "VARIABLE / CONSUMPTION": ["Groceries", "Household Essentials", "Beauty & Cosmetics", "Clothing & Fashion", "Fuel & Vehicle Maint.", "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars"],
        "INVESTMENT / ASSET": ["Medical & Pharmacy", "Health Supplements", "Education & Books", "Hobby & Skill Dev.", "Events & Gifts"],
        "DISCRETIONARY / LOSS": ["Travel & Accommodation", "Movies & Shows", "Games & Digital Goods", "Taxi Convenience", "Fees & Penalties", "Unclassified"],
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
    2. **Analyze & Accumulate:** Results are added to the cumulative record.
    3. **Review & Chat:** Check the integrated report, spending charts, and get personalized financial advice.
    4. **Report Generation:** Generate a comprehensive PDF report based on analysis and chat history.
    """)
    
    st.markdown("---")
    if st.session_state.all_receipts_items:
        st.info(f"Currently tracking {len(st.session_state.all_receipts_summary)} receipts.") 
        
st.title("🧾 Smart Receipt Analyzer & Tracker")
st.markdown("---")


# 📢 Fetch rates once at app startup
EXCHANGE_RATES = get_exchange_rates()


# --- 1. Gemini Analysis Function (Prompt Remains English) ---
def analyze_receipt_with_gemini(_image: Image.Image):
    """
    Calls the Gemini model to extract data and categorize items from a receipt image.
    (Function body omitted for brevity, assumed correct)
    """
    prompt_template = """
    You are an expert in receipt analysis and ledger recording.
    Analyze the following items from the receipt image and **you must extract them in JSON format**.
    
    **CRITICAL INSTRUCTION:** The response must only contain the **JSON code block wrapped in backticks (```json)**. Do not include any explanations, greetings, or additional text outside the JSON code block.
    
    1. store_name: Store Name (text)
    2. date: Date (YYYY-MM-DD format). **If not found, use YYYY-MM-DD format based on today's date.**
    3. store_location: Store location/address (text). **If not found, use "Seoul".**
    4. total_amount: Final amount settled/paid via card or cash (numbers only, no commas). **CRITICAL: You MUST extract the FINAL '합계' (Total) amount settled by the customer, which reflects tax and discount.** 5. tax_amount: Tax or VAT amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    6. tip_amount: Tip amount recognized on the receipt (numbers only, no commas). Must be 0 if not present.
    7. discount_amount: Total discount amount applied to the entire receipt (numbers only, no commas). **CRITICAL: Extract this as a POSITIVE number (e.g., if the discount is -18,000 KRW, output 18000). Must be 0 if not present.**
    8. currency_unit: Official currency code shown on the receipt (e.g., KRW, USD, EUR).
    9. items: List of purchased items. Each item must include:
        - name: Item Name (text)
        - price: Unit Price (numbers only, no commas). **This must be the final, VAT-INCLUSIVE price displayed next to the item name (before final discount allocation).** - quantity: Quantity (numbers only)
        - category: The most appropriate **Detailed Sub-Category** for this item, which must be **automatically classified** by you.
    
    **Classification Guide (Choose ONE sub-category for 'category' field):**
    - **FIXED / ESSENTIAL:** Rent & Mortgage, Communication Fees, Public Utilities, Public Transit, Fuel & Vehicle Maint., Parking & Tolls
    - **VARIABLE / CONSUMPTION (Planned):** Groceries, Household Essentials 
    - **VARIABLE / CONSUMPTION (Experience):** Dining Out, Travel & Accommodation, Movies & Shows, Beauty & Cosmetics, Clothing & Fashion 
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

# --- 2. AI Analysis Report Generation Function (English remains unchanged) ---

def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float, currency_unit: str, detailed_items_text: str):
    """
    Generates an AI analysis report based on aggregated spending data and detailed items.
    """
    # [Function body omitted for brevity, assumed correct]
    summary_text = summary_df.to_string(index=False)

    prompt_template = f"""
    You are a supportive, friendly, and highly knowledgeable Financial Psychologist and Advisor. Your role is to analyze the user's spending habits from a **psychological and behavioral economics perspective**, and provide personalized advice on overcoming impulse spending and optimizing happiness per won. Your tone should be consistently polite and helpful, like a professional mentor.

    The user's **all accumulated spending** amounts to {total_amount:,.0f} {currency_unit}.
    
    Below is the category breakdown of all accumulated spending (Unit: {currency_unit}):
    --- Spending Summary Data (Category, Amount) ---
    {summary_text}
    ---
    
    **CRITICAL DETAILED DATA:** Below are the individual item names, their original AI categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores, or refer to high-frequency, low-value items that drive the Impulse Index).
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

# 📢 [NEW] Chat Summary Function
def generate_chat_summary(chat_history: list, total_spent: float, impulse_index: float, high_impulse_cat: str) -> str:
    """
    Calls the Gemini model to summarize the main financial advice and alternatives from the chat history.
    """
    
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    prompt_template = f"""
    You are summarizing a financial consultation transcript.
    The user's spending profile: Total Spent {total_spent:,.0f} KRW, Impulse Index {impulse_index:.2f}, Highest Impulse Category '{high_impulse_cat}'.
    
    **Instructions:**
    1. Analyze the full chat history below.
    2. Extract the **main psychological insight** shared with the user (e.g., spending patterns, index interpretation).
    3. Summarize the **2-3 most critical, specific, and actionable low-cost alternatives or efficiency tips** recommended to the user.
    4. Provide the summary as a concise, professional, and objective paragraph.

    --- Chat Transcript ---
    {history_text}
    ---
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template],
        )
        return response.text
        
    except Exception as e:
        return "Failed to generate chat summary report due to an AI processing error."


# 📢 [NEW] PDF 생성 클래스 (fpdf2 기반)
class PDF(FPDF):
    def header(self):
        # 📢 [FIX] Nanum Gothic으로 폰트 설정
        self.set_font('Nanum', 'B', 15)
        self.cell(0, 10, 'Personal Spending Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Nanum', '', 8) # 📢 [FIX] Italic removed
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        # 📢 [FIX] title argument added
        self.set_font('Nanum', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Nanum', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_table(self, data: pd.DataFrame, header_titles: list):
        self.set_font('Nanum', 'B', 8)
        
        # 📢 [FIX] Column width calculation
        num_cols = len(header_titles)
        col_width = 190 / num_cols
        
        # Header
        for i, title in enumerate(header_titles):
            self.cell(col_width, 7, title, 1, 0, 'C')
        self.ln()

        # Data rows
        self.set_font('Nanum', '', 8)
        for _, row in data.iterrows():
            row_list = [str(item) for item in row.iloc[:len(header_titles)]]
            
            # Truncate content for table layout
            row_list = [item[:25] if len(item) > 25 else item for item in row_list]
            
            for i, item in enumerate(row_list):
                self.cell(col_width, 6, item, 1, 0, 'C')
            self.ln()


# 📢 [FIX] Removed cache decorator
def register_pdf_fonts(pdf_instance):
    """Registers Nanum fonts with FPDF, returns False if failed."""
    # 📢 [FIX] Removed cache decorator to prevent UnhashableParamError
    try:
         # Uses relative path from app root/fonts folder
         pdf_instance.add_font('Nanum', '', 'fonts/NanumGothic.ttf', uni=True) 
         pdf_instance.add_font('Nanum', 'B', 'fonts/NanumGothicBold.ttf', uni=True)
         return True
    except Exception as e:
         return False 


# ----------------------------------------------------------------------
# 📌 4. Streamlit UI: Tab Setup (Translated)
# ----------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Analysis & Tracking", "💬 Financial Expert Chat", "📄 PDF Report"])


# ======================================================================
#     		 	TAB 1: ANALYSIS & TRACKING (MODIFIED)
# ======================================================================
with tab1:
    
    st.subheader("📁 Data Input & AI Analysis")
    
    col_csv, col_img = st.columns(2)
    
    # 1. CSV Upload Section (Left Column)
    with col_csv:
        st.markdown("**Load Previous Record (CSV Upload)**")
        
        if 'csv_load_triggered' not in st.session_state:
            st.session_state.csv_load_triggered = False
            
        uploaded_csv_file = st.file_uploader(
            "Upload a previously downloaded ledger CSV file",
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
                    st.error("❌ Uploaded CSV file is missing required columns. Please upload a correctly formatted file.")
                else:
                    st.session_state.all_receipts_items.append(imported_df)
                    
                    summary_data = regenerate_summary_data(imported_df)
                    if summary_data:
                        st.session_state.all_receipts_summary.append(summary_data)
                        st.success(f"🎉 CSV file **{uploaded_csv_file.name}** record (**{len(imported_df)} items**) successfully loaded and accumulated.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to regenerate Summary data from CSV file.")
                
            except Exception as e:
                st.error(f"❌ Error processing CSV file: {e}")

    # 2. Image Upload Section (Right Column)
    with col_img:
        st.markdown("**Upload Receipt Image (AI Analysis)**")
        uploaded_file = st.file_uploader(
            "Upload one receipt image (jpg, png) at a time.", 
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=False,
            key='receipt_uploader' 
        )


    st.markdown("---")
    # --- 📢 [NEW] CSV/Image Upload Section End ---

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        
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
                
                st.warning(f"⚠️ This receipt ({uploaded_file.name}) is already analyzed. Prevent recording the same data multiple times")
                analyze_button = st.button("✨ Start Receipt Analysis", disabled=True)
                
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
                
                st.info(f"Cumulative Total (KRW): **{existing_summary.get('Total', 0):,.0f} KRW** (Excluding Tax)")
                st.markdown("---")

            else:
                analyze_button = st.button("✨ Start Receipt Analysis")


            if analyze_button and not is_already_analyzed:
                
                st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
                with st.spinner('AI is reading the receipt...'):
                    
                    json_data_text = analyze_receipt_with_gemini(image)

                    if json_data_text:
                        try:
                            # JSON Cleaning Logic
                            cleaned_text = json_data_text.strip()
                            if cleaned_text.startswith("```json"):
                                cleaned_text = cleaned_text.lstrip("```json")
                            if cleaned_text.endswith("```"):
                                cleaned_text = cleaned_text.rstrip("```")
                            
                            receipt_data = json.loads(cleaned_text.strip()) 
                            
                            # Data Validation and Defaults
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
                                st.warning("⚠️ AI date recognition failed, defaulting to today's date.")
                                
                            final_location = store_location_str if store_location_str else "Seoul"

                            
                            # --- Amount Validation and Override ---
                            if 'items' in receipt_data and receipt_data['items']:
                                items_df = pd.DataFrame(receipt_data['items'])
                                
                                items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                                items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                                items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                                
                                calculated_original_total = (items_df['Unit Price'] * items_df['Quantity']).sum()
                                total_discount = safe_get_amount(receipt_data, 'discount_amount') 
                                
                                calculated_final_total = calculated_original_total - total_discount
                                
                                if abs(calculated_final_total - total_amount) > 100 and calculated_final_total > 0:
                                    st.warning(
                                        f"⚠️ AI total ({total_amount:,.0f} {display_unit}) differs significantly from item sum ({calculated_final_total:,.0f} {display_unit}). "
                                        f"**Overriding total with item sum.**"
                                    )
                                    total_amount = calculated_final_total
                                
                                
                                # --- Main Information Display ---
                                st.success("✅ Analysis Complete! Check the ledger data below.")
                                
                                st.markdown(f"**🏠 Store Name:** {receipt_data.get('store_name', 'N/A')}")
                                st.markdown(f"**📍 Location:** {final_location}") 
                                st.markdown(f"**📅 Date:** {final_date}") 
                                st.subheader(f"💰 Total Amount Paid (Corrected): {total_amount:,.0f} {display_unit}")

                                if discount_amount > 0:
                                    discount_display = f"{discount_amount:,.2f} {display_unit}"
                                    st.markdown(f"**🎁 Total Discount:** {discount_display}") 

                                if tax_amount > 0 or tip_amount > 0:
                                    tax_display = f"{tax_amount:,.2f} {display_unit}"
                                    tip_display = f"{tip_amount:,.2f} {display_unit}"
                                    st.markdown(f"**🧾 Tax/VAT:** {tax_display} | **💸 Tip:** {tip_display}")
                                
                                if display_unit != 'KRW':
                                    applied_rate = EXCHANGE_RATES.get(display_unit, 1.0)
                                    st.info(f"**📢 Applied Exchange Rate:** 1 {display_unit} = {applied_rate:,.4f} KRW (Rate fetched from API/Fallback)")
                                    
                                st.markdown("---")

                                # 📢 Discount Allocation Logic
                                items_df['Total Spend Original'] = items_df['Unit Price'] * items_df['Quantity']
                                items_df['Discount Applied'] = 0.0
                                items_df['Total Spend'] = items_df['Total Spend Original']
                                
                                total_item_original = items_df['Total Spend Original'].sum()
                                
                                if total_discount > 0 and total_item_original > 0:
                                    discount_rate = total_discount / total_item_original
                                    items_df['Discount Applied'] = items_df['Total Spend Original'] * discount_rate
                                    items_df['Total Spend'] = items_df['Total Spend Original'] - items_df['Discount Applied']
                                    st.info(f"💡 Discount of {total_discount:,.0f} {display_unit} successfully allocated across items.")
                                else:
                                    pass
                                    
                                
                                st.subheader("🛒 Detailed Item Breakdown (Category Editable)")
                                
                                edited_df = st.data_editor(
                                    items_df.drop(columns=['Total Spend Original', 'Discount Applied', 'Total Spend']), 
                                    column_config={
                                        "AI Category": st.column_config.SelectboxColumn(
                                            "Final Category",
                                            help="Select the correct sub-category for this item.",
                                            width="medium",
                                            options=ALL_CATEGORIES,
                                            required=True,
                                        ),
                                    },
                                    disabled=['Item Name', 'Unit Price', 'Quantity'], 
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                edited_df['Total Spend'] = items_df['Total Spend']
                                edited_df['Total Spend Numeric'] = pd.to_numeric(edited_df['Total Spend'], errors='coerce').fillna(0)
                                edited_df['Currency'] = display_unit
                                
                                # 📢 Currency Conversion for Accumulation (AI Analysis)
                                edited_df['Currency'] = display_unit
                                edited_df['Total Spend Numeric'] = pd.to_numeric(edited_df['Total Spend'], errors='coerce').fillna(0)
                                edited_df['KRW Total Spend'] = edited_df.apply(
                                    lambda row: convert_to_krw(row['Total Spend Numeric'], row['Currency'], EXCHANGE_RATES), axis=1
                                )
                                edited_df = edited_df.drop(columns=['Total Spend Numeric'])

                                krw_tax_total = convert_to_krw(tax_amount, display_unit, EXCHANGE_RATES) 
                                krw_tip_total = convert_to_krw(tip_amount, display_unit, EXCHANGE_RATES)
                                
                                lat, lon = geocode_address(final_location)
                                
                                # ** Accumulate Data: Store the edited DataFrame **
                                st.session_state.all_receipts_items.append(edited_df)
                                
                                final_total_krw = edited_df['KRW Total Spend'].sum() + krw_tip_total
                                
                                st.session_state.all_receipts_summary.append({
                                    'id': file_id, 
                                    'filename': uploaded_file.name,
                                    'Store': receipt_data.get('store_name', 'N/A'),
                                    'Total': final_total_krw, 
                                    'Tax_KRW': krw_tax_total, 
                                    'Tip_KRW': krw_tip_total, 
                                    'Currency': 'KRW', 
                                    'Date': final_date, 
                                    'Location': final_location, 
                                    'Original_Total': total_amount, 
                                    'Original_Currency': display_unit,
                                    'latitude': lat,
                                    'longitude': lon
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
            manual_category = st.selectbox("📌 Category (Sub-Category)", 
                                 options=ALL_CATEGORIES, 
                                 index=ALL_CATEGORIES.index('Unclassified'))
            manual_currency = st.selectbox("Currency Unit", options=['KRW', 'USD', 'EUR', 'JPY'], index=0)
            manual_location = st.text_input("📍 Location/City", placeholder="e.g., Gangnam, Seoul") 
            
        submitted = st.form_submit_button("✅ Add to Ledger")

        if submitted:
            if manual_description and manual_amount > 0 and manual_category:
                
                krw_total = convert_to_krw(manual_amount, manual_currency, EXCHANGE_RATES)
                applied_rate = EXCHANGE_RATES.get(manual_currency, 1.0)

                final_location = manual_location if manual_location else "Manual Input Location"
                lat, lon = geocode_address(final_location)
                
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
                    'Tax_KRW': 0.0, 
                    'Tip_KRW': 0.0, 
                    'Currency': 'KRW', 
                    'Date': manual_date.strftime('%Y-%m-%d'),
                    'Location': final_location, 
                    'Original_Total': manual_amount, 
                    'Original_Currency': manual_currency,
                    'latitude': lat,
                    'longitude': lon
                }
                
                # 3. Accumulate Data
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
    
    # ----------------------------------------------------------------------
    # --- 5. Cumulative Data Analysis Section (ALL ANALYSIS IS KRW BASED) ---
    # ----------------------------------------------------------------------

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


        # A. Display Accumulated Receipts Summary Table (Translated/Modified)
        st.subheader(f"Total {len(st.session_state.all_receipts_summary)} Receipts Logged (Summary)")
        summary_df = pd.DataFrame(st.session_state.all_receipts_summary)
        
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
        if 'latitude' not in summary_df.columns:
            summary_df['latitude'] = 37.5665
        if 'longitude' not in summary_df.columns:
            summary_df['longitude'] = 126.9780
            
        def format_amount_paid(row):
            krw_amount = f"{row['Total']:,.0f} KRW"
            
            if row['Original_Currency'] != 'KRW':
                original_amount = f"{row['Original_Total']:,.2f} {row['Original_Currency']}"
                return f"{original_amount} / {krw_amount}"
            
            return krw_amount
        
        summary_df['Amount Paid'] = summary_df.apply(format_amount_paid, axis=1)

        
        summary_df = summary_df.drop(columns=['id'])
        summary_df_display = summary_df[['Date', 'Store', 'Location', 'Amount Paid', 'Tax_KRW', 'Tip_KRW', 'filename']] 
        summary_df_display.columns = ['Date', 'Store', 'Location', 'Amount Paid', 'Tax (KRW)', 'Tip (KRW)', 'Source'] 

        st.dataframe(
            summary_df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Tax (KRW)": st.column_config.NumberColumn(
                    "Tax (KRW)", 
                    format="%.0f KRW" 
                ),
                "Tip (KRW)": st.column_config.NumberColumn(
                    "Tip (KRW)", 
                    format="%.0f KRW" 
                ),
            }
        )
        
        st.markdown("---")
        
        # 📢 [MODIFIED] Spending Trend and Map Visualization in Parallel
        col_trend, col_map = st.columns(2)
        
        with col_trend:
            # --- Spending Trend Over Time Chart (KRW based) ---
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
        
        with col_map:
            # --- Spending Map Visualization Section ---
            st.subheader("📍 Spending Map Visualization")
            
            map_df = summary_df.copy()
            map_df.columns = [col.replace('latitude', 'lat').replace('longitude', 'lon') for col in map_df.columns]
    
            if not map_df.empty and 'lat' in map_df.columns and 'lon' in map_df.columns:
                
                map_data = map_df[map_df['Total'] > 0].dropna(subset=['lat', 'lon'])
                
                if not map_data.empty:
                    st.map(
                        map_data, 
                        latitude='lat', 
                        longitude='lon', 
                        color='#ff6347', 
                        zoom=11, 
                        use_container_width=True
                    )
                
                else:
                    st.warning("No valid coordinate data found to display the map.")
            else:
                st.warning("Location data or coordinate columns are not available.")


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
        
        total_tax_krw = summary_df['Tax_KRW'].sum()
        total_tip_krw = summary_df['Tip_KRW'].sum()
        
        if total_tax_krw > 0:
            category_summary.loc[len(category_summary)] = ['Tax/VAT', total_tax_krw]
        if total_tip_krw > 0:
            category_summary.loc[len(category_summary)] = ['Tip', total_tip_krw]
            
        # --- Display Summary Table ---
        st.subheader("💰 Spending Summary by Category (Items + Tax + Tip)") 
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
# 		 	TAB 2: FINANCIAL EXPERT CHAT (MODIFIED)
# ======================================================================
with tab2:
    st.header("💬 Financial Expert Chat")
    
    if not st.session_state.all_receipts_items:
        st.warning("Please analyze at least one receipt or load a CSV in the 'Analysis & Tracking' tab before starting a consultation.")
    else:
        # --- Chat Data Preparation (Calculation logic remains English) ---
        current_data_hash = hash(tuple(item['id'] for item in st.session_state.all_receipts_summary))
        
        if 'last_data_hash' not in st.session_state or st.session_state.last_data_hash != current_data_hash:
            st.session_state.chat_history = []
            st.session_state.last_data_hash = current_data_hash
            st.info("📊 New spending data detected. Chat history is being reset for fresh analysis.")
        
        all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        if 'KRW Total Spend' not in all_items_df.columns:
             all_items_df['KRW Total Spend'] = all_items_df.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )

        all_items_df['Psychological Category'] = all_items_df['AI Category'].apply(get_psychological_category)
        psychological_summary = all_items_df.groupby('Psychological Category')['KRW Total Spend'].sum().reset_index()
        psychological_summary.columns = ['Category', 'KRW Total Spend']

        summary_df_for_chat = pd.DataFrame(st.session_state.all_receipts_summary)
        tax_tip_only_total = 0.0
        if 'Tip_KRW' in summary_df_for_chat.columns:
            tax_tip_only_total += summary_df_for_chat['Tip_KRW'].sum()
        
        if tax_tip_only_total > 0:
            fixed_cost_index = psychological_summary[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[3]].index
            if not fixed_cost_index.empty:
                psychological_summary.loc[fixed_cost_index[0], 'KRW Total Spend'] += tax_tip_only_total 
            else:
                new_row = pd.DataFrame([{'Category': PSYCHOLOGICAL_CATEGORIES[3], 'KRW Total Spend': tax_tip_only_total}])
                psychological_summary = pd.concat([psychological_summary, new_row], ignore_index=True)

        total_spent = psychological_summary['KRW Total Spend'].sum()
        
        impulse_spending = psychological_summary.loc[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[2], 'KRW Total Spend'].sum()
        
        total_transactions = len(all_items_df)
        impulse_transactions = len(all_items_df[all_items_df['Psychological Category'] == PSYCHOLOGICAL_CATEGORIES[2]])
        
        if total_spent > 0 and total_transactions > 0:
            amount_ratio = impulse_spending / total_spent
            frequency_ratio_factor = np.sqrt(impulse_transactions / total_transactions)
            impulse_index = amount_ratio * frequency_ratio_factor
        else:
            impulse_index = 0.0
        
        psychological_summary_text = psychological_summary.to_string(index=False)
        
        highest_impulse_category = ""
        highest_impulse_amount = 0
        
        impulse_items_df = all_items_df[all_items_df['Psychological Category'] == PSYCHOLOGICAL_CATEGORIES[2]]
        if not impulse_items_df.empty:
            impulse_category_sum = impulse_items_df.groupby('AI Category')['KRW Total Spend'].sum()
            if not impulse_category_sum.empty:
                highest_impulse_category = impulse_category_sum.idxmax()
                highest_impulse_amount = impulse_category_sum.max()
        
        detailed_items_for_chat = all_items_df[['Psychological Category', 'Item Name', 'KRW Total Spend']]
        items_text_for_chat = detailed_items_for_chat.to_string(index=False)
        
        # MODIFIED SYSTEM INSTRUCTION (CRITICAL)
        system_instruction = f"""
        You are a supportive, friendly, and highly knowledgeable Financial Psychologist and Advisor. Your role is to analyze the user's spending habits from a **psychological and behavioral economics perspective**, and provide personalized advice on overcoming impulse spending and optimizing happiness per won. Your tone should be consistently polite and helpful, like a professional mentor.
        
        The user's cumulative spending data for the current session (All converted to KRW) is analyzed by its **Psychological Spending Nature**:
        - **Total Accumulated Spending**: {total_spent:,.0f} KRW
        - **Calculated Impulse Spending Index (Refined)**: {impulse_index:.2f} (Target: < 0.15 for Refined Index)
        - **Psychological Category Breakdown (Category, Amount)**:
        {psychological_summary_text}
        
        **CRITICAL DETAILED DATA:** Below are the individual item names, their original AI categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores, or refer to high-frequency, low-value items that drive the Impulse Index).
        --- Detailed Items Data (Psychological Category, Item Name, KRW Total Spend) ---
        {items_text_for_chat}
        ---

        --- Alternative Recommendation Task (NEW - Utility Optimization) ---
        The user's highest impulse/loss spending is in the **'{highest_impulse_category}'** category, amounting to **{highest_impulse_amount:,.0f} KRW**.
        
        When the user asks for alternatives or efficiency advice, you MUST prioritize and perform the following:
        1. Identify the core utility (e.g., comfort, energy, pleasure, time-saving, social belonging) the user gains from spending on **'{highest_impulse_category}'** or a specific high-frequency impulse item.
        2. Propose 2-3 specific, actionable, and low-cost alternatives that satisfy the *same core utility* while aiming to **reduce the expense by at least 30%**.
        3. Examples of alternatives: *Home-brewed coffee for routine, pre-planning walking route instead of taxi, frozen meal kit instead of dining out.*

        Base all your advice and responses on this data. Your analysis MUST start with a professional interpretation of the **Impulse Spending Index (Refined)**. Provide actionable, psychological tips to convert 'Impulse Loss' spending into 'Investment/Asset' spending. Always include the currency unit (KRW) when referring to monetary amounts.
        """

        # 💡 Initial Message (Translated)
        if not st.session_state.chat_history or (len(st.session_state.chat_history) == 1 and st.session_state.chat_history[0]["content"].startswith("Hello! I am your AI Financial Psychology Expert")):
              st.session_state.chat_history = []
              
              if highest_impulse_category:
                  impulse_info = f"Your highest impulse spending is in the **{highest_impulse_category}** category, totaling **{highest_impulse_amount:,.0f} KRW**."
              else:
                  impulse_info = "Impulse spending items have not been clearly analyzed yet."

              initial_message = f"""
              Hello! I am your AI Financial Psychology Expert, here to analyze your spending patterns. 🧠
              Your total spending accumulated so far is **{total_spent:,.0f} KRW**.
              Your **Calculated Impulse Index** stands at **{impulse_index:.2f}** (Target: below 0.15).
              {impulse_info}

              What specific psychological advice would you like? For example, you can ask:

              * **"What does my Impulse Index of {impulse_index:.2f} signify?"**
              * **"Could you recommend alternatives to reduce the cost of my biggest impulse item ({highest_impulse_category}, etc.)?"**
              * "How can I convert my spending into **'Investment / Asset'**?"
              """
              st.session_state.chat_history.append({"role": "assistant", "content": initial_message})

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
                        combined_contents = []
                        history_items = st.session_state.chat_history 
                        
                        for item in history_items:
                            gemini_role = "user" if item["role"] == "user" else "model" 
                            
                            combined_contents.append({
                                "role": gemini_role, 
                                "parts": [{"text": item["content"]}]
                            })
                        
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=combined_contents, 
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Chatbot API call failed: {e}")

# ======================================================================
# 		 	TAB 3: PDF REPORT GENERATOR (MODIFIED)
# ======================================================================
with tab3:
    st.header("📄 Comprehensive Spending Report (PDF)")

    st.warning("🚨 **Nanum Gothic Font Required:** PDF generation requires the **Nanum Gothic** font files (`NanumGothic.ttf`, `NanumGothicBold.ttf`) to be placed in the **`fonts/` folder** of your project.")

    if not st.session_state.all_receipts_items:
        st.warning("Spending data is required to generate the report. Please analyze data in the 'Analysis & Tracking' tab.")
    else:
        
        # 1. Data Preparation
        summary_list = st.session_state.all_receipts_summary
        items_list = st.session_state.all_receipts_items
        
        items_with_meta = []
        for item_df, summary in zip(items_list, summary_list):
            item_df_copy = item_df.copy()
            
            if 'Date' not in item_df_copy.columns:
                item_df_copy['Date'] = summary.get('Date', 'N/A')
            if 'Store' not in item_df_copy.columns:
                item_df_copy['Store'] = summary.get('Store', 'N/A')
                
            items_with_meta.append(item_df_copy)
            
        all_items_df = pd.concat(items_with_meta, ignore_index=True)
        
        all_items_df['Psychological Category'] = all_items_df['AI Category'].apply(get_psychological_category)
        
        psychological_summary_pdf = all_items_df.groupby('Psychological Category')['KRW Total Spend'].sum().reset_index()
        psychological_summary_pdf.columns = ['Category', 'Amount (KRW)']
        total_spent = psychological_summary_pdf['Amount (KRW)'].sum()
        
        impulse_spending = psychological_summary_pdf.loc[psychological_summary_pdf['Category'] == PSYCHOLOGICAL_CATEGORIES[2], 'Amount (KRW)'].sum()
        total_transactions = len(all_items_df)
        impulse_transactions = len(all_items_df[all_items_df['Psychological Category'] == PSYCHOLOGICAL_CATEGORIES[2]])
        
        if total_spent > 0 and total_transactions > 0:
            amount_ratio = impulse_spending / total_spent
            frequency_ratio_factor = np.sqrt(impulse_transactions / total_transactions)
            impulse_index = amount_ratio * frequency_ratio_factor
        else:
            impulse_index = 0.0

        highest_impulse_category = "N/A"
        impulse_items_df = all_items_df[all_items_df['Psychological Category'] == PSYCHOLOGICAL_CATEGORIES[2]]
        if not impulse_items_df.empty:
            highest_impulse_category_calc = impulse_items_df.groupby('AI Category')['KRW Total Spend'].sum()
            if not highest_impulse_category_calc.empty:
                highest_impulse_category = highest_impulse_category_calc.idxmax()
        
        
        # 2. PDF Creation Function
        def create_pdf_report(psycho_summary, total_spent, impulse_index, high_impulse_cat, chat_history_list):
            pdf = PDF(orientation='P', unit='mm', format='A4')
            
            # 📢 [FONT LOAD FIX] Load fonts and check for failure
            font_loaded = register_pdf_fonts(pdf)
            
            if not font_loaded:
                 st.error(f"❌ PDF Font Load Failed: NanumGothic font files missing in 'fonts/' folder.")
                 return None 
            
            pdf.set_font('Nanum', '', 10) 
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Section 1: Executive Summary
            pdf.chapter_title("1. Executive Summary & Key Metrics")
            
            summary_body = (
                f"Total Accumulated Spending: {total_spent:,.0f} KRW\n"
                f"Refined Impulse Spending Index: {impulse_index:.2f} (Target: below 0.15)\n"
                f"Highest Impulse Category: {high_impulse_cat}\n\n"
                f"This report analyzes your spending patterns from a psychological perspective and provides tailored advice for effective financial goal achievement."
            )
            pdf.chapter_body(summary_body)

            # Section 2: Consumption Profile
            pdf.chapter_title("2. Psychological Consumption Profile")
            pdf.chapter_body("Summary of psychological spending breakdown (Investment/Experience/Habit/Fixed Cost):")
            
            # Simpler table for PDF
            psycho_summary_display = psycho_summary.copy()
            psycho_summary_display['Amount (KRW)'] = psycho_summary_display['Amount (KRW)'].apply(lambda x: f"{x:,.0f}")
            
            pdf.add_table(psycho_summary_display, ['Category', 'Amount (KRW)'])

            # Section 3: Chat Consultation History
            pdf.chapter_title("3. Financial Expert Consultation Summary")
            
            # 📢 [NEW] Generate concise summary using AI
            if chat_history_list:
                summary_text = generate_chat_summary(chat_history_list, total_spent, impulse_index, high_impulse_cat)
                pdf.chapter_body(summary_text)
            else:
                pdf.chapter_body("No consultation history found. Start a conversation in the 'Financial Expert Chat' tab.")
            
            # Section 4: Detailed Transaction Data (ALL ITEMS)
            pdf.chapter_title("4. Detailed Transaction History")
            pdf.chapter_body(f"Total {len(all_items_df)} detailed transaction records:")
            
            detailed_data = all_items_df[['Date', 'Store', 'Item Name', 'AI Category', 'KRW Total Spend']].copy() 
            detailed_data['KRW Total Spend'] = detailed_data['KRW Total Spend'].apply(lambda x: f"{x:,.0f}")
            
            pdf.add_table(detailed_data, ['Date', 'Store', 'Item Name', 'Category', 'Amount (KRW)'])
            
            # 📢 [CRITICAL FIX] Convert output to bytes()
            pdf_result = bytes(pdf.output(dest='S')) 
            return pdf_result


        # 3. Streamlit Download Button
        pdf_output = create_pdf_report(
            psychological_summary_pdf, 
            total_spent, 
            impulse_index, 
            highest_impulse_category, 
            st.session_state.chat_history
        )
        
        if pdf_output:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_output,
                file_name=f"Financial_Report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
                mime='application/pdf',
            )
