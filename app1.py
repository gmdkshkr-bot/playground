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

# ----------------------------------------------------------------------
# 📌 0. Currency Conversion Setup & Globals
# ----------------------------------------------------------------------

try:
    # 🚨 주의: 이 키들은 Streamlit Secrets에 설정되어 있어야 합니다.
    API_KEY = st.secrets["GEMINI_API_KEY"]
    EXCHANGE_API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"] 
    # 📢 [NEW] 카카오 API 키 로드
    KAKAO_REST_API_KEY = st.secrets["KAKAO_REST_API_KEY"]
except KeyError:
    st.error("❌ Please set 'GEMINI_API_KEY', 'EXCHANGE_RATE_API_KEY', and 'KAKAO_REST_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)

# --- 📢 [UPDATED] Geocoding Helper Function (Kakao API 적용) ---
@st.cache_data(ttl=datetime.timedelta(hours=48))
def geocode_address(address: str) -> tuple[float, float]:
    """
    카카오 로컬 API를 사용하여 주소를 위도와 경도로 변환합니다. (Kakao Maps API)
    """
    if not address or address == "Manual Input Location" or address == "Imported Location":
        # 유효하지 않은 주소는 서울 중심의 기본 좌표를 반환
        return 37.5665, 126.9780
    
    # 📢 Kakao Local API 호출 설정
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and data.get('documents'):
            # 첫 번째 검색 결과 사용
            document = data['documents'][0]
            # Kakao API는 경도(x)를 먼저, 위도(y)를 나중에 반환합니다.
            lat = float(document.get('y', 0))
            lon = float(document.get('x', 0))
            
            # 유효성 검사
            if lat != 0 and lon != 0:
                return lat, lon

    except requests.exceptions.RequestException as e:
        # API 요청 오류 (네트워크, 4xx, 5xx 오류)
        st.sidebar.error(f"❌ Kakao Geocoding API Error for '{address}'. Using fallback: {e}")
    except Exception as e:
        # JSON 파싱 등 기타 오류
        st.sidebar.warning(f"⚠️ Geocoding Processing Error for '{address}'. Using fallback: {e}")

    # 모든 실패 시나리오에서 서울 기본 좌표 반환
    return 37.5665, 126.9780


# 💡 헬퍼 함수: 단일 값을 안전하게 추출하고, 숫자가 아니거나 누락된 경우 0.0을 반환합니다.
def safe_get_amount(data, key):
    """단일 값을 안전하게 추출하고, 숫자가 아니거나 누락된 경우 0.0을 반환합니다."""
    value = data.get(key, 0)
    # pd.to_numeric을 사용하여 숫자로 변환 시도. 변환 실패 시 NaN 반환.
    numeric_value = pd.to_numeric(value, errors='coerce')
    # NaN이면 0.0을 사용하고, 아니면 해당 숫자 값을 사용
    return numeric_value if not pd.isna(numeric_value) else 0.0

# 💡 헬퍼 함수: 업로드된 아이템 데이터프레임에서 Summary 데이터를 재구성하는 헬퍼 함수
def regenerate_summary_data(item_df: pd.DataFrame) -> dict:
    """아이템 DataFrame에서 Summary 단위를 추출하고 재구성합니다. (CSV Import 전용)"""
    
    # 🚨 필수 컬럼 존재 여부 확인 (내보낸 CSV 파일 기준)
    required_cols = ['Item Name', 'AI Category', 'KRW Total Spend']
    if not all(col in item_df.columns for col in required_cols):
        return None

    # KRW Total Spend 합계 = Total (KRW)
    final_total_krw = item_df['KRW Total Spend'].sum()
    
    # CSV Import 기록은 메타데이터가 없으므로 임의의 값 또는 기본값을 사용
    current_date = datetime.date.today().strftime('%Y-%m-%d')
    
    # 📢 [NEW] CSV Import 시 임시 좌표 사용
    lat, lon = geocode_address("Imported Location")
    
    summary_data = {
        'id': f"imported-{pd.Timestamp.now().timestamp()}",
        'filename': 'Imported CSV',
        'Store': 'Imported Record',
        'Total': final_total_krw, 
        # 💡 U+00A0 제거 후 일반 공백 사용: CSV 상세 기록에는 Tax/Tip 정보가 없으므로 0으로 가정
        'Tax_KRW': 0.0, 
        'Tip_KRW': 0.0,
        'Currency': 'KRW', 
        'Date': current_date, 
        'Location': 'Imported Location', 
        'Original_Total': final_total_krw, 
        'Original_Currency': 'KRW',
        # 📢 [NEW] 좌표 추가
        'latitude': lat,
        'longitude': lon
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


@st.cache_data(ttl=datetime.timedelta(hours=24))
def get_exchange_rates():
    """
    Fetches real-time exchange rates using ExchangeRate-API (USD Base).
    Returns a dictionary: {currency_code: 1 Foreign Unit = X KRW}
    """
    
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/USD"
    # Fallback Rates는 1 단위 외화당 KRW 값입니다. (보다 현실적인 환율로 조정)
    FALLBACK_RATES = {'KRW': 1.0, 'USD': 1350.00, 'EUR': 1450.00, 'JPY': 9.20} 
    exchange_rates = {'KRW': 1.0} 

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        conversion_rates = data.get('conversion_rates', {})
        
        # 1. KRW Rate (USD -> KRW) 추출
        krw_per_usd = conversion_rates.get('KRW', 0)
        usd_per_usd = conversion_rates.get('USD', 1.0) 

        # 데이터 유효성 검사 강화
        if krw_per_usd == 0 or data.get('result') != 'success':
             raise ValueError("API returned incomplete or failed data or KRW rate is missing.")

        # 2. Store USD rate: 1 USD = krw_per_usd KRW
        exchange_rates['USD'] = krw_per_usd / usd_per_usd 
        
        # 3. Calculate EUR rate: 1 EUR = (KRW/USD) / (EUR/USD)
        eur_rate_vs_usd = conversion_rates.get('EUR', 0)
        if eur_rate_vs_usd > 0:
            exchange_rates['EUR'] = krw_per_usd / eur_rate_vs_usd
        
        # 4. Calculate JPY rate: 1 JPY = (KRW/USD) / (JPY/USD)
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
    
    # 0으로 나누는 오류 방지
    if rate == 0:
        return amount * rates.get('USD', 1300) 
        
    return amount * rate

# Global Categories (Internal classification names remain Korean for consistency with AI analysis prompt)
# Global Categories (Updated for professional, detailed analysis)
ALL_CATEGORIES = [
    "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars", 
    "Groceries", "Household Goods", "Medical & Pharmacy", "Health Supplements",
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
    # FIXED / ESSENTIAL (고정/필수)
    "Rent & Mortgage": "Fixed_Essential",
    "Communication Fees": "Fixed_Essential",
    "Public Utilities": "Fixed_Essential",
    "Public Transit": "Fixed_Essential",
    "Parking & Tolls": "Fixed_Essential",
    
    # INVESTMENT / ASSET (미래 투자)
    "Medical & Pharmacy": "Investment_Asset",
    "Health Supplements": "Investment_Asset",
    "Education & Books": "Investment_Asset",
    "Hobby & Skill Dev.": "Investment_Asset",
    "Events & Gifts": "Investment_Asset", # Social Capital
    
    # PLANNED CONSUMPTION / VARIABLE (계획적 소비/변동비)
    "Groceries": "Consumption_Planned",
    "Household Goods": "Consumption_Planned",
    "Fuel & Vehicle Maint.": "Consumption_Planned", # Essential Variable
    
    # EXPERIENCE / DISCRETIONARY (경험적/선택적)
    "Dining Out": "Consumption_Experience",
    "Travel & Accommodation": "Consumption_Experience",
    "Movies & Shows": "Consumption_Experience",
    
    # IMPULSE / LOSS (충동/손실)
    "Casual Dining": "Impulse_Habitual", # 잦은 습관성 소액 지출
    "Coffee & Beverages": "Impulse_Habitual",
    "Alcohol & Bars": "Impulse_Habitual",
    "Games & Digital Goods": "Impulse_Habitual",
    "Taxi Convenience": "Impulse_Convenience", # 비효율적 편의 지출
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
    # 💡 이 함수도 새로운 카테고리에 맞춰 영어로 업데이트합니다.
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
    4. **Export & Continue:** Export the current record in CSV, load the CSV to continue recording.
    """)
    
    st.markdown("---")
    if st.session_state.all_receipts_items:
        st.info(f"Currently tracking {len(st.session_state.all_receipts_summary)} receipts.") # Summary 기준으로 갯수 표시
        
st.title("🧾 Receipt Recorder powered by AI")
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

# --- 2. AI Analysis Report Generation Function ---
# generate_ai_analysis 함수는 tab1에서 사용되지 않으므로, 코드를 그대로 유지하여 tab2에서 사용할 수 있게 합니다.
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float, currency_unit: str, detailed_items_text: str):
    """
    Generates an AI analysis report based on aggregated spending data and detailed items.
    """
    # 🌟 추가/수정: summary_df를 문자열로 변환하여 summary_text 변수 정의
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
# 📌 4. Streamlit UI: Tab Setup (Translated)
# ----------------------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Analysis & Tracking", "💬 Financial Expert Chat"])


# ======================================================================
#     		 	TAB 1: ANALYSIS & TRACKING
# ======================================================================
with tab1:
    
    # --- 📢 [NEW] CSV/Image Upload Section (Parallel Columns) ---
    st.subheader("📁 Data Input & AI Analysis")
    
    col_csv, col_img = st.columns(2)
    
    # 1. CSV Upload Section (Left Column)
    with col_csv:
        st.markdown("**Load Previous Record (CSV Upload)**")
        
        # 파일을 불러온 후, 처리 상태를 저장할 임시 키
        if 'csv_load_triggered' not in st.session_state:
            st.session_state.csv_load_triggered = False
            
        uploaded_csv_file = st.file_uploader(
            "Upload a previously downloaded ledger CSV file",
            type=['csv'],
            accept_multiple_files=False,
            key='csv_uploader', 
            on_change=lambda: st.session_state.__setitem__('csv_load_triggered', True)
        )

        # 💡 로직 분리: 파일이 업로드되었고, 아직 처리되지 않았다면 처리 시작
        if st.session_state.csv_load_triggered and uploaded_csv_file is not None:
            
            st.session_state.csv_load_triggered = False # 재실행 방지를 위해 즉시 초기화
            
            try:
                # CSV 파일을 DataFrame으로 읽기
                imported_df = pd.read_csv(uploaded_csv_file)
                
                # 필수 컬럼 검증
                required_cols = ['Item Name', 'Unit Price', 'Quantity', 'AI Category', 'Total Spend', 'Currency', 'KRW Total Spend']
                
                if not all(col in imported_df.columns for col in required_cols):
                    st.error("❌ 업로드된 CSV 파일에 필수 컬럼이 부족합니다. 올바른 형식의 파일을 업로드해주세요.")
                else:
                    # 1. 아이템 목록에 추가
                    st.session_state.all_receipts_items.append(imported_df)
                    
                    # 2. Summary 데이터 재구성 및 추가
                    summary_data = regenerate_summary_data(imported_df)
                    if summary_data:
                        st.session_state.all_receipts_summary.append(summary_data)
                        st.success(f"🎉 CSV 파일 **{uploaded_csv_file.name}**의 기록 (**{len(imported_df)}개 아이템**)이 성공적으로 불러와져 누적되었습니다.")
                        st.rerun()
                    else:
                        st.error("❌ CSV 파일에서 Summary 데이터를 재구성하는 데 실패했습니다.")
                
            except Exception as e:
                st.error(f"❌ CSV 파일을 처리하는 중 오류가 발생했습니다: {e}")

    # 2. Image Upload Section (Right Column)
    with col_img:
        st.markdown("**Upload Receipt Image (AI Analysis)**")
        uploaded_file = st.file_uploader(
            "Upload one receipt image (jpg, png) at a time.", 
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=False,
            key='receipt_uploader' # CSV Uploader와 키 충돌 방지
        )


    st.markdown("---")
    # --- 📢 [NEW] CSV/Image Upload Section End ---

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        
        # 💡 중복 파일 체크
        existing_summary = next((s for s in st.session_state.all_receipts_summary if s.get('id') == file_id), None)
        is_already_analyzed = existing_summary is not None
        
        # UI 레이아웃 변경 (이미지 표시 및 분석 결과)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Uploaded Receipt")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True) 

        with col2:
            st.subheader("📊 Analysis and Recording")
            
            if is_already_analyzed:
                
                # 💡 중복된 경우, 경고 메시지 표시 및 저장된 결과 표시
                st.warning(f"⚠️ This receipt ({uploaded_file.name}) is already analyzed. Prevent recording the same data multiple times")
                analyze_button = st.button("✨ Start Receipt Analysis", disabled=True)
                
                # 💡 저장된 Summary 데이터로 분석 결과를 바로 표시
                display_unit = existing_summary['Original_Currency']
                applied_rate = EXCHANGE_RATES.get(display_unit, 1.0)
                
                st.markdown(f"**🏠 Store Name:** {existing_summary.get('Store', 'N/A')}")
                st.markdown(f"**📍 Location:** {existing_summary.get('Location', 'N/A')}")
                st.markdown(f"**📅 Date:** {existing_summary.get('Date', 'N/A')}")
                st.subheader(f"💰 Total Amount Paid: {existing_summary.get('Original_Total', 0):,.0f} {display_unit}")
                
                krw_tax = existing_summary.get('Tax_KRW', 0)
                krw_tip = existing_summary.get('Tip_KRW', 0)
                
                if krw_tax > 0 or krw_tip > 0:
                    # 원화 기준 금액을 다시 원화로 표시
                    tax_display = f"{krw_tax:,.0f} KRW"
                    tip_display = f"{krw_tip:,.0f} KRW"
                    st.markdown(f"**🧾 Tax/VAT (KRW):** {tax_display} | **💸 Tip (KRW):** {tip_display}")
                
                st.info(f"누적 기록 총액 (KRW): **{existing_summary.get('Total', 0):,.0f} KRW** (부가세 제외)")
                st.markdown("---")

                # 중복이므로 추가적인 분석 로직은 실행하지 않음
                pass 
                
            else:
                # 중복이 아닌 경우, 분석 버튼 활성화
                analyze_button = st.button("✨ Start Receipt Analysis")


            if analyze_button and not is_already_analyzed:
                
                st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
                with st.spinner('AI is reading the receipt...'):
                    
                    json_data_text = analyze_receipt_with_gemini(image)

                    if json_data_text:
                        try:
                            # 💡 JSON 클리닝 로직 강화
                            cleaned_text = json_data_text.strip()
                            if cleaned_text.startswith("```json"):
                                cleaned_text = cleaned_text.lstrip("```json")
                            if cleaned_text.endswith("```"):
                                cleaned_text = cleaned_text.rstrip("```")
                            
                            receipt_data = json.loads(cleaned_text.strip()) 
                            
                            # 데이터 유효성 검사 및 기본값 설정 (safe_get_amount 사용)
                            total_amount = safe_get_amount(receipt_data, 'total_amount')
                            tax_amount = safe_get_amount(receipt_data, 'tax_amount')
                            tip_amount = safe_get_amount(receipt_data, 'tip_amount')
                            discount_amount = safe_get_amount(receipt_data, 'discount_amount') # ⬅️ **[추가: 할인액 추출]**
                            
                            currency_unit = receipt_data.get('currency_unit', '').strip()
                            display_unit = currency_unit if currency_unit else 'KRW'
                            
                            # 💡 날짜와 위치 기본값 처리 로직 추가 (강력한 포맷 검사 포함)
                            receipt_date_str = receipt_data.get('date', '').strip()
                            store_location_str = receipt_data.get('store_location', '').strip()
                            
                            try:
                                # ISO 8601 형식 (YYYY-MM-DD)으로 강제 변환 시도
                                date_object = pd.to_datetime(receipt_date_str, format='%Y-%m-%d', errors='raise').date()
                                final_date = date_object.strftime('%Y-%m-%d')
                            except (ValueError, TypeError):
                                # 변환에 실패하면 오늘 날짜를 기본값으로 사용
                                final_date = datetime.date.today().strftime('%Y-%m-%d')
                                st.warning("⚠️ AI가 인식한 날짜가 유효하지 않아 오늘 날짜로 대체되었습니다.")
                                
                            # 위치 기본값: 유효하지 않거나 빈 문자열이면 "Seoul" 사용
                            final_location = store_location_str if store_location_str else "Seoul"

                            
                            # --- 📢 [NEW] 금액 검증 및 덮어쓰기 로직 시작 (OVRRIDE) ---
                            # 1. 아이템 데이터프레임 생성 및 기본 계산
                            if 'items' in receipt_data and receipt_data['items']:
                                items_df = pd.DataFrame(receipt_data['items'])
                                
                                items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                                items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                                items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                                
                                # 2. 아이템 원가 총합 (할인 적용 전, Tax 포함) 계산
                                calculated_original_total = (items_df['Unit Price'] * items_df['Quantity']).sum()
                                total_discount = safe_get_amount(receipt_data, 'discount_amount') 
                                
                                # 3. 아이템 합계를 기반으로 최종 지불액 재계산 (이론적 합계)
                                calculated_final_total = calculated_original_total - total_discount
                                
                                # 4. AI가 추출한 total_amount와 비교하여 덮어쓰기
                                # 오차 허용 범위: 100원
                                if abs(calculated_final_total - total_amount) > 100 and calculated_final_total > 0:
                                    st.warning(
                                        f"⚠️ AI 추출 총액({total_amount:,.0f} {display_unit})이 아이템 합계({calculated_final_total:,.0f} {display_unit})와 크게 다릅니다. "
                                        f"**아이템 합계로 총액을 교정합니다.**"
                                    )
                                    # AI가 잘못 읽은 total_amount를 아이템 합계로 덮어씁니다.
                                    total_amount = calculated_final_total
                                
                                # --- 📢 [NEW] 금액 검증 및 덮어쓰기 로직 종료 ---
                            
                                
                                # --- Main Information Display ---
                                st.success("✅ Analysis Complete! Check the ledger data below.")
                                
                                st.markdown(f"**🏠 Store Name:** {receipt_data.get('store_name', 'N/A')}")
                                st.markdown(f"**📍 Location:** {final_location}") 
                                st.markdown(f"**📅 Date:** {final_date}") 
                                # 교정된 total_amount를 표시합니다.
                                st.subheader(f"💰 Total Amount Paid (Corrected): {total_amount:,.0f} {display_unit}")

                                if discount_amount > 0:
                                    discount_display = f"{discount_amount:,.2f} {display_unit}"
                                    st.markdown(f"**🎁 Total Discount:** {discount_display}") 

                                
                                # 💡 세금/팁 정보 표시
                                if tax_amount > 0 or tip_amount > 0:
                                    tax_display = f"{tax_amount:,.2f} {display_unit}"
                                    tip_display = f"{tip_amount:,.2f} {display_unit}"
                                    st.markdown(f"**🧾 Tax/VAT:** {tax_display} | **💸 Tip:** {tip_display}")
                                
                                # 💡 Display Applied Exchange Rate for AI Analysis
                                if display_unit != 'KRW':
                                    applied_rate = EXCHANGE_RATES.get(display_unit, 1.0)
                                    st.info(f"**📢 Applied Exchange Rate:** 1 {display_unit} = {applied_rate:,.4f} KRW (Rate fetched from API/Fallback)")
                                    
                                st.markdown("---")

                                # 📢 할인 안분(Allocation) 로직 시작! - 로직 안정화 (Robust Initialization)
                                # items_df는 이제 `calculated_original_total`이 계산된 상태입니다.
                                items_df['Total Spend Original'] = items_df['Unit Price'] * items_df['Quantity']
                                items_df['Discount Applied'] = 0.0
                                items_df['Total Spend'] = items_df['Total Spend Original']
                                
                                total_item_original = items_df['Total Spend Original'].sum()
                                
                                # 🌟 2단계: 할인이 있을 경우에만 재계산
                                # total_discount는 AI가 추출한 양수 값입니다.
                                if total_discount > 0 and total_item_original > 0:
                                    # 할인 비율 계산: 품목 원가 총합 대비 할인액 비율
                                    discount_rate = total_discount / total_item_original
                                    
                                    # 품목별 할인액 계산 및 실제 지출액 (Total Spend)으로 업데이트
                                    items_df['Discount Applied'] = items_df['Total Spend Original'] * discount_rate
                                    items_df['Total Spend'] = items_df['Total Spend Original'] - items_df['Discount Applied']
                                    st.info(f"💡 Discount of {total_discount:,.0f} {display_unit} successfully allocated across items.")
                                else:
                                    pass
                                    
                                # 📢 할인 안분 로직 종료. Total Spend는 이제 할인이 반영된 금액입니다.
                                
                                st.subheader("🛒 Detailed Item Breakdown (Category Editable)")
                                
                                # 데이터 에디터에 할인 전 금액, 할인액, 최종 지출 금액을 보여줍니다.
                                edited_df = st.data_editor(
                                    items_df.drop(columns=['Total Spend Original', 'Discount Applied', 'Total Spend']), # 임시로 제외
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
                                
                                # 📢 할인 안분 로직을 통과한 'Total Spend' 컬럼을 다시 edited_df에 합칩니다.
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

                                # 💡 세금과 팁도 원화로 환산
                                krw_tax_total = convert_to_krw(tax_amount, display_unit, EXCHANGE_RATES) 
                                krw_tip_total = convert_to_krw(tip_amount, display_unit, EXCHANGE_RATES)
                                
                                # 📢 [NEW] 위치 정보에 대한 좌표 추출
                                # geocode_address_placeholder 대신 실제 API 호출 함수를 사용합니다.
                                lat, lon = geocode_address(final_location)
                                
                                # ** Accumulate Data: Store the edited DataFrame **
                                st.session_state.all_receipts_items.append(edited_df)
                                
                                # 💡 최종 수정: 한국 영수증의 경우 Tax_KRW는 Total 금액에 다시 합산하지 않고 Tip만 합산합니다.
                                final_total_krw = edited_df['KRW Total Spend'].sum() + krw_tip_total
                                
                                st.session_state.all_receipts_summary.append({
                                    'id': file_id, 
                                    'filename': uploaded_file.name,
                                    'Store': receipt_data.get('store_name', 'N/A'),
                                    'Total': final_total_krw, # 아이템 총합 + Tip만 더함 (Tax 제외)
                                    'Tax_KRW': krw_tax_total, 
                                    'Tip_KRW': krw_tip_total, 
                                    'Currency': 'KRW', 
                                    'Date': final_date, 
                                    'Location': final_location, 
                                    'Original_Total': total_amount, # 교정된 total_amount 사용
                                    'Original_Currency': display_unit,
                                    # 📢 [NEW] 좌표 추가
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
                
                # 📢 Currency Conversion for Manual Input
                krw_total = convert_to_krw(manual_amount, manual_currency, EXCHANGE_RATES)
                applied_rate = EXCHANGE_RATES.get(manual_currency, 1.0)

                # 📢 [NEW] 위치 정보에 대한 좌표 추출
                final_location = manual_location if manual_location else "Manual Input Location"
                # geocode_address_placeholder 대신 실제 API 호출 함수를 사용합니다.
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
                    'Total': krw_total, # 수동 입력은 총액을 그대로 사용 (Tip/Tax는 0)
                    'Tax_KRW': 0.0, 
                    'Tip_KRW': 0.0, 
                    'Currency': 'KRW', 
                    'Date': manual_date.strftime('%Y-%m-%d'),
                    'Location': final_location, 
                    'Original_Total': manual_amount, 
                    'Original_Currency': manual_currency,
                    # 📢 [NEW] 좌표 추가
                    'latitude': lat,
                    'longitude': lon
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
        # 📢 [NEW] 좌표 컬럼 호환성 확보
        if 'latitude' not in summary_df.columns:
            summary_df['latitude'] = 37.5665
        if 'longitude' not in summary_df.columns:
            summary_df['longitude'] = 126.9780
            
        # Conditional formatting for Amount Paid
        def format_amount_paid(row):
            krw_amount = f"{row['Total']:,.0f} KRW"
            
            if row['Original_Currency'] != 'KRW':
                original_amount = f"{row['Original_Total']:,.2f} {row['Original_Currency']}"
                return f"{original_amount} / {krw_amount}"
            
            return krw_amount
        
        summary_df['Amount Paid'] = summary_df.apply(format_amount_paid, axis=1)

        
        summary_df = summary_df.drop(columns=['id'])
        # 💡 Location 컬럼을 추가하여 표시
        summary_df_display = summary_df[['Date', 'Store', 'Location', 'Amount Paid', 'Tax_KRW', 'Tip_KRW', 'filename']] 
        summary_df_display.columns = ['Date', 'Store', 'Location', 'Amount Paid', 'Tax (KRW)', 'Tip (KRW)', 'Source'] 

        st.dataframe(
            summary_df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Tax (KRW)": st.column_config.NumberColumn(
                    "Tax (KRW)", 
                    format="%.0f KRW" # 소수점 없이 KRW 표시
                ),
                "Tip (KRW)": st.column_config.NumberColumn(
                    "Tip (KRW)", 
                    format="%.0f KRW" # 소수점 없이 KRW 표시
                ),
            }
        )
        
        st.markdown("---")
        
        # --- 📢 [NEW] Map Visualization Section ---
        st.subheader("📍 Spending Map Visualization")
        
        map_df = summary_df.copy()
        # st.map은 'lat'과 'lon' 컬럼을 기대합니다.
        map_df.columns = [col.replace('latitude', 'lat').replace('longitude', 'lon') for col in map_df.columns]

        if not map_df.empty and 'lat' in map_df.columns and 'lon' in map_df.columns:
            
            # 📢 [CRITICAL FIX] lat/lon 컬럼의 결측치(NaN)가 StreamlitAPIException을 발생시키므로,
            #    유효한 좌표를 가진 행만 필터링합니다.
            map_data = map_df[map_df['Total'] > 0].dropna(subset=['lat', 'lon'])
            
            if not map_data.empty:
                # 중앙 위치 계산 (전체 데이터의 평균)
                # center_lat = map_data['lat'].mean()
                # center_lon = map_data['lon'].mean()
                
                st.map(
                    map_data, 
                    latitude='lat', 
                    longitude='lon', 
                    color='#ff6347', # 산호색
                    zoom=11, 
                    use_container_width=True
                )
            
            else:
                st.warning("유효한 좌표 정보가 있는 지출 기록이 없어 지도를 표시할 수 없습니다.")
        else:
            st.warning("위치 정보가 없거나 좌표 컬럼이 유효하지 않아 지도를 표시할 수 없습니다.")

        st.markdown("---")
        # --- 📢 [NEW] Map Visualization Section End ---
        
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
        
        # 💡 세금과 팁도 별도의 카테고리로 합산하여 표시
        total_tax_krw = summary_df['Tax (KRW)'].sum()
        total_tip_krw = summary_df['Tip (KRW)'].sum()
        
        if total_tax_krw > 0:
            category_summary.loc[len(category_summary)] = ['세금/부가세 (Tax/VAT)', total_tax_krw]
        if total_tip_krw > 0:
            category_summary.loc[len(category_summary)] = ['팁 (Tip)', total_tip_krw]
            
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
# 		 	TAB 2: FINANCIAL EXPERT CHAT (유지)
# ======================================================================
with tab2:
    st.header("💬 Financial Expert Chat")
    
    if not st.session_state.all_receipts_items:
        st.warning("Please analyze at least one receipt or load a CSV in the 'Analysis & Tracking' tab before starting a consultation.")
    else:
        # --- 🌟 Chat History Reset Logic (Fix 2) 🌟 ---
        # 현재 영수증 기록을 기반으로 해시 생성
        # all_receipts_summary의 'id' 목록으로 데이터 변경 여부를 감지합니다.
        current_data_hash = hash(tuple(item['id'] for item in st.session_state.all_receipts_summary))
        
        # 데이터가 변경되었는지 확인
        if 'last_data_hash' not in st.session_state or st.session_state.last_data_hash != current_data_hash:
            # 데이터가 변경된 경우, 채팅 기록과 해시를 리셋합니다.
            st.session_state.chat_history = []
            st.session_state.last_data_hash = current_data_hash
            st.info("📊 새로운 지출 내역이 감지되었습니다. 신선한 분석을 위해 채팅 기록이 초기화됩니다.")
        all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        # Defensive check for KRW Total Spend column
        if 'KRW Total Spend' not in all_items_df.columns:
             all_items_df['KRW Total Spend'] = all_items_df.apply(
                 lambda row: convert_to_krw(row['Total Spend'], row['Currency'], EXCHANGE_RATES), axis=1
             )

        # 1. Add Psychological Category to the detailed DataFrame
        all_items_df['Psychological Category'] = all_items_df['AI Category'].apply(get_psychological_category)

        # 2. Group by the new Psychological Category
        psychological_summary = all_items_df.groupby('Psychological Category')['KRW Total Spend'].sum().reset_index()
        psychological_summary.columns = ['Category', 'KRW Total Spend']

        # 3. Add Tip only to Fixed/Essential Cost 
        summary_df_for_chat = pd.DataFrame(st.session_state.all_receipts_summary)
        
        tax_tip_only_total = 0.0
        
        if 'Tip_KRW' in summary_df_for_chat.columns:
            tax_tip_only_total += summary_df_for_chat['Tip_KRW'].sum() # Tip만 합산합니다.
        
        # Add Tip (Only) to the 'Fixed / Essential Cost' category
        if tax_tip_only_total > 0:
            # Find or create the Fixed / Essential Cost entry
            fixed_cost_index = psychological_summary[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[3]].index
            if not fixed_cost_index.empty:
                # Tip만 Fixed Cost에 합산
                psychological_summary.loc[fixed_cost_index[0], 'KRW Total Spend'] += tax_tip_only_total 
            else:
                new_row = pd.DataFrame([{'Category': PSYCHOLOGICAL_CATEGORIES[3], 'KRW Total Spend': tax_tip_only_total}])
                psychological_summary = pd.concat([psychological_summary, new_row], ignore_index=True)

        total_spent = psychological_summary['KRW Total Spend'].sum()
        
        # Calculate the Impulse Spending Index
        impulse_spending = psychological_summary.loc[psychological_summary['Category'] == PSYCHOLOGICAL_CATEGORIES[2], 'KRW Total Spend'].sum()
        impulse_index = impulse_spending / total_spent if total_spent > 0 else 0.0
        
        psychological_summary_text = psychological_summary.to_string(index=False)
        
        # Prepare detailed item data for the chatbot's system instruction
        detailed_items_for_chat = all_items_df[['Psychological Category', 'Item Name', 'KRW Total Spend']]
        items_text_for_chat = detailed_items_for_chat.to_string(index=False)
        
        # MODIFIED SYSTEM INSTRUCTION (CRITICAL)
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

        # 💡 초기 메시지 추가 (UX 개선)
        if not st.session_state.chat_history or (len(st.session_state.chat_history) == 1 and st.session_state.chat_history[0]["content"].startswith("안녕하세요! 저는 귀하의 지출 패턴을 분석하는")):
              # 챗 기록이 없거나, 이전 버전의 초기 메시지만 있을 경우 재설정
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

        # Display chat history
        # ... (이하 기존 채팅 history display 및 prompt input 로직 유지)
        
        # ... (이하 기존 채팅 history display 및 prompt input 로직 유지)
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
                        # --- 🌟 수정된 역할 매핑 로직 시작 🌟 ---
                        combined_contents = []
                        history_items = st.session_state.chat_history # 모든 기록을 사용 (마지막 user prompt 포함)
                        
                        for item in history_items:
                            # Streamlit 역할(user, assistant)을 Gemini 역할(user, model)로 매핑합니다.
                            gemini_role = "user" if item["role"] == "user" else "model" 
                            
                            combined_contents.append({
                                "role": gemini_role, 
                                "parts": [{"text": item["content"]}]
                            })
                        
                        # Note: st.session_state.chat_history에 마지막 user prompt가 이미 추가되어 있으므로,
                        # combined_contents는 마지막까지 정확히 구성됩니다.
                        
                        # --- 🌟 수정된 역할 매핑 로직 종료 🌟 ---

                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=combined_contents, # ⬅️ 이제 올바른 역할(user/model)이 포함됨
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Chatbot API call failed: {e}")
