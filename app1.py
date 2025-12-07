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
    # 🚨 주의: 이 키들은 Streamlit Secrets에 설정되어 있어야 합니다.
    API_KEY = st.secrets["GEMINI_API_KEY"]
    EXCHANGE_RATE_API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"] 
    # 📢 [NEW] 카카오 API 키 로드
    KAKAO_REST_API_KEY = st.secrets["KAKAO_REST_API_KEY"]
except KeyError:
    st.error("❌ Please set 'GEMINI_API_KEY', 'EXCHANGE_RATE_API_KEY', and 'KAKAO_REST_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)

# --- 📢 [UPDATED] Geocoding Helper Function (Kakao API 최적화) ---
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
        # st.sidebar.error(f"❌ Kakao Geocoding API Error for '{address}'. Using fallback: {e}") # 사이드바에 에러가 너무 많이 뜨는 것을 방지
        pass
    except Exception as e:
        # JSON 파싱 등 기타 오류
        pass

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
# 📢 [MODIFIED] Household Goods 카테고리 세분화
ALL_CATEGORIES = [
    "Dining Out", "Casual Dining", "Coffee & Beverages", "Alcohol & Bars", 
    "Groceries", 
    "Household Essentials", "Beauty & Cosmetics", "Clothing & Fashion", # 📢 세분화된 카테고리
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
# 📢 [MODIFIED] SPENDING_NATURE 재매핑
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
    "Household Essentials": "Consumption_Planned", # 📢 [MODIFIED] 필수 생활용품은 계획 소비로 분류
    "Fuel & Vehicle Maint.": "Consumption_Planned", # Essential Variable
    
    # EXPERIENCE / DISCRETIONARY (경험적/선택적)
    "Dining Out": "Consumption_Experience",
    "Travel & Accommodation": "Consumption_Experience",
    "Movies & Shows": "Consumption_Experience",
    "Beauty & Cosmetics": "Consumption_Experience", # 📢 [MODIFIED]
    "Clothing & Fashion": "Consumption_Experience", # 📢 [MODIFIED]
    
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
        # 📢 [MODIFIED] 카테고리 가이드 업데이트
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
    - **VARIABLE / CONSUMPTION (Planned):** Groceries, Household Essentials # 📢 수정됨
    - **VARIABLE / CONSUMPTION (Experience):** Dining Out, Travel & Accommodation, Movies & Shows, Beauty & Cosmetics, Clothing & Fashion # 📢 수정됨
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

# 📢 [NEW] PDF 생성 클래스 (fpdf2 기반)
class PDF(FPDF):
    def header(self):
        # 📢 [FIX] Nanum Gothic으로 폰트 설정
        self.set_font('Nanum', 'B', 15)
        self.cell(0, 10, 'Personal Spending Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Nanum', '', 8) # 📢 [FIX] 이탤릭('I') 제거
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self):
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
        
        # 📢 [FIX] 테이블 너비 자동 계산 (PDF 너비 190mm 기준)
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
            
            # 셀 내용이 너무 길어지지 않도록 조정 (테이블 레이아웃 유지)
            row_list = [item[:25] if len(item) > 25 else item for item in row_list]
            
            for i, item in enumerate(row_list):
                self.cell(col_width, 6, item, 1, 0, 'C')
            self.ln()


# 📢 [NEW] 폰트 로딩을 캐시하는 함수 (FPDFException 방지)
@st.cache_resource
def load_pdf_fonts(pdf_instance):
    """Nanum 폰트를 FPDF에 등록하며, 실패 시 None을 반환합니다."""
    try:
         # 폰트 파일이 'fonts/' 폴더 안에 있다고 가정하고 상대 경로를 지정합니다.
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
