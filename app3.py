import streamlit as st
import json
import pandas as pd
from PIL import Image
import io
# Import Google GenAI library
from google import genai
# Corrected import path for types
from google.genai.types import HarmCategory, HarmBlockThreshold 
import numpy as np
import plotly.express as px # Plotly for interactive Pie Chart

# ----------------------------------------------------------------------
# 📌 1. Initialize session state for cumulative receipt data & chat history
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    # Space to store detailed item data (list of DataFrames)
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    # Space to store receipt summaries (total, store, ID etc.)
    st.session_state.all_receipts_summary = []
if 'chat_history' not in st.session_state:
    # Space to store the conversation history for the chat bot
    st.session_state.chat_history = []


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart Receipt Analyzer & Tracker 🧾",
    layout="wide"
)


# ----------------------------------------------------------------------
# 💡 Sidebar (About This App)
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("About This App")
    st.markdown("---")
    
    st.subheader("How to Use")
    st.markdown("""
    This application helps you manage your household ledger easily by using AI.
    1. **Upload:** Upload one receipt image (JPG, PNG) at a time.
    2. **Analyze:** Click 'Start Receipt Analysis' to extract store, date, items, and total amount.
    3. **Accumulate:** The results are automatically added to the cumulative record.
    4. **Review & Chat:** Check the integrated report, spending charts, and get personalized financial advice from the Chatbot.
    """)
    
    st.subheader("APIs Used")
    st.markdown("""
    - **Google Gemini API:** Utilized for Multimodal analysis (OCR and categorization) and conversational analysis.
    - **Streamlit:** Used for creating the interactive web application interface.
    - **Pandas/Plotly:** Used for data manipulation, accumulation, and visualization (charts).
    """)
    
    st.markdown("---")
    if st.session_state.all_receipts_items:
        st.info(f"Currently tracking {len(st.session_state.all_receipts_items)} receipts.")
        
st.title("🧾 AI Household Ledger: Receipt Analysis & Cumulative Tracking")
st.markdown("---")


# --- 0. API Key Configuration (Using Streamlit Secrets) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("❌ Please set 'GEMINI_API_KEY' in Streamlit Secrets.")
    st.stop()

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)


# --- 1. Gemini Analysis Function ---
def analyze_receipt_with_gemini(_image: Image.Image):
    """
    Calls the Gemini model to extract data and categorize items from a receipt image.
    """
    
    # Prompt for data extraction and AI category classification (JSON format enforced)
    # **Updated Prompt for detailed sub-categories**
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
                # Safety filter configuration
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
    
    # Updated prompt to include detailed item data for richer analysis
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
    4. **CRITICAL:** When mentioning the total spending amount in the analysis, **you must include the currency unit** (e.g., "총 1,500,000 KRW 지출").
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template],
        )
        return response.text
        
    except Exception as e:
        # st.error(f"AI analysis report generation failed: {e}") # Suppress error in chat mode
        return "Failed to generate analysis report."


# ----------------------------------------------------------------------
# --- 3. Streamlit UI: Tab Setup ---
# ----------------------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Analysis & Tracking", "💬 Financial Expert Chat"])

# Define all possible categories for the data_editor Selectbox
all_categories = [
    "외식", "식재료", "카페/음료", "주류", 
    "생필품", "의료/건강", "교육/서적", "통신", "공과금",
    "대중교통", "유류비", "택시", "주차비", 
    "영화/공연", "여행", "취미", "게임", 
    "경조사", "이체/수수료", "비상금", "미분류"
]


# ======================================================================
#             TAB 1: ANALYSIS & TRACKING
# ======================================================================
with tab1:
    
    # 1. File Uploader (Single file mode)
    uploaded_file = st.file_uploader(
        "📸 Upload one receipt image (jpg, png) at a time. (Data will accumulate in the current session)", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=False 
    )


    if uploaded_file is not None:
        # 2. Generate unique file ID (to prevent re-analysis after reruns)
