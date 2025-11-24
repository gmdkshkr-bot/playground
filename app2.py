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

# ----------------------------------------------------------------------
# 📌 1. Initialize session state for cumulative receipt data (Runs once on app start)
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    # Space to store detailed item data (list of DataFrames)
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    # Space to store receipt summaries (total, store, ID etc.)
    st.session_state.all_receipts_summary = []


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart Receipt Analyzer & Tracker 🧾",
    layout="wide"
)

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
        - category: The most appropriate category for this item (e.g., 'Food', 'Transport', 'Household Goods', 'Culture/Leisure', 'Other') which must be **automatically classified** by you.
    
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
        # Model call (gemini-2.5-flash is fast and efficient for multimodal processing)
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
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float):
    """
    Generates an AI analysis report based on aggregated spending data.
    """
    st.info("💡 Generating AI analysis report on spending patterns...")
    
    # Convert DataFrame to a string suitable for the prompt
    summary_text = summary_df.to_string(index=False)
    
    prompt_template = f"""
    You are an AI ledger analyst providing professional financial advice.
    The user's **all accumulated spending** amounts to {total_amount:,.0f}.
    Below is the category breakdown of **all accumulated spending**.
    
    --- Spending Summary Data ---
    {summary_text}
    ---
    
    Follow these instructions and provide an analysis report in a friendly and professional tone:
    1. Summarize the main characteristic of this total spending (e.g., the largest spending category) in one sentence.
    2. Provide 2-3 sentences of helpful and friendly advice or commentary for the user (e.g., a suggestion for future budget management).
    3. The response must only contain the analysis content, starting directly with the summary, without any greetings or additional explanations.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template],
        )
        return response.text
        
    except Exception as e:
        st.error(f"AI analysis report generation failed: {e}")
        return "Failed to generate analysis report."


# ----------------------------------------------------------------------
# --- 3. Streamlit UI and Main Logic ---
# ----------------------------------------------------------------------

# 1. File Uploader (Single file mode)
uploaded_file = st.file_uploader(
    "📸 Upload one receipt image (jpg, png) at a time. (Data will accumulate)", 
    type=['jpg', 'png', 'jpeg'],
    accept_multiple_files=False 
)


if uploaded_file is not None:
    # 2. Generate unique file ID (to prevent re-analysis after reruns)
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    is_already_analyzed = any(s.get('id') == file_id for s in st.session_state.all_receipts_summary)

    # 3. File Preview and Analysis Button
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Uploaded Receipt")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True) 

    with col2:
        st.subheader("📊 Analysis and Recording")
        
        # Disable button if file is already analyzed
        if is_already_analyzed:
            st.warning("⚠️ This receipt has already been analyzed and added to the record. Please upload a different file.")
            analyze_button = st.button("✨ Start Receipt Analysis", disabled=True)
        else:
            analyze_button = st.button("✨ Start Receipt Analysis")


        # 4. Execute analysis on button click
        if analyze_button and not is_already_analyzed:
            
            st.info("💡 Starting Gemini analysis. This may take 10-20 seconds.")
            with st.spinner('AI is meticulously reading the receipt...'):
                
                # Gemini analysis call
                json_data_text = analyze_receipt_with_gemini(image)

                if json_data_text:
                    try:
                        # Defense logic: extract JSON code block only
                        if json_data_text.strip().startswith("```json"):
                            json_data_text = json_data_text.strip().lstrip("```json").rstrip("```").strip()
                        
                        receipt_data = json.loads(json_data_text)
                        
                        # Data type normalization
                        if not isinstance(receipt_data.get('total_amount'), (int, float)):
                             # Handle cases where amount is missing or not a number
                             receipt_data['total_amount'] = np.nan 

                        # --- Main Information Display ---
                        st.success("✅ Analysis Complete! Check the ledger data below.")
                        
                        currency_unit = receipt_data.get('currency_unit', '').strip()
                        display_unit = currency_unit if currency_unit else 'KRW'
                        total_amount = receipt_data.get('total_amount', 0)
                        
                        st.markdown(f"**🏠 Store Name:** {receipt_data.get('store_name', 'N/A')}")
                        st.markdown(f"**📅 Date:** {receipt_data.get('date', 'N/A')}")
                        st.subheader(f"💰 Total Amount Paid: {total_amount:,.0f} {display_unit}")
                        st.markdown("---")


                        # --- Item DataFrame Creation and Accumulation ---
                        if 'items' in receipt_data and receipt_data['items']:
                            items_df = pd.DataFrame(receipt_data['items'])
                            
                            # Data normalization and total spend calculation
                            items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                            items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                            items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                            items_df['Total Spend'] = items_df['Unit Price'] * items_df['Quantity']
                            
                            # ** Accumulate Data **
                            st.session_state.all_receipts_items.append(items_df)
                            st.session_state.all_receipts_summary.append({
                                'id': file_id, # Unique ID for deduplication
                                'filename': uploaded_file.name,
                                'Store': receipt_data.get('store_name', 'N/A'),
                                'Total': total_amount,
                                'Currency': display_unit,
                                'Date': receipt_data.get('date', 'N/A')
                            })

                            st.subheader("🛒 Detailed Item Breakdown")
                            st.dataframe(items_df, use_container_width=True, hide_index=True)
                            st.success(f"🎉 Data from {uploaded_file.name} successfully added to the cumulative record!")

                        else:
                            st.warning("Item list could not be found in the analysis result.")

                    except json.JSONDecodeError:
                        st.error("❌ Gemini analysis result is not a valid JSON format. (JSON parsing error)")
                    except Exception as e:
                        st.error(f"Unexpected error occurred during data processing: {e}")
                else:
                    st.error("Analysis failed to complete. Please try again.")


# ----------------------------------------------------------------------
# --- 4. Cumulative Data Analysis Section (Always displayed if data exists) ---
# ----------------------------------------------------------------------

if st.session_state.all_receipts_items:
    st.markdown("---")
    st.title("📚 Cumulative Spending Analysis Report")

    # 1. Create a single DataFrame from all accumulated items
    all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
    
    st.subheader(f"({len(st.session_state.all_receipts_items)} Receipts) Integrated Data")
    st.dataframe(all_items_df[['Item Name', 'Unit Price', 'Quantity', 'AI Category', 'Total Spend']], use_container_width=True, hide_index=True)

    # 2. Aggregate spending by category and visualize
    category_summary = all_items_df.groupby('AI Category')['Total Spend'].sum().reset_index()
    category_summary.columns = ['Category', 'Amount']
    
    st.markdown("---")
    st.subheader("💰 Spending Summary by Category")
    st.dataframe(category_summary, use_container_width=True, hide_index=True)
    # Note: Streamlit's bar_chart uses the index as the x-axis label
    st.bar_chart(category_summary.set_index('Category'))
    
    # 3. Generate AI Analysis Report
    st.markdown("---")
    st.subheader("🤖 AI Expert's Advice on Total Spending")
    
    total_spent = category_summary['Amount'].sum()
    
    ai_report = generate_ai_analysis(
        summary_df=category_summary,
        store_name="Multiple Stores",
        total_amount=total_spent
    )
    
    st.info(ai_report)
    
    # 4. Reset and Download Buttons
    st.markdown("---")
    
    @st.cache_data
    def convert_df_to_csv(df):
        # Convert the entire DataFrame to CSV format (UTF-8-sig encoding for compatibility)
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = convert_df_to_csv(all_items_df)
    
    st.download_button(
        label="⬇️ Download Full Cumulative Ledger Data (CSV)",
        data=csv,
        file_name=f"all_receipts_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

    if st.button("🧹 Reset Record", help="Clears all accumulated receipt analysis records in the app."):
        st.session_state.all_receipts_items = []
        st.session_state.all_receipts_summary = []
        st.experimental_rerun()
