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
# This function is used to summarize the overall spending in the Analysis tab
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float, currency_unit: str): # 💡 Added currency_unit
    """
    Generates an AI analysis report based on aggregated spending data for the main analysis tab.
    """
    
    # Convert DataFrame to a string suitable for the prompt
    summary_text = summary_df.to_string(index=False)
    
    # 💡 Included currency_unit in the total spending amount and mentioned it for the category breakdown
    prompt_template = f"""
    You are an AI ledger analyst providing professional financial advice.
    The user's **all accumulated spending** amounts to {total_amount:,.0f} {currency_unit}.
    Below is the category breakdown of **all accumulated spending**. (All amounts are in {currency_unit})
    
    --- Spending Summary Data ---
    {summary_text}
    ---

    **CRITICAL DETAILED DATA:** Below are the individual item names, their categories, and total costs. Use this data to provide qualitative and specific advice (e.g., mention specific products or stores if patterns are observed).
    --- Detailed Items Data (AI Category, Item Name, Total Spend) ---
    {detailed_items_text}
    ---
    
    Follow these instructions and provide an analysis report in a friendly and professional tone:
    1. Summarize the main characteristic of this total spending (e.g., the largest spending category) in one sentence.
    2. Provide 2-3 sentences of helpful and friendly advice or commentary for the user (e.g., a suggestion for future budget management).
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


                            # --- Item DataFrame Creation and Accumulation (Using data_editor for user category selection) ---
                            if 'items' in receipt_data and receipt_data['items']:
                                items_df = pd.DataFrame(receipt_data['items'])
                                
                                # Data normalization and total spend calculation
                                items_df.columns = ['Item Name', 'Unit Price', 'Quantity', 'AI Category']
                                items_df['Unit Price'] = pd.to_numeric(items_df['Unit Price'], errors='coerce').fillna(0)
                                items_df['Quantity'] = pd.to_numeric(items_df['Quantity'], errors='coerce').fillna(1)
                                items_df['Total Spend'] = items_df['Unit Price'] * items_df['Quantity']
                                
                                st.subheader("🛒 Detailed Item Breakdown (Category Editable)")
                                
                                # Use st.data_editor to allow users to modify the 'AI Category' field
                                edited_df = st.data_editor(
                                    items_df,
                                    column_config={
                                        "AI Category": st.column_config.SelectboxColumn(
                                            "Final Category",
                                            help="Select the correct sub-category for this item.",
                                            width="medium",
                                            options=all_categories,
                                            required=True,
                                        )
                                    },
                                    disabled=['Item Name', 'Unit Price', 'Quantity', 'Total Spend'], # Other columns are read-only
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                # Add Currency column to the edited_df before accumulation
                                edited_df['Currency'] = display_unit

                                # ** Accumulate Data: Store the edited DataFrame **
                                st.session_state.all_receipts_items.append(edited_df)
                                st.session_state.all_receipts_summary.append({
                                    'id': file_id, # Unique ID for deduplication
                                    'filename': uploaded_file.name,
                                    'Store': receipt_data.get('store_name', 'N/A'),
                                    'Total': total_amount,
                                    'Currency': display_unit,
                                    'Date': receipt_data.get('date', 'N/A')
                                })

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
    # --- 5. Cumulative Data Analysis Section (Always displayed if data exists) ---
    # ----------------------------------------------------------------------

    if st.session_state.all_receipts_items:
        st.markdown("---")
        st.title("📚 Cumulative Spending Analysis Report")
        
        # 1. Create a single DataFrame from all accumulated items
        all_items_df_numeric = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        
        # 🚀 SOLUTION FOR KEY ERROR: Defensive coding to ensure 'Currency' column exists
        if 'Currency' not in all_items_df_numeric.columns:
            default_currency = st.session_state.all_receipts_summary[-1]['Currency'] if st.session_state.all_receipts_summary else 'KRW'
            all_items_df_numeric['Currency'] = default_currency
        # -------------------------------------------------------------------------------
        
        # Get the currency unit of the last receipt for consistent labeling in the cumulative report
        # We can safely access 'Currency' now
        display_currency_label = all_items_df_numeric['Currency'].iloc[-1] if not all_items_df_numeric.empty else 'KRW'


        # A. Display Accumulated Receipts Summary Table
        st.subheader(f"Total {len(st.session_state.all_receipts_summary)} Receipts Logged (Summary)")
        summary_df = pd.DataFrame(st.session_state.all_receipts_summary)
        
        # Drop 'id' and reorder columns for presentation
        summary_df = summary_df.drop(columns=['id'])
        
        # ⭐️ Combine Total and Currency for better display ⭐️
        summary_df['Total'] = summary_df['Total'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else 'N/A')
        summary_df['Amount Paid'] = summary_df['Total'] + ' ' + summary_df['Currency']
        
        # Select columns to display
        summary_df = summary_df[['Date', 'Store', 'Amount Paid', 'filename']] 
        summary_df.columns = ['Date', 'Store', 'Amount Paid', 'Original File'] 

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🛒 Integrated Detail Items") # Title for the detailed item list
        
        # Create a display version to show currency unit next to monetary values
        all_items_df_display = all_items_df_numeric.copy()
        all_items_df_display['Unit Price'] = all_items_df_display.apply(
            lambda row: f"{row['Unit Price']:,.0f} {row['Currency']}" if pd.notnull(row['Unit Price']) else f"N/A {row['Currency']}", axis=1
        )
        all_items_df_display['Total Spend'] = all_items_df_display.apply(
            lambda row: f"{row['Total Spend']:,.0f} {row['Currency']}" if pd.notnull(row['Total Spend']) else f"N/A {row['Currency']}", axis=1
        )
        
        st.dataframe(
            all_items_df_display[['Item Name', 'Unit Price', 'Quantity', 'AI Category', 'Total Spend']], 
            use_container_width=True, 
            hide_index=True
        )

        # 2. Aggregate spending by category and visualize
        # Use the numeric DataFrame for aggregation
        category_summary = all_items_df_numeric.groupby('AI Category')['Total Spend'].sum().reset_index()
        category_summary.columns = ['Category', 'Amount']
        
        # --- Display Summary Table ---
        st.subheader("💰 Spending Summary by Category")
        
        # Format the Amount column for display with currency unit
        category_summary_display = category_summary.copy()
        category_summary_display['Amount'] = category_summary_display['Amount'].apply(lambda x: f"{x:,.0f} {display_currency_label}")
        
        st.dataframe(category_summary_display, use_container_width=True, hide_index=True)

        # --- Visualization ---
        
        col_chart, col_pie = st.columns(2)
        
        with col_chart:
            st.subheader(f"Bar Chart Visualization (Unit: {display_currency_label})") # Updated subtitle
            # Bar Chart (uses numeric category_summary)
            st.bar_chart(category_summary.set_index('Category'))
            
        with col_pie:
            st.subheader(f"Pie Chart Visualization (Unit: {display_currency_label})") # Updated subtitle
            # Pie Chart using Plotly Express for better visualization
            
            # Ensure only positive amounts are included in the chart
            chart_data = category_summary[category_summary['Amount'] > 0] 
            
            if not chart_data.empty:
                fig = px.pie(
                    chart_data, 
                    values='Amount', 
                    names='Category', 
                    title=f'Spending Distribution by Category (Unit: {display_currency_label})', # Updated title
                    # Set hole for a donut chart appearance
                    hole=.3, 
                )
                # Update layout for better appearance
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No spending data found to generate the pie chart.")


        # 3. Generate AI Analysis Report (for main analysis summary)
        st.markdown("---")
        st.subheader("🤖 AI Expert's Analysis Summary")

        # 💡 새로운 변수 준비: 분석에 필요한 핵심 항목만 추출합니다.
        detailed_items_for_ai = all_items_df_numeric[['AI Category', 'Item Name', 'Total Spend']]
        items_text = detailed_items_for_ai.to_string(index=False) # 텍스트로 변환
        
        total_spent = category_summary['Amount'].sum()
        display_currency_label = all_items_df_numeric['Currency'].iloc[-1] if not all_items_df_numeric.empty else 'KRW'
        
        # 💡 Passed the currency unit to the analysis function
        ai_report = generate_ai_analysis(
            summary_df=category_summary,
            store_name="Multiple Stores",
            total_amount=total_spent,
            currency_unit=display_currency_label,
            detailed_items_text=items_text # 👈 새 파라미터 추가!
        )
        
        st.info(ai_report)
        
        # 4. Reset and Download Buttons
        st.markdown("---")
        
        @st.cache_data
        def convert_df_to_csv(df):
            # Convert the entire DataFrame to CSV format (UTF-8-sig encoding for compatibility)
            return df.to_csv(index=False, encoding='utf-8-sig')

        csv = convert_df_to_csv(all_items_df_numeric) # Use the numeric dataframe for CSV download
        
        st.download_button(
            label="⬇️ Download Full Cumulative Ledger Data (CSV)",
            data=csv,
            file_name=f"all_receipts_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )

        if st.button("🧹 Reset Record", help="Clears all accumulated receipt analysis records in the app."):
            st.session_state.all_receipts_items = []
            st.session_state.all_receipts_summary = []
            st.session_state.chat_history = [] # Reset chat history too!
            st.rerun() # Corrected function name

# ======================================================================
#             TAB 2: FINANCIAL EXPERT CHAT
# ======================================================================
with tab2:
    st.header("💬 Financial Expert Chat")
    
    if not st.session_state.all_receipts_items:
        st.warning("Please analyze at least one receipt in the 'Analysis & Tracking' tab before starting a consultation.")
    else:
        # Get accumulated data summary for the system prompt
        all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
        # 🚀 Defensive check for 'Currency' column (if chat is opened before analysis section)
        if 'Currency' not in all_items_df.columns:
            default_currency = st.session_state.all_receipts_summary[-1]['Currency'] if st.session_state.all_receipts_summary else 'KRW'
            all_items_df['Currency'] = default_currency
        # -------------------------------------------------------------------------------
        
        category_summary = all_items_df.groupby('AI Category')['Total Spend'].sum().reset_index()
        total_spent = category_summary['Total Spend'].sum()
        summary_text = category_summary.to_string(index=False)
        display_currency_label_chat = all_items_df['Currency'].iloc[-1] if not all_items_df.empty else 'KRW' # Currency for Chatbot
        
        # System instruction is generated based on the user's current data
        system_instruction = f"""
        You are a supportive, friendly, and highly knowledgeable Financial Expert. Your role is to provide personalized advice on saving money, budgeting, and making smarter consumption choices.
        
        The user's cumulative spending data for the current session is as follows:
        - Total Accumulated Spending: {total_spent:,.0f} {display_currency_label_chat}
        - Category Breakdown (Category, Amount, all in {display_currency_label_chat}):
        {summary_text}
        
        Base all your advice and responses on this data. When asked for advice, refer directly to their spending patterns (e.g., "I see 'Food' is your largest expense..."). Keep your tone professional yet encouraging. **Always include the currency unit ({display_currency_label_chat}) when referring to monetary amounts.**
        """

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process user input
        if prompt := st.chat_input("Ask for financial advice or review your spending..."):
            
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Gemini response
            with st.chat_message("assistant"):
                with st.spinner("Expert is thinking..."):
                    try:
                        # Construct conversation contents for Gemini
                        contents = [
                            {"role": "user", "parts": [{"text": msg["content"]}]} 
                            for msg in st.session_state.chat_history
                        ]
                        
                        # Generate response
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=contents,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        
                        # Display response
                        st.markdown(response.text)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                        
                    except Exception as e:
                        st.error(f"Chatbot API call failed: {e}")
