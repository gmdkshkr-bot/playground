# app.py

import streamlit as st
import json
import pandas as pd
from PIL import Image
import io
# Google GenAI 라이브러리 임포트
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="Smart Household Account Book 🧾",
    layout="wide"
)

st.title("🧾 Smart Household Account Book")
st.markdown("---")


# --- 0. API 키 설정 (Streamlit Secrets 사용) ---
# Streamlit Cloud 배포 시 st.secrets['GEMINI_API_KEY']를 사용합니다.
# 로컬 테스트 시에는 'GEMINI_API_KEY' 환경 변수 또는 secrets.toml 파일을 사용하세요.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("❌ 'GEMINI_API_KEY'를 Streamlit Secrets에 설정해 주세요.")
    st.stop()

# GenAI 클라이언트 초기화
client = genai.Client(api_key=API_KEY)


# --- 1. Gemini 분석 함수 ---

#@st.cache_data(show_spinner=False)
def analyze_receipt_with_gemini(_image: Image.Image): # image 앞에 언더바('_') 추가!
    """
    Gemini 모델을 호출하여 영수증 이미지에서 데이터를 추출하고 카테고리를 분류합니다.
    """
    st.info("💡 Gemini API를 사용하여 영수증 분석을 시작합니다. (약 10~20초 소요)")
    # 🎯 데이터 추출 및 AI 카테고리 분류를 위한 프롬프트 (JSON 형식 강제)
   # app.py (analyze_receipt_with_gemini 함수 내부)

    prompt_template = """
    당신은 영수증 분석 및 가계부 기록 전문가입니다. 
    이 영수증 이미지에서 다음 항목들을 분석하여 **반드시 JSON 형식**으로 추출해 주세요. 

    **가장 중요한 지시:** 응답은 오직 **백틱(```json)으로 감싸진 JSON 코드 블록**으로만 제공해야 합니다. 어떤 형태의 설명, 인사, 추가 문구도 JSON 코드 블록 앞뒤에 포함하지 마세요.

    1. store_name: 상호명 (텍스트)
    2. date: 날짜 (YYYY-MM-DD 형식)
    3. total_amount: 총 결제 금액 (숫자만, 쉼표 없이)
    4. currency_unit: 영수증에 표기된 **통화의 공식 코드** (예: **USD**, KRW, EUR 등)를 추출해 주세요.
    5. items: 구매 품목 리스트. 각 품목에 대해 다음 정보를 포함해야 합니다.
        - name: 품목명 (텍스트)
        - price: 단가 (숫자만, 쉼표 없이)
        - quantity: 수량 (숫자만)
        - category: 해당 품목에 가장 적절한 카테고리 (예: '식비', '교통', '생활용품', '문화/여가', '기타')를 **자동으로 분류**해서 넣어주세요.

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
        # 모델 호출 (gemini-2.5-flash는 멀티모달 처리가 빠르고 효율적입니다.)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template, image],
            config=genai.types.GenerateContentConfig(
                # 안전 필터 조정 (영수증 분석은 일반적으로 유해성이 없으므로 기본 설정 유지)
                safety_settings=[
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                ]
            )
        )
        return response.text
    
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
        return None


# --- 2. Streamlit UI 및 로직 ---

uploaded_file = st.file_uploader("📸 분석할 영수증 사진(jpg, png)을 업로드해 주세요.",
                                 type=['jpg', 'png', 'jpeg']) # heic, heif 추가 

if uploaded_file is not None:
    # 파일을 PIL Image 객체로 변환
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"이미지 파일 로드 오류: {e}")
        return
    # 이제 'image' 변수는 PIL Image 객체이며, 다음 분석 로직으로 넘어갑니다.
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ 업로드된 영수증")
        # 🚨 수정: use_column_width 대신 use_container_width 사용!
        st.image(image, use_container_width=True) 
    
    with col2:
        st.subheader("📊 분석 및 기록")
        if st.button("✨ 영수증 분석 시작하기"):
            with st.spinner('AI가 영수증을 꼼꼼히 읽고 있습니다...'):
                # image 인자를 전달할 때 함수 정의에 맞게 이름은 그대로 사용합니다.
                json_data_text = analyze_receipt_with_gemini(image)

                if json_data_text:
                    try:
                        # 1. JSON 코드 블록만 추출하는 방어 로직 추가
                        if json_data_text.startswith("```json"):
                            # 응답이 코드 블록으로 시작하는 경우, 블록 내부만 추출
                            json_data_text = json_data_text.strip().lstrip("```json").rstrip("```").strip()
                        
                        # 텍스트 응답을 JSON 객체로 파싱
                        receipt_data = json.loads(json_data_text)

                        # --- 통화 단위 추출 ---
                        # 영수증에서 추출한 통화 단위를 변수에 저장합니다.
                        #currency_unit = receipt_data.get('currency_unit', '원')
                        currency_unit = receipt_data.get('currency_unit', '').strip()
                        display_unit = currency_unit if currency_unit else '원'
                        
                        # --- 메인 정보 표시 ---
                        st.success("✅ 분석 완료! 아래 가계부 데이터를 확인해 보세요.")
                        
                        # 메인 요약 정보를 표시
                        st.markdown(f"**🏠 상호명:** {receipt_data.get('store_name', '정보 없음')}")
                        st.markdown(f"**📅 날짜:** {receipt_data.get('date', '정보 없음')}")
                        #st.subheader(f"💰 총 결제 금액: {receipt_data.get('total_amount', 0):,} 원")
                        st.subheader(f"💰 총 결제 금액: {receipt_data.get('total_amount', 0):,} {display_unit}")
                        st.markdown("---")

                        # --- 품목별 데이터프레임 생성 ---
                        if 'items' in receipt_data and receipt_data['items']:
                            items_df = pd.DataFrame(receipt_data['items'])
                            
                            # 데이터프레임 컬럼 이름 변경 (사용자 친화적으로)
                            items_df.columns = ['품목명', '단가', '수량', 'AI 카테고리']
                            
                            st.subheader("🛒 품목별 상세 내역")
                            st.dataframe(items_df, use_container_width=True, hide_index=True)
                            
                            # --- 3. 데이터 다운로드 기능 추가 ---
                            
                            @st.cache_data
                            def convert_df_to_csv(df):
                                # DataFrame을 CSV 형식으로 변환 (인코딩: UTF-8-sig)
                                return df.to_csv(index=False, encoding='utf-8-sig')

                            csv = convert_df_to_csv(items_df)
                            
                            st.download_button(
                                label="⬇️ 가계부 CSV 파일 다운로드",
                                data=csv,
                                file_name=f"{receipt_data.get('date', 'receipt')}_{receipt_data.get('store_name', 'data')}.csv",
                                mime='text/csv',
                            )
                        else:
                            st.warning("분석 결과에서 품목 리스트를 찾을 수 없습니다.")

                    except json.JSONDecodeError:
                        st.error("❌ Gemini 분석 결과가 올바른 JSON 형식이 아닙니다. 영수증 이미지를 더 선명하게 올려주세요.")
                    except Exception as e:
                        st.error(f"데이터 처리 중 예상치 못한 오류 발생: {e}")
                else:
                    st.error("분석을 완료하지 못했습니다. 다시 시도해 주세요.")
