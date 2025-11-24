# app.py

import streamlit as st
import json
import pandas as pd
from PIL import Image
import io
# Google GenAI 라이브러리 임포트
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold

# ----------------------------------------------------------------------
# 📌 1. 전체 영수증 데이터를 저장할 세션 상태 초기화
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    # 품목별 상세 데이터(DataFrame의 리스트)를 저장할 공간
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    # 영수증별 요약 데이터 (총액, 상호 등)를 저장할 공간
    st.session_state.all_receipts_summary = []


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
# app.py (기존 file_uploader 부분 수정)

uploaded_file = st.file_uploader("📸 분석할 영수증 사진(jpg, png)을 업로드해 주세요.",
                                 type=['jpg', 'png', 'jpeg'],
                                 accept_multiple_files=True # 다중 파일 허용
                                )


if uploaded_files:
    if st.button("🔍 영수증 분석 시작하기"):
        
        with st.spinner("⏳ 선택된 영수증들을 순차적으로 분석 중입니다..."):
            
            # --- 2. 다중 파일 반복 처리 ---
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"--- **[{i+1}/{len(uploaded_files)}]** {uploaded_file.name} 분석 시작 ---")
                
                # 1. 이미지 로드
                try:
                    image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"{uploaded_file.name} 파일 로드 오류: {e}")
                    continue # 다음 파일로 넘어감
                
                # 2. Gemini 분석 호출
                receipt_data = analyze_receipt_with_gemini(image)
                
                if not receipt_data or 'items' not in receipt_data:
                    st.warning(f"⚠️ {uploaded_file.name}: 분석 결과를 얻지 못했습니다.")
                    continue
                
                # 3. 데이터프레임 생성 및 금액 계산 (기존 로직 재활용)
                items_df = pd.DataFrame(receipt_data['items'])
                items_df['단가'] = pd.to_numeric(items_df['단가'], errors='coerce').fillna(0)
                items_df['수량'] = pd.to_numeric(items_df['수량'], errors='coerce').fillna(1)
                items_df['총 지출'] = items_df['단가'] * items_df['수량']
                
                # 4. 분석 결과 누적 저장
                st.session_state.all_receipts_items.append(items_df)
                
                # 영수증별 요약 정보 저장
                st.session_state.all_receipts_summary.append({
                    '파일명': uploaded_file.name,
                    '상호': receipt_data.get('store_name', 'N/A'),
                    '총액': receipt_data.get('total_amount', 0),
                    '통화': receipt_data.get('currency', 'N/A'),
                    '날짜': receipt_data.get('date', 'N/A')
                })
                
                # 분석 직후 개별 영수증 미리보기
                st.subheader(f"✅ {uploaded_file.name} 분석 완료")
                st.dataframe(items_df, use_container_width=True, hide_index=True)

        st.success(f"🎉 총 {len(uploaded_files)}개의 영수증 분석이 완료되었습니다!")

# --- 누적 데이터 분석 섹션 시작 ---
if st.session_state.all_receipts_items:
    st.markdown("---")
    st.title("📚 누적된 전체 지출 분석 리포트")

    # 1. 전체 품목 데이터프레임 생성
    all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
    
    st.subheader("모든 영수증 품목 통합 데이터")
    st.dataframe(all_items_df, use_container_width=True, hide_index=True)

    # 2. 카테고리별 집계 (통합된 데이터 사용)
    category_summary = all_items_df.groupby('AI 카테고리')['총 지출'].sum().reset_index()
    category_summary.columns = ['카테고리', '금액']
    
    st.markdown("---")
    st.subheader("💰 전체 누적 카테고리별 지출 요약")
    st.dataframe(category_summary, use_container_width=True, hide_index=True)
    st.bar_chart(category_summary.set_index('카테고리'))
    
    # 3. AI 분석 리포트 생성 (가장 큰 지출 카테고리와 총액 정보를 전달)
    st.markdown("---")
    st.subheader("🤖 AI 분석 전문가의 전체 지출 조언")
    
    # 누적 총 지출 금액 계산
    total_spent = category_summary['금액'].sum()
    
    # AI 분석 함수 호출 (함수 정의는 기존대로 유지)
    ai_report = generate_ai_analysis(
        summary_df=category_summary,
        store_name="다수 상점", # 다중 분석임을 명시
        total_amount=total_spent
    )
    
    st.info(ai_report)
    
    if st.button("🧹 기록 초기화"):
        st.session_state.all_receipts_items = []
        st.session_state.all_receipts_summary = []
        st.experimental_rerun() # 앱을 새로고침하여 초기화된 상태를 반영
