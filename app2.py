import streamlit as st
import json
import pandas as pd
from PIL import Image
import io
# Google GenAI 라이브러리 임포트
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
import numpy as np

# ----------------------------------------------------------------------
# 📌 1. 전체 영수증 데이터를 저장할 세션 상태 초기화 (앱 시작 시 한 번만 실행)
# ----------------------------------------------------------------------
if 'all_receipts_items' not in st.session_state:
    # 품목별 상세 데이터(DataFrame의 리스트)를 저장할 공간
    st.session_state.all_receipts_items = [] 
if 'all_receipts_summary' not in st.session_state:
    # 영수증별 요약 데이터 (총액, 상호, ID 등)를 저장할 공간
    st.session_state.all_receipts_summary = []


# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="Smart Household Account Book 🧾",
    layout="wide"
)

st.title("🧾 AI 가계부 도우미: 영수증 분석 및 누적 기록")
st.markdown("---")


# --- 0. API 키 설정 (Streamlit Secrets 사용) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("❌ 'GEMINI_API_KEY'를 Streamlit Secrets에 설정해 주세요.")
    st.stop()

# GenAI 클라이언트 초기화
client = genai.Client(api_key=API_KEY)


# --- 1. Gemini 분석 함수 ---
# @st.cache_data를 사용하지 않습니다. 대신 중복 분석 방지 로직을 수동으로 처리합니다.
def analyze_receipt_with_gemini(_image: Image.Image):
    """
    Gemini 모델을 호출하여 영수증 이미지에서 데이터를 추출하고 카테고리를 분류합니다.
    """
    
    # 🎯 데이터 추출 및 AI 카테고리 분류를 위한 프롬프트 (JSON 형식 강제)
    prompt_template = """
    당신은 영수증 분석 및 가계부 기록 전문가입니다. 
    이 영수증 이미지에서 다음 항목들을 분석하여 **반드시 JSON 형식**으로 추출해 주세요. 
    
    **가장 중요한 지시:** 응답은 오직 **백틱(```json)으로 감싸진 JSON 코드 블록**으로만 제공해야 합니다. 어떤 형태의 설명, 인사, 추가 문구도 JSON 코드 블록 앞뒤에 포함하지 마세요.
    
    1. store_name: 상호명 (텍스트)
    2. date: 날짜 (YYYY-MM-DD 형식)
    3. total_amount: 총 결제 금액 (숫자만, 쉼표 없이)
    4. currency_unit: 영수증에 표기된 **통화의 공식 코드** (예: KRW, USD, EUR 등)를 추출해 주세요.
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
        st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
        return None

# --- 2. AI 분석 리포트 생성 함수 (새로 추가) ---
def generate_ai_analysis(summary_df: pd.DataFrame, store_name: str, total_amount: float):
    """
    집계된 카테고리별 지출 데이터를 기반으로 AI 분석 리포트를 생성합니다.
    """
    st.info("💡 지출 패턴에 대한 AI 분석 리포트를 생성 중입니다...")
    
    summary_text = summary_df.to_string(index=False)
    
    prompt_template = f"""
    당신은 전문적인 재정 조언을 해주는 AI 가계부 분석가입니다.
    사용자는 **최근 여러 영수증**을 통해 총 {total_amount:,.0f}만큼 지출했습니다.
    아래는 이 **전체 지출**의 카테고리별 요약 데이터입니다.
    
    --- 지출 요약 데이터 ---
    {summary_text}
    ---
    
    다음 지침을 따라 친근하고 공손한 말투로 분석 리포트를 생성해 주세요:
    1. 이 전체 지출의 주요 특징 (예: 가장 큰 지출 카테고리)을 한 줄로 요약해 주세요.
    2. 사용자에게 도움이 될 만한 친절한 조언이나 코멘트 (예: 다음 지출 관리 방향)를 2~3줄로 제공해 주세요.
    3. 응답은 오직 분석 내용만 포함해야 하며, 인사말이나 추가 설명 없이 바로 요약부터 시작하세요.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_template],
        )
        return response.text
        
    except Exception as e:
        st.error(f"AI 분석 리포트 생성 중 오류가 발생했습니다: {e}")
        return "분석 리포트를 생성하지 못했습니다."


# ----------------------------------------------------------------------
# --- 3. Streamlit UI 및 메인 로직 ---
# ----------------------------------------------------------------------

# 1. 파일 업로더 (단일 파일 모드로 유지)
uploaded_file = st.file_uploader(
    "📸 분석할 영수증 사진(jpg, png)을 하나씩 업로드해 주세요. (데이터가 누적됩니다)", 
    type=['jpg', 'png', 'jpeg'],
    accept_multiple_files=False 
)


if uploaded_file is not None:
    # 2. 업로드된 파일의 고유 ID 생성 (중복 분석 방지용)
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    is_already_analyzed = any(s.get('id') == file_id for s in st.session_state.all_receipts_summary)

    # 3. 파일 미리보기 및 분석 버튼
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ 업로드된 영수증")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True) 

    with col2:
        st.subheader("📊 분석 및 기록")
        
        # 이미 분석된 파일일 경우 버튼 비활성화 및 경고 표시
        if is_already_analyzed:
            st.warning("⚠️ 이 영수증은 이미 분석되어 기록에 추가되었습니다. 다른 파일을 올려주세요.")
            analyze_button = st.button("✨ 영수증 분석 시작하기", disabled=True)
        else:
            analyze_button = st.button("✨ 영수증 분석 시작하기")


        # 4. 분석 버튼 클릭 시 실행
        if analyze_button and not is_already_analyzed:
            
            with st.spinner('AI가 영수증을 꼼꼼히 읽고 있습니다...'):
                
                # Gemini 분석 호출
                json_data_text = analyze_receipt_with_gemini(image)

                if json_data_text:
                    try:
                        # JSON 코드 블록만 추출하는 방어 로직
                        if json_data_text.strip().startswith("```json"):
                            json_data_text = json_data_text.strip().lstrip("```json").rstrip("```").strip()
                        
                        receipt_data = json.loads(json_data_text)
                        
                        # 금액 관련 데이터 타입 정규화 (Pandas 처리를 위해)
                        if not isinstance(receipt_data.get('total_amount'), (int, float)):
                             receipt_data['total_amount'] = np.nan

                        # --- 메인 정보 표시 ---
                        st.success("✅ 분석 완료! 아래 가계부 데이터를 확인해 보세요.")
                        
                        currency_unit = receipt_data.get('currency_unit', '').strip()
                        display_unit = currency_unit if currency_unit else '원'
                        total_amount = receipt_data.get('total_amount', 0)
                        
                        st.markdown(f"**🏠 상호명:** {receipt_data.get('store_name', '정보 없음')}")
                        st.markdown(f"**📅 날짜:** {receipt_data.get('date', '정보 없음')}")
                        st.subheader(f"💰 총 결제 금액: {total_amount:,.0f} {display_unit}")
                        st.markdown("---")


                        # --- 품목별 데이터프레임 생성 및 누적 ---
                        if 'items' in receipt_data and receipt_data['items']:
                            items_df = pd.DataFrame(receipt_data['items'])
                            
                            # 데이터 정규화 및 총 지출 계산
                            items_df.columns = ['품목명', '단가', '수량', 'AI 카테고리']
                            items_df['단가'] = pd.to_numeric(items_df['단가'], errors='coerce').fillna(0)
                            items_df['수량'] = pd.to_numeric(items_df['수량'], errors='coerce').fillna(1)
                            items_df['총 지출'] = items_df['단가'] * items_df['수량']
                            
                            # ** 누적 저장 **
                            st.session_state.all_receipts_items.append(items_df)
                            st.session_state.all_receipts_summary.append({
                                'id': file_id, # 중복 방지 ID
                                'filename': uploaded_file.name,
                                '상호': receipt_data.get('store_name', 'N/A'),
                                '총액': total_amount,
                                '통화': display_unit,
                                '날짜': receipt_data.get('date', 'N/A')
                            })

                            st.subheader("🛒 품목별 상세 내역")
                            st.dataframe(items_df, use_container_width=True, hide_index=True)
                            st.success(f"🎉 {uploaded_file.name}의 데이터가 누적 기록에 성공적으로 추가되었습니다!")

                        else:
                            st.warning("분석 결과에서 품목 리스트를 찾을 수 없습니다.")

                    except json.JSONDecodeError:
                        st.error("❌ Gemini 분석 결과가 올바른 JSON 형식이 아닙니다. (JSON 파싱 오류)")
                    except Exception as e:
                        st.error(f"데이터 처리 중 예상치 못한 오류 발생: {e}")
                else:
                    st.error("분석을 완료하지 못했습니다. 다시 시도해 주세요.")


# ----------------------------------------------------------------------
# --- 4. 누적 데이터 분석 섹션 (항상 표시) ---
# ----------------------------------------------------------------------

if st.session_state.all_receipts_items:
    st.markdown("---")
    st.title("📚 누적된 전체 지출 분석 리포트")

    # 1. 전체 품목 데이터프레임 생성
    all_items_df = pd.concat(st.session_state.all_receipts_items, ignore_index=True)
    
    st.subheader(f"({len(st.session_state.all_receipts_items)}개 영수증) 통합 데이터")
    st.dataframe(all_items_df[['품목명', '단가', '수량', 'AI 카테고리', '총 지출']], use_container_width=True, hide_index=True)

    # 2. 카테고리별 집계 및 시각화
    category_summary = all_items_df.groupby('AI 카테고리')['총 지출'].sum().reset_index()
    category_summary.columns = ['카테고리', '금액']
    
    st.markdown("---")
    st.subheader("💰 전체 누적 카테고리별 지출 요약")
    st.dataframe(category_summary, use_container_width=True, hide_index=True)
    st.bar_chart(category_summary.set_index('카테고리'))
    
    # 3. AI 분석 리포트 생성
    st.markdown("---")
    st.subheader("🤖 AI 분석 전문가의 전체 지출 조언")
    
    total_spent = category_summary['금액'].sum()
    
    ai_report = generate_ai_analysis(
        summary_df=category_summary,
        store_name="다수 상점",
        total_amount=total_spent
    )
    
    st.info(ai_report)
    
    # 4. 기록 초기화 및 다운로드 버튼
    st.markdown("---")
    
    @st.cache_data
    def convert_df_to_csv(df):
        # 전체 데이터프레임을 CSV 형식으로 변환
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = convert_df_to_csv(all_items_df)
    
    st.download_button(
        label="⬇️ 전체 누적 가계부 데이터 (CSV) 다운로드",
        data=csv,
        file_name=f"all_receipts_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

    if st.button("🧹 기록 초기화", help="앱에 누적된 모든 영수증 분석 기록을 지웁니다."):
        st.session_state.all_receipts_items = []
        st.session_state.all_receipts_summary = []
        st.experimental_rerun()
