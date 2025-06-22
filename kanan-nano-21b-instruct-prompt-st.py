import streamlit as st
from transformers import pipeline
import torch

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40" style="margin-right: 10px;">
        <h1 style="margin: 0; font-size: 1.8em;">Text Generator with Kanana-Nano-Instruct-2.1B</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 모델 로딩
@st.cache_resource
def load_pipeline():
    model_name = "kakaocorp/kanana-nano-2.1b-instruct"
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        device=0,
        torch_dtype=torch.bfloat16,
    )

translator = load_pipeline()

# UI 시작
# st.title("Text Generator with Kanana-Nano-Instruct-2.1B")
st.markdown("프롬프트를 입력하면 텍스트를 생성합니다. (예: 요약, 이야기, 설명, 창작 등 자유롭게!)")

# 고급 생성 옵션
# st.subheader("고급 생성 옵션")
st.markdown(
    """
    <h3 style="font-size: 1.0rem; font-weight: 500; margin-bottom: 0.5rem;">
        고급 생성 옵션
    </h3>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("max_length (토큰 수)", 20, 300, 100, 10)
    temperature = st.slider("Temperature (창의성)", 0.1, 2.0, 1.0, 0.1)
with col2:
    top_k = st.slider("Top-K", 0, 100, 50, 1)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.95, 0.01)
    repetition_penalty = st.slider("repetition_penalty (반복 억제)", 1.0, 2.0, 1.2, 0.1)

# 프롬프트 입력
# st.subheader("프롬프트 입력")
st.markdown(
    """
    <h3 style="font-size: 1.0rem; font-weight: 500; margin-bottom: 0.5rem;">
        프롬프트 입력
    </h3>
    """,
    unsafe_allow_html=True
)

text_input = st.text_area(
    "프롬프트를 입력하세요:",
    placeholder="예: '로봇이 인간처럼 말하게 되는 이야기 써줘' 또는 '이 문장을 요약해줘: ...'",
    height=200
)

# 생성
if st.button("생성하기"):
    if text_input.strip() == "":
        st.warning("프롬프트를 입력해주세요.")
    else:
        with st.spinner("텍스트 생성 중..."):
            prompt = text_input.strip()  # 일반 프롬프트 사용
            output = translator(
                prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )[0]
            full_text = output.get("generated_text", output.get("text", "")).strip()

            # 입력 프롬프트 이후 결과만 추출 (가능한 경우)
            result = full_text.split(prompt, 1)[-1].strip()

        # 출력 (문단 정렬)
        st.success("생성된 결과:")
        st.markdown(
            f"""
            <div style="padding: 1em; background-color: #f9f9f9;
                        border: 1px solid #ccc; border-radius: 6px;
                        color: #1a1a1a; font-weight: 500; font-size: 1rem;
                        line-height: 1.8; text-align: justify;">
                {result.replace('\n', ' ')}
            </div>
            """,
            unsafe_allow_html=True
        )
        
