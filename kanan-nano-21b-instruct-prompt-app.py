# Simple AI Assistant with Kakao's Kanana-nano-2.1b-Instruct Model
import streamlit as st
from transformers import pipeline
import torch

MAX_CONTEXT = 8192  # 모델 컨텍스트 한계(max_position_embeddings): 입력+출력 합산

st.markdown(
    """
    <div style="text-align:center; margin-bottom:20px;">
      <h1 style="display:inline; font-size:1.6rem; margin:0;">
        🤗 Simple AI Assistant with <span style="color:#FF6D00;">Kakao Kanana-Nano 2.1B Instruct</span>
      </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model pipeline once
def load_pipeline():
    model_name = "kakaocorp/kanana-nano-2.1b-instruct"
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,  # transformers 5.x: torch_dtype → dtype
    )

# 모델 로딩 실패 시 앱이 트레이스백으로 죽지 않도록 방어
try:
    translator = st.cache_resource(load_pipeline)()
except Exception as e:
    st.error(f"모델 로딩에 실패했습니다: {e}")
    st.stop()

# Prompt input section
st.markdown("##### 프롬프트를 입력하세요:")
text_input = st.text_area(
    label="프롬프트",
    label_visibility="collapsed",
    placeholder="예: 로봇이 인간처럼 말하게 되는 이야기 써줘",
    height=150
)

# Generation options sliders
st.markdown("##### 생성 옵션")
max_new_tokens = st.slider("생성할 토큰 수", min_value=10, max_value=4096, value=256, step=1)
temperature = st.slider("Temperature (창의성)", min_value=0.10, max_value=1.50, value=0.80, step=0.01)
top_k = st.slider("Top-k", min_value=0, max_value=100, value=50, step=1)
top_p = st.slider("Top-p (nucleus)", min_value=0.05, max_value=1.00, value=0.95, step=0.01)  # 0.0은 degenerate → 최소 0.05

# Generate button and output
generate = st.button("텍스트 생성")
if generate:
    if not text_input.strip():
        st.warning("프롬프트를 입력해주세요.")
    else:
        with st.spinner("생성 중..."):
            # instruct 모델: chat template을 적용하도록 메시지 형식으로 전달
            # (파이프라인이 role 기반 입력을 받으면 chat template을 자동 적용)
            messages = [{"role": "user", "content": text_input.strip()}]
            # 컨텍스트 한계(8192) 초과 방지: chat template 적용 후 프롬프트 토큰 수를 빼서 동적 제한
            prompt_len = len(translator.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True))
            safe_max = max(1, min(max_new_tokens, MAX_CONTEXT - prompt_len))
            if safe_max < max_new_tokens:
                st.warning(f"프롬프트({prompt_len}토큰)가 길어 생성 토큰을 {safe_max}개로 제한했습니다 "
                           f"(모델 컨텍스트 {MAX_CONTEXT}토큰).")
            out = translator(
                messages,
                max_new_tokens=safe_max,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[0]

            # 채팅 입력의 경우 generated_text는 전체 대화(list[dict])로 반환됨
            # → 마지막 assistant 메시지 content만 추출
            generated = out.get("generated_text", "")
            if isinstance(generated, list):
                result = generated[-1]["content"].strip()
            else:
                result = str(generated).strip()

        st.markdown("##### 생성된 결과")
        st.markdown(result)
