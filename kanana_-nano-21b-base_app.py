# Kanana 2.1B BASE MODEL

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Kanana 텍스트 생성기", layout="wide")

st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://huggingface.co/kakaocorp/kanana-nano-2.1b-base/resolve/main/assets/logo/kanana-logo.png" alt="Kanana Logo" width="120">
        <h1 style='font-size: 28px; margin-top: 10px;'>Text Generator with <span style="color:#ff6600;">Kakao Kanana-Nano 2.1B</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

prompt = st.text_area("프롬프트를 입력하세요:", "한국의 수도는")
max_tokens  = st.slider("생성할 토큰 수", 10, 4096, 256)
temperature = st.slider("Temperature (창의성)", 0.1, 1.5, 0.8)
top_k       = st.slider("Top-k", 0, 100, 50)
top_p       = st.slider("Top-p (nucleus)", 0.05, 1.0, 0.95)  # 0.0은 degenerate → 최소 0.05
generate_button = st.button("텍스트 생성")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CONTEXT = 8192  # 모델 컨텍스트 한계(max_position_embeddings): 입력+출력 합산

@st.cache_resource
def load_model():
    # Hub에서 자동 다운로드 (로컬 캐시 사용). 로컬 경로 쓰려면 "./local-kanana"로 교체
    MODEL_NAME = "kakaocorp/kanana-nano-2.1b-base"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )

    # pad_token 을 eos_token 으로 설정
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # transformers 5.x: torch_dtype → dtype (GPU 메모리 절약)
        device_map=None
    )
    # model에도 pad_token_id 설정
    model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    return tokenizer, model

# 모델 로딩 실패 시 앱이 트레이스백으로 죽지 않도록 방어
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"모델 로딩에 실패했습니다: {e}")
    st.stop()

if generate_button:
    if not prompt.strip():
        st.error("프롬프트를 입력해 주세요.")
    else:
        with st.spinner("생성 중..."):
            # 입력 토큰화 및 디바이스로 이동
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # 컨텍스트 한계(8192) 초과 방지: 프롬프트 길이를 빼서 생성 토큰 수를 동적으로 제한
            prompt_len = inputs["input_ids"].shape[1]
            safe_max = max(1, min(max_tokens, MAX_CONTEXT - prompt_len))
            if safe_max < max_tokens:
                st.warning(f"프롬프트({prompt_len}토큰)가 길어 생성 토큰을 {safe_max}개로 제한했습니다 "
                           f"(모델 컨텍스트 {MAX_CONTEXT}토큰).")
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=safe_max,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            # 입력 프롬프트를 제외하고 새로 생성된 토큰만 디코딩
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        st.markdown("### 생성된 텍스트:")
        st.success(output_text)
