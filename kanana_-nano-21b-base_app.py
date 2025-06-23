# Kanan 2.1B BASE MODEL

import streamlit as st
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Kanana 텍스트 생성기", layout="wide")
# st.title("Text Generator with Kanana-Nano 2.1b base model")

# st.markdown(
#     "<h1 style='font-size: 28px;'>Text Generator with Kakao Kanana-Nano 2.1b base model</h1>",
#     unsafe_allow_html=True
# )

# https://huggingface.co/kakaocorp/kanana-nano-2.1b-base/resolve/main/assets/logo/kanana-logo.png

st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" alt="Hugging Face Logo" width="120">
        <h1 style='font-size: 28px; margin-top: 10px;'>Text Generator with <span style="color:#ff6600;">Kakao Kanana-Nano 2.1B</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)



prompt = st.text_area("프롬프트를 입력하세요:", "한국의 수도는")
max_tokens  = st.slider("생성할 토큰 수", 10, 200, 50)
temperature = st.slider("Temperature (창의성)", 0.1, 1.5, 0.8)
top_k       = st.slider("Top-k", 0, 100, 50)
top_p       = st.slider("Top-p (nucleus)", 0.0, 1.0, 0.95)
generate_button = st.button("텍스트 생성")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # MODEL_NAME = "kakaocorp/kanana-nano-2.1b-base"  # 또는 "./local-kanana", kakaocorp/kanana-1.5-8b-base
    MODEL_NAME = "./local-kanana"  # 또는 "./local-kanana", kakaocorp/kanana-1.5-8b-base
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # pad_token 을 eos_token 으로 설정
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # GPU 메모리 절약
        device_map=None
    )
    # model에도 pad_token_id 설정
    model.config.pad_token_id = tokenizer.eos_token_id
    
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model()
# model.to(device)

if generate_button:
    if not prompt.strip():
        st.error("프롬프트를 입력해 주세요.")
    else:
        with st.spinner("생성 중..."):
            # 입력 토큰화 및 디바이스로 이동
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                    # max_time=20  # 필요시 재활성화
                )
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        st.markdown("### 생성된 텍스트:")
        st.success(output_text)
