import streamlit as st
from transformers import pipeline
import torch
import re

# Header with emoji and styled title
# st.markdown(
#     """
#     <div style="text-align:center; margin-bottom:20px;">
#       <span style="font-size:4rem;">🤗</span>
#     </div>
#     <div style="text-align:center; margin-bottom:10px;">
#       <h1 style="margin:0; font-size:1.6rem;">
#         Text Generator with <span style="color:#FF6D00;">Kakao Kanana-Nano 2.1B Instruct</span>
#       </h1>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <div style="text-align:center; margin-bottom:20px;">
      <h1 style="display:inline; font-size:1.6rem; margin:0;">
        🤗 AI Assistant with <span style="color:#FF6D00;">Kakao Kanana-Nano 2.1B Instruct</span>
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
        torch_dtype=torch.bfloat16,
    )
translator = st.cache_resource(load_pipeline)()

# Prompt input section
st.markdown("##### 프롬프트를 입력하세요:")
text_input = st.text_area(
    label="", 
    placeholder="예: 로봇이 인간처럼 말하게 되는 이야기 써줘", 
    height=150
)

# Generation options sliders
st.markdown("##### 생성 옵션")
max_new_tokens = st.slider("생성할 토큰 수", min_value=10, max_value=200, value=100, step=1)
temperature = st.slider("Temperature (창의성)", min_value=0.10, max_value=1.50, value=0.80, step=0.01)
top_k = st.slider("Top-k", min_value=0, max_value=100, value=50, step=1)
top_p = st.slider("Top-p (nucleus)", min_value=0.00, max_value=1.00, value=0.95, step=0.01)

# Generate button and output
generate = st.button("텍스트 생성")
if generate:
    if not text_input.strip():
        st.warning("프롬프트를 입력해주세요.")
    else:
        with st.spinner("생성 중..."):
            out = translator(
                text_input.strip(),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_full_text=False 
            )[0]
            
            # 모델이 생성한 순수 텍스트만 가져옵니다
            full_text = out.get("generated_text", "").strip()
            
            # ——— 여기가 핵심: “작성해줘” 한 줄만 잘라내기 ———
            # 정규표현식으로 첫 번째 문장부(마침표·개행 포함)를 통째로 제거
            result = re.sub(r'^[^\.]*\.\s*', '', full_text, count=1).strip()
            
            # Normalize code fences: ensure all fences specify python
            result = re.sub(r'```(?:\w*)', '```python', result, count=1).strip()

        st.markdown("##### 생성된 결과")    
        st.markdown(result)
