# HuggingFace 실습 모음 (BIPA-AI-2025)

Hugging Face Transformers 기반 한국어 LLM 실습 코드 모음입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `Simple_HF_Transformer_Pipleline.ipynb` | Transformers `pipeline` 기본 사용 예제 |
| `huggingFaceIntro-pipeline.ipynb` | Hugging Face 입문 / pipeline 소개 |
| `kanana_-nano-21b-base_app.py` | Kakao **Kanana-Nano 2.1B base** 모델 텍스트 생성 Streamlit 앱 |
| `kanan-nano-21b-instruct-prompt-aPP.py` | Kanana-Nano 2.1B **instruct** 프롬프트 앱 |
| `kanan-nano-21b-instruct-trans.ipynb` | Kanana-Nano 2.1B instruct 노트북 |
| `kogpt2_text_generator.ipynb` | KoGPT2 텍스트 생성 예제 |

## 실행 방법 (Streamlit 앱)

```bash
streamlit run kanana_-nano-21b-base_app.py
```

> 모델(`kakaocorp/kanana-nano-2.1b-base`)은 최초 실행 시 Hugging Face Hub에서 자동 다운로드됩니다.

## 변경 내역 (Changelog)

### 2026-06-17 — `kanana_-nano-21b-base_app.py` 수정
- **모델 경로**: 로컬 경로 `./local-kanana` → Hub 이름 `kakaocorp/kanana-nano-2.1b-base`로 변경 (Hub 자동 다운로드)
- **dtype**: deprecated된 `torch_dtype=` → `dtype=` 으로 교체 (transformers 5.x 대응)
- **출력**: 입력 프롬프트를 제외하고 새로 생성된 토큰만 디코딩하도록 수정 (프롬프트 중복 제거)
- **샘플링**: `top_p` 슬라이더 최소값 `0.0` → `0.05` (degenerate 샘플링 방지)
- **안정성**: 모델 로딩을 `try/except` + `st.error`/`st.stop` 으로 감싸 실패 시 앱이 트레이스백으로 죽지 않도록 처리
- **정리**: 미사용 `AutoConfig` import 제거, Kanana 로고/주석 오타 정리
