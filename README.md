# HuggingFace 실습 모음 (BIPA-AI-2025)

Hugging Face Transformers 기반 한국어 LLM 실습 코드 모음입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `Simple_HF_Transformer_Pipleline.ipynb` | Transformers `pipeline` 기본 사용 예제 |
| `huggingFaceIntro-pipeline.ipynb` | Hugging Face 입문 / pipeline 소개 |
| `kanana_-nano-21b-base_app.py` | Kakao **Kanana-Nano 2.1B base** 모델 텍스트 생성 Streamlit 앱 |
| `kanan-nano-21b-instruct-prompt-app.py` | Kanana-Nano 2.1B **instruct** 프롬프트 앱 |
| `kanan-nano-21b-instruct-trans.ipynb` | Kanana-Nano 2.1B instruct 노트북 |
| `kogpt2_text_generator.ipynb` | KoGPT2 텍스트 생성 예제 |
| `트랜스포머로 무엇을 할 수 있나요.ipynb` | Transformers `pipeline` 8종(감정분석·zero-shot·생성·fill-mask·NER·QA·요약·번역) 튜토리얼 |

## 실행 방법 (Streamlit 앱)

```bash
streamlit run kanana_-nano-21b-base_app.py
streamlit run kanan-nano-21b-instruct-prompt-app.py
```

> 모델(`kakaocorp/kanana-nano-2.1b-base`, `kakaocorp/kanana-nano-2.1b-instruct`)은 최초 실행 시 Hugging Face Hub에서 자동 다운로드됩니다.
> 두 모델 모두 컨텍스트 길이는 **8192 토큰**(입력+출력 합산)입니다.

## 변경 내역 (Changelog)

### 2026-06-17 — `트랜스포머로 무엇을 할 수 있나요.ipynb` 수정
- **NER**: 5.x에서 제거된 `grouped_entities=True` → `aggregation_strategy="simple"`로 교체
- **한국어 NER**: 파인튜닝 안 된 `kykim/bert-kor-base`(랜덤 헤드 → 무의미한 결과)를 실제 NER 모델 `Leo97/KoELECTRA-small-v3-modu-ner`로 교체, 출력 갱신
- **pip install 셀**: 사전 구성 환경을 깨뜨릴 수 있어(fsspec/cuda 충돌) 주석 처리
- **QA·요약·번역 재작성**: 해당 `pipeline`들은 transformers 5.0에서 제거됨 → `AutoModelForQuestionAnswering` / `AutoModelForSeq2SeqLM` 직접 호출로 재작성해 transformers 5.3에서 동작
  - 한국어 요약(kobart): `AutoTokenizer`가 잘못된 `RobertaTokenizer`를 로드해 출력이 깨지던 문제 → `PreTrainedTokenizerFast` 명시로 정상화
- **검증**: transformers 5.3에서 전체 셀 실행(에러 0), 모든 출력 재생성
- **정리**: 오타(`strat`→`start`, `단의의`→`단어의`), 커널 `torch_211_env` 지정

### 2026-06-17 — 파일명 변경
- `kanan-nano-21b-instruct-prompt-aPP.py` → `kanan-nano-21b-instruct-prompt-app.py` (소문자 `app`로 통일)

### 2026-06-17 — `kanan-nano-21b-instruct-prompt-app.py` 수정
- **Chat template 적용**: 입력을 평문 대신 메시지 형식(`[{"role":"user", ...}]`)으로 전달해 instruct 모델이 지시를 제대로 따르도록 수정
- **출력 손상 버그 제거**: 정상 응답까지 잘라먹던 정규식 2개(첫 마침표까지 삭제, 코드펜스 강제 치환)를 제거하고, 반환된 대화의 마지막 assistant 메시지에서 응답 추출
- **dtype**: deprecated된 `torch_dtype=` → `dtype=` 으로 교체 (transformers 5.x 대응)
- **샘플링**: `top_p` 슬라이더 최소값 `0.0` → `0.05` (degenerate 방지)
- **토큰 한계**: 생성 토큰 슬라이더 `10~200` → `10~4096`(기본 256), 컨텍스트(8192) 초과 방지용 동적 클램프(`8192 - 프롬프트_토큰`) 추가
- **안정성**: 모델 로딩을 `try/except` + `st.error`/`st.stop` 으로 처리
- **정리**: `text_area` 라벨 접근성(`label_visibility="collapsed"`), 미사용 import 및 주석 오타 정리

### 2026-06-17 — `kanana_-nano-21b-base_app.py` 수정
- **모델 경로**: 로컬 경로 `./local-kanana` → Hub 이름 `kakaocorp/kanana-nano-2.1b-base`로 변경 (Hub 자동 다운로드)
- **dtype**: deprecated된 `torch_dtype=` → `dtype=` 으로 교체 (transformers 5.x 대응)
- **출력**: 입력 프롬프트를 제외하고 새로 생성된 토큰만 디코딩하도록 수정 (프롬프트 중복 제거)
- **샘플링**: `top_p` 슬라이더 최소값 `0.0` → `0.05` (degenerate 샘플링 방지)
- **토큰 한계**: 생성 토큰 슬라이더 `10~200` → `10~4096`(기본 256), 컨텍스트(8192) 초과 방지용 동적 클램프(`8192 - 프롬프트_토큰`) 추가
- **안정성**: 모델 로딩을 `try/except` + `st.error`/`st.stop` 으로 감싸 실패 시 앱이 트레이스백으로 죽지 않도록 처리
- **정리**: 미사용 `AutoConfig` import 제거, Kanana 로고/주석 오타 정리
