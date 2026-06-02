# 논문 PDF Q&A 챗봇

> 한국어 학술 논문 PDF를 벡터 DB에 저장하고, 자연어 질문으로 내용을 검색·답변하는 RAG 시스템

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-LLM-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 시스템 개요

**ingest 파이프라인 — PDF를 벡터 DB에 저장:**

```
[data/*.pdf]
    ↓ PyMuPDF
[페이지별 텍스트]
    ↓ RecursiveCharacterTextSplitter (500자, overlap 50자)
[청크 리스트]
    ↓ ko-sroberta (로컬, 무료)
[768차원 벡터]
    ↓ ChromaDB.upsert()
[chroma_db/ 로컬 저장]
```

**query 파이프라인 — 질문에 답변:**

```
[사용자 질문]
    ↓ ko-sroberta (동일 모델)
[질문 벡터]
    ↓ ChromaDB.query(top-3)
[관련 청크 + 출처(파일명·페이지)]
    ↓ PromptTemplate
[컨텍스트 + 질문]
    ↓ Gemini 2.5 Flash
[답변 + 출처 표시]
```

---

## 기술 스택

| 분류 | 라이브러리 | 비고 |
|------|-----------|------|
| PDF 파싱 | PyMuPDF | 페이지 번호를 metadata로 추출 가능 |
| 청킹 | LangChain RecursiveCharacterTextSplitter | 문장 경계 우선 분할 |
| 임베딩 | jhgan/ko-sroberta-multitask | 한국어 특화, 로컬 실행(무료) |
| 벡터 DB | ChromaDB | 로컬 영구 저장, 별도 서버 불필요 |
| LLM | Gemini 2.5 Flash | 무료 티어, 카드 등록 불필요 |
| 평가 | RAGAS | 청크 크기별 검색 품질 정량 평가 |

---

## 파일 구조

```
paper_chatbot/
├── .env                    # GOOGLE_API_KEY (gitignore)
├── .gitignore
├── config.py               # 전역 설정값 (모델명, 청크 크기, 경로 등)
├── embeddings.py           # 임베딩 모델 공통 모듈
├── db.py                   # ChromaDB 연결 공통 모듈
├── ingest.py               # PDF → 청크 → 벡터 저장
├── query.py                # 질문 → 검색 → 답변 출력
├── ingest_experiment.py    # 청킹 비교 실험용 ingest
├── query_experiment.py     # 청킹 비교 실험용 query
├── evaluate_ragas.py       # RAGAS 정량 평가
├── ragas_results.csv       # 평가 결과
├── peek.py                 # DB 내 저장 청크 확인용 유틸
├── chroma_db/              # 벡터 DB 로컬 저장소 (gitignore)
└── data/                   # 논문 PDF 원본 (gitignore)
```

---

## 실험 결과

### 청킹 전략 비교 실험

청크 크기(300자 / 500자 / 700자)에 따라 검색·답변 품질이 어떻게 달라지는지 동일 질문 5개로 주관 평가.

**질문별 답변 품질 (1=나쁨, 5=좋음):**

| 질문 | 300자 | 500자 | 700자 | 비고 |
|------|:-----:|:-----:|:-----:|------|
| Q1 — 키워드 vs 임베딩 검색 차이점 | 4 | **5** | 2 | 700자는 핵심 문장이 주변 내용에 묻힘 |
| Q2 — 프롬프트 제약 방식 | **5** | 4 | 4 | 300자가 수치까지 포착 |
| Q3 — 공공서비스 적용 장단점 | 1 | 2 | 1 | 논문에 해당 내용 없음 (정상 동작) |
| Q4 — 임베딩 모델 역할 | 1 | 2 | **4** | 긴 설명이 큰 청크에서만 포착 |
| Q5 — 비용 비교 (할루시네이션 체크) | 5 | 5 | 5 | 전 크기 "없다"고 정확히 답변 |

> 질문 유형에 따라 최적 청크 크기가 다름을 확인. 짧고 구체적인 수치/정의는 300자, 일반 질문은 500자, 긴 설명이 연속된 개념은 700자가 유리. 전체적으로 500자가 가장 안정적.

### RAGAS 정량 평가

RAGAS 프레임워크로 Faithfulness와 Answer Relevancy를 청크 크기별로 자동 측정.

**측정 결과 (유효 수치 기준):**

| 청크 크기 | Faithfulness | Answer Relevancy (유효 평균) | 유효 행 수 |
|---------|:------------:|:---------------------------:|:--------:|
| 300자 | 1.00 | 0.891 | 2 / 5 |
| 500자 | 0.933 | **0.919** | 3 / 5 |
| 700자 | 1.00 | 0.938 | 2 / 5 |

> - **Faithfulness**: 평가 LLM이 생성 LLM과 동일(Gemini)하여 자기 채점 구조 — 판별력 없음
> - **Answer Relevancy**: RAGAS 내부의 역질문 생성이 한국어에서 불안정하여 15행 중 7행이 0.0으로 측정됨. 이는 답변 품질 문제가 아닌 **한국어 처리 한계**
> - 유효 수치 기준으로는 500자 우세 경향이 주관 평가와 일치

---

## 실행 방법

### 1. 저장소 클론

```powershell
git clone https://github.com/eze-hong/paper_chatbot.git
cd paper_chatbot
```

### 2. 가상환경 생성 및 활성화

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. 패키지 설치

```powershell
# torch는 CPU 버전을 먼저 설치 (순서 중요)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. `.env` 파일 생성

```powershell
New-Item .env
```

`.env` 파일에 아래 내용을 추가한다. Gemini API 키는 [Google AI Studio](https://aistudio.google.com)에서 무료로 발급받을 수 있다.

```
GOOGLE_API_KEY=발급받은키
```

### 5. PDF 파일 추가

```powershell
# data/ 폴더에 논문 PDF를 넣는다
Copy-Item "논문.pdf" data\
```

### 6. ingest 실행

```powershell
python ingest.py
```

> ko-sroberta 모델은 첫 실행 시 자동으로 다운로드된다 (약 500MB). 이후 실행부터는 로컬 캐시를 사용한다.

### 7. 질문 실행

```powershell
python query.py
```

---

## 한계 및 배운 점

### 한계

- CLI 전용 — UI 없음 (Streamlit 연결 미완)
- 단일 PDF 컬렉션 — 논문 추가 시 전체 재ingest 필요
- RAGAS 한국어 지원 불안정으로 정량 평가 일부 신뢰도 낮음

### 배운 점

- 임베딩 모델과 벡터 DB를 공통 모듈로 분리하면 변경 시 단일 지점만 수정하면 된다는 것 (`embeddings.py` / `db.py` 설계)
- 청크 크기는 "클수록 좋다 / 작을수록 좋다"가 아니라 질문 유형에 따라 최적값이 다름
- RAGAS 같은 자동 평가 프레임워크도 언어·도메인 적합성 검토가 필요함
