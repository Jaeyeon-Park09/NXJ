# RAG 시스템 (nxj_llm)

Agent 기반 RAG (Retrieval-Augmented Generation) 시스템 준비 중

## 주요 기능

- 문서 검색 및 답변 생성
- LangGraph 기반 에이전트 시스템
- Function Calling을 통한 도구 실행
- Gradio 웹 인터페이스 -> 프론트엔드 미구현

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. OpenAI API 키 설정:
# YOUR_API_KEY에 API 키 입력(ex. sk-000100...)
```bash
export OPENAI_API_KEY='YOUR_API_KEY'
```
or
# bashrc에 API 키 정보 입력
```bash
echo "export OPENAI_API_KEY='YOUR_API_KEY'" >> ~/.bashrc
```
# 절대 코드 내에 API 키를 입력하고, 이를 공유하지 마세요.

3. 임베딩 데이터 준비:
- `embeddings/` 디렉토리에 임베딩 파일 배치
- 또는 `utils/load_all_embeddings.py` 수정
- 실습생 폴더 /shared/embedding

## 실행 방법

### 시스템 테스트
```bash
python test_rag.py
```

### gradio 실행
```bash
python run_gradio.py
```

## 파일 구조

- `graph_builder.py`: 메인 그래프 로직
- `state_types.py`: State 타입 정의
- `run_gradio.py`: Gradio UI
- `registry.py`: 도구 레지스트리
- `utils/`: 도구 및 유틸리티 함수
  - `tool_search.py`: 문서 검색 도구
  - `tool_answer.py`: 답변 생성 도구
  - `tool_diagnose.py`: 시스템 진단 도구