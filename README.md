# RAG 기반 Report Generation 시스템

Report Generation Tool을 탑재한 Agent 기반 RAG (Retrieval-Augmented Generation) 시스템 준비 중

## 주요 기능

- 문서 검색 및 답변 생성
- LangGraph 기반 에이전트 시스템
- Function Calling을 통한 도구 실행

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Ollama 모델 준비:

```bash
ollama serve
```

```bash
ollama run llama3:8b
```

* 본 프로젝트는 **llama3:8b** 모델을 기반으로 동작합니다.
* Ollama 미설치 시 [Ollama 공식 문서](https://ollama.com/) 참고

3. 임베딩 데이터 준비:

* `llm_repo_opr/embedding` 디렉토리에 임베딩 파일 배치


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
