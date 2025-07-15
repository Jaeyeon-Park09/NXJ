# RAG 기반 Report Generation 시스템

Report Generation Tool을 탑재한 Agent 기반 RAG 시스템

LangChain의 딥리서치 오픈소스 [open\_deep\_research](https://github.com/langchain-ai/open_deep_research)를 기반으로 제작함

테스트용 질문: *의료용 휠체어란 무엇인가요? 특히 수동식 휠체어의 정의는 무엇인가요? 보고서를 작성해주세요.*


## 주요 기능

- 문서 검색 및 답변 생성
- Function Calling을 통한 각종 도구 실행
- Report Generation Tool으로 RAG 기반 리서치 보고서 출력


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

* `llm_repo_opr/embedding` 디렉토리에 임베딩 JSON 파일 배치해야 코드 실행 가능


## 파일 구조 (주요 파일)

- `embedding/`: 임베딩 데이터셋 (.json) - 미포함(따로 파일 집어넣어야 모델 구동 가능)
- `test_embed_data.py`: 'embedding/' 디렉토리 내 JSON 파일 개요 파악
- `test_report.py`: RAG 기반 보고서 생성 기능 테스트
- `utils/`: 도구 및 유틸리티 함수
  - `tool_search.py`: 문서 검색 도구
  - `tool_answer.py`: 답변 생성 도구
  - `tool_section.py`: 보고서 목차 생성 도구
  - `tool_report.py`: 보고서 본문 생성 도구
