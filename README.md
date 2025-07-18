# RAG 기반 Report Generation 시스템

Report Generation Tool을 탑재한 Agent 기반 RAG 시스템

LangChain의 딥리서치 오픈소스 [open\_deep\_research](https://github.com/langchain-ai/open_deep_research)를 기반으로 제작함


## 주요 기능

- RAG 기반 문서 탐색 및 리서치 보고서 출력
- LLM이 직접 소목차 생성 및 내용단락 구획
- LLM을 보고서 내 목차마다 따로 호출하여 본문 작성 정확도를 높임
- 유사도 검색 관련 Hyperparmeters 튜닝


## 구현 사례

예시1) *의료용 휠체어, 특히 수동식 휠체어가 무엇인지 보고서를 작성해주세요.*

<img width="978" height="735" alt="image" src="https://github.com/user-attachments/assets/b96bc400-f01a-4da8-b479-72bb62c9e581" />

예시2) *흡수성 마그네슘 합금 관련 의약품의 인허가에 대한 보고서를 작성하라.*

<img width="989" height="599" alt="image" src="https://github.com/user-attachments/assets/d3639a46-a1f9-47aa-9159-9f9c7c3aedee" />

예시3) *혈액제제 GMP 평가 기준에 대한 리포트 작성해줘.*

<img width="1005" height="509" alt="image" src="https://github.com/user-attachments/assets/75f472fc-8e1e-409d-b8fb-e886dfe03dbe" />

예시4) *국내에서 한약 제제의 신약으로 허가받기 위해 필요한 비임상 시험 항목과 관련 가이드라인은 무엇인지 보고서를 작성하세요.*
<img width="1598" height="1265" alt="image" src="https://github.com/user-attachments/assets/bdbf646b-a031-4c31-a8a0-495e76ca4d1c" />

예시5) *의약품 임상시험 결과보고서 작성 시 구성과 내용에 대한 가이드라인을 알려주세요.*
<img width="1591" height="1042" alt="image" src="https://github.com/user-attachments/assets/c5cbb666-7d43-45dc-885a-fe0f5a386b3a" />



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

- `embedding/`: 임베딩 데이터셋 (.json) - 미포함   **(따로 파일 집어넣어야 모델 구동 가능)**
- `test_embed_data.py`: 'embedding/' 디렉토리 내 JSON 파일 개요 파악
- `test_report.py`: RAG 기반 보고서 생성 기능 테스트
- `utils/`: 도구 및 유틸리티 함수
  - `tool_search.py`: 문서 검색 도구
  - `tool_answer.py`: 답변 생성 도구
  - `tool_section.py`: 보고서 목차 생성 도구
  - `tool_report.py`: 보고서 본문 생성 도구
