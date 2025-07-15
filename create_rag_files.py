#!/usr/bin/env python3
"""
RAG 시스템의 모든 수정된 파일을 생성하는 스크립트
실행: python create_rag_files.py
"""
import os
from pathlib import Path
import sys
import importlib.util
sys.path.append('./utils')

# tool_reportgen 동적 import
spec = importlib.util.spec_from_file_location("tool_reportgen", os.path.join(os.path.dirname(__file__), "utils", "tool_reportgen.py"))
if spec is not None and spec.loader is not None:
    tool_reportgen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_reportgen)
    report_generation = tool_reportgen.report_generation
else:
    raise ImportError("Cannot import tool_reportgen module.")

def create_file(filepath, content):
    """파일 생성 함수"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")

# 프로젝트 구조 생성
print("RAG 시스템 파일 생성 시작...\n")

# 1. graph_builder.py
graph_builder_content = '''from __future__ import annotations
import json
from typing import Dict, Any, List, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from utils.tool_search import search_documents
from utils.tool_answer import final_answer
from utils.tool_diagnose import tool_diagnose
from state_types import GraphState
import os
os.environ["OPENAI_API_KEY"] = ""

# Tool registry - 이름 통일
search_documents.name = "search_documents"  # 명시적으로 이름 설정
final_answer.name = "final_answer"  # 이름 통일
tool_diagnose.name = "tool_diagnose"
report_generation.name = "report_generation"

TOOLS = [search_documents, final_answer, tool_diagnose, report_generation]
tool_node = ToolNode(TOOLS)  # ToolNode 사용

# System prompt - Function Calling 방식으로 수정
system_prompt = """
당신은 대한민국 식품의약품안전처에서 의약제품 인허가를 지원하기 위해 개발된 매우 강력한 인허스턴트 nxj_llm입니다.

<role>
** 중요: 주어진 문서를 활용해 답변해야 합니다. 문서에 없는 내용을 지어내지 마세요.
** 중요: 추가 검증 전까지 이 정보는 참고용입니다.
</role>

<available_tools>
당신은 다음 도구들을 사용할 수 있습니다:
1. search_documents: 사용자 질문과 관련된 문서를 검색합니다.
   - 입력: query (문자열)
   - 출력: documents, sources, paths를 포함한 딕셔너리

2. final_answer: 검색 결과를 기반으로 최종 답변을 생성합니다.
   - 입력: answer (문자열), sources (리스트), paths (리스트)
   - 출력: 최종 답변 딕셔너리

3. tool_diagnose: 문제 발생 시 진단합니다.
   - 입력: last_input (문자열), context (딕셔너리)

4. report_generation: 사용자가 보고서/리포트 형태의 답변을 요구할 때 보고서 플래너 및 섹션 생성 프롬프트를 반환합니다.
   - 입력: topic (문자열), report_organization (문자열, 옵션), context (문자열, 옵션), feedback (문자열, 옵션)
   - 출력: 보고서 플래너 프롬프트(문자열)
</available_tools>

<instructions>
1. 사용자가 "보고서" 또는 "리포트" 형태의 답변을 요구하면 반드시 report_generation tool을 먼저 호출하세요.
2. 그 외에는 기존 Q&A 플로우(search_documents → final_answer)로 동작하세요.
3. 문서에 정보가 없으면 그 사실을 명시하세요.
</instructions>
"""

# LLM setup with function calling
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.2,
    max_tokens=2048,
).bind_functions(TOOLS)  # Function binding

# Simplified State
class State(GraphState):
    messages: List[BaseMessage] = []
    documents: List[str] = []
    sources: List[str] = []
    paths: List[str] = []
    answer: str = ""

# Agent node
def agent(state: State) -> Dict[str, Any]:
    """LLM이 다음 행동을 결정하는 노드"""
    messages = state.get("messages", [])
    
    # System prompt 추가 (첫 번째 메시지인 경우)
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # LLM 호출
    response = llm.invoke(messages)
    
    # 메시지 업데이트
    return {"messages": [response]}

# Router function
def should_continue(state: State) -> Literal["tools", "end"]:
    """다음 노드를 결정하는 라우터"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    # Function call이 있으면 tools로, 없으면 end로
    if last_message and hasattr(last_message, "additional_kwargs"):
        if last_message.additional_kwargs.get("function_call"):
            return "tools"
    
    return "end"

# Result processor
def process_result(state: State) -> Dict[str, Any]:
    """최종 결과를 처리하는 노드"""
    messages = state.get("messages", [])
    
    # 마지막 메시지에서 답변 추출
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            # final_answer 도구의 결과가 있는지 확인
            if "answer" in state and state["answer"]:
                return {
                    "messages": messages,
                    "answer": state["answer"]
                }
            # 일반 AI 메시지
            return {
                "messages": messages,
                "answer": message.content
            }
    
    return {
        "messages": messages,
        "answer": "죄송합니다. 답변을 생성할 수 없습니다."
    }

# Tool result handler
def handle_tool_result(state: State) -> Dict[str, Any]:
    """도구 실행 결과를 State에 반영"""
    messages = state.get("messages", [])
    
    # 마지막 메시지가 tool 응답인지 확인
    if messages and hasattr(messages[-1], "content"):
        try:
            # Tool 응답 파싱
            content = messages[-1].content
            if isinstance(content, str) and content.startswith("{"):
                result = json.loads(content)
                
                # search_documents 결과 처리
                if "documents" in result:
                    return {
                        "documents": result.get("documents", []),
                        "sources": result.get("sources", []),
                        "paths": result.get("paths", []),
                        "messages": messages
                    }
                
                # final_answer 결과 처리
                if "answer" in result:
                    return {
                        "answer": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "paths": result.get("paths", []),
                        "messages": messages
                    }
        except:
            pass
    
    return {"messages": messages}

# Build graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.add_node("process_result", process_result)
workflow.add_node("handle_tool_result", handle_tool_result)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": "process_result"
    }
)
workflow.add_edge("tools", "handle_tool_result")
workflow.add_edge("handle_tool_result", "agent")
workflow.add_edge("process_result", END)

# Compile
app = workflow.compile()

# Simple interface
def respond(state: dict) -> dict:
    """
    간단한 인터페이스 함수
    """
    # 입력 검증
    messages = state.get("messages", [])
    query = state.get("query", "") or state.get("last_user", "")
    
    if not messages and query:
        messages = [HumanMessage(content=query)]
    
    if not messages:
        return {
            "messages": [],
            "output": "❌ 입력 오류: 질문을 입력해 주세요.",
            "answer": ""
        }
    
    # 초기 상태 설정
    initial_state = {
        "messages": messages,
        "documents": [],
        "sources": [],
        "paths": [],
        "answer": ""
    }
    
    try:
        # Graph 실행
        result = app.invoke(initial_state)
        
        # 결과 추출
        answer = result.get("answer", "")
        if not answer:
            # 메시지에서 답변 추출 시도
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content
                    break
        
        return {
            "messages": result.get("messages", []),
            "output": answer,
            "answer": answer,
            "sources": result.get("sources", []),
            "paths": result.get("paths", [])
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 오류 진단
        diagnosis = tool_diagnose._run(
            last_input=query,
            context={"error": str(e), "state": initial_state}
        )
        
        return {
            "messages": messages,
            "output": f"❌ 시스템 오류: {str(e)}\\n\\n🔍 진단 결과:\\n{diagnosis}",
            "answer": ""
        }
'''

# 2. state_types.py
state_types_content = '''from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import BaseMessage

class GraphState(TypedDict, total=False):
    """LangGraph State 정의 - 필수 필드만 유지"""
    # 핵심 필드
    messages: List[BaseMessage]  # LangGraph 메시지 히스토리
    
    # 검색 관련 필드
    documents: List[str]  # 검색된 문서 내용
    sources: List[str]    # 문서 출처 (파일명)
    paths: List[str]      # 문서 경로
    
    # 결과 필드
    answer: str          # 최종 답변
    
    # 선택적 필드 (기존 코드 호환성)
    query: str           # 사용자 질문
    last_user: str       # 마지막 사용자 입력 (호환성)
'''

# 3. run_gradio.py
run_gradio_content = '''import gradio as gr
from graph_builder import respond as graph_respond
from langchain_core.messages import HumanMessage, AIMessage
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# 전체 대화 히스토리 관리
conversation_history: list[dict] = []

def extract_answer(response: dict) -> str:
    """응답에서 답변 추출"""
    # 1. answer 필드 확인
    if response.get("answer"):
        return response["answer"]
    
    # 2. output 필드 확인
    if response.get("output"):
        return response["output"]
    
    # 3. messages에서 마지막 AI 메시지 추출
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # JSON 형태의 응답인지 확인
            content = msg.content
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    data = json.loads(content)
                    if "answer" in data:
                        return data["answer"]
                except:
                    pass
            return content
    
    return "❌ 죄송합니다. 답변을 생성할 수 없습니다."

def respond_wrapper(message: str, chat_history: list | None = None):
    """Gradio 인터페이스를 위한 응답 함수"""
    logging.info(f"USER: {message}")
    
    if not message.strip():
        return chat_history or []
    
    # 히스토리 관리
    history = chat_history or []
    user_query = message.strip()
    
    # 대화 히스토리에 추가
    conversation_history.append({"role": "user", "content": user_query})
    
    # LangChain 메시지 형식으로 변환
    messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Graph 상태 생성
    state = {
        "messages": messages,
        "query": user_query,
        "last_user": user_query
    }
    
    try:
        # Graph 실행
        response = graph_respond(state)
        logging.debug(f"Graph response: {response}")
        
        # 답변 추출
        answer = extract_answer(response)
        
        # 대화 히스토리 업데이트
        conversation_history.append({"role": "assistant", "content": answer})
        
        # UI에 표시할 히스토리 (최근 대화만)
        visible_history = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer}
        ]
        
        logging.info(f"ASSISTANT: {answer[:100]}...")
        return visible_history
        
    except Exception as e:
        logging.exception("Error in graph_respond")
        error_msg = f"⚠️ 시스템 오류가 발생했습니다: {str(e)}"
        
        visible_history = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": error_msg}
        ]
        return visible_history

def reset_conversation():
    """대화 초기화"""
    global conversation_history
    conversation_history = []
    return []

# Gradio 인터페이스
with gr.Blocks(title="nxj_llm RAG 시스템") as demo:
    gr.Markdown("""
    # nxj_llm RAG 시스템
    
    의료제품 인허가 관련 질문을 입력하세요. 시스템이 관련 문서를 검색하여 답변을 제공합니다.
    """)
    
    chatbot = gr.Chatbot(
        height=500,
        type="messages",
        label="대화",
        elem_id="chatbot"
    )
    
    msg = gr.Textbox(
        label="질문 입력",
        placeholder="예: 의료용 휠체어란?",
        lines=2
    )
    
    with gr.Row():
        submit = gr.Button("전송", variant="primary")
        clear = gr.Button("대화 초기화")
    
    # 예제 질문들
    gr.Examples(
        examples=[
            "의료용 휠체어란?",
            "체외진단기기의 등급 분류 기준은?",
            "의료기기 임상시험 절차는?",
            "GMP 인증 요구사항은?"
        ],
        inputs=msg
    )
    
    # 이벤트 핸들러
    msg.submit(respond_wrapper, [msg, chatbot], [chatbot]).then(
        lambda: "", None, [msg]
    )
    submit.click(respond_wrapper, [msg, chatbot], [chatbot]).then(
        lambda: "", None, [msg]
    )
    clear.click(reset_conversation, None, [chatbot])
    
    # 상태 표시
    gr.Markdown("""
    ---
    💡 **사용 팁**:
    - 구체적인 질문을 입력하면 더 정확한 답변을 받을 수 있습니다.
    - 답변에는 출처 정보가 함께 제공됩니다.
    - 대화 초기화 버튼으로 새로운 대화를 시작할 수 있습니다.
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7862, 
        share=False,
        show_error=True
    )
'''

# 4. registry.py
registry_content = '''"""
도구 레지스트리
RAG 시스템에서 사용하는 모든 도구를 관리합니다.
"""
from utils.tool_search import search_documents
from utils.tool_answer import final_answer
from utils.tool_diagnose import tool_diagnose

# 활성화된 도구 목록
TOOLS = [
    search_documents,  # 문서 검색 도구
    final_answer,      # 최종 답변 생성 도구
    tool_diagnose      # 시스템 진단 도구
]

# 도구 이름으로 매핑 (선택적)
TOOL_MAP = {tool.name: tool for tool in TOOLS}

# 도구 정보 출력 (디버깅용)
def print_tool_info():
    """등록된 도구 정보를 출력합니다."""
    print("=== 등록된 도구 목록 ===")
    for tool in TOOLS:
        print(f"- {tool.name}: {tool.description}")
    print(f"총 {len(TOOLS)}개의 도구가 등록되었습니다.")

if __name__ == "__main__":
    print_tool_info()
'''

# 5. utils/tool_search.py
tool_search_content = '''import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.embed_query import embed_query
from utils.load_all_embeddings import load_all_embeddings
from pathlib import Path
from collections import defaultdict
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class SearchDocumentsTool(BaseTool):
    name: str = "search_documents"
    description: str = "쿼리에 가장 관련성이 높은 문서 청크와 메타데이터를 검색합니다."

    def _run(self, query: str) -> str:
        """
        문서 검색을 수행하고 결과를 JSON 문자열로 반환합니다.
        
        Args:
            query: 검색 쿼리
            
        Returns:
            JSON 문자열 형태의 검색 결과
        """
        try:
            logger.info(f"[SearchDocuments] Query: {query}")
            
            # 1. 임베딩 데이터 로드
            docs_all, embs_all, srcs_all = load_all_embeddings()
            logger.info(f"[SearchDocuments] Loaded {len(embs_all)} chunks from {len(set(srcs_all))} files")
            
            if len(embs_all) == 0:
                result = {
                    "documents": [],
                    "sources": [],
                    "paths": [],
                    "message": "검색할 문서가 없습니다."
                }
                return json.dumps(result, ensure_ascii=False)
            
            # 2. 쿼리 임베딩 생성
            query_vec = embed_query(query)
            if query_vec is None or not isinstance(query_vec, np.ndarray):
                result = {
                    "documents": [],
                    "sources": [],
                    "paths": [],
                    "error": "쿼리 임베딩 생성 실패"
                }
                return json.dumps(result, ensure_ascii=False)
            
            # 3. 유효한 임베딩만 필터링
            filtered = [(d, e, s) for d, e, s in zip(docs_all, embs_all, srcs_all) if e is not None]
            if not filtered:
                result = {
                    "documents": [],
                    "sources": [],
                    "paths": [],
                    "message": "유효한 임베딩이 없습니다."
                }
                return json.dumps(result, ensure_ascii=False)
            
            docs, embs, srcs = zip(*filtered)
            
            # 4. 코사인 유사도 계산
            embs_arr = np.vstack(embs)
            sims = cosine_similarity([query_vec], embs_arr)[0]
            
            logger.info(f"[SearchDocuments] Similarity - max: {sims.max():.4f}, mean: {sims.mean():.4f}")
            
            # 5. 상위 청크 선택
            top_k_chunks = min(100, len(sims))
            top_idx = sims.argsort()[-top_k_chunks:][::-1]
            
            docs = [docs[i] for i in top_idx]
            srcs = [srcs[i] for i in top_idx]
            sims = sims[top_idx]
            
            # 6. 문서별 평균 점수로 상위 파일 선정
            file_scores = defaultdict(list)
            for sim, src in zip(sims, srcs):
                file_scores[src].append(sim)
            
            avg_scores = [(src, sum(scores)/len(scores)) for src, scores in file_scores.items()]
            top_files = [src for src, _ in sorted(avg_scores, key=lambda x: x[1], reverse=True)[:10]]
            
            # 7. 최종 결과 선택 (상위 5개)
            final_results = []
            for src in top_files:
                file_chunks = [(sim, doc) for sim, doc, s in zip(sims, docs, srcs) if s == src]
                for sim, doc in sorted(file_chunks, key=lambda x: x[0], reverse=True):
                    final_results.append((sim, doc, src))
            
            # 상위 5개만 선택
            final_results = sorted(final_results, key=lambda x: x[0], reverse=True)[:5]
            
            # 8. 결과 포맷팅
            result = {
                "documents": [doc for _, doc, _ in final_results],
                "sources": [Path(src).name for _, _, src in final_results],
                "paths": [str(Path(src)) for _, _, src in final_results],
                "scores": [float(sim) for sim, _, _ in final_results]  # 점수도 포함
            }
            
            logger.info(f"[SearchDocuments] Found {len(result['documents'])} relevant documents")
            logger.info(f"[SearchDocuments] Sources: {result['sources']}")
            
            # JSON 문자열로 반환
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[SearchDocuments] Error: {e}", exc_info=True)
            result = {
                "documents": [],
                "sources": [],
                "paths": [],
                "error": f"검색 중 오류 발생: {str(e)}"
            }
            return json.dumps(result, ensure_ascii=False)

    async def _arun(self, query: str) -> str:
        return self._run(query)

# 인스턴스 생성
search_documents = SearchDocumentsTool()
'''

# 6. utils/tool_answer.py
tool_answer_content = '''from langchain_core.tools import BaseTool
from typing import List
import json


class FinalAnswerTool(BaseTool):
    name: str = "final_answer"  # 이름 통일
    description: str = "최종 답변과 출처 및 경로를 포함한 사전(dict)을 생성합니다."

    def _run(self, answer: str, sources: List[str] = None, paths: List[str] = None) -> str:
        """
        최종 답변을 생성합니다.
        
        Args:
            answer: 최종 답변 텍스트
            sources: 출처 파일명 리스트 (옵션)
            paths: 파일 경로 리스트 (옵션)
            
        Returns:
            JSON 문자열 형태의 답변
        """
        # 기본값 처리
        sources = sources or []
        paths = paths or []
        
        # 출처가 있는 경우 답변에 추가
        if sources:
            answer_with_sources = f"{answer}\\n\\n📚 출처: {', '.join(sources)}"
        else:
            answer_with_sources = answer
            
        output = {
            "answer": answer_with_sources,
            "sources": sources,
            "paths": paths
        }
        
        # JSON 문자열로 반환 (ToolNode가 파싱할 수 있도록)
        return json.dumps(output, ensure_ascii=False)

    async def _arun(self, answer: str, sources: List[str] = None, paths: List[str] = None) -> str:
        return self._run(answer, sources, paths)

# 인스턴스 생성
final_answer = FinalAnswerTool()
'''

# 7. utils/tool_diagnose.py
tool_diagnose_content = '''from langchain_core.tools import BaseTool
import json
from typing import Dict, Any
import traceback

class ToolDiagnoseTool(BaseTool):
    name: str = "tool_diagnose"
    description: str = "시스템 문제를 진단하고 원인을 분석합니다."

    def _run(self, last_input: str = "", context: Dict[str, Any] = None) -> str:
        """
        시스템 문제를 진단합니다.
        
        Args:
            last_input: 마지막 사용자 입력
            context: 진단에 필요한 컨텍스트 정보
            
        Returns:
            JSON 문자열 형태의 진단 결과
        """
        context = context or {}
        diagnosis = {
            "status": "diagnosed",
            "issues": [],
            "recommendations": []
        }
        
        # 1. 입력 검증
        if not last_input:
            diagnosis["issues"].append("❗ 사용자 입력이 없습니다.")
            diagnosis["recommendations"].append("질문을 입력해주세요.")
        
        # 2. 상태 검증
        if "messages" in context:
            msg_count = len(context["messages"])
            if msg_count == 0:
                diagnosis["issues"].append("❗ 메시지 히스토리가 비어있습니다.")
                diagnosis["recommendations"].append("새로운 대화를 시작해주세요.")
            else:
                diagnosis["issues"].append(f"✓ 메시지 수: {msg_count}개")
        
        # 3. 에러 정보 확인
        if "error" in context:
            error_msg = str(context["error"])
            diagnosis["issues"].append(f"❗ 오류 발생: {error_msg}")
            
            # 일반적인 오류 패턴 분석
            if "API" in error_msg or "api" in error_msg:
                diagnosis["recommendations"].append("API 키를 확인해주세요.")
            elif "embedding" in error_msg or "임베딩" in error_msg:
                diagnosis["recommendations"].append("임베딩 데이터베이스를 확인해주세요.")
            elif "connection" in error_msg or "연결" in error_msg:
                diagnosis["recommendations"].append("네트워크 연결을 확인해주세요.")
            else:
                diagnosis["recommendations"].append("시스템 로그를 확인해주세요.")
        
        # 4. 도구 상태 확인
        if "tools_status" in context:
            for tool_name, status in context["tools_status"].items():
                if status == "failed":
                    diagnosis["issues"].append(f"❗ {tool_name} 도구 실행 실패")
                else:
                    diagnosis["issues"].append(f"✓ {tool_name} 도구 정상")
        
        # 5. 임베딩 데이터 확인
        try:
            from utils.load_all_embeddings import load_all_embeddings
            docs, embs, srcs = load_all_embeddings()
            if len(docs) == 0:
                diagnosis["issues"].append("❗ 임베딩 데이터베이스가 비어있습니다.")
                diagnosis["recommendations"].append("문서를 임베딩하여 데이터베이스를 구축해주세요.")
            else:
                diagnosis["issues"].append(f"✓ 임베딩 데이터: {len(docs)}개 문서, {len(set(srcs))}개 파일")
        except Exception as e:
            diagnosis["issues"].append(f"❗ 임베딩 로드 실패: {str(e)}")
            diagnosis["recommendations"].append("임베딩 파일 경로와 형식을 확인해주세요.")
        
        # 6. LLM 연결 확인
        if "llm_error" in context:
            diagnosis["issues"].append("❗ LLM 연결 오류")
            diagnosis["recommendations"].append("OpenAI API 키와 네트워크를 확인해주세요.")
        
        # 7. 최종 진단
        if not diagnosis["issues"]:
            diagnosis["issues"].append("✓ 시스템 상태가 정상입니다.")
        
        if not diagnosis["recommendations"]:
            diagnosis["recommendations"].append("시스템이 정상 작동 중입니다.")
        
        # 진단 결과 포맷팅
        result = {
            "diagnosis": "\\n".join(diagnosis["issues"]),
            "recommendations": "\\n".join(diagnosis["recommendations"]),
            "details": diagnosis
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def _arun(self, last_input: str = "", context: Dict[str, Any] = None) -> str:
        return self._run(last_input, context)

# 인스턴스 생성
tool_diagnose = ToolDiagnoseTool()
'''

# 8. test_rag.py
test_rag_content = '''#!/usr/bin/env python3
"""
RAG 시스템 테스트 스크립트
각 컴포넌트를 개별적으로 테스트하여 문제점을 찾습니다.
"""
import os
import sys
import logging
from langchain_core.messages import HumanMessage

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s"
)

# API 키 설정
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"  # 실제 키로 교체 필요

def test_tools():
    """도구들이 제대로 작동하는지 테스트"""
    print("\\n=== 도구 테스트 ===")
    
    # 1. search_documents 테스트
    try:
        from utils.tool_search import search_documents
        print(f"✓ search_documents 로드 성공 (name: {search_documents.name})")
        
        # 검색 테스트
        result = search_documents._run("의료용 휠체어")
        print(f"  검색 결과: {result[:100]}...")
    except Exception as e:
        print(f"✗ search_documents 오류: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. final_answer 테스트
    try:
        from utils.tool_answer import final_answer
        print(f"\\n✓ final_answer 로드 성공 (name: {final_answer.name})")
        
        # 답변 생성 테스트
        result = final_answer._run(
            answer="테스트 답변입니다.",
            sources=["test.pdf"],
            paths=["/path/to/test.pdf"]
        )
        print(f"  답변 결과: {result}")
    except Exception as e:
        print(f"✗ final_answer 오류: {e}")
    
    # 3. tool_diagnose 테스트
    try:
        from utils.tool_diagnose import tool_diagnose
        print(f"\\n✓ tool_diagnose 로드 성공 (name: {tool_diagnose.name})")
    except Exception as e:
        print(f"✗ tool_diagnose 오류: {e}")

def test_llm():
    """LLM 연결 테스트"""
    print("\\n\\n=== LLM 테스트 ===")
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
        
        response = llm.invoke([HumanMessage(content="Hello, are you working?")])
        print(f"✓ LLM 응답: {response.content}")
    except Exception as e:
        print(f"✗ LLM 오류: {e}")
        print("  → API 키를 확인하세요")

def test_graph():
    """Graph 전체 테스트"""
    print("\\n\\n=== Graph 테스트 ===")
    
    try:
        from graph_builder import respond
        
        # 간단한 테스트
        state = {
            "messages": [HumanMessage(content="의료용 휠체어란 무엇인가요?")],
            "query": "의료용 휠체어란 무엇인가요?"
        }
        
        print("입력:", state)
        result = respond(state)
        print("\\n출력:")
        print(f"  - answer: {result.get('answer', 'N/A')[:100]}...")
        print(f"  - sources: {result.get('sources', [])}")
        print(f"  - output: {result.get('output', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"✗ Graph 오류: {e}")
        import traceback
        traceback.print_exc()

def test_embeddings():
    """임베딩 데이터 확인"""
    print("\\n\\n=== 임베딩 데이터 테스트 ===")
    
    try:
        from utils.load_all_embeddings import load_all_embeddings
        docs, embs, srcs = load_all_embeddings()
        
        print(f"✓ 총 문서 수: {len(docs)}")
        print(f"✓ 총 파일 수: {len(set(srcs))}")
        
        if docs:
            print(f"✓ 첫 번째 문서 샘플: {docs[0][:100]}...")
            print(f"✓ 첫 번째 소스: {srcs[0]}")
    except Exception as e:
        print(f"✗ 임베딩 로드 오류: {e}")

def main():
    """모든 테스트 실행"""
    print("RAG 시스템 진단 시작...")
    
    # 1. 임베딩 데이터 확인
    test_embeddings()
    
    # 2. 도구 테스트
    test_tools()
    
    # 3. LLM 테스트
    test_llm()
    
    # 4. 전체 시스템 테스트
    test_graph()
    
    print("\\n\\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
'''

# 9. utils/embed_query.py (샘플)
embed_query_content = '''import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

def embed_query(query: str) -> np.ndarray:
    """쿼리를 임베딩 벡터로 변환"""
    try:
        embedding = embeddings_model.embed_query(query)
        return np.array(embedding)
    except Exception as e:
        print(f"임베딩 오류: {e}")
        return None
'''

# 10. utils/load_all_embeddings.py (샘플)
load_all_embeddings_content = '''import pickle
import numpy as np
from pathlib import Path

def load_all_embeddings():
    """저장된 임베딩 데이터 로드"""
    # 실제 구현에 맞게 수정 필요
    embeddings_dir = Path("embeddings")
    
    docs_all = []
    embs_all = []
    srcs_all = []
    
    # 예시: pickle 파일에서 로드
    if embeddings_dir.exists():
        for emb_file in embeddings_dir.glob("*.pkl"):
            try:
                with open(emb_file, "rb") as f:
                    data = pickle.load(f)
                    docs_all.extend(data.get("documents", []))
                    embs_all.extend(data.get("embeddings", []))
                    srcs_all.extend(data.get("sources", []))
            except Exception as e:
                print(f"Error loading {emb_file}: {e}")
    
    # 테스트용 더미 데이터 (실제 데이터가 없을 경우)
    if not docs_all:
        print("⚠️  실제 임베딩 데이터가 없습니다. 테스트용 더미 데이터 사용")
        docs_all = ["의료용 휠체어는 보행이 불편한 환자를 위한 의료기기입니다."]
        embs_all = [np.random.rand(1536)]  # OpenAI 임베딩 차원
        srcs_all = ["dummy_document.pdf"]
    
    return docs_all, embs_all, srcs_all
'''

# 11. requirements.txt
requirements_content = '''langchain>=0.1.0
langchain-openai>=0.0.5
langgraph>=0.0.37
gradio>=4.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
'''

# 12. .env (샘플)
env_content = '''# OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here

# 기타 환경 변수
LOG_LEVEL=INFO
'''

# 13. README.md
readme_content = '''# RAG 시스템 (nxj_llm)

대한민국 식품의약품안전처 의료제품 인허가 지원을 위한 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- 문서 검색 및 답변 생성
- LangGraph 기반 에이전트 시스템
- Function Calling을 통한 도구 실행
- Gradio 웹 인터페이스

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. OpenAI API 키 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 입력
```

3. 임베딩 데이터 준비:
- `embeddings/` 디렉토리에 임베딩 파일 배치
- 또는 `utils/load_all_embeddings.py` 수정

## 실행 방법

### 시스템 테스트
```bash
python test_rag.py
```

### 웹 인터페이스 실행
```bash
python run_gradio.py
```

브라우저에서 http://localhost:7862 접속

## 파일 구조

- `graph_builder.py`: 메인 그래프 로직
- `state_types.py`: State 타입 정의
- `run_gradio.py`: Gradio UI
- `registry.py`: 도구 레지스트리
- `utils/`: 도구 및 유틸리티 함수
  - `tool_search.py`: 문서 검색 도구
  - `tool_answer.py`: 답변 생성 도구
  - `tool_diagnose.py`: 시스템 진단 도구

## 문제 해결

임베딩 데이터가 없는 경우, `utils/load_all_embeddings.py`의 더미 데이터가 사용됩니다.
실제 사용을 위해서는 문서를 임베딩하여 데이터베이스를 구축해야 합니다.
'''

# 파일 생성
create_file("graph_builder.py", graph_builder_content)
create_file("state_types.py", state_types_content)
create_file("run_gradio.py", run_gradio_content)
create_file("registry.py", registry_content)
create_file("utils/tool_search.py", tool_search_content)
create_file("utils/tool_answer.py", tool_answer_content)
create_file("utils/tool_diagnose.py", tool_diagnose_content)
create_file("test_rag.py", test_rag_content)
create_file("utils/embed_query.py", embed_query_content)
create_file("utils/load_all_embeddings.py", load_all_embeddings_content)
create_file("requirements.txt", requirements_content)
create_file(".env.example", env_content)
create_file("README.md", readme_content)

# utils/__init__.py 생성 (패키지 인식용)
create_file("utils/__init__.py", "# Utils package")

print("\n✅ 모든 파일이 생성되었습니다!")
print("\n다음 단계:")
print("1. OpenAI API 키 설정: .env.example을 .env로 복사하고 키 입력")
print("2. 패키지 설치: pip install -r requirements.txt")
print("3. 시스템 테스트: python test_rag.py")
print("4. UI 실행: python run_gradio.py")