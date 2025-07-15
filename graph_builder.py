from __future__ import annotations
import json
from typing import Dict, Any, List, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os

# Tool registry - 이름 통일
from utils.tool_search import search_documents
from utils.tool_answer import final_answer
from utils.tool_diagnose import tool_diagnose
from state_types import GraphState # GraphState 클래스를 임포트합니다.

TOOLS = [search_documents, final_answer, tool_diagnose]
tool_node = ToolNode(TOOLS)

# System prompt
system_prompt = """
당신은 대한민국 식품의약품안전처에서 의료제품 인허가를 지원하기 위해 개발된 매우 강력한 인허스턴트 nxj_llm입니다.

<role>
** 중요: 주어진 문서를 활용해 답변해야 합니다. 문서에 없는 내용을 지어내지 마세요.
** 중요: 자신의 데이터베이스의 정보나 지식을 사용하지 마세요. 오직 주어진 문서에만 의존해야 합니다.
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
</available_tools>

<strict_rules>
- search_documents는 반드시 1번만 호출
- search_documents 후 즉시 final_answer 호출
- 재검색 절대 금지
**위반 시 시스템이 점검합니다.**
</strict_rules>

<instructions>
1. 사용자 질문을 받으면 먼저 search_documents로 관련 문서를 검색하세요.
2. 검색 결과를 바탕으로 답변을 작성하세요.
3. final_answer 도구를 사용해 최종 답변을 생성하세요.
4. 문서에 정보가 없으면 그 사실을 명시하세요.
** 최종 답변 시에 '+', '.json' 문자열은 제거하세요. **
</instructions>
"""

# LLM setup with function calling
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=2048,
).bind_tools(TOOLS)

# Agent node
def agent(state: GraphState) -> Dict[str, Any]:
    """LLM이 다음 행동을 결정하는 노드"""
    messages = state.get("messages", [])
    
    print("DEBUG: Agent node received messages (before LLM invoke):")
    for i, msg in enumerate(messages):
        tool_calls_info = getattr(msg, 'tool_calls', 'N/A')
        print(f"  [{i}] Type: {type(msg).__name__}, Role: {msg.type}, Content: {str(msg.content)[:70]}..., Tool Calls: {tool_calls_info}")
    
    response = llm.invoke(messages)
    
    print(f"DEBUG: LLM Response type: {type(response)}")
    print(f"DEBUG: LLM Response content: {response.content[:70]}...")
    print(f"DEBUG: LLM Response additional_kwargs: {response.additional_kwargs}")
    
    updated_messages = add_messages(messages, response)
    
    new_state = state.copy()
    new_state["messages"] = updated_messages
    
    print("DEBUG: Agent node returning state with updated messages.")
    return new_state

# Router function
def should_continue(state: GraphState) -> Literal["tools", "end"]:
    """다음 노드를 결정하는 라우터"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if last_message and isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("DEBUG: Router decided: tools (AIMessage with tool_calls detected)")
        return "tools"
    
    print("DEBUG: Router decided: end (no tool_calls detected or final answer)")
    return "end"

# Result processor
def process_result(state: GraphState) -> Dict[str, Any]:
    """최종 결과를 처리하는 노드"""
    messages = state.get("messages", [])
    
    if "answer" in state and state["answer"]:
        return {
            "messages": messages,
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "paths": state.get("paths", [])
        }

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return {
                "messages": messages,
                "answer": message.content,
                "sources": state.get("sources", []),
                "paths": state.get("paths", [])
            }
    
    return {
        "messages": messages,
        "answer": "죄송합니다. 답변을 생성할 수 없습니다.",
        "sources": state.get("sources", []),
        "paths": state.get("paths", [])
    }

# Tool result handler
def handle_tool_result(state: GraphState) -> Dict[str, Any]:
    """도구 실행 결과를 State에 반영"""
    messages = state.get("messages", []) # 노드가 받은 현재 메시지 리스트를 가져옵니다.
    
    print("DEBUG: handle_tool_result received state messages:")
    for i, msg in enumerate(messages):
        tool_calls_info = getattr(msg, 'tool_calls', 'N/A')
        print(f"  [{i}] Type: {type(msg).__name__}, Role: {msg.type}, Content: {str(msg.content)[:70]}..., Tool Calls: {tool_calls_info}")

    new_state = state.copy() # 현재 상태를 복사합니다.
    # new_state["messages"] = messages # 이 줄은 아래에서 조건부로 업데이트되므로, 여기서는 제거합니다.

    if messages and isinstance(messages[-1], ToolMessage):
        try:
            content = messages[-1].content
            if isinstance(content, str) and content.strip().startswith("{") and content.strip().endswith("}"):
                result = json.loads(content)
                
                # 메시지 리스트는 ToolNode에 의해 이미 업데이트되었으므로, 여기서는 다른 필드만 업데이트합니다.
                # 그리고 반드시 messages 필드를 명시적으로 반환해야 합니다.
                if "documents" in result:
                    new_state["documents"] = result.get("documents", [])
                    new_state["sources"] = result.get("sources", [])
                    new_state["paths"] = result.get("paths", [])
                    print(f"DEBUG: handle_tool_result updated documents/sources/paths. Docs: {len(new_state['documents'])}")
                
                if "answer" in result:
                    new_state["answer"] = result.get("answer", "")
                    new_state["sources"] = result.get("sources", [])
                    new_state["paths"] = result.get("paths", [])
                    print(f"DEBUG: handle_tool_result updated answer: {new_state['answer'][:70]}...")
            else:
                print(f"DEBUG: handle_tool_result received non-JSON or empty content: '{content[:50]}...'")
        except json.JSONDecodeError as e:
            print(f"ERROR: handle_tool_result failed to parse JSON: {e}")
            print(f"ERROR: Problematic content (last ToolMessage): {messages[-1].content[:100]}...")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in handle_tool_result: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("DEBUG: handle_tool_result did not find ToolMessage as last message or messages list was empty.")
    
    # 이 노드에서 messages 필드를 직접 변경하지는 않았지만,
    # 다음 노드(agent)로 전달될 때 messages 필드가 올바르게 유지되도록 명시적으로 할당합니다.
    # ToolNode가 messages를 이미 업데이트했으므로, 그 상태를 그대로 가져옵니다.
    new_state["messages"] = messages 

    print("DEBUG: handle_tool_result returning state messages:")
    for i, msg in enumerate(new_state["messages"]):
        print(f"  [{i}] Type: {type(msg).__name__}, Role: {msg.type}, Content: {str(msg.content)[:70]}...")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"    Tool Calls: {msg.tool_calls}")
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            print(f"    Tool Call ID: {msg.tool_call_id}")

    return new_state # 변경된 new_state를 반환

# Build graph
workflow = StateGraph(GraphState)

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
workflow.add_edge("handle_tool_result", "agent") # 도구 실행 후 다시 agent가 다음 행동 결정

workflow.add_edge("process_result", END)

try:
    app = workflow.compile()
    print(f"DEBUG: app after compile: {app}")
    if app is None:
        print("ERROR: workflow.compile()이 None을 반환했습니다. 그래프 컴파일에 실패했습니다.")
        raise RuntimeError("LangGraph 컴파일 실패: app이 None입니다.")
except Exception as e:
    print(f"CRITICAL ERROR: workflow.compile() 중 예외 발생: {e}")
    import traceback
    traceback.print_exc()
    raise

# Simple interface
def respond(state: dict) -> dict:
    """
    간단한 인터페이스 함수
    """
    # 입력 검증
    messages_input = state.get("messages", [])
    query = state.get("query", "") or state.get("last_user", "")
    
    # messages_input이 비어있으면 새로 시작하는 대화이므로 SystemMessage와 HumanMessage를 추가
    if not messages_input:
        if query:
            messages_input = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        else: # query도 없고 messages도 없으면 오류
            return {
                "messages": [],
                "output": "❌ 입력 오류: 질문을 입력해 주세요.",
                "answer": "",
                "sources": [],
                "paths": []
            }
    else:
        # 기존 메시지가 있지만 SystemMessage가 첫 메시지가 아니거나 내용이 다르면 추가/교체
        if not isinstance(messages_input[0], SystemMessage) or messages_input[0].content != system_prompt:
            messages_input = [SystemMessage(content=system_prompt)] + messages_input
    
    # 초기 상태 설정
    initial_state: GraphState = {
        "messages": messages_input,
        "documents": state.get("documents", []), # 기존 상태의 documents, sources, paths 유지
        "sources": state.get("sources", []),
        "paths": state.get("paths", []),
        "answer": state.get("answer", "")
    }
    
    print("DEBUG: Initial state for Graph execution (respond function entry):")
    for i, msg in enumerate(initial_state["messages"]):
        print(f"  [{i}] Type: {type(msg).__name__}, Role: {msg.type}, Content: {str(msg.content)[:70]}...")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"    Tool Calls: {msg.tool_calls}")
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            print(f"    Tool Call ID: {msg.tool_call_id}")
                
    try:
        result: GraphState = app.invoke(initial_state)
        
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        paths = result.get("paths", [])

        if not answer:
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content
                    break
        
        if not answer:
            answer = "죄송합니다. 답변을 생성할 수 없습니다."
        
        return {
            "messages": result.get("messages", []),
            "output": answer,
            "answer": answer,
            "sources": sources,
            "paths": paths
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        diagnosis_result = tool_diagnose._run(
            last_input=query,
            context={"error": str(e), "state": initial_state}
        )
        
        try:
            parsed_diagnosis = json.loads(diagnosis_result)
            diagnosis_str = f"🔍 진단 결과:\\n{parsed_diagnosis.get('diagnosis', '진단 정보 없음')}\\n\\n✅ 권장 사항:\\n{parsed_diagnosis.get('recommendations', '권장 사항 없음')}"
        except json.JSONDecodeError:
            diagnosis_str = f"🔍 진단 결과 파싱 실패: {diagnosis_result}"

        return {
            "messages": messages_input,
            "output": f"❌ 시스템 오류: {str(e)}\\n\\n{diagnosis_str}",
            "answer": "시스템 오류가 발생했습니다. 자세한 내용은 진단 결과를 확인해주세요.",
            "sources": [],
            "paths": []
        }
