import gradio as gr
from graph_builder import respond as graph_respond
from langchain_core.messages import HumanMessage, AIMessage
import json
import logging
import importlib.util
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# 전체 대화 히스토리 관리
conversation_history: list[dict] = []

def extract_answer(response: dict) -> str:
    """
    응답에서 답변 추출
    """
    # 1. answer 필드 확인
    if response.get("answer"):
        answer = response["answer"]
        if isinstance(answer, list):
            return "\n".join(str(a) for a in answer)
        return answer
    
    # 2. output 필드 확인
    if response.get("output"):
        output = response["output"]
        if isinstance(output, list):
            return "\n".join(str(a) for a in output)
        return output
    
    # 3. messages에서 마지막 AI 메시지 추출
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, list):
                return "\n".join(str(a) for a in content)
            # JSON 형태의 응답인지 확인
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    data = json.loads(content)
                    if "answer" in data:
                        answer = data["answer"]
                        if isinstance(answer, list):
                            return "\n".join(str(a) for a in answer)
                        return answer
                except:
                    pass
            return content
    
    return "❌ 죄송합니다. 답변을 생성할 수 없습니다."

def respond_wrapper(message: str, chat_history: list | None = None) -> list:
    """Gradio 인터페이스를 위한 응답 함수"""
    logging.info(f"USER: {message}")
    
    if not message.strip():
        return chat_history or []
    
    # 히스토리 관리
    history = chat_history or []
    user_query = message.strip()
    
    # 대화 히스토리에 추가
    conversation_history.append({"role": "user", "content": user_query})
    
    # '보고서' 또는 '리포트' 키워드가 포함된 경우 report_generation tool을 우선 호출
    if any(keyword in user_query for keyword in ["보고서", "리포트"]):
        # report_generation tool 동적 import
        spec = importlib.util.spec_from_file_location("tool_reportgen", os.path.join(os.path.dirname(__file__), "utils", "tool_reportgen.py"))
        if spec is not None and spec.loader is not None:
            tool_reportgen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_reportgen)
            report_generation = tool_reportgen.report_generation
            report_prompt = report_generation._run(topic=user_query)
            answer = report_prompt
            conversation_history.append({"role": "assistant", "content": answer})
            visible_history = [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": answer}
            ]
            logging.info(f"ASSISTANT: {answer[:100]}...")
            return visible_history
        else:
            error_msg = "❌ report_generation tool import 오류"
            conversation_history.append({"role": "assistant", "content": error_msg})
            visible_history = [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": error_msg}
            ]
            return visible_history
    
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
with gr.Blocks(title="nxj_llm RAG test") as demo:
    gr.Markdown("""
    # RAG test
    
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
        share=True,
        show_error=True
    )
