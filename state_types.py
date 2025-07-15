from typing_extensions import TypedDict, Annotated # Annotated 임포트 추가
from typing import List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages # add_messages 임포트 추가

class GraphState(TypedDict, total=False):
    """LangGraph State 정의 - 필수 필드만 유지"""
    # 핵심 필드
    # LangGraph에 messages 필드를 처리하는 방식을 명시적으로 알려줍니다.
    messages: Annotated[List[BaseMessage], add_messages] # messages 필드에 add_messages 적용
    
    # 검색 관련 필드
    documents: List[str]  # 검색된 문서 내용
    sources: List[str]    # 문서 출처 (파일명)
    paths: List[str]      # 문서 경로
    
    # 결과 필드
    answer: str          # 최종 답변
    
    # 선택적 필드 (기존 코드 호환성)
    query: str           # 사용자 질문
    last_user: str       # 마지막 사용자 입력 (호환성)

