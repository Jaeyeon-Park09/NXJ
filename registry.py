"""
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
