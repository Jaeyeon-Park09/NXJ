from langchain_core.tools import BaseTool
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
            # 1. 각 출처 파일명의 .json 확장자를 제거하고 '+'를 공백으로 변환합니다.
            # 2. 번호 목록 형식으로 만듭니다.
            formatted_sources = [
                f"{i}) {src.replace('.json', '').replace('+', ' ')}" 
                for i, src in enumerate(sources, 1)
            ]
            
            # 3. 최종 답변에 제목과 함께 줄바꿈으로 연결된 목록을 추가합니다.
            sources_str = "\n".join(formatted_sources)
            answer_with_sources = f"{answer}\n\n📚 출처:\n{sources_str}"
        else:
            answer_with_sources = answer
            
        output = {
            "answer": answer_with_sources,
            "sources": sources, # 원본 소스 정보는 그대로 유지
            "paths": paths
        }
        
        # JSON 문자열로 반환 (ToolNode가 파싱할 수 있도록)
        return json.dumps(output, ensure_ascii=False)

    async def _arun(self, answer: str, sources: List[str] = None, paths: List[str] = None) -> str:
        return self._run(answer, sources, paths)

# 인스턴스 생성
final_answer = FinalAnswerTool()
