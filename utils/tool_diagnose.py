from langchain_core.tools import BaseTool
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
            "diagnosis": "\n".join(diagnosis["issues"]),
            "recommendations": "\n".join(diagnosis["recommendations"]),
            "details": diagnosis
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def _arun(self, last_input: str = "", context: Dict[str, Any] = None) -> str:
        return self._run(last_input, context)

# 인스턴스 생성
tool_diagnose = ToolDiagnoseTool()
