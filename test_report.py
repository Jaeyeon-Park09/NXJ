import os
import json
from utils.tool_search import search_context_for_rag
from utils.tool_report import write_detailed_report, write_body_sections_prompt
from utils.tool_section import write_section_titles_prompt, fix_section_markers
from langchain_community.llms import Ollama
import difflib
import re

def filter_context(raw_context):
    lines = raw_context.split("\n")
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("|") or line.endswith("|"):
            continue
        if "Unnamed" in line:
            continue
        if set(line) <= set("| -:."):
            continue
        if len(line) < 10:
            continue
        filtered.append(line)
    if not filtered:
        filtered = [l.strip() for l in lines if l.strip()][:3]
    result = ""
    for line in filtered:
        if len(result) + len(line) + 1 > 2000:
            break
        result += line + "\n"
    return result.strip()

# 사람이 읽기 쉬운 파일명 매핑 테이블
PRETTY_FILENAME_MAP = {
    'data1.txt': '2024년 6월 매출 데이터',
    'data2.txt': '고객 설문 결과',
    'data3.txt': '경쟁사 분석 보고서',
    # 필요시 추가 - 소스 목록 불필요
}

def prettify_source_filename(filename):
    # 확장자 제거, +를 공백으로, 매핑 테이블 우선 적용
    base = filename.replace('+', ' ')
    if base.endswith('.json'):
        base = base[:-5]
    if base in PRETTY_FILENAME_MAP:
        return PRETTY_FILENAME_MAP[base]
    if base.endswith('.txt'):
        base = base[:-4]
    return base

# context와 sources를 세부 위치(행번호)까지 매핑
# context: 여러 청크(문단)로 구성, 각 청크별로 (파일명, 행번호) 정보 필요
# search_context_for_rag에서 반환하는 paths를 활용해 세부 위치 추출

def postprocess_citations(report_text, context, sources):
    """
    context의 각 문장이 답변 본문에 부분 문자열로 포함되면 해당 문장 뒤에 (출처: 파일명) 표기.
    이미 출처가 붙은 경우는 중복 표기하지 않음.
    본문 내 출처가 하나도 없으면, 마지막에 참고문헌(파일명 리스트) 자동 추가.
    """
    from collections import OrderedDict
    context_lines = [line.strip() for line in context.split("\n") if line.strip()]
    context_map = OrderedDict()
    for i, line in enumerate(context_lines):
        if line and i < len(sources):
            pretty_src = prettify_source_filename(sources[i])
            context_map[line] = pretty_src

    processed = report_text
    cited_files = set()
    # 각 context 문장이 답변 본문에 substring으로 포함되면 출처 표기
    for ctx, src in sorted(context_map.items(), key=lambda x: -len(x[0])):
        # 이미 출처가 붙은 경우는 건너뜀
        pattern = re.escape(ctx) + r'\s*\(출처:'
        if re.search(pattern, processed):
            continue
        # 부분 문자열로 포함되어 있으면 출처 표기
        if ctx in processed:    
            # 해당 문장 뒤에 (출처: 파일명) 삽입 (여러 번 등장해도 모두 표기)
            processed = re.sub(r'(' + re.escape(ctx) + r')(?!\s*\(출처:)', r'\1 (출처: ' + src + ')', processed)
            cited_files.add(src)
    # 본문 내 출처가 하나도 없으면 참고문헌 자동 추가
    if not cited_files and len(context_map) > 0:
        unique_files = list(OrderedDict.fromkeys(context_map.values()))
        processed += "\n\n참고문헌:\n" + "\n".join(f"- {f}" for f in unique_files)
    return processed

def check_toc_and_sections(report_text):
    """
    [목차] 마커, [본문 시작] 마커 기반으로 목차/본문을 추출하고,
    목차가 서론, 소제목3개, 결론 순서로 한 줄씩, 줄바꿈 포함해 있는지,
    소제목 3개가 서로 겹치지 않는지, 각 소제목 본문이 300자 이상인지 검증
    """
    # 1. [목차] 마커 추출
    toc_marker = '[목차]'
    body_marker = '[본문 시작]'
    if toc_marker not in report_text or body_marker not in report_text:
        return False, "[목차] 또는 [본문 시작] 마커가 없음"
    toc_block = report_text.split(toc_marker, 1)[-1].split(body_marker, 1)[0].strip()
    toc_lines = [line.strip() for line in toc_block.split('\n') if line.strip()]
    if len(toc_lines) != 5:
        return False, f"목차 줄 개수 오류: {len(toc_lines)}줄"
    if toc_lines[0] != '서론' or toc_lines[-1] != '결론':
        return False, "목차 첫 줄이 '서론', 마지막 줄이 '결론'이 아님"
    section_titles = toc_lines[1:4]
    # 소제목 3개가 서로 겹치지 않는지(유사도 0.85 이상이면 겹침으로 간주)
    for i in range(3):
        for j in range(i+1, 3):
            s1, s2 = section_titles[i], section_titles[j]
            ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
            if ratio > 0.85:
                return False, f"소제목 {i+1}과 {j+1}이 유사함: '{s1}' vs '{s2}'"
    # 2. [본문 시작] 이후 본문 추출
    body_block = report_text.split(body_marker, 1)[-1].strip()
    # 서론, 소제목1~3, 결론 본문 분리 (마크다운 제목 기준)
    # 서론: #
    # 소제목: ##
    # 결론: ## 결론
    # 서론
    intro_match = re.search(r"# (.+?)\n([\s\S]+?)(?=^## |^## 결론|\Z)", body_block, re.MULTILINE)
    if not intro_match:
        return False, "서론(# 제목)이 제대로 시작되지 않음"
    # 소제목 3개
    section_matches = list(re.finditer(r"^## (.+?)\n([\s\S]+?)(?=^## |^## 결론|\Z)", body_block, re.MULTILINE))
    if len(section_matches) < 3:
        return False, f"소제목(## 제목) 개수 부족: {len(section_matches)}개"
    # 결론
    conclusion_match = re.search(r"^## 결론\n([\s\S]+)$", body_block, re.MULTILINE)
    if not conclusion_match:
        return False, "결론(## 결론)이 제대로 시작되지 않음"
    # 각 소제목 본문 300자 이상 체크
    for idx, m in enumerate(section_matches[:3]):
        content = m.group(2).replace('\n', '').strip()
        if len(content) < 300:
            return False, f"소제목 {idx+1} 본문이 300자 미만임 ({len(content)}자)"
    return True, "OK"

def parse_body_sections(text, section_titles):
    """
    LLM이 출력한 본문을 마커([서론], [소제목1], [소제목2], [소제목3], [결론]) 기준으로 파싱하여 dict로 반환
    """
    import re
    if text is None:
        text = ""
    sec1, sec2, sec3 = section_titles
    result = {}
    patterns = [
        ("서론", rf"\[서론\]\s*([\s\S]*?)(?=\[{re.escape(sec1)}\]|\[{re.escape(sec2)}\]|\[{re.escape(sec3)}\]|\[결론\]|$)"),
        (sec1, rf"\[{re.escape(sec1)}\]\s*([\s\S]*?)(?=\[{re.escape(sec2)}\]|\[{re.escape(sec3)}\]|\[결론\]|$)"),
        (sec2, rf"\[{re.escape(sec2)}\]\s*([\s\S]*?)(?=\[{re.escape(sec3)}\]|\[결론\]|$)"),
        (sec3, rf"\[{re.escape(sec3)}\]\s*([\s\S]*?)(?=\[결론\]|$)"),
        ("결론", r"\[결론\]\s*([\s\S]*)")
    ]
    for name, pat in patterns:
        m = re.search(pat, text)
        if m:
            result[name] = m.group(1).strip()
        else:
            result[name] = ""
    return result

def remove_introductory_sentences(text, section_title):
    """
    본문에서 '이 장에서는', '이 절에서는', '이 소제목에서는', '설명하겠습니다', '살펴보겠습니다' 등 안내/예고 문장 제거
    """
    import re
    # 다양한 안내/예고 패턴
    patterns = [
        rf"이 (장|절|소제목|부분|섹션)[^\n\.!?]*({re.escape(section_title)})?[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*설명(하겠|드리겠|합니다)[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*살펴보(겠|도록)[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*다루(겠|도록)[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*소개(하겠|합니다)[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*중심으로[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*목적은[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*설명하고자[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*살펴보고자[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*다루고자[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*소개하고자[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*중점적으로[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*집중적으로[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*설명될[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*살펴볼[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*다룰[^\n\.!?]*[\n\.!?]",
        r"이 (장|절|소제목|부분|섹션)[^\n\.!?]*소개할[^\n\.!?]*[\n\.!?]",
        r"이 절은[^\n\.!?]*[\n\.!?]",
        r"이 장은[^\n\.!?]*[\n\.!?]",
        r"이 소제목은[^\n\.!?]*[\n\.!?]",
        r"이 부분은[^\n\.!?]*[\n\.!?]",
        r"이 섹션은[^\n\.!?]*[\n\.!?]",
        r"설명하겠습니다[\n\.!?]",
        r"살펴보겠습니다[\n\.!?]",
        r"다루겠습니다[\n\.!?]",
        r"소개하겠습니다[\n\.!?]",
        r"설명합니다[\n\.!?]",
        r"살펴봅니다[\n\.!?]",
        r"다룹니다[\n\.!?]",
        r"소개합니다[\n\.!?]",
    ]
    result = text
    for pat in patterns:
        result = re.sub(pat, '', result, flags=re.MULTILINE)
    return result.strip()

def remove_duplicate_sentences(text):
    # 문장 단위로 중복 제거
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])', text) if s.strip()]
    seen = set()
    result = []
    for s in sentences:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return ' '.join(result)

def is_valid_subtitle(line):
    # 안내문, 영어, 불필요한 줄 필터링
    invalid_patterns = [
        r"Based on the question",
        r"I have selected",
        r"Here they are",
        r"아래는",
        r"소제목을 선정",
        r"subtitles",
        r"reference document",
        r"^\s*$"
    ]
    for pat in invalid_patterns:
        if re.search(pat, line, re.IGNORECASE):
            return False
    return True

def main():
    print("질문을 입력하세요.")
    question = input(">> ").strip()
    if not question:
        print("질문을 입력해야 합니다.")
        return

    # 1. 임베딩 기반 문서 검색 (context 전체)
    search_result = search_context_for_rag(question, top_k=5)
    raw_context = search_result.get("context", "")
    sources = search_result.get("sources", [])
    paths = search_result.get("paths", [])
    rag_logs = search_result.get("rag_logs", [])  # RAG-LOG 리스트
    # 인허가/행정 절차 관련 context 추가
    permit_keywords = ["인허가", "허가", "승인", "등록", "신고", "행정 절차", "법적 요건", "제도"]
    permit_query = question + " " + " ".join(permit_keywords)
    permit_search_result = search_context_for_rag(permit_query, top_k=2)
    permit_context = permit_search_result.get("context", "")
    # context 합치기
    context = filter_context(raw_context + "\n" + permit_context)
    MAX_CONTEXT_LEN = 2000
    if len(context) > MAX_CONTEXT_LEN:
        print(f"[경고] context가 너무 깁니다({len(context)}자). {MAX_CONTEXT_LEN}자까지만 사용합니다.")
        context = context[:MAX_CONTEXT_LEN]

    llm = Ollama(
        model="llama3:8b",
        temperature=0.2,
        base_url="http://localhost:11434"
    )

    # 2. 소제목 3개만 생성 (LLM이 생성, 반복 없이 한 번만)
    print(f"\n[소제목 3개 생성 프롬프트]")
    title_prompt = write_section_titles_prompt(question, context)
    print(title_prompt)
    titles_text = ""
    for chunk in llm.stream(title_prompt):
        if chunk and isinstance(chunk, str) and chunk.strip():
            print(chunk, end="", flush=True)
            titles_text += chunk
    print("\n[소제목 생성 완료]")
    titles = [line.strip() for line in titles_text.split('\n') if line.strip()]
    titles = [t for t in titles if is_valid_subtitle(t)]
    section_titles = titles[:3] if len(titles) >= 3 else ["소제목1", "소제목2", "소제목3"]

    # 3. 목차를 코드에서 직접 출력
    print("\n[목차]")
    print("서론")
    for t in section_titles:
        print(t)
    print("결론\n")
    print("[본문 시작]")

    # 4. 본문 생성 프롬프트 (마커 기반)
    report_prompt = write_body_sections_prompt(question, context, section_titles)
    report_prompt += "\n\n[중요] 반드시 아래 규칙을 지키세요.\n"
    report_prompt += "- 각 소제목 본문에는 '이 장에서는', '이 절에서는', '설명하겠습니다', '살펴보겠습니다' 등 안내/예고 문장을 쓰지 말고, 해당 주제에 대한 구체적 진술만 작성하세요.\n"
    report_prompt += "- 각 소제목 본문은 100자 이상, 서론/결론도 50자 이상으로 작성하세요.\n"
    report_prompt += "- 답변 내에 (출처: ~) 표기는 직접 작성하지 마세요. 출처 표기는 시스템이 자동으로 붙입니다.\n"

    print(f"\n[보고서 본문 생성 프롬프트]")
    print("="*40)
    print(report_prompt)
    print("="*40)
    answer_text = ""
    for chunk in llm.stream(report_prompt):
        if chunk and isinstance(chunk, str) and chunk.strip():
            print(chunk, end="", flush=True)
            answer_text += chunk
    print("\n[보고서 생성 완료]")
    # 마커 자동 치환: [소제목1] 등 → 실제 소제목, [[소제목]] 등도 한 쌍으로 통일
    answer_text = fix_section_markers(answer_text, section_titles)
    # 마커별 파싱
    body_sections = parse_body_sections(answer_text, section_titles)
    # 서론 안내문/영문 등 제거 및 중복 문장 제거
    body_sections["서론"] = remove_introductory_sentences(body_sections["서론"], "서론")
    body_sections["서론"] = remove_duplicate_sentences(body_sections["서론"])
    # 검증: 각 본문이 존재하고, 분량 조건(서론/결론 50자 이상, 소제목 100자 이상)
    ok = True
    reasons = []
    if len(body_sections["서론"]) < 50:
        ok = False
        reasons.append(f"서론 분량 오류: {len(body_sections['서론'])}자")
    for idx, t in enumerate(section_titles):
        if len(body_sections[t]) < 100:
            ok = False
            reasons.append(f"소제목{idx+1}({t}) 본문 100자 미만: {len(body_sections[t])}자")
    if len(body_sections["결론"]) < 50:
        ok = False
        reasons.append(f"결론 분량 오류: {len(body_sections['결론'])}자")
    # 피드백/재생성 루프 완전 제거: 한 번 생성된 본문 그대로 사용

    # 5. 후처리: context 기반 자동 출처 표기 (파일명만, 부분 일치/참고문헌 자동 추가)
    for k in body_sections:
        body_sections[k] = postprocess_citations(body_sections[k], context, sources)
        if k in section_titles:
            body_sections[k] = remove_introductory_sentences(body_sections[k], k)

    # 6. 최종 보고서 출력 (목차+본문)
    print("\n[최종 보고서]")
    print("="*40)
    print("[목차]")
    print("서론")
    for t in section_titles:
        print(t)
    print("결론\n")
    print("[본문 시작]")
    for name in ["서론"] + section_titles + ["결론"]:
        print(name)
        print(body_sections[name])
        print()
    print("="*40)

    # --- log 파일 저장 ---
    import datetime
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"reportgen_nosection_{now_str}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"질문: {question}\n\n")
        f.write("[보고서 생성 프롬프트]\n" + "="*40 + "\n")
        f.write(report_prompt + "\n" + "="*40 + "\n")
        f.write("[임베딩 기반 참고 문서]\n")
        # 파일명만 넘버링해서 출력
        for idx, src in enumerate(sources, 1):
            pretty_src = prettify_source_filename(src)
            f.write(f"{idx}. {pretty_src}\n")
        # RAG-LOG도 함께 기록
        if rag_logs:
            f.write("\n[RAG-LOG: 선택된 문서 및 점수]\n")
            for logline in rag_logs:
                f.write(logline + "\n")
        f.write("\n[최종 보고서]\n" + "="*40 + "\n")
        for name in ["서론"] + section_titles + ["결론"]:
            f.write(f"{name}\n")
            f.write(body_sections[name] + "\n")
            f.write("\n")
    print(f"\n[log 파일 저장 완료: {log_path}")

if __name__ == "__main__":
    main() 