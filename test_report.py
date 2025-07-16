import os
import json
from utils.tool_search import search_context_for_rag
from utils.tool_report import (
    write_section_titles_prompt, write_intro_prompt, write_section_body_prompt, write_conclusion_prompt
)
from utils.tool_section import fix_section_markers
from langchain_community.llms import Ollama
import difflib
import re

def filter_context(raw_context):
    lines = [l.strip() for l in raw_context.split("\n") if l.strip()]
    filtered = [
        line for line in lines
        if not (line.startswith("|") or line.endswith("|") or "Unnamed" in line or set(line) <= set("| -:.") or len(line) < 10)
    ]
    if not filtered:
        filtered = lines[:3]
    result = ""
    for line in filtered:
        if len(result) + len(line) + 1 > 2000:
            break
        result += line + "\n"
    return result.strip()

def prettify_source_filename(filename):
    base = filename.replace('+', ' ')
    if base.endswith('.json'):
        base = base[:-5]
    if base.endswith('.txt'):
        base = base[:-4]
    return base

def postprocess_citations(report_text, context, sources):
    from collections import OrderedDict
    context_lines = [line.strip() for line in context.split("\n") if line.strip()]
    context_map = OrderedDict((line, prettify_source_filename(sources[i]) if i < len(sources) else "") for i, line in enumerate(context_lines))
    processed = re.sub(r'\n*참고문헌:.*?(?=\n\[|\n#|\n##|\n결론|\Z)', '', report_text, flags=re.DOTALL)
    processed = re.sub(r'\n*참고문헌:.*', '', processed, flags=re.DOTALL)
    processed = re.sub(r'\n*출처:.*?(?=\n\[|\n#|\n##|\n결론|\Z)', '', processed, flags=re.DOTALL)
    processed = re.sub(r'\n*출처:.*', '', processed, flags=re.DOTALL)
    cited_files = set()
    for ctx, src in sorted(context_map.items(), key=lambda x: -len(x[0])):
        pattern = re.escape(ctx) + r'\s*\(출처:'
        if re.search(pattern, processed):
            continue
        if ctx in processed:    
            processed = re.sub(r'(' + re.escape(ctx) + r')(?!\s*\(출처:)', r'\1 (출처: ' + src + ')', processed)
            cited_files.add(src)
    if context_map:
        unique_files = list(OrderedDict.fromkeys(context_map.values()))
        processed = processed.rstrip() + "\n\n참고문헌:\n" + "\n".join(f"- {f}" for f in unique_files)
    return processed

def remove_introductory_sentences(text, section_title):
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
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])', text) if s.strip()]
    seen = set()
    result = []
    for s in sentences:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return ' '.join(result)

def is_valid_subtitle(line):
    invalid_patterns = [
        r"아래는", r"소제목을 선정", r"소제목 후보", r"참고 문서", r"참고자료", r"목차는", r"소제목", r"^\s*$",
        r"Based on", r"I have selected", r"Here are the subtopics", r"based reference document", r"subtopics", r"reference document"
    ]
    return not any(re.search(pat, line, re.IGNORECASE) for pat in invalid_patterns)

def call_llm_for_section(llm, section_name, question, context, section_title=None, all_section_titles=[]):
    if section_name == "서론":
        prompt = write_intro_prompt(question, context)
    elif section_name == "결론":
        prompt = write_conclusion_prompt(question, context)
    elif section_name == "목차":
        prompt = write_section_titles_prompt(question, context)
    elif section_name == "소제목" and section_title is not None:
        prompt = write_section_body_prompt(question, context, section_title, all_section_titles)
    else:
        raise ValueError("section_title이 필요합니다.")
    answer = ""
    for chunk in llm.stream(prompt):
        if chunk and isinstance(chunk, str) and chunk.strip():
            print(chunk, end="", flush=True)
            answer += chunk
    print(f"\n[{section_name} 생성 완료]")
    return answer.strip()

def remove_leading_section_title(text, section_title):
    import re
    pattern = rf"^(#*\s*\**\s*{re.escape(section_title)}(\s*\([^)]+\))?\**\s*:?[\-–—]?\s*)"
    return re.sub(pattern, '', text, flags=re.IGNORECASE).lstrip()

def main():
    question = input("질문을 입력하세요.\n>> ").strip()
    if not question:
        print("질문을 입력해야 합니다.")
        return
    search_result = search_context_for_rag(question, top_k=5)
    raw_context = search_result.get("context", "")
    sources = search_result.get("sources", [])
    rag_logs = search_result.get("rag_logs", [])
    permit_keywords = ["인허가", "허가", "승인", "등록", "신고", "행정 절차", "법적 요건", "제도"]
    permit_query = question + " " + " ".join(permit_keywords)
    permit_context = search_context_for_rag(permit_query, top_k=2).get("context", "")
    context = filter_context(raw_context + "\n" + permit_context)
    context = context[:2000] if len(context) > 2000 else context
    llm = Ollama(model="llama3:8b", temperature=0.2, base_url="http://localhost:11434")
    print(f"\n[소제목 3개 생성 프롬프트]")
    titles_text = call_llm_for_section(llm, "목차", question, context)
    print(titles_text)
    print("\n[소제목 생성 완료]")
    import re
    raw_titles = [line.strip() for line in titles_text.split('\n') if is_valid_subtitle(line.strip())]
    titles = []
    for line in raw_titles:
        titles += [t.strip("•*-·, .") for t in re.split(r"[•*\-,·]", line) if is_valid_subtitle(t.strip("•*-·, ."))]
    # 중복 제거 및 3개만 추출
    titles = [t for i, t in enumerate(titles) if t and t not in titles[:i]]
    section_titles = (titles + ["소제목1", "소제목2", "소제목3"])[:3]
    print("\n[서론 생성]")
    intro = call_llm_for_section(llm, "서론", question, context)
    print("\n[소제목1 생성]")
    section1 = call_llm_for_section(llm, "소제목", question, context, section_titles[0], section_titles)
    print("\n[소제목2 생성]")
    section2 = call_llm_for_section(llm, "소제목", question, context, section_titles[1], section_titles)
    print("\n[소제목3 생성]")
    section3 = call_llm_for_section(llm, "소제목", question, context, section_titles[2], section_titles)
    print("\n[결론 생성]")
    conclusion = call_llm_for_section(llm, "결론", question, context)
    body_sections = {
        "서론": remove_duplicate_sentences(remove_introductory_sentences(intro, "서론")),
        section_titles[0]: remove_leading_section_title(remove_duplicate_sentences(remove_introductory_sentences(section1, section_titles[0])), section_titles[0]),
        section_titles[1]: remove_leading_section_title(remove_duplicate_sentences(remove_introductory_sentences(section2, section_titles[1])), section_titles[1]),
        section_titles[2]: remove_leading_section_title(remove_duplicate_sentences(remove_introductory_sentences(section3, section_titles[2])), section_titles[2]),
        "결론": remove_duplicate_sentences(remove_introductory_sentences(conclusion, "결론")),
    }

    print(body_sections[section_titles[0]])
    print(body_sections[section_titles[1]])
    print(body_sections[section_titles[2]])
    # 참고문헌은 결론 이후 한 번만 출력
    def make_references(context, sources):
        from collections import OrderedDict
        context_lines = [line.strip() for line in context.split("\n") if line.strip()]
        context_map = OrderedDict((line, prettify_source_filename(sources[i]) if i < len(sources) else "") for i, line in enumerate(context_lines))
        unique_files = list(OrderedDict.fromkeys(context_map.values()))
        if unique_files:
            return "참고문헌:\n" + "\n".join(f"- {f}" for f in unique_files)
        return ""
    references_block = make_references(context, sources)
    print("\n[최종 보고서]")
    print("="*40)
    print("[목차]")
    print("서론")
    for t in section_titles:
        print(t)
    print("결론\n")
    print("[본문 시작]")
    print("서론\n" + body_sections["서론"] + "\n")
    for t in section_titles:
        print(f"{t}\n{body_sections[t]}\n")
    print("결론\n" + body_sections["결론"] + "\n")
    if references_block:
        print(references_block)
    print("="*40)
    import datetime
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"reportgen_sectionwise_{now_str}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"질문: {question}\n\n")
        f.write("[임베딩 기반 참고 문서]\n")
        for idx, src in enumerate(sources, 1):
            pretty_src = prettify_source_filename(src)
            f.write(f"{idx}. {pretty_src}\n")
        if rag_logs:
            f.write("\n[RAG-LOG: 선택된 문서 및 점수]\n")
            for logline in rag_logs:
                f.write(logline + "\n")
        f.write("\n[최종 보고서]\n" + "="*40 + "\n")
        f.write("[목차]\n서론\n")
        for t in section_titles:
            f.write(f"{t}\n")
        f.write("결론\n\n[본문 시작]\n")
        f.write(f"서론\n{body_sections['서론']}\n\n")
        for t in section_titles:
            f.write(f"{t}\n{body_sections[t]}\n\n")
        f.write(f"결론\n{body_sections['결론']}\n\n")
        if references_block:
            f.write(references_block + "\n")
    print(f"\n[log 파일 저장 완료: {log_path}")

if __name__ == "__main__":
    main() 