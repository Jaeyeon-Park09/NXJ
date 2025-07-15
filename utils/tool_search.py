# utils/tool_search.py
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.embed_query import embed_query # Assuming this provides embed_query function
from utils.load_all_embeddings import load_all_embeddings
from pathlib import Path
from collections import defaultdict
from langchain_core.tools import BaseTool
import os

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    # 콘솔 핸들러만 유지
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class SearchDocumentsTool(BaseTool):
    name: str = "search_documents"
    description: str = "쿼리에 가장 관련성이 높은 문서 청크와 메타데이터를 검색합니다."

    def _run(self, query: str) -> str:
        try:
            logger.info(f"[SearchDocuments] Query: {query}")
            
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
            
            query_vec = embed_query(query)
            if query_vec is None or len(query_vec) == 0:
                result = {
                    "documents": [],
                    "sources": [],
                    "paths": [],
                    "error": "쿼리 임베딩 생성 실패"
                }
                return json.dumps(result, ensure_ascii=False)

            # 1. 코사인 유사도 계산
            sims = cosine_similarity(query_vec.reshape(1, -1), np.array(embs_all)).flatten()
            
            # 초기 결과 조합 (유사도, 문서 내용, 원본 파일 경로)
            initial_results = [(sim, doc, src) for sim, doc, src in zip(sims, docs_all, srcs_all)]

            # 2. 키워드 매칭 점수 추가 (재랭킹)
            # 쿼리에서 중요한 키워드를 추출하는 간단한 방법
            query_keywords = set(query.lower().split()) 
            
            reranked_results = []
            for sim, doc, src in initial_results:
                keyword_match_score = 0
                doc_lower = doc.lower()
                for keyword in query_keywords:
                    if keyword in doc_lower:
                        keyword_match_score += 0.03 # 키워드 하나당 0.1점 추가 (가중치 조절 필요)
                
                # 최종 점수 = 코사인 유사도 + 키워드 매칭 점수
                # 가중치를 조절하여 코사인 유사도의 중요도를 유지
                final_score = sim + keyword_match_score 
                reranked_results.append((final_score, doc, src, sim, keyword_match_score))

            # 3. 소스별 상위 문서 선택 (기존 로직 유지)
            # 각 파일에서 가장 유사도가 높은 청크를 가져오는 방식으로 변경하여 전체적인 문서 관련성 향상
            # 단순히 상위 5개 청크를 가져오기보다, 각 파일의 가장 관련성 높은 청크를 포함
            
            # 각 소스별 최고 점수를 가진 청크를 찾기
            best_chunks_per_source = defaultdict(lambda: {'score': -1, 'doc': '', 'src': '', 'sim': 0, 'kw': 0})
            for score, doc, src, sim, kw in reranked_results:
                if score > best_chunks_per_source[src]['score']:
                    best_chunks_per_source[src] = {'score': score, 'doc': doc, 'src': src, 'sim': sim, 'kw': kw}

            # 모든 소스에서 최고 점수 청크들을 리스트로 만듦
            final_results = []
            for src_info in best_chunks_per_source.values():
                final_results.append((src_info['score'], src_info['doc'], src_info['src'], src_info['sim'], src_info['kw']))

            # 최종적으로 전체 중 상위 5개 선택
            final_results = sorted(final_results, key=lambda x: x[0], reverse=True)[:3]
            
            logger.info(f"[SearchDocuments] Similarity - max: {final_results[0][0] if final_results else 0:.4f}, mean: {np.mean([res[0] for res in final_results]) if final_results else 0:.4f}")
            logger.info(f"[SearchDocuments] Found {len(final_results)} relevant documents")
            logger.info(f"[SearchDocuments] Sources: {[Path(src).name for _, _, src in final_results]}")
            
            result = {
                "documents": [doc for _, doc, _ in final_results],
                "sources": [Path(src).name for _, _, src in final_results],
                "paths": [str(Path(src)) for _, _, src in final_results],
                "scores": [float(sim) for _, _, sim, _, _ in final_results]
            }
            
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

def search_context_for_rag(query: str, top_k: int = 5):
    """
    쿼리에 대해 임베딩 기반으로 context(문서 청크)와 출처 리스트를 반환.
    ollama_deep_researcher의 web_research 단계 대체용.
    """
    docs_all, embs_all, srcs_all = load_all_embeddings()
    if len(embs_all) == 0:
        return {
            "context": "",
            "sources": [],
            "paths": [],
            "message": "검색할 문서가 없습니다."
        }
    query_vec = embed_query(query)
    if query_vec is None or len(query_vec) == 0:
        return {
            "context": "",
            "sources": [],
            "paths": [],
            "error": "쿼리 임베딩 생성 실패"
        }
    sims = cosine_similarity(query_vec.reshape(1, -1), np.array(embs_all)).flatten()
    initial_results = [(sim, doc, src) for sim, doc, src in zip(sims, docs_all, srcs_all)]
    query_keywords = set(query.lower().split())
    reranked_results = []
    for sim, doc, src in initial_results:
        keyword_match_score = 0
        doc_lower = doc.lower()
        for keyword in query_keywords:
            if keyword in doc_lower:
                keyword_match_score += 0.03 # Hyper parameter. 기존 0.1이었으나 너무 높아 낮춤.
        final_score = sim + keyword_match_score
        reranked_results.append((final_score, doc, src, sim, keyword_match_score))
    best_chunks_per_source = defaultdict(lambda: {'score': -1, 'doc': '', 'src': '', 'sim': 0, 'kw': 0})
    for score, doc, src, sim, kw in reranked_results:
        if score > best_chunks_per_source[src]['score']:
            best_chunks_per_source[src] = {'score': score, 'doc': doc, 'src': src, 'sim': sim, 'kw': kw}
    final_results = []
    for src_info in best_chunks_per_source.values():
        final_results.append((src_info['score'], src_info['doc'], src_info['src'], src_info['sim'], src_info['kw']))
    final_results = sorted(final_results, key=lambda x: x[0], reverse=True)[:top_k]
    # 로그 추가: 어떤 문서가 뽑혔는지, 유사도/키워드 점수 등
    rag_logs = []
    for idx, (score, doc, src, sim, kw) in enumerate(final_results, 1):
        doc_snippet = doc.replace('\n', ' ')[:80]
        if len(doc.replace('\n', ' ')) > 80:
            doc_snippet += '...'
        logline = f"[RAG-LOG] 선택 {idx}: 파일={Path(src).name}, 유사도={sim:.4f}, 키워드가중치={kw:.4f}, 최종점수={score:.4f}, 청크={doc_snippet}"
        logger.info(logline)
        rag_logs.append(logline)
    context = "\n\n".join([doc for _, doc, _, _, _ in final_results])
    sources = [Path(src).name for _, _, src, _, _ in final_results]
    paths = [str(Path(src)) for _, _, src, _, _ in final_results]
    return {
        "context": context,
        "sources": sources,
        "paths": paths,
        "scores": [float(score) for score, _, _, _, _ in final_results],
        "rag_logs": rag_logs
    }

# 인스턴스 생성
search_documents = SearchDocumentsTool()