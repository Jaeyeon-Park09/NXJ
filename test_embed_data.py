from utils.load_all_embeddings import load_all_embeddings

docs, embs, srcs = load_all_embeddings("/home/james4u1/llm_repo_opr/embedding")  # embedding 디렉토리 경로 지정

print(f"문서 청크 개수: {len(docs)}")
print(f"임베딩 벡터 개수: {len(embs)}")
print(f"소스 파일 개수: {len(set(srcs))}")