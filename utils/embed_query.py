import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def embed_query(query: str) -> np.ndarray:
    """쿼리를 임베딩 벡터로 변환"""
    try:
        embedding = embedding_model.encode(query, convert_to_numpy=True)
        return np.array(embedding)
    except Exception as e:
        print(f"임베딩 오류: {e}")
        return None
