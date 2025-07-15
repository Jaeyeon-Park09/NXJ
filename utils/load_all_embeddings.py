import json
import numpy as np
from pathlib import Path

def load_all_embeddings(embedding_dir_path: str = "/home/james4u1/llm_repo_opr/embedding"):
    """
    임베딩 디렉토리에서 모든 문서 임베딩을 로드합니다.
    Args:
        embedding_dir_path (str): 임베딩 JSON 파일들이 있는 디렉토리 경로.
    Returns:
        all_docs (list): 모든 문서의 텍스트 리스트
        all_embs (np.ndarray): 모든 문서의 임베딩 벡터 배열
        all_filenames (list): 각 문서의 파일 이름 리스트 (상대 경로)
    """
    embedding_dir = Path(embedding_dir_path)
    all_docs, all_embs, all_filenames = [], [], []

    if not embedding_dir.exists():
        print(f"⚠️  임베딩 디렉토리가 존재하지 않습니다: {embedding_dir}")
        # 테스트용 더미 데이터 (선택 사항)
        print("⚠️  실제 임베딩 데이터가 없습니다. 테스트용 더미 데이터 사용")
        all_docs = ["의료용 휠체어는 보행이 불편한 환자를 위한 의료기기입니다."]
        all_embs = [np.random.rand(1536)]  # OpenAI 임베딩 차원 (예시)
        all_filenames = ["dummy_document.json"]
        return all_docs, np.array(all_embs), all_filenames

    for json_path in embedding_dir.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 데이터가 리스트 형태이고, 각 항목이 딕셔너리인지 확인
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                docs = [item.get("text") for item in data]
                embs = [item.get("embedding") for item in data]
                
                # text 또는 embedding이 None인 경우를 필터링하거나 오류 처리
                valid_indices = [i for i, (d, e) in enumerate(zip(docs, embs)) if d is not None and e is not None]
                
                if len(valid_indices) < len(data):
                    print(f"Warning: 일부 항목에 'text' 또는 'embedding' 키가 없거나 값이 None입니다 in {json_path.name}. 유효한 항목만 로드합니다.")

                docs = [docs[i] for i in valid_indices]
                embs = [embs[i] for i in valid_indices]

                if docs: # 유효한 문서가 있을 경우에만 추가
                    rel_name = json_path.relative_to(embedding_dir).as_posix()
                    all_docs.extend(docs)
                    all_embs.extend(embs)
                    all_filenames.extend([rel_name] * len(docs))
                elif data: # 데이터는 있지만 유효한 text/embedding 쌍이 없는 경우
                    print(f"Warning: {json_path.name} 파일에 유효한 'text'/'embedding' 쌍이 없습니다. 건너뜁니다.")
            else:
                print(f"Warning: {json_path.name} 파일의 형식이 예상과 다릅니다 (list of dicts). 건너뜁니다. 실제 타입: {type(data)}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {json_path}: {e}")
        except Exception as e:
            print(f"Error processing file {json_path}: {e}")

    if not all_docs:
        print("⚠️  임베딩 디렉토리에서 로드된 문서가 없습니다.")
        # 필요하다면 여기서도 더미 데이터 반환 로직을 추가할 수 있습니다.
        # 예:
        # print("⚠️  테스트용 더미 데이터 사용")
        # all_docs = ["의료용 휠체어는 보행이 불편한 환자를 위한 의료기기입니다."]
        # all_embs = [np.random.rand(1536)]
        # all_filenames = ["dummy_document.json"]
        # return all_docs, np.array(all_embs), all_filenames
        
    return all_docs, np.array(all_embs, dtype=object), all_filenames # dtype=object for potentially ragged arrays before conversion