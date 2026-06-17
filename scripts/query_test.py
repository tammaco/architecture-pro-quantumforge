import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")

INDEX_PATH = Path(__file__).parent.parent / "faiss_index.pkl"

with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)

faiss_index = index_data["faiss_index"]
chunks = index_data["chunks"]

sdk = YCloudML(folder_id=YANDEX_CLOUD_FOLDER, auth=YANDEX_CLOUD_API_KEY)
query_embedder = sdk.models.text_embeddings('query')

def search(query: str, top_k: int = 3):
    result = query_embedder.run(query)
    query_emb = np.array(result.embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_emb, top_k)
    
    print(f"\n{query}\n")
    for i, idx in enumerate(indices[0]):
        print(f"[{i+1}] {chunks[idx]['source_file']} (dist: {distances[0][i]:.4f})")
        print(f"    {chunks[idx]['text']}\n") 

if __name__ == "__main__":
    search("Марья")
    search("Крузинштерн")