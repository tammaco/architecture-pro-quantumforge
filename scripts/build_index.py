import pickle
import numpy as np
import faiss
from pathlib import Path

EMBEDDINGS_INPUT = Path(__file__).parent / "embeddings_cache.pkl"
INDEX_OUTPUT = Path(__file__).parent.parent / "faiss_index.pkl"

if not EMBEDDINGS_INPUT.exists():
    exit(1)

with open(EMBEDDINGS_INPUT, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
chunks = data["chunks"]

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

index_data = {
    "faiss_index": faiss_index,
    "chunks": chunks,
    "dimension": dimension,
    "total_vectors": faiss_index.ntotal
}

with open(INDEX_OUTPUT, "wb") as f:
    pickle.dump(index_data, f)

print(f"Размерность: {dimension}")
print(f"Количество векторов: {faiss_index.ntotal}")