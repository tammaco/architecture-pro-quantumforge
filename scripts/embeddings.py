import pickle
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")

CACHE_INPUT = Path(__file__).parent.parent / "chunks_cache.pkl"
CACHE_OUTPUT = Path(__file__).parent.parent / "embeddings_cache.pkl"

if not CACHE_INPUT.exists():
    exit(1)

with open(CACHE_INPUT, "rb") as f:
    cached = pickle.load(f)

chunks = cached["chunks"]

sdk = YCloudML(folder_id=YANDEX_CLOUD_FOLDER, auth=YANDEX_CLOUD_API_KEY)
doc_embedder = sdk.models.text_embeddings('query')

embeddings_list = []

start_time = time.time()
for i, chunk in enumerate(chunks):
    result = doc_embedder.run(chunk["text"])
    embeddings_list.append(result.embedding)
    
    if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
        print(f"   Прогресс: {i+1}/{len(chunks)} чанков")

elapsed = time.time() - start_time
print(f"\n Время векторизации: {elapsed:.1f} с")

import numpy as np
embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

with open(CACHE_OUTPUT, "wb") as f:
    pickle.dump({
        "embeddings": embeddings_matrix,
        "chunks": chunks,
        "count": len(embeddings_matrix),
        "dimension": embeddings_matrix.shape[1]
    }, f)

print(f"Всего эмбеддингов: {embeddings_matrix.shape[0]}")
print(f"Размерность: {embeddings_matrix.shape[1]}")