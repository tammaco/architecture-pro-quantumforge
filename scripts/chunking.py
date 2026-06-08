import pathlib
import pickle
import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

KB_PATH = pathlib.Path(__file__).parent.parent / "knowledge_base"
CACHE_PATH = pathlib.Path(__file__).parent.parent / "chunks_cache.pkl"  # корень

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_SEPARATORS = json.loads(os.getenv("CHUNK_SEPARATORS", '["\n\n", "\n", ". ", " ", ""]'))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=CHUNK_SEPARATORS,
    length_function=len,
)

all_chunks = []

for file_path in sorted(KB_PATH.glob("*.txt")):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    doc = Document(page_content=content, metadata={"source": file_path.name})
    chunks = text_splitter.split_documents([doc])
    
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk.page_content,
            "source_file": file_path.name,
            "chunk_index": i,
            "char_count": len(chunk.page_content),
        })
    
    print(f"   {file_path.name}: {len(chunks)} чанков")

with open(CACHE_PATH, "wb") as f:
    pickle.dump({
        "chunks": all_chunks,
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "total_chunks": len(all_chunks)
        }
    }, f)

print(f"\nВсего чанков: {len(all_chunks)}")