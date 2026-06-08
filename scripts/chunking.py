import pathlib
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

KB_PATH = pathlib.Path(__file__).parent.parent / "knowledge_base"
CACHE_PATH = pathlib.Path(__file__).parent.parent / "chunks_cache.pkl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
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
            "start_index": i * (CHUNK_SIZE - CHUNK_OVERLAP), 
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