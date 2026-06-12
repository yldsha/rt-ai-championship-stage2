"""
Генерация эмбеддингов для чанков из RAG/chunks_ast.jsonl.
Используется модель BGE-M3 от BAAI.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb

MODEL_NAME = "BAAI/bge-m3"             
CHUNKS_FILE = Path("RAG/chunks.jsonl")
CHROMA_DB_DIR = "RAG/chroma_db"
COLLECTION_NAME = "gymhero_code"

def get_embedding(text: str, model: SentenceTransformer) -> list:
    """
    Превращает текст в список чисел (эмбеддинг)
    """
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def main():
    print(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Чтение чанков из {CHUNKS_FILE}...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"Загружено {len(chunks)} чанков")
    
    print("Генерация эмбеддингов...")
    
    # Инициализация ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for i, chunk in enumerate(tqdm(chunks)):
        # Собираем текст для эмбеддинга
        text_parts = []
        if chunk.get("docstring"):
            text_parts.append(chunk["docstring"])
        if chunk.get("code"):
            text_parts.append(chunk["code"])
        
        text = "\n".join(text_parts) if text_parts else chunk.get("code", "")
        
        if not text:
            text = " ".join([f"{k}: {v}" for k, v in chunk.items() 
                            if k not in ["chunk_id", "start_line", "end_line"]])
        
        embedding = get_embedding(text, model)
        
        # Получаем ID чанка или генерируем новый
        chunk_id = chunk.get("chunk_id", str(i))
        
        # Фильтруем метадату для ChromaDB (только str, int, float, bool)
        meta = {}
        for k, v in chunk.items():
            if k not in ["code", "docstring"] and isinstance(v, (str, int, float, bool)):
                meta[k] = v
        
        ids.append(chunk_id)
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(meta)

    # Сохраняем в ChromaDB батчами
    print(f"Сохранение эмбеддингов в ChromaDB (папка {CHROMA_DB_DIR})...")
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size)):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    
    print(f"Готово! Сгенерировано и сохранено {len(ids)} эмбеддингов в ChromaDB.")
    if embeddings:
        print(f"Размерность эмбеддинга: {len(embeddings[0])}")


if __name__ == "__main__":
    main()