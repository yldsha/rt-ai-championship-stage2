"""
Генерация эмбеддингов для чанков из RAG/chunks_ast.jsonl.
Используется модель BGE-M3 от BAAI.
"""

import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-m3"             
CHUNKS_FILE = Path("RAG/chunks_ast.jsonl")
OUTPUT_JSON = Path("RAG/embeddings.json")
OUTPUT_PKL = Path("RAG/embeddings_list.pkl")

def get_embedding(text: str, model: SentenceTransformer) -> list:
    """
    Превращает текст в список чисел (эмбеддинг)
    """
    
    # Генерируем эмбеддинг
    embedding = model.encode(text, normalize_embeddings=True)
    
    # Превращаем numpy-массив в обычный список
    return embedding.tolist()


def main():
    print(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Читаем чанки из JSONL
    print(f"Чтение чанков из {CHUNKS_FILE}...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"Загружено {len(chunks)} чанков")
    
    # Генерируем эмбеддинги
    print("Генерация эмбеддингов...")
    embeddings_list = []
    
    for chunk in tqdm(chunks):
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
        
        embeddings_list.append({
            "chunk": chunk,
            "embedding": embedding
        })
    
    # Сохраняем в JSON
    print(f"Сохранение в {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(embeddings_list, f, ensure_ascii=False, indent=2)
    
    # Сохраняем в pickle
    print(f"Сохранение в {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(embeddings_list, f)
    
    print(f"Готово! Сгенерировано {len(embeddings_list)} эмбеддингов.")
    if embeddings_list:
        print(f"Размерность эмбеддинга: {len(embeddings_list[0]['embedding'])}")


if __name__ == "__main__":
    main()