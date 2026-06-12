import json
from pathlib import Path
import chromadb
import sentence_transformers

model = sentence_transformers.SentenceTransformer("BAAI/bge-m3")
QUERIES_FILE = Path("../data/eval_questions.json")

# 1. Подключаемся к той же папке
client = chromadb.PersistentClient(path="RAG/chroma_db")
collection = client.get_collection("gymhero_code")

# 2. Переводим запрос пользователя в вектор с помощью той же модели
query_vector = model.encode("как проверяется токен авторизации?").tolist()

# 3. Поиск по ChromaDB
results = collection.query(
    query_embeddings=[query_vector],
    n_results=3, # Вернуть топ-3 похожих чанка
    # where={"path": "gymhero/api/dependencies.py"} # Можно наложить фильтр
)

print(results['documents']) # Тут будут лежать готовые тексты найденного кода