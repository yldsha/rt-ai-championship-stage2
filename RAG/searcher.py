import json

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridSearcher:
    def __init__(self, db_path="RAG/chroma_db", chunks_file="RAG/data/chunks.jsonl"):
        # 1 Загрузка данных
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        # 2
        self.model = SentenceTransformer("BAAI/bge-m3")

        # 3 BM25 (по всему корпусу)
        corpus = [chunk["code"] for chunk in self.data]
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 4 Инициализация ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="gymhero_code")

    def _min_max_scale(self, scores):
        scores = np.array(scores)
        if np.max(scores) - np.min(scores) == 0:
            return np.zeros_like(scores)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def search(self, query: str, alpha: float = 0.5, top_k: int = 5):
        if not query.strip():
            return []

        # 1 Векторный поиск
        # Используем модель для получения вектора запроса нужной размерности
        query_embedding = self.model.encode(query)
        vec_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=len(self.data),
            include=["distances"],
        )

        # 2 BM25 для всего корпуса
        bm25_scores = self.bm25.get_scores(query.split())

        # 3 Подготовка векторов (заполняем скорами по индексам)
        all_vec_scores = np.zeros(len(self.data))

        # Создаем маппинг ID чанка -> индекс в self.data
        id_to_idx = {item["chunk_id"]: idx for idx, item in enumerate(self.data)}

        # vec_results['ids'] содержит список списков, берем [0]
        # В новых версиях Chroma IDs возвращаются автоматически
        ids = vec_results.get("ids", [[]])[0]
        distances = vec_results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            dist = distances[i]
            # Chroma: 0 - полное совпадение, 2 - полная противоположность
            score = 1.0 - (dist / 2.0)

            idx = id_to_idx.get(chunk_id)
            if idx is not None:
                all_vec_scores[idx] = score

        # 4 Нормализация и объединение
        norm_vec = self._min_max_scale(all_vec_scores)
        norm_bm25 = self._min_max_scale(bm25_scores)

        final_scores = (alpha * norm_vec) + ((1 - alpha) * norm_bm25)

        # 5 Сортировка всего списка
        results = sorted(zip(final_scores, self.data), key=lambda x: x[0], reverse=True)
        return results[:top_k]
