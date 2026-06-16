from RAG.searcher import HybridSearcher

def run_test():
    # Инициализируем наш поисковик
    searcher = HybridSearcher()
    
    # Запрос, который содержит и смысл, и специфическое слово
    query = "как реализовать авторизацию" 
    
    # Проверяем гибридный поиск (alpha=0.5 — баланс 50/50)
    print(f"--- Тестируем запрос: '{query}' ---")
    results = searcher.search(query, alpha=0.5, top_k=3)
    
    for i, (score, chunk) in enumerate(results):
        print(f"\nРезультат #{i+1} (Score: {score:.4f})")
        print(f"Файл: {chunk.get('path', 'unknown')}")
        print(f"Код: {chunk['code'][:100]}...") # выводим начало кода

if __name__ == "__main__":
    run_test()