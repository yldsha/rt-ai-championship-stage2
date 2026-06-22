import time

from RAG.src.searcher import HybridSearcher


def test_latency(queries: list[str]) -> float:
    searcher = HybridSearcher()

    latencies = []
    for query in queries:
        start_time = time.time()
        searcher.search(query["query"], alpha=0.5, top_k=5)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
    return sum(latencies)/len(latencies)

def main():
    with open("RAG/data/sample_queries.txt", "r", encoding="utf-8") as f:
        queries = [{"query": line.strip()} for line in f if line.strip()]
    avg_latency = test_latency(queries)
    print(f"Latency: {avg_latency:.4f} seconds")

if __name__ == "__main__":
    main()
