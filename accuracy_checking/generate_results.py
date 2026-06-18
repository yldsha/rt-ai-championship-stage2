import json
import argparse

from RAG.searcher import HybridSearcher


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG")
    parser.add_argument("--questions", type=str, default="accuracy_checking/eval_questions.json", help="Path to evaluation questions file")
    parser.add_argument("--output", type=str, default="accuracy_checking/results.json", help="Path to output results file")
    args = parser.parse_args()

    searcher = HybridSearcher()

    with open(args.questions, "r", encoding="utf-8") as f:
        queries = json.load(f)

    output_results = []
    
    for q in queries:
        q_id = q["question_id"]
        query = q["query"]
        
        results = searcher.search(query, alpha=0.8, top_k=5)
        chunks = []

        for (score, chunk) in results:
            chunks.append(chunk["chunk_id"])
        
        output_results.append({
            "question_id": q_id,
            "top_5_chunks": chunks
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()