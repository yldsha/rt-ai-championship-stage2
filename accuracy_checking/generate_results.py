import argparse
import json

import chromadb
import sentence_transformers
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RAG.searcher import HybridSearcher

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG")
    parser.add_argument("--questions", type=str, default="accuracy_checking/eval_questions.json")
    parser.add_argument("--output", type=str, default="accuracy_checking/results.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    searcher = HybridSearcher(
        db_path="RAG/chroma_db",
        chunks_file="RAG/data/chunks.jsonl"
    )

    with open(args.questions, "r", encoding="utf-8") as f:
        queries = json.load(f)

    output_results = []
    for q in queries:
        results = searcher.search(q["query"], alpha=args.alpha, top_k=5)
        # results — список (score, chunk_dict)
        top5 = [chunk["chunk_id"] for score, chunk in results]
        output_results.append({
            "question_id": q["question_id"],
            "top_5_chunks": top5
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(output_results)} results to {args.output}")


if __name__ == "__main__":
    main()
