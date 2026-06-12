import json
import argparse
import chromadb
import sentence_transformers

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG")
    parser.add_argument("--questions", type=str, default="data/eval_questions.json", help="Path to evaluation questions file")
    parser.add_argument("--output", type=str, default="RAG/results.json", help="Path to output results file")
    args = parser.parse_args()

    model = sentence_transformers.SentenceTransformer("BAAI/bge-m3")

    client = chromadb.PersistentClient(path="RAG/chroma_db")
    collection = client.get_collection("gymhero_code")

    with open(args.questions, "r", encoding="utf-8") as f:
        queries = json.load(f)

    output_results = []
    
    for q in queries:
        q_id = q["question_id"]
        query_text = q["query"]
        
        query_vector = model.encode(query_text).tolist()

        db_results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
        )
        
        chunks = db_results['ids'][0]
        output_results.append({
            "question_id": q_id,
            "top_5_chunks": chunks
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()