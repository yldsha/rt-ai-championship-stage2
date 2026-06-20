#!/usr/bin/env python3
"""
CodeLens RAG — Alpha Parameter Tuner.
Runs evaluation across multiple alpha values and plots Precision@5 curve.
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from RAG.searcher import HybridSearcher
from accuracy_checking.score import score_question

def main():
    questions_path = Path("accuracy_checking/eval_questions.json")
    
    if not questions_path.exists():
        print(f"Eval questions file not found: {questions_path}")
        return

    print("Initializing HybridSearcher...")
    searcher = HybridSearcher()
    
    with open(questions_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Successfully loaded questions for testing: {len(queries)}")

    alpha_grid = np.linspace(0.0, 1.0, 21)
    mean_scores = []

    print("\n=== Running optimization process ===")

    searchtime = []
    
    for alpha in alpha_grid:
        alpha = round(alpha, 2) 
        per_question_scores = []
        
        for q in queries:
            query_text = q["query"]
            correct_chunks = q.get("correct_chunk_ids", [])

            start_time = time.time()
            
            results = searcher.search(query_text, alpha=alpha, top_k=5)

            end_time = time.time()
            
            searchtime.append(end_time - start_time)

            predicted_chunks = [chunk["chunk_id"] for (_, chunk) in results]
            
            score = score_question(predicted_chunks, correct_chunks)
            per_question_scores.append(score)
            
        mean_p5 = sum(per_question_scores) / len(per_question_scores)
        mean_scores.append(mean_p5)
        
        print(f"Alpha: {alpha:<4} | Mean Precision@5: {mean_p5:.4f}")

    best_idx = np.argmax(mean_scores)
    best_alpha = alpha_grid[best_idx]
    best_score = mean_scores[best_idx]
    
    print("\n=== Optimization Results ===")
    print(f"Best Alpha coefficient: {best_alpha:.2f}")
    print(f"Maximum Mean Precision@5: {best_score:.4f}")

    print("\nGenerating dependency graph...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(alpha_grid, mean_scores, marker='o', linestyle='-', color='#2ca02c', linewidth=2, label='Precision@5')
    
    plt.scatter(best_alpha, best_score, color='red', s=120, zorder=5, label=f'Best Alpha ({best_alpha:.1f})')
    
    plt.title("Optimization of Hybrid Search: Dependency of Precision@5 on Alpha", fontsize=14, pad=15)
    plt.xlabel("Alpha Coefficient (1.0 = Only Vectors, 0.0 = Only BM25)", fontsize=12)
    plt.ylabel("Mean Precision@5", fontsize=12)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower center')
    
    plt.annotate(f"Max: {best_score:.3f}", 
                 xy=(best_alpha, best_score), 
                 xytext=(best_alpha, best_score + 0.015),
                 ha="center",                             
                 va="bottom",                             
                 weight="bold",
                 color="red")

    output_img = "accuracy_checking/alpha_optimization_curve.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graph successfully saved to: {output_img}")
    print(f"Time: {sum(searchtime) / len(searchtime)}")

if __name__ == "__main__":
    main()