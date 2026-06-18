#!/usr/bin/env python3
"""
CodeLens RAG — Alpha Parameter Tuner.
Runs evaluation across multiple alpha values and plots Precision@5 curve.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Импортируем компоненты твоей системы
from RAG.searcher import HybridSearcher
from accuracy_checking.score import score_question

def main():
    # 1. Настройка путей (используем те же дефолты, что и в твоих скриптах)
    questions_path = Path("accuracy_checking/eval_questions.json")
    
    if not questions_path.exists():
        print(f"[-] Файл с вопросами не найден по пути: {questions_path}")
        return

    print("[+] Инициализация поискового движка HybridSearcher...")
    searcher = HybridSearcher()
    
    # Загружаем валидационные вопросы
    with open(questions_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"[+] Успешно загружено вопросов для теста: {len(queries)}")

    # 2. Определяем сетку значений alpha (от 0.0 до 1.0 с шагом 0.1)
    # Получим список: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_grid = np.linspace(0.0, 1.0, 21)
    mean_scores = []

    print("\n=== Запуск процесса оптимизации ===")
    
    # Слой инференса и подсчета метрик в памяти
    for alpha in alpha_grid:
        alpha = round(alpha, 2)  # Избавляемся от хвостов вещественных чисел плавающей точки
        per_question_scores = []
        
        for q in queries:
            query_text = q["query"]
            correct_chunks = q.get("correct_chunk_ids", [])
            
            # Прогоняем поиск с текущим значением alpha
            results = searcher.search(query_text, alpha=alpha, top_k=5)
            
            # Вытаскиваем только ID предсказанных чанков
            predicted_chunks = [chunk["chunk_id"] for (_, chunk) in results]
            
            # Считаем Precision@5 для конкретного вопроса по правилам из score.py
            score = score_question(predicted_chunks, correct_chunks)
            per_question_scores.append(score)
            
        # Считаем среднее значение Precision@5 по всему датасету для текущей alpha
        mean_p5 = sum(per_question_scores) / len(per_question_scores)
        mean_scores.append(mean_p5)
        
        print(f"Alpha: {alpha:<4} | Mean Precision@5: {mean_p5:.4f}")

    best_idx = np.argmax(mean_scores)
    best_alpha = alpha_grid[best_idx]
    best_score = mean_scores[best_idx]
    
    print("\n=== Результаты оптимизации ===")
    print(f"🔥 Лучший коэффициент Alpha: {best_alpha:.2f}")
    print(f"🎯 Максимальный Mean Precision@5: {best_score:.4f}")

    print("\n[+] Генерация графика зависимости...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(alpha_grid, mean_scores, marker='o', linestyle='-', color='#2ca02c', linewidth=2, label='Precision@5')
    
    plt.scatter(best_alpha, best_score, color='red', s=120, zorder=5, label=f'Best Alpha ({best_alpha:.1f})')
    
    plt.title("Оптимизация гибридного поиска: Зависимость Precision@5 от Alpha", fontsize=14, pad=15)
    plt.xlabel("Коэффициент Alpha (1.0 = Только Векторы, 0.0 = Только BM25)", fontsize=12)
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

    # Сохраняем результат в файл на диск
    output_img = "accuracy_checking/alpha_optimization_curve.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[+] График успешно сохранен в: {output_img}")

if __name__ == "__main__":
    main()