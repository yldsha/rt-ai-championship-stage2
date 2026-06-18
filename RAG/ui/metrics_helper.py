"""
Вспомогательный файл для работы с метриками.
Использует generate_results.py и score.py для расчета Precision@5
"""

import json
import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_precision(searcher=None, alpha: float = 0.5):
    """
    Генерирует results.json через generate_results.py и запускает score.py.
    Возвращает (precision: float, debug_log: str)
    """
    debug_lines = []

    try:
        gen_script = os.path.join(PROJECT_ROOT, "accuracy_checking", "generate_results.py")
        eval_file = os.path.join(PROJECT_ROOT, "accuracy_checking", "eval_questions.json")
        results_file = os.path.join(PROJECT_ROOT, "accuracy_checking", "results.json")
        score_script = os.path.join(PROJECT_ROOT, "score.py")

        debug_lines.append(f"PROJECT_ROOT: {PROJECT_ROOT}")
        debug_lines.append(f"Python interpreter: {sys.executable}")
        debug_lines.append(f"gen_script exists: {os.path.exists(gen_script)}")
        debug_lines.append(f"eval_file exists: {os.path.exists(eval_file)}")
        debug_lines.append(f"score_script exists: {os.path.exists(score_script)}")

        if not os.path.exists(gen_script):
            debug_lines.append(f"ОШИБКА: не найден {gen_script}")
            return 0.0, "\n".join(debug_lines)
        if not os.path.exists(eval_file):
            debug_lines.append(f"ОШИБКА: не найден {eval_file}")
            return 0.0, "\n".join(debug_lines)


        result_gen = subprocess.run(
            [
                sys.executable,
                gen_script,
                "--questions", eval_file,
                "--output", results_file,
                "--alpha", str(alpha),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        debug_lines.append("--- generate_results.py stdout ---")
        debug_lines.append(result_gen.stdout or "(пусто)")
        debug_lines.append("--- generate_results.py stderr ---")
        debug_lines.append(result_gen.stderr or "(пусто)")

        if result_gen.returncode != 0:
            debug_lines.append(f"generate_results.py завершился с кодом {result_gen.returncode}")
            return 0.0, "\n".join(debug_lines)

        if not os.path.exists(results_file):
            debug_lines.append(f"ОШИБКА: results.json не был создан по пути {results_file}")
            return 0.0, "\n".join(debug_lines)

        result_score = subprocess.run(
            [
                sys.executable,
                score_script,
                "--predictions", results_file,
                "--questions", eval_file,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        debug_lines.append("--- score.py stdout ---")
        debug_lines.append(result_score.stdout or "(пусто)")
        debug_lines.append("--- score.py stderr ---")
        debug_lines.append(result_score.stderr or "(пусто)")

        if result_score.returncode != 0:
            debug_lines.append(f"score.py завершился с кодом {result_score.returncode}")
            return 0.0, "\n".join(debug_lines)

        for line in result_score.stdout.split("\n"):
            if "Mean Precision@5:" in line or "Total score:" in line:
                value = float(line.split(":")[-1].strip())
                debug_lines.append(f"Найдено значение: {value}")
                return value, "\n".join(debug_lines)

        debug_lines.append("Не удалось найти строку с метрикой в выводе score.py")
        return 0.0, "\n".join(debug_lines)

    except Exception as e:
        debug_lines.append(f"Исключение: {type(e).__name__}: {e}")
        return 0.0, "\n".join(debug_lines)