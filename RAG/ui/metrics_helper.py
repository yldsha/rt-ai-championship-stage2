"""
Вспомогательный файл для работы с метриками.
Использует generate_results.py и score.py для расчета Precision@5
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_precision(searcher=None, alpha: float = 0.5):
    debug_lines = []
    metrics = {"overall": 0.0, "ru": None, "en": None}

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
            return metrics, "\n".join(debug_lines)
        if not os.path.exists(eval_file):
            debug_lines.append(f"ОШИБКА: не найден {eval_file}")
            return metrics, "\n".join(debug_lines)

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
            return metrics, "\n".join(debug_lines)

        if not os.path.exists(results_file):
            debug_lines.append(f"ОШИБКА: results.json не был создан по пути {results_file}")
            return metrics, "\n".join(debug_lines)

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
            return metrics, "\n".join(debug_lines)

        in_language_block = False

        for raw_line in result_score.stdout.split("\n"):
            line = raw_line.strip()

            if "Mean Precision@5:" in line or "Total score:" in line:
                metrics["overall"] = float(line.split(":")[-1].strip())
                continue

            if line.startswith("By language"):
                in_language_block = True
                continue

            if in_language_block:
                if line.startswith("ru:"):
                    metrics["ru"] = float(line.split(":")[1].strip().split(" ")[0])
                elif line.startswith("en:"):
                    metrics["en"] = float(line.split(":")[1].strip().split(" ")[0])
                elif line == "" or line.startswith("Per-question") or line.startswith("By "):
                    in_language_block = False

        debug_lines.append(f"Итоговые метрики: {metrics}")

        if metrics["overall"] == 0.0:
            debug_lines.append("Не удалось найти строку с общей метрикой в выводе score.py")

        return metrics, "\n".join(debug_lines)

    except Exception as e:
        debug_lines.append(f"Исключение: {type(e).__name__}: {e}")
        return metrics, "\n".join(debug_lines)