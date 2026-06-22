#!/bin/bash
set -e

# Если база ещё не проиндексирована — индексируем
if [ ! -f "RAG/data/chunks.jsonl" ]; then
    echo ">>> Индексация кодовой базы..."
    python -m RAG.src.index dataset/gymhero --project-prefix gymhero
    echo ">>> Индексация завершена."
else
    echo ">>> База уже проиндексирована, пропускаем."
fi

echo ">>> Запуск Streamlit..."
exec streamlit run RAG/src/ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.fileWatcherType=none