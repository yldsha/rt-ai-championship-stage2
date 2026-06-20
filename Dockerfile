# CodeLens RAG — Dockerfile
# Базовый образ с Python 3.11 (соответствует версии в CI)
FROM python:3.11-slim

# Системные зависимости, нужные для torch/chromadb/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только requirements.txt — это позволяет Docker
# кэшировать слой с зависимостями и не переустанавливать их
# при каждом изменении кода (только при изменении requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Теперь копируем весь остальной проект
COPY . .

# Порт, на котором Streamlit слушает по умолчанию
EXPOSE 8501

# Healthcheck — Docker сможет понять, что приложение реально запустилось
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск Streamlit. --server.address=0.0.0.0 обязателен для Docker —
# иначе приложение будет слушать только localhost ВНУТРИ контейнера
# и не будет доступно снаружи.
ENTRYPOINT ["streamlit", "run", "RAG/ui/app.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.fileWatcherType=none"]