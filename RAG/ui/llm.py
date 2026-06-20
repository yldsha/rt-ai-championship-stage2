"""
Модуль для интеграции с LLM через облачный API (Groq).
Используется в чат-режиме интерфейса — берёт найденные фрагменты кода
и просит модель объяснить их на человеческом языке.

Не требует скачивания модели локально — работает через интернет.
Groq даёт бесплатный доступ с высокими лимитами и очень быстрым ответом.

Настройка:
    1. Зайдите на https://console.groq.com и войдите через Google
    2. API Keys -> Create API Key, скопируйте ключ (начинается с gsk_)
    3. Задайте переменную окружения GROQ_API_KEY (рекомендуется),
       либо впишите ключ напрямую в GROQ_API_KEY ниже —
       но тогда НЕ коммитьте этот файл с ключом в публичный репозиторий.
"""

import os

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Актуальный список моделей: https://console.groq.com/docs/models
MODEL_NAME = "llama-3.1-8b-instant"


def ask_llm(question: str, chunks: list[dict], timeout: int = 30) -> str:
    """
    Отправляет вопрос пользователя вместе с найденными фрагментами кода
    в облачную модель через Groq и возвращает объяснение на русском.

    Параметры:
        question — вопрос пользователя
        chunks   — список словарей вида {"chunk_id": str, "code": str}
        timeout  — максимальное время ожидания ответа в секундах

    Возвращает:
        строку с ответом модели, либо понятное сообщение об ошибке.
    """
    if not GROQ_API_KEY:
        return (
            "⚠️ Не задан API-ключ Groq. Получи бесплатный ключ на "
            "console.groq.com и задай переменную окружения GROQ_API_KEY, "
            "либо впиши его напрямую в RAG/llm.py."
        )

    if not chunks:
        return "Не нашлось подходящих фрагментов кода для этого вопроса."

    context = "\n\n---\n\n".join(
        f"# {c['chunk_id']}\n```\n{c['code']}\n```"
        for c in chunks
    )

    prompt = f"""Ты ассистент по анализу исходного кода в проекте, проиндексированном гибридной RAG-системой (векторный поиск + BM25). Кодовая база может содержать фрагменты на разных языках программирования — определяй язык по расширению файла в chunk_id (например .py, .js, .ts, .cpp, .go, .java).

Отвечай на русском языке. Будь вежлив, краток и по делу.

Вопрос пользователя: {question}

Найденные фрагменты кода (получены гибридным поиском как наиболее релевантные вопросу):
{context}

Объясни, как этот код отвечает на вопрос. Обязательно упомяни конкретные имена функций/классов и пути к файлам из фрагментов выше. Если фрагменты на разных языках — уточни это. Не выдумывай ничего, чего нет в переданном коде. Уложись в 3-5 предложений."""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer if answer else "Модель вернула пустой ответ. Попробуйте переформулировать вопрос."

    except requests.exceptions.ConnectionError:
        return "⚠️ Нет подключения к интернету. Облачная LLM требует интернет."
    except requests.exceptions.Timeout:
        return "⚠️ Модель не ответила за отведённое время. Попробуйте ещё раз."
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 401:
            return "⚠️ Неверный API-ключ Groq. Проверь GROQ_API_KEY."
        if status == 429:
            return "⚠️ Превышен лимит бесплатных запросов. Подожди немного и попробуй снова."
        return f"⚠️ Ошибка API: {e}"
    except Exception as e:
        return f"⚠️ Ошибка при обращении к LLM: {e}"


def check_ollama_status() -> bool:
    return bool(GROQ_API_KEY)


if __name__ == "__main__":
    # Быстрый тест из командной строки:
    #   export GROQ_API_KEY=gsk_...
    #   python RAG/llm.py
    test_chunks = [
        {
            "chunk_id": "gymhero/security.py:create_access_token:12",
            "code": (
                "def create_access_token(subject, expires_delta=None):\n"
                "    expire = datetime.utcnow() + expires_delta\n"
                "    to_encode = {'exp': expire, 'sub': str(subject)}\n"
                "    return jwt.encode(to_encode, settings.SECRET_KEY)"
            ),
        }
    ]
    print("API-ключ задан:", check_ollama_status())
    print("\nОтвет модели:\n")
    print(ask_llm("как создаётся токен доступа?", test_chunks))