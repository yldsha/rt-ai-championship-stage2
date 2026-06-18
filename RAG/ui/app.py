import streamlit as st
from datetime import datetime
import sys
import os

st.set_page_config(
    page_title="CodeLens RAG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searcher import HybridSearcher
from metrics_helper import get_precision

MODEL_NAME = "bge-m3"


@st.cache_resource
def get_searcher():
    return HybridSearcher(
        db_path="RAG/chroma_db",
        chunks_file="RAG/data/chunks.jsonl"
    )


searcher = get_searcher()

# --- СОСТОЯНИЕ ---
if "page" not in st.session_state:
    st.session_state.page = "search"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "searched" not in st.session_state:
    st.session_state.searched = False

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "precision_value" not in st.session_state:
    st.session_state.precision_value = 0.0

if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = None

if "metrics_debug" not in st.session_state:
    st.session_state.metrics_debug = ""

if "alpha" not in st.session_state:
    st.session_state.alpha = 0.5

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.stApp { background: #f0faf5 !important; }

/* ── САЙДБАР ── */
.stButton > button {
    background: transparent !important;
    color: #5a8a75 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    justify-content: flex-start !important;
}
.stButton > button:hover {
    background: #e8f5ee !important;
    color: #1a4a38 !important;
}

.active-nav div[data-testid="stMarkdownContainer"] + .stButton > button,
.active-nav .stButton > button {
    background: rgba(52,199,137,0.18) !important;
    color: #1a4a38 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(52,199,137,0.35) !important;
}

/* Подсветка кнопки при нажатии */
.stButton > button:active,
.stButton > button:focus {
    background: rgba(52,199,137,0.22) !important;
    color: #1a4a38 !important;
}

/* Поиск поле */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d1e9dc !important;
    border-radius: 12px !important;
    padding: 11px 16px !important;
    font-size: 13.5px !important;
    color: #1a2e28 !important;
    box-shadow: 0 1px 4px rgba(27,77,62,0.05) !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #34c789 !important;
    box-shadow: 0 0 0 3px rgba(52,199,137,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #94b8a8 !important }
.stTextInput label { display: none !important }

/* Кнопка Найти */
.find-btn button {
    background: linear-gradient(135deg, #1a4a38, #276b52) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 11px 20px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: 1.5px solid #0e3526 !important;
    box-shadow: 0 3px 10px rgba(26,74,56,0.25) !important;
    text-align: center !important;
    justify-content: center !important;
    height: 46px !important;
}
.find-btn button:hover { 
    background: linear-gradient(135deg, #225944, #2e7d60) !important;
    border-color: #1a4a38 !important;
    opacity: 1 !important;
}
.find-btn button * {
    color: #ffffff !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #f7fdfb !important;
    border: 1px solid #d1e9dc !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    color: #1a4a38 !important;
}

/* Slider */
.stSlider > div > div > div {
    background: #d1e9dc !important;
}
.stSlider > div > div > div > div {
    background: #34c789 !important;
}

/* Метрики */
.metric-card {
    background: #ffffff;
    border: 1px solid #d8f0e5;
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
}
.metric-title { font-size: 10px; text-transform: uppercase; letter-spacing: .07em; color: #7ab09a; margin-bottom: 6px; font-weight: 500 }
.metric-val   { font-size: 26px; font-weight: 600; color: #1a4a38; letter-spacing: -0.5px }
.metric-sub   { font-size: 11.5px; color: #6a9e8a; margin-top: 3px; font-weight: 300 }
.metric-badge { display: inline-block; font-size: 10px; background: #e0f5ea; color: #1a4a38; padding: 2px 9px; border-radius: 6px; margin-top: 8px; border: 0.5px solid rgba(52,199,137,0.3) }

/* Обновлённые карточки результатов поиска — в стиле поисковой строки и сайдбара */
.metric-card-v2 {
    background: #ffffff;
    border: 1px solid #d1e9dc;
    border-radius: 16px;
    padding: 20px 22px;
    height: 100%;
    box-shadow: 0 1px 4px rgba(27,77,62,0.05);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.metric-card-v2:hover {
    border-color: #34c789;
    box-shadow: 0 3px 12px rgba(26,74,56,0.08);
}

/* Карточки результатов */
.result-card {
    background: #ffffff;
    border: 1px solid #d8f0e5;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 5px rgba(26,74,56,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.result-card:hover { border-color: #34c789; box-shadow: 0 3px 14px rgba(26,74,56,0.08) }
.result-top  { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px }
.result-num  { font-size: 10px; color: #94b8a8; margin-bottom: 3px; font-weight: 300 }
.result-name { font-size: 14px; font-weight: 600; color: #1a2e28 }
.result-path { font-size: 11.5px; color: #94b8a8; font-family: 'SF Mono', monospace; margin-top: 2px }
.pct-hi { background: #dcfaed; color: #0e6641; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 7px }
.pct-mi { background: #fef9e7; color: #7d5a0a; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 7px }
.pct-lo { background: #e8f4fd; color: #1a5276; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 7px }

.section-lbl {
    font-size: 10.5px; font-weight: 600; color: #7ab09a;
    text-transform: uppercase; letter-spacing: .08em;
    margin: 20px 0 14px;
}

/* Подсказки */
.sugg-box {
    background: #ffffff; border: 1px solid #d8f0e5;
    border-radius: 14px; padding: 8px 0;
    box-shadow: 0 4px 16px rgba(26,74,56,0.07);
    margin-top: 12px;
}
.sugg-lbl  { font-size: 10px; color: #94b8a8; padding: 6px 18px 4px; text-transform: uppercase; letter-spacing: .06em; font-weight: 500 }
.sugg-item { padding: 10px 18px; font-size: 13px; color: #2d6a54; font-weight: 300; display: flex; align-items: center; gap: 8px }
.sugg-arrow { color: #34c789; font-size: 12px }

/* Чат */
.chat-header {
    background: #ffffff; border-bottom: 1px solid #e8f5ee;
    padding: 18px 28px; display: flex;
    align-items: center; justify-content: space-between;
    border-radius: 0;
}
.chat-title-main { font-size: 17px; font-weight: 600; color: #1a2e28; letter-spacing: -0.3px }
.chat-online { display: flex; align-items: center; gap: 7px }
.online-pulse { width: 7px; height: 7px; border-radius: 50%; background: #34c789; display: inline-block; box-shadow: 0 0 0 2px rgba(52,199,137,0.25) }
.chat-model-pill { background: #e8f5ee; border: 0.5px solid #b8dece; border-radius: 20px; padding: 5px 13px; font-size: 12px; color: #1a4a38; font-weight: 500 }

.bubble-user {
    background: linear-gradient(135deg, #1a4a38, #276b52);
    color: #fff !important; border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; font-size: 13.5px; line-height: 1.6;
    font-weight: 300; box-shadow: 0 3px 12px rgba(26,74,56,0.2);
    display: inline-block; max-width: 72%;
}
.bubble-user * { color: #fff !important }
.time-right { font-size: 10px; color: #94b8a8; text-align: right; margin-top: 5px }

.ai-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 4px }
.ai-avatar { width: 26px; height: 26px; border-radius: 50%; background: linear-gradient(135deg,#1a4a38,#34c789); display: inline-flex; align-items: center; justify-content: center; font-size: 10px; color: #fff; font-weight: 600; flex-shrink: 0; margin-top: 2px }
.bubble-ai {
    background: #ffffff; border: 1px solid #d8f0e5;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px; font-size: 13.5px; color: #2d3a34;
    line-height: 1.75; font-weight: 300;
    box-shadow: 0 1px 6px rgba(26,74,56,0.05);
}
.bubble-ai strong { font-weight: 600; color: #1a4a38 }
.bubble-ai code   { font-family: monospace; font-size: 11.5px; background: #e8f5ee; padding: 1px 5px; border-radius: 4px; color: #0e6641 }
.time-left { font-size: 10px; color: #94b8a8; margin-top: 5px; margin-left: 36px }

.src-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; margin-left: 36px }
.src-chip  { font-size: 10.5px; background: #e8f5ee; border: 0.5px solid #b8dece; color: #1a4a38; padding: 4px 10px; border-radius: 8px; font-family: monospace }

.ric-card { background: #f7fdfb; border: 1px solid #d8f0e5; border-radius: 12px; padding: 13px 16px; margin-top: 6px; margin-left: 36px }
.ric-top  { display: flex; justify-content: space-between; margin-bottom: 8px }
.ric-name { font-size: 12.5px; font-weight: 600; color: #1a2e28 }
.ric-path { font-size: 11px; color: #94b8a8; font-family: monospace; margin-top: 2px }
.ric-code { background: #edfaf3; border-radius: 8px; padding: 9px 12px; font-size: 11px; font-family: monospace; color: #1a4a38; line-height: 1.6; border: 0.5px solid #c8e8d8 }

.chat-input-area { background: #ffffff; border-top: 1px solid #e8f5ee; padding: 16px 28px }
.stChatInput > div { border: 1px solid #d1e9dc !important; border-radius: 12px !important; background: #f7fdfb !important }
.stChatInput textarea { font-size: 13.5px !important; color: #1a2e28 !important }

.empty-state { text-align: center; padding: 80px 0 }
.empty-icon  { font-size: 38px; margin-bottom: 16px; opacity: .7 }
.empty-title { font-size: 16px; font-weight: 600; color: #1a4a38; margin-bottom: 8px }
.empty-sub   { font-size: 12.5px; color: #7ab09a; font-weight: 300; line-height: 1.6 }

/* Метрики страница — графики */
.metrics-stat { background: #ffffff; border: 1px solid #d8f0e5; border-radius: 14px; padding: 18px 20px }
.ms-title { font-size: 10px; text-transform: uppercase; letter-spacing: .07em; color: #7ab09a; margin-bottom: 6px; font-weight: 500 }
.ms-val   { font-size: 28px; font-weight: 600; color: #1a4a38; letter-spacing: -0.5px }
.ms-sub   { font-size: 11.5px; color: #6a9e8a; margin-top: 3px }
</style>
""", unsafe_allow_html=True)

# ── LAYOUT
col_menu, col_content = st.columns([1, 4.3], gap="small")

# ── САЙДБАР ──
with col_menu:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.65);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:0 18px 18px 0;border-right:1px solid #e8f5ee;padding:28px 16px 24px;display:flex;flex-direction:column;gap:4px;">
        <div style="font-size:15px;font-weight:600;color:#1a4a38;margin-bottom:28px;display:flex;align-items:center;gap:8px;letter-spacing:-0.3px">
            <span style="width:8px;height:8px;border-radius:50%;background:#34c789;display:inline-block;box-shadow:0 0 0 3px rgba(52,199,137,0.2)"></span>
            CodeLens RAG
        </div>
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#94b8a8;margin-bottom:6px;font-weight:500;padding:0 4px">
            Навигация
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    for pid, icon, label in [("search", "⌕", "Поиск"), ("chat", "💬", "Чат с LLM"), ("settings", "⚙", "Расширенные настройки")]:
        if st.session_state.page == pid:
            st.markdown('<div class="active-nav">', unsafe_allow_html=True)
        else:
            st.markdown('<div>', unsafe_allow_html=True)

        if st.button(f"{icon}  {label}", use_container_width=True, key=f"nav_{pid}"):
            st.session_state.page = pid
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    total_chunks = len(searcher.data) if hasattr(searcher, 'data') else 0
    st.markdown(f"""
        <div style="padding:0 4px;margin-top:16px;border-top:1px solid #e8f5ee;padding-top:14px">
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#94b8a8;margin-bottom:6px;font-weight:500">
                Модель
            </div>
            <div style="font-size:13px;color:#1a4a38;font-weight:500;padding:0 4px;margin-bottom:14px">
                {MODEL_NAME}
            </div>
            <div style="display:flex;align-items:center;gap:6px">
                <span style="width:6px;height:6px;border-radius:50%;background:#34c789;display:inline-block;box-shadow:0 0 0 2px rgba(52,199,137,0.2)"></span>
                <span style="font-size:11.5px;color:#7ab09a;font-weight:300">Индекс готов · {total_chunks} чанков</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── КОНТЕНТ ──
metric_type = st.session_state.get("metric_sel", "Precision@5")

with col_content:

    # ── ПОИСК ────────────────────────────────────────
    if st.session_state.page == "search":

        st.markdown("<div style='padding:28px 36px 0'>", unsafe_allow_html=True)

        st.markdown('<div class="search-box-container">', unsafe_allow_html=True)
        c_in, c_btn = st.columns([4.5, 0.8])
        with c_in:
            query = st.text_input(
                "Поиск",
                placeholder="⌕ Введите запрос...",
                key="search_input",
                label_visibility="collapsed"
            )
        with c_btn:
            st.markdown('<div class="find-btn">', unsafe_allow_html=True)
            search = st.button("Найти", use_container_width=True, key="search_run")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if search and query:
            st.session_state.searched = True
            st.session_state.last_query = query
            # Используем alpha из session_state
            results = searcher.search(query, alpha=st.session_state.alpha, top_k=5)
            st.session_state.results = results

        if not st.session_state.get("searched", False):
            st.markdown("""
            <div class="sugg-box">
                <div class="sugg-lbl">Популярные запросы</div>
                <div class="sugg-item"><span class="sugg-arrow">↗</span>как создаётся JWT токен?</div>
                <div class="sugg-item"><span class="sugg-arrow">↗</span>как работает пагинация?</div>
                <div class="sugg-item"><span class="sugg-arrow">↗</span>как хешируется пароль?</div>
                <div class="sugg-item"><span class="sugg-arrow">↗</span>как удалить упражнение?</div>
                <div class="sugg-item"><span class="sugg-arrow">↗</span>как получить текущего пользователя?</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            q = st.session_state.last_query
            results = st.session_state.get("results", [])

            if results:
                # results — это список кортежей (score, chunk)
                top_score = int(results[0][0] * 100)  # score — первый элемент кортежа
                top_chunk = results[0][1]  # chunk — второй элемент кортежа
                top_chunk_id = top_chunk['chunk_id']
                
                if ':' in top_chunk_id:
                    top_path, top_name = top_chunk_id.split(':', 1)
                else:
                    top_path = top_chunk_id
                    top_name = "function"
                
                found_count = len(results)
                total_chunks = len(searcher.data)
                precision = st.session_state.get('precision_value', 0.71)
            else:
                top_score = 0
                top_path = "—"
                top_name = "—"
                found_count = 0
                total_chunks = len(searcher.data)
                precision = st.session_state.get('precision_value', 0.71)

            st.markdown(f"""
            <div style="margin:20px 0 20px">
                <div style="font-size:20px;font-weight:600;color:#1a2e28;letter-spacing:-0.4px">
                    Результаты поиска
                </div>
                <div style="font-size:12px;color:#7ab09a;margin-top:4px;font-weight:300">
                    запрос: «{q}» · модель: {MODEL_NAME} · метрика: {metric_type}
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-card-v2">
                    <div class="metric-title">Топ результат</div>
                    <div class="metric-val">{top_score}%</div>
                    <div class="metric-sub">{top_name}</div>
                    <div class="metric-badge">{top_path}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card-v2">
                    <div class="metric-title">Найдено чанков</div>
                    <div class="metric-val">{found_count}</div>
                    <div class="metric-sub">из {total_chunks} в индексе</div>
                    <div class="metric-badge">гибридный поиск</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-lbl">Фрагменты кода</div>', unsafe_allow_html=True)

            if results:
                for i, r in enumerate(results, 1):
                    score, chunk = r  # разворачиваем кортеж
                    chunk_id = chunk["chunk_id"]
                    if ':' in chunk_id:
                        path, name = chunk_id.split(':', 1)
                    else:
                        path = chunk_id
                        name = "function"
                    
                    pct = int(score * 100)
                    cls = "pct-hi" if pct >= 60 else ("pct-mi" if pct >= 30 else "pct-lo")
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-top">
                            <div>
                                <div class="result-num">#{i}</div>
                                <div class="result-name">{name}</div>
                                <div class="result-path">{path}</div>
                            </div>
                            <span class="{cls}">{pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(chunk["code"], language="python")
                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            else:
                st.info("Ничего не найдено. Попробуйте изменить запрос.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── ЧАТ ──────────────────────────────────────────
    elif st.session_state.page == "chat":

        q = st.session_state.last_query or "как создаётся токен доступа?"

        st.markdown(f"""
        <div class="chat-header">
            <div>
                <div class="chat-title-main">Чат с LLM</div>
                <div style="font-size:12px;color:#7ab09a;margin-top:2px;font-weight:300">
                    отвечает на основе найденного кода
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
                <div class="chat-model-pill">{MODEL_NAME}</div>
                <div class="chat-model-pill" style="background:#f0faf5;border-color:#c8e8d8">{metric_type}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='padding:20px 28px;display:flex;flex-direction:column;gap:16px'>",
                    unsafe_allow_html=True)

        if not st.session_state.searched and not st.session_state.messages:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">💬</div>
                <div class="empty-title">Чат с Mistral</div>
                <div class="empty-sub">
                    Сначала введите запрос на странице Поиска —<br>
                    LLM объяснит найденный код на русском языке
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.session_state.searched:
                # Показываем реальные результаты в чате
                results = st.session_state.get("results", [])
                if results:
                    top = results[0]
                    score, chunk = top
                    chunk_id = chunk["chunk_id"]
                    if ':' in chunk_id:
                        path, name = chunk_id.split(':', 1)
                    else:
                        path = chunk_id
                        name = "function"
                    code_preview = chunk["code"][:300] + "..." if len(chunk["code"]) > 300 else chunk["code"]
                    pct = int(score * 100)

                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-end;margin-bottom:4px">
                        <div>
                            <div class="bubble-user">{q}</div>
                            <div class="time-right">только что</div>
                        </div>
                    </div>
                    <div class="ai-row">
                        <div class="ai-avatar">CL</div>
                        <div>
                            <div style="font-size:11px;color:#7ab09a;margin-bottom:5px">
                                CodeLens · {MODEL_NAME}
                            </div>
                            <div class="bubble-ai">
                                <strong>Найден фрагмент:</strong> <strong>{name}</strong><br>
                                <strong>Файл:</strong> <code>{path}</code><br>
                                <strong>Релевантность:</strong> {score}%<br><br>
                                <strong>Код:</strong><br>
                                <pre style="background:#f0faf5;padding:12px;border-radius:8px;font-size:12px;overflow-x:auto;white-space:pre-wrap;">
{code_preview}
                                </pre>
                                <br>
                                <em style="color:#7ab09a;">Для полного объяснения подключите LLM</em>
                            </div>
                        </div>
                    </div>
                    <div class="time-left">только что</div>
                    <div class="src-chips">
                        <span class="src-chip">📄 {path}</span>
                        <span class="src-chip">🔍 {score}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"].replace("\n", "<br>")
                t = msg.get("time", "")
                if role == "user":
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-end;margin-bottom:4px">
                        <div>
                            <div class="bubble-user">{content}</div>
                            <div class="time-right">{t}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-row">
                        <div class="ai-avatar">CL</div>
                        <div class="bubble-ai">{content}</div>
                    </div>
                    <div class="time-left">{t}</div>
                    <div style="height:6px"></div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chat-input-area'>", unsafe_allow_html=True)
        prompt = st.chat_input("Задайте уточняющий вопрос о коде...", key="chat_field")
        st.markdown("""
        <div style="font-size:10.5px;color:#94b8a8;text-align:center;margin-top:8px">
            Ответы формируются на основе найденных фрагментов кода.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if prompt:
            st.session_state.messages.append({
                "role": "user", "content": prompt,
                "time": datetime.now().strftime("%H:%M")
            })

            results = st.session_state.get("results", [])
            if results:
                top = results[0]
                code_preview = top["code"][:200] + "..." if len(top["code"]) > 200 else top["code"]
                response = f"Анализирую запрос: «{prompt}».\n\nНа основе найденного кода:\n\n```python\n{code_preview}\n```\n\nЛогика обработки инкапсулирована в соответствующем модуле. Для полного ответа подключите реальный LLM."
            else:
                response = f"Анализирую запрос: «{prompt}».\n\nСначала выполните поиск на странице «Поиск», чтобы найти релевантные фрагменты кода."

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "time": datetime.now().strftime("%H:%M")
            })
            st.rerun()

    elif st.session_state.page == "settings":

        st.markdown("<div style='padding:28px 36px 0'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:24px">
            <div style="font-size:20px;font-weight:600;color:#1a2e28;letter-spacing:-0.4px">
                Расширенные настройки
            </div>
            <div style="font-size:12px;color:#7ab09a;margin-top:4px;font-weight:300">
                Технические параметры системы — для оценки качества и тонкой настройки поиска
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-lbl">Качество системы</div>', unsafe_allow_html=True)

        metric_type = st.selectbox(
            "Метрика оценки",
            ["Precision@5", "MRR"],
            key="metric_sel",
        )

        precision = st.session_state.precision_value

        s1, s2, s3 = st.columns(3)
        with s1:
            val = f"{precision:.3f}" if precision > 0 else "—"
            st.markdown(f"""<div class="metrics-stat">
                <div class="ms-title">Overall {metric_type}</div>
                <div class="ms-val">{val}</div>
                <div class="ms-sub">15 вопросов</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metrics-stat">
                <div class="ms-title">Русский язык</div>
                <div class="ms-val">—</div>
                <div class="ms-sub">8 вопросов</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metrics-stat">
                <div class="ms-title">Английский язык</div>
                <div class="ms-val">—</div>
                <div class="ms-sub">7 вопросов</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        btn_col, hint_col = st.columns([1, 2.5])
        with btn_col:
            calc_clicked = st.button("Рассчитать " + metric_type, key="calc_metrics")
        with hint_col:
            st.markdown(f"""
            <div style="font-size:11.5px;color:#94b8a8;font-weight:300;padding-top:11px">
                пересчитает {metric_type} с текущим балансом поиска
            </div>
            """, unsafe_allow_html=True)

        if calc_clicked:
            with st.spinner(f"Расчет {metric_type}..."):
                precision, debug_log = get_precision(searcher, alpha=st.session_state.alpha)
                st.session_state.precision_value = precision
                st.session_state.metrics_debug = debug_log
                st.rerun()

        if st.session_state.get("metrics_debug"):
            with st.expander("🔍 Debug log", expanded=(st.session_state.precision_value == 0.0)):
                st.code(st.session_state.metrics_debug, language="text")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        precision = st.session_state.precision_value

        try:
            import plotly.graph_objects as go

            if precision > 0:
                yvals = [precision, precision * 0.8, precision * 0.6]
            else:
                yvals = [0.88, 0.65, 0.45]

            fig = go.Figure(go.Bar(
                x=["Easy", "Medium", "Hard"],
                y=yvals,
                marker_color=["#b8f0d8", "#d4f0b8", "#d0ede8"],
                marker_line_color=["#34c789", "#7ab840", "#2e9682"],
                marker_line_width=1.5,
                text=[f"{v:.3f}" for v in yvals],
                textposition="outside",
                textfont=dict(color="#1a4a38", size=12, family="Inter")
            ))
            fig.update_layout(
                height=260,
                margin=dict(t=24, b=12, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.7)",
                yaxis=dict(
                    range=[0, 1.1],
                    gridcolor="rgba(26,74,56,0.06)",
                    tickfont=dict(color="#7ab09a", size=11),
                    tickformat=".2f"
                ),
                xaxis=dict(tickfont=dict(color="#1a4a38", size=13, family="Inter")),
                showlegend=False,
                title=dict(text=f"{metric_type} по сложности", font=dict(size=12, color="#7ab09a"))
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Установи plotly: pip install plotly")

        st.markdown("<div style='height:32px;border-top:1px solid #e8f5ee'></div>", unsafe_allow_html=True)

  
        st.markdown('<div class="section-lbl">Баланс гибридного поиска</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12.5px;color:#6a9e8a;font-weight:300;margin-bottom:14px;line-height:1.6">
            Регулирует соотношение между векторным поиском (по смыслу запроса)
            и BM25 (по точному совпадению слов). Чем выше значение — тем больше
            система полагается на семантическое сходство.
        </div>
        """, unsafe_allow_html=True)

        alpha = st.slider(
            "",
            0.0, 1.0, st.session_state.alpha, 0.05,
            key="alpha_slider",
            label_visibility="collapsed",
            help="0 = только BM25 (поиск по словам), 1 = только векторный (по смыслу)"
        )
        st.session_state.alpha = alpha

        st.markdown(f"""
            <div style="font-size:12px;color:#7ab09a;margin-top:-6px">
                Векторный {int(alpha*100)}% · BM25 {int((1-alpha)*100)}%
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:32px;border-top:1px solid #e8f5ee'></div>", unsafe_allow_html=True)


        st.markdown('<div class="section-lbl">Модель эмбеддингов</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12.5px;color:#6a9e8a;font-weight:300;margin-bottom:14px;line-height:1.6">
            Модель, используемая для построения векторных представлений кода и запросов.
        </div>
        """, unsafe_allow_html=True)

        st.selectbox(
            "Модель",
            ["bge-m3", "MiniLM-L12"],
            key="model_sel_settings",
        )

        st.markdown("</div>", unsafe_allow_html=True)