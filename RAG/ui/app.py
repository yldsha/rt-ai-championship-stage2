# -*- coding: utf-8 -*-
"""
CodeLens RAG — семантический поиск по кодовой базе gymhero.
Дизайн: светло-зелёная тема, боковое меню,
вкладки Поиск / Чат с LLM / История / Расширенные настройки.

Запуск (из корня проекта):
    streamlit run RAG/ui/app.py
"""

import os
import sys
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="CodeLens RAG", page_icon="🟢", layout="wide")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import ask_llm  # noqa: E402
from metrics_helper import get_precision  # noqa: E402
from searcher import HybridSearcher  # noqa: E402

DEFAULT_MODEL = "bge-m3"


@st.cache_resource
def get_searcher():
    return HybridSearcher(
        db_path="RAG/chroma_db",
        chunks_file="RAG/data/chunks.jsonl"
    )


searcher = get_searcher()


# ======================================================================
#  ОФОРМЛЕНИЕ
# ======================================================================

CSS = """
<style>

:root, .stApp {
    color-scheme: light !important;
    --background-color: #EAF6F4 !important;
    --secondary-background-color: #FFFFFF !important;
    --text-color: #1a1a1a !important;
    --primary-color: #1B4D3E !important;
}
* { color-scheme: light !important; }


[data-testid="stHeader"] { display: none !important; background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.stApp > header { display: none !important; }

/* ---- общий фон и шрифт ---- */
.stApp { background: #eef5ec !important; }
html, body, [class*="css"] { font-family: 'Geist', system-ui, -apple-system, sans-serif; }

p, span, div, label, h1, h2, h3, h4, h5, h6, li {
    color: #1a1a1a;
}
input, textarea {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* ---- сайдбар ---- */
section[data-testid="stSidebar"] { background: #e6f0e4; border-right: 1px solid #d4e3d2; }
section[data-testid="stSidebar"] .block-container { padding-top: 20px; }

/* логотип */
.cl-logo { display:flex; align-items:center; gap:11px; padding:4px 4px 0 4px; }
.cl-logo .mark { width:36px; height:36px; border-radius:10px;
    background:linear-gradient(150deg,#23ab63,#138047); display:flex;
    align-items:center; justify-content:center; color:#fff;
    font-family:'JetBrains Mono',monospace; font-weight:600; font-size:15px;
    box-shadow:0 4px 10px rgba(20,128,71,.30); }
.cl-logo .name { font-size:16px; font-weight:700; letter-spacing:-.01em; color:#16271b; }
.cl-logo .sub  { font-size:11px; color:#6d7e70; }
.cl-divider { height:1px; background:#d4e3d2; margin:18px 2px; }

/* навигация (radio в сайдбаре) */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:3px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding:9px 12px; border-radius:9px; cursor:pointer; width:100%;
    transition:background .12s; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#dcefe1; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size:13.5px; font-weight:500; color:#5d6f60; }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }
/* выбранный пункт */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background:#d6ecdb; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color:#0f3d22; font-weight:600; }

/* футер сайдбара */
.cl-foot { border-top:1px solid #d4e3d2; padding-top:12px; margin-top:8px; }
.cl-foot .row { display:flex; justify-content:space-between; font-size:12px; margin-bottom:8px; }
.cl-foot .k { color:#7c8a7e; }
.cl-foot .v { font-family:'JetBrains Mono',monospace; font-weight:600; color:#3a4d40; }

/* ---- заголовки страниц ---- */
.cl-h1 { font-size:21px; font-weight:700; letter-spacing:-.01em; color:#16271b; margin:0; }
.cl-sub { font-size:13.5px; color:#5d6f60; margin:4px 0 0 0; }

/* ---- поле поиска ---- */
div[data-testid="stTextInput"] input {
    height:48px; border:1.5px solid #cfe0cf !important; border-radius:11px !important;
    background:#fff !important; font-size:14.5px; color:#16271b; }
div[data-testid="stTextInput"] input:focus {
    border-color:#1f9d57 !important; box-shadow:0 0 0 3px rgba(31,157,87,.15) !important; }

/* ---- кнопки ---- */
.stButton > button, .stFormSubmitButton > button {
    background:linear-gradient(180deg,#22a861,#178a4c) !important; color:#fff !important;
    border:none !important; border-radius:11px !important; height:48px;
    font-weight:600 !important; font-size:14.5px !important;
    box-shadow:0 6px 14px rgba(23,138,76,.32), 0 1px 0 rgba(255,255,255,.25) inset !important; }
.stButton > button:hover, .stFormSubmitButton > button:hover { filter:brightness(1.05); }

/* ---- карточка результата ---- */
.cl-card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:2px; }
.cl-path { font-family:'JetBrains Mono',monospace; font-size:12.5px; color:#3a4d40; }
.cl-path .dir { color:#9aa89c; }
.cl-path .file { font-weight:600; color:#16271b; }
.cl-badge { display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
    border-radius:999px; background:#def0e2; }
.cl-badge .dot { width:6px; height:6px; border-radius:50%; background:#17a356; }
.cl-badge .pct { font-size:12px; font-weight:700; color:#127a42;
    font-family:'JetBrains Mono',monospace; }
.cl-badge.low { background:#eef2ec; }
.cl-badge.low .dot { background:#7fae8e; }
.cl-badge.low .pct { color:#5a6f60; }

/* контейнеры-карточки */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff; border:1px solid #e1eadf !important; border-radius:12px;
    box-shadow:0 1px 3px rgba(30,50,35,.05); }

/* подпись «найдено N» */
.cl-meta { font-size:13px; color:#5d6f60; margin:6px 2px 4px 2px; }
.cl-meta b { color:#16271b; }

/* блок метрик на странице настроек */
.cl-metric { background:#fff; border:1px solid #e1eadf; border-radius:12px; padding:14px 16px; }
.cl-metric .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:#7c8a7e; margin-bottom:5px; font-weight:600 }
.cl-metric .val { font-size:22px; font-weight:700; color:#16271b; }
.cl-metric .sub { font-size:11px; color:#7c8a7e; margin-top:3px }

/* ---- чат: переписка как в мессенджере ---- */
.cl-chat-window { display:flex; flex-direction:column; gap:14px; padding:6px 2px 18px 2px; }

.cl-msg-row { display:flex; align-items:flex-end; gap:9px; }
.cl-msg-row.user { justify-content:flex-end; }
.cl-msg-row.assistant { justify-content:flex-start; }

.cl-avatar {
    width:32px; height:32px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700;
}
.cl-avatar.user {
    background:#cfe0cf; color:#16271b;
}
.cl-avatar.assistant {
    background:linear-gradient(150deg,#23ab63,#138047); color:#fff;
    font-family:'JetBrains Mono',monospace; box-shadow:0 3px 8px rgba(20,128,71,.28);
}

.cl-bubble {
    max-width:68%; padding:11px 15px; font-size:13.5px; line-height:1.55;
    color:#16271b; position:relative;
}
.cl-bubble.user {
    background:linear-gradient(180deg,#22a861,#178a4c); color:#fff;
    border-radius:16px 16px 4px 16px;
    box-shadow:0 4px 10px rgba(23,138,76,.25);
}
.cl-bubble.user .cl-bubble-text { color:#fff; }
.cl-bubble.assistant {
    background:#fff; border:1px solid #e1eadf;
    border-radius:16px 16px 16px 4px;
    box-shadow:0 1px 3px rgba(30,50,35,.06);
}
.cl-bubble-name {
    font-size:11px; font-weight:700; color:#178a4c; margin-bottom:3px;
}
.cl-bubble-text code {
    background:rgba(0,0,0,.06); padding:1px 5px; border-radius:4px;
    font-family:'JetBrains Mono',monospace; font-size:12px;
}
.cl-bubble.user .cl-bubble-text code {
    background:rgba(255,255,255,.18); color:#fff;
}
.cl-bubble-text pre {
    background:#0f1f15; color:#d7ecd9; padding:10px 12px; border-radius:8px;
    font-size:12px; overflow-x:auto; margin:8px 0;
    font-family:'JetBrains Mono',monospace;
}
.cl-bubble-time {
    font-size:10px; color:#9aa89c; margin-top:5px; text-align:right;
}
.cl-bubble.user .cl-bubble-time { color:rgba(255,255,255,.7); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def fmt_pct(score: float) -> int:
    return int(round(score * 100))


# ======================================================================
#  СОСТОЯНИЕ
# ======================================================================
ss = st.session_state
ss.setdefault("messages", [
    {"role": "assistant",
     "content": "Привет! Я помогу найти и объяснить код в кодовой базе gymhero. "
                "Спросите, например, как создаётся токен доступа или как работает пагинация.",
     "time": ""},
])
ss.setdefault("history", [])
ss.setdefault("top_k", 5)
ss.setdefault("alpha", 0.8)
ss.setdefault("model", DEFAULT_MODEL)
ss.setdefault("metric_type", "Precision@5")
if not isinstance(ss.get("metrics_result"), dict) or "mrr" not in ss.get("metrics_result", {}):
    ss.metrics_result = {"overall": 0.0, "mrr": 0.0, "ru": None, "en": None, "mrr_ru": None, "mrr_en": None}
ss.setdefault("metrics_debug", "")
ss.setdefault("last_results", [])
ss.setdefault("last_query", "")


def run_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
    """
    Обёртка над HybridSearcher.search(), которая приводит результат
    (score, chunk_dict) к плоскому словарю для отображения в карточках.
    """
    if not query.strip():
        return []
    raw = searcher.search(query, alpha=alpha, top_k=top_k)
    results = []
    for score, chunk in raw:
        results.append({
            "chunk_id": chunk["chunk_id"],
            "path": chunk["chunk_id"],
            "code": chunk.get("code", ""),
            "score": score,
        })
    return results


def total_chunks() -> int:
    return len(searcher.data) if hasattr(searcher, "data") else 0


# ======================================================================
#  САЙДБАР
# ======================================================================
with st.sidebar:
    st.markdown(
        '<div class="cl-logo"><div class="mark">&lt;/&gt;</div>'
        '<div><div class="name">CodeLens RAG</div>'
        '<div class="sub">поиск по кодовой базе gymhero</div></div></div>'
        '<div class="cl-divider"></div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Навигация",
        ["🔍  Поиск кода", "💬  Чат с LLM", "🕘  История", "⚙️  Расширенные настройки"],
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="cl-foot">'
        f'<div class="row"><span class="k">Чанков в базе</span>'
        f'<span class="v">{total_chunks()}</span></div>'
        f'<div class="row"><span class="k">Модель</span>'
        f'<span class="v">{ss.model}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ======================================================================
#  СТРАНИЦА: ПОИСК
# ======================================================================
if page.endswith("Поиск кода"):
    st.markdown('<p class="cl-h1">Поиск фрагментов кода</p>'
                '<p class="cl-sub">Опишите словами, что ищете — найдём похожие фрагменты в базе.</p>',
                unsafe_allow_html=True)
    st.write("")

    col_in, col_btn = st.columns([5, 1])
    with col_in:
        query = st.text_input("Запрос", placeholder="Например: как создаётся токен доступа",
                              label_visibility="collapsed", key="search_query_input")
    with col_btn:
        do_search = st.button("🔍  Найти", use_container_width=True, key="search_run_btn")

    if do_search and query:
        ss.last_query = query
        ss.last_results = run_search(query, top_k=ss.top_k, alpha=ss.alpha)
        ss.history.insert(0, {
            "q": query,
            "n": len(ss.last_results),
            "t": datetime.now().strftime("сегодня, %H:%M"),
        })

    if ss.last_query:
        results = ss.last_results
        st.markdown(f'<p class="cl-meta">Найдено <b>{len(results)}</b> '
                    f'{"фрагмент" if len(results)==1 else "фрагмента" if 2<=len(results)<=4 else "фрагментов"} '
                    f'по запросу «{ss.last_query}»</p>',
                    unsafe_allow_html=True)

        for r in results:
            with st.container(border=True):
                chunk_id = r["chunk_id"]
                if ":" in chunk_id:
                    path, name = chunk_id.split(":", 1)
                else:
                    path, name = chunk_id, ""
                d, f = path.rsplit("/", 1) if "/" in path else ("", path)
                display_file = f"{f}:{name}" if name else f
                pct = fmt_pct(r["score"])
                low = "" if pct >= 60 else "low"
                st.markdown(
                    f'<div class="cl-card-head">'
                    f'<span class="cl-path"><span class="dir">{d}/</span>'
                    f'<span class="file">{display_file}</span></span>'
                    f'<span class="cl-badge {low}"><span class="dot"></span>'
                    f'<span class="pct">{pct}%</span></span></div>',
                    unsafe_allow_html=True,
                )
                # Подсветка синтаксиса — одной строкой:
                st.code(r["code"], language="python")
    else:
        st.info("Введите запрос, чтобы начать поиск.")


# ======================================================================
#  СТРАНИЦА: ЧАТ С LLM
# ======================================================================
elif page.endswith("Чат с LLM"):
    st.markdown('<p class="cl-h1">Чат с LLM</p>'
                '<p class="cl-sub">Задавайте вопросы о коде — система ищет по базе и отвечает.</p>',
                unsafe_allow_html=True)
    st.write("")

    # ---- история переписки ----
    st.markdown('<div class="cl-chat-window">', unsafe_allow_html=True)

    for m in ss.messages:
        ts = m.get("time", "")
        if m["role"] == "user":
            st.markdown(f"""
            <div class="cl-msg-row user">
                <div class="cl-bubble user">
                    <div class="cl-bubble-text">{m['content']}</div>
                    <div class="cl-bubble-time">{ts}</div>
                </div>
                <div class="cl-avatar user">Вы</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cl-msg-row assistant">
                <div class="cl-avatar assistant">&lt;/&gt;</div>
                <div class="cl-bubble assistant">
                    <div class="cl-bubble-name">CodeLens</div>
                    <div class="cl-bubble-text">{m['content']}</div>
                    <div class="cl-bubble-time">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- поле ввода ----
    prompt = st.chat_input("Спросите что-нибудь о коде…")
    if prompt:
        now = datetime.now().strftime("%H:%M")
        ss.messages.append({"role": "user", "content": prompt, "time": now})

        results = run_search(prompt, top_k=ss.top_k, alpha=ss.alpha)

        if results:
            top_chunks = [
                {"chunk_id": r["chunk_id"], "code": r["code"]}
                for r in results[:3]
            ]

            with st.spinner("CodeLens анализирует код…"):
                llm_answer = ask_llm(prompt, top_chunks)

            # Список источников снизу ответа — компактные ссылки на файлы
            sources_html = " &nbsp; ".join(
                f"<code>{r['chunk_id'].split(':', 1)[0]}</code>"
                for r in results[:3]
            )

            answer = (
                f"{llm_answer}<br><br>"
                f"<span style='color:#9aa89c;font-size:11.5px'>Источники: {sources_html}</span>"
            )
        else:
            answer = (
                "Сначала выполните поиск на странице «Поиск кода», "
                "либо переформулируйте вопрос — релевантных фрагментов не найдено."
            )

        ss.messages.append({"role": "assistant", "content": answer, "time": datetime.now().strftime("%H:%M")})
        st.rerun()


# ======================================================================
#  СТРАНИЦА: ИСТОРИЯ
# ======================================================================
elif page.endswith("История"):
    st.markdown('<p class="cl-h1">История запросов</p>'
                '<p class="cl-sub">Недавние поиски в этой сессии.</p>',
                unsafe_allow_html=True)
    st.write("")

    if not ss.history:
        st.info("Пока нет запросов — выполните поиск на странице «Поиск кода».")
    else:
        for i, h in enumerate(ss.history):
            with st.container(border=True):
                c1, c2, c3 = st.columns([6, 1.2, 1.6])
                c1.markdown(f"🔍  {h['q']}")
                c2.markdown(f"<span style='font-family:JetBrains Mono,monospace;"
                            f"font-size:12px;color:#7c8a7e'>{h['n']} рез.</span>",
                            unsafe_allow_html=True)
                c3.markdown(f"<span style='font-size:12px;color:#9aa89c'>{h['t']}</span>",
                            unsafe_allow_html=True)


# ======================================================================
#  СТРАНИЦА: РАСШИРЕННЫЕ НАСТРОЙКИ
# ======================================================================
elif page.endswith("Расширенные настройки"):
    st.markdown('<p class="cl-h1">Расширенные настройки</p>'
                '<p class="cl-sub">Параметры поиска, модель и качество системы.</p>',
                unsafe_allow_html=True)
    st.write("")

    # ── Качество системы ──
    with st.container(border=True):
        st.markdown("**Качество системы**")

        ss.metric_type = st.selectbox(
            "Метрика оценки",
            ["Precision@5", "MRR"],
            index=["Precision@5", "MRR"].index(ss.metric_type),
        )

        m = ss.metrics_result
        if ss.metric_type == "MRR":
            overall = m["mrr"]
            ru_val = m.get("mrr_ru")
            en_val = m.get("mrr_en")
        else:
            overall = m["overall"]
            ru_val = m["ru"]
            en_val = m["en"]

        c1, c2, c3 = st.columns(3)
        with c1:
            val = f"{overall:.3f}" if overall > 0 else "—"
            st.markdown(f"""<div class="cl-metric">
                <div class="lbl">Overall {ss.metric_type}</div>
                <div class="val">{val}</div>
                <div class="sub">15 вопросов</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            ru_display = f"{ru_val:.3f}" if ru_val is not None else "—"
            st.markdown(f"""<div class="cl-metric">
                <div class="lbl">Русский язык</div>
                <div class="val">{ru_display}</div>
                <div class="sub">8 вопросов{" (только для P@5)" if ss.metric_type == "MRR" else ""}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            en_display = f"{en_val:.3f}" if en_val is not None else "—"
            st.markdown(f"""<div class="cl-metric">
                <div class="lbl">Английский язык</div>
                <div class="val">{en_display}</div>
                <div class="sub">7 вопросов{" (только для P@5)" if ss.metric_type == "MRR" else ""}</div>
            </div>""", unsafe_allow_html=True)

        st.write("")
        btn_col, hint_col = st.columns([1, 2.5])
        with btn_col:
            calc_clicked = st.button("Рассчитать " + ss.metric_type, key="calc_metrics_btn")
        with hint_col:
            st.markdown(f"""
            <div style="font-size:11.5px;color:#7c8a7e;font-weight:400;padding-top:13px">
                пересчитает {ss.metric_type} с текущим балансом поиска
            </div>
            """, unsafe_allow_html=True)

        if calc_clicked:
            with st.spinner(f"Расчет {ss.metric_type}..."):
                metrics_result, debug_log = get_precision(searcher, alpha=ss.alpha)
                ss.metrics_result = metrics_result
                ss.metrics_debug = debug_log
                st.rerun()

        if ss.metrics_debug:
            with st.expander("🔍 Debug log", expanded=(ss.metrics_result["overall"] == 0.0)):
                st.code(ss.metrics_debug, language="text")

    st.write("")

    # ── Поиск ──
    with st.container(border=True):
        st.markdown("**Поиск**")
        vw = st.slider("Соотношение поиска (доля векторного)",
                       0, 100, int(ss.alpha * 100), step=5,
                       help="Слева — ключевые слова (BM25), справа — векторный поиск (по смыслу).")
        ss.alpha = vw / 100
        st.caption(f"Вектор {vw}%  ·  ключевые слова {100 - vw}%")


    st.write("")

 