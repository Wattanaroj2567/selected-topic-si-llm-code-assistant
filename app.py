"""Streamlit chat UI for the local hybrid RAG assistant."""

import os

import streamlit as st

from hybrid_rag import HybridRAG

st.set_page_config(
    page_title="Local Coding Assistant",
    page_icon="💻",
    layout="wide",
)


@st.cache_resource
def load_assistant(data_dir: str | None = None) -> HybridRAG:
    if data_dir:
        return HybridRAG.from_data_dir(data_dir)
    return HybridRAG.from_data_dir()


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    mode = str(message.get("mode", "vector")).title()
    elapsed = float(message.get("elapsed", 0.0))
    st.caption(f"Mode: {mode} · {elapsed:.2f}s")
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                file = source.get("file", "unknown")
                line = source.get("line", "?")
                symbol = source.get("symbol", "unknown")
                st.markdown(f"**{symbol}** — `{file}:{line}`")
                if source.get("score") is not None:
                    st.caption(f"Similarity: {float(source['score']):.4f}")
                snippet = source.get("snippet")
                if snippet:
                    st.code(snippet, language="python")


st.title("Local Coding Assistant")
st.write(
    "Ask semantic or code-relationship questions about the indexed Python project."
)
st.markdown(
    """### ตัวอย่างคำถามภาษาไทย

**Vector — ถามว่าโค้ดทำอะไร**
- `ฟังก์ชัน mean ทำงานอย่างไร`
- `อธิบายการทำงานของ run_pipeline`

**Graph — ถามความสัมพันธ์ของโค้ด**
- `ใครเรียก mean`
- `run_pipeline เรียกฟังก์ชันอะไรบ้าง`
- `pipeline นำเข้าอะไรบ้าง`

พิมพ์ `สวัสดี` หรือ `ถามอะไรได้บ้าง` เพื่อดูคำแนะนำได้เช่นกัน
"""
)

try:
    assistant = load_assistant(os.getenv("RAG_DATA_DIR"))
    startup_error = None
except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
    assistant = None
    startup_error = str(error)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    if assistant is not None:
        st.success("Index ready")
    else:
        st.error("Index unavailable")
        st.caption(startup_error)
        st.code("uv run python setup.py --project test_codebase", language="powershell")
    st.caption("Router selects Vector or Graph for every question.")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input(
    "Ask about the indexed Python codebase...",
    disabled=assistant is None,
):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving code and asking Ollama..."):
            result = assistant.query(prompt)
        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "mode": result["mode"],
            "sources": result["sources"],
            "elapsed": result["elapsed"],
            "error": result["error"],
        }
        render_assistant_message(assistant_message)
        if result["error"]:
            st.warning(
                "Check that Ollama is running and the required models are installed."
            )
    st.session_state.messages.append(assistant_message)
