import os
import json
import re
import uuid
import time
from datetime import datetime
import streamlit as st
import requests


def api_base() -> str:  
    env = (os.environ.get("BACKEND_URL") or "").strip().rstrip("/")
    if env:
        return env
    try:
        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            return str(st.secrets["BACKEND_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return "http://localhost:8000"


# PAGE CONFIG
st.set_page_config(
    page_title="⚡ SPARK AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #111827 40%, #1a1040 100%);
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }

    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    .stChatInput > div {
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px);
    }

    /* Capability Cards */
    .cap-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 40px auto;
        max-width: 1000px;
    }
    @media (max-width: 900px) {
        .cap-grid { grid-template-columns: 1fr; }
    }
    .cap-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .cap-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.4);
    }
    .cap-icon { font-size: 2.4rem; margin-bottom: 12px; }
    .cap-title { color: #f1f5f9; font-weight: 600; font-size: 1.1rem; }
    .cap-desc {
        color: #94a3b8; font-size: 0.88rem;
        margin-top: 8px; line-height: 1.5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.97) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.1);
    }

    /* Gradient Text */
    .spark-gradient {
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Pipeline Stepper */
    .pipeline-stepper {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .pipe-step {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.7rem;
        color: #64748b;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(100, 116, 139, 0.15);
        transition: all 0.3s ease;
    }
    .pipe-step.active {
        color: #c084fc;
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.3);
        animation: step-pulse 1.2s infinite;
    }
    .pipe-step.done {
        color: #34d399;
        background: rgba(52, 211, 153, 0.08);
        border-color: rgba(52, 211, 153, 0.2);
    }
    .pipe-arrow { color: #475569; font-size: 0.65rem; }

    @keyframes step-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Pipeline Status Badge */
    .pipe-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #c084fc;
        font-family: 'Outfit', monospace;
        padding: 5px 12px;
        border-radius: 8px;
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.15);
        margin-bottom: 12px;
    }
    .pipe-dot {
        width: 6px; height: 6px;
        background: #a855f7;
        border-radius: 50%;
        animation: pulse-dot 1.2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Suggestion Chips */
    .chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-top: 30px;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }
    .suggestion-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 10px 18px;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.15);
        color: #cbd5e1;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .suggestion-chip:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: rgba(139, 92, 246, 0.4);
        color: #f1f5f9;
        transform: translateY(-2px);
    }

    /* Source Citation Box */
    .source-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(139, 92, 246, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .source-item {
        padding: 6px 0;
        border-bottom: 1px solid rgba(100, 116, 139, 0.1);
    }
    .source-item:last-child { border-bottom: none; }
    .source-rank { color: #a855f7; font-weight: 600; margin-right: 8px; }
    .source-file { color: #60a5fa; }

    /* Timestamp style */
    .msg-timestamp {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Stats card */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 0.82rem;
    }
    .stat-label { color: #94a3b8; }
    .stat-value { color: #e2e8f0; font-weight: 500; }

    /* Hide native Streamlit uploader file list */
    [data-testid="stFileUploaderFile"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# HELPERS
NODE_LABELS = {
    "stress_test": "🔒 Security",
    "simple_answer": "💬 Reply",
    "planner": "📋 Planning",
    "retrieve": "🔍 Retrieval",
    "router": "🔀 Routing",
    "worker": "✍️ Generating",
    "validation": "✅ Validating",
    "evaluation": "✨ Polishing",
    "error": "❌ Error",
}

PIPELINE_STEPS = ["stress_test", "planner", "retrieve", "router", "worker", "validation", "evaluation"]


def _backend_alive() -> bool:
    try:
        return requests.get(f"{api_base()}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _render_pipeline_stepper(completed_nodes: set, active_node: str = None):
    """Render horizontal pipeline stepper."""
    parts = []
    for i, step in enumerate(PIPELINE_STEPS):
        label = NODE_LABELS.get(step, step)
        if step in completed_nodes:
            parts.append(f'<span class="pipe-step done">✓ {label}</span>')
        elif step == active_node:
            parts.append(f'<span class="pipe-step active">● {label}</span>')
        else:
            parts.append(f'<span class="pipe-step">{label}</span>')
        if i < len(PIPELINE_STEPS) - 1:
            parts.append('<span class="pipe-arrow">›</span>')
    return '<div class="pipeline-stepper">' + ''.join(parts) + '</div>'


# SESSION INIT
if "app_initialized" not in st.session_state:
    try:
        requests.post(f"{api_base()}/clear_session", timeout=10)
    except Exception:
        pass
    st.session_state.app_initialized = True
    st.session_state.messages = []
    st.session_state.uploaded_filenames = set()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())


# SIDEBAR
with st.sidebar:
    st.markdown("## ⚡ <span class='spark-gradient'>SPARK AI</span>", unsafe_allow_html=True)
    st.caption("Agentic RAG Pipeline")

    if _backend_alive():
        st.success("System Online", icon="🟢")
    else:
        st.error("System Offline", icon="🔴")

    st.divider()
    use_stream = st.toggle("Stream Pipeline Progress", value=True)

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True, type="primary"):
        with st.spinner("Clearing..."):
            try:
                requests.post(f"{api_base()}/clear_session", timeout=10)
            except Exception:
                pass
            st.session_state.messages = []
            st.session_state.uploaded_filenames = set()
            st.session_state.uploader_key = str(uuid.uuid4())
            st.rerun()

    st.divider()




    st.divider()

    # Document Upload
    st.markdown("### 📎 Documents")

    uploaded_file = st.file_uploader(
        "Upload", type=["pdf", "txt"],
        label_visibility="collapsed", key=st.session_state.uploader_key,
        help="Max file size: 10MB"
    )

    if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_filenames:
        if not _backend_alive():
            st.error("Backend is offline.")
        else:
            with st.spinner("Processing & chunking..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{api_base()}/upload_doc", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["chunks_added"] > 0:
                            st.session_state.uploaded_filenames.add(uploaded_file.name)
                            st.success(f"✅ {data['chunks_added']} chunks ready")
                            st.session_state.uploader_key = str(uuid.uuid4())
                            st.rerun()
                        else:
                            st.warning(data["message"])
                    else:
                        st.error(f"Failed: HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Document List
    try:
        resp = requests.get(f"{api_base()}/documents", timeout=5)
        if resp.status_code == 200:
            docs = resp.json().get("documents", [])
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(f"📄 {doc['filename']} ({doc.get('chunks', '?')} chunks)")
                    with col2:
                        if st.button("✕", key=f"d_{doc['filename']}"):
                            try:
                                requests.delete(f"{api_base()}/documents/{doc['filename']}", timeout=5)
                                st.session_state.uploaded_filenames.discard(doc["filename"])
                            except Exception:
                                pass
                            st.rerun()
    except Exception:
        pass

    st.divider()

    # Session Analytics (Feature 12)
    with st.expander("📊 Session Stats", expanded=False):
        try:
            analytics_resp = requests.get(f"{api_base()}/analytics", timeout=5)
            if analytics_resp.status_code == 200:
                stats = analytics_resp.json()
                st.markdown(f"""
                <div>
                    <div class="stat-row"><span class="stat-label">Messages Sent</span><span class="stat-value">{stats.get('messages_sent', 0)}</span></div>
                    <div class="stat-row"><span class="stat-label">Responses</span><span class="stat-value">{stats.get('messages_received', 0)}</span></div>
                    <div class="stat-row"><span class="stat-label">Documents</span><span class="stat-value">{stats.get('documents_uploaded', 0)}</span></div>
                    <div class="stat-row"><span class="stat-label">Chunks Indexed</span><span class="stat-value">{stats.get('total_chunks_indexed', 0)}</span></div>
                    <div class="stat-row"><span class="stat-label">Avg Response</span><span class="stat-value">{stats.get('avg_response_time_ms', 0):.0f}ms</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Analytics unavailable")
        except Exception:
            st.caption("Analytics unavailable")

    # Chat Export (Feature 1)
    if st.session_state.messages:
        st.divider()
        if st.button("📥 Export Chat", use_container_width=True):
            try:
                export_resp = requests.post(
                    f"{api_base()}/export_chat",
                    json={"messages": st.session_state.messages},
                    timeout=10
                )
                if export_resp.status_code == 200:
                    md_content = export_resp.json().get("content", "")
                    st.download_button(
                        "⬇️ Download Markdown",
                        data=md_content,
                        file_name=f"spark_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            except Exception:
                st.error("Export failed")


# MAIN CHAT AREA

# Welcome screen (Feature 2 — Enhanced)
if not st.session_state.messages:
    st.markdown("")
    st.markdown(
        "<h1 style='text-align:center; margin-top:50px; font-size:2.4rem;'>"
        "⚡ <span class='spark-gradient'>SPARK AI</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#64748b; margin-bottom:36px; font-size:1.05rem;'>"
        "High-Performance Agentic RAG Pipeline</p>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="cap-grid">
        <div class="cap-card">
            <div class="cap-icon">🧠</div>
            <div class="cap-title">Chat & Reason</div>
            <div class="cap-desc">
                Multi-agent pipeline with planning, routing, validation, and evaluation.
            </div>
        </div>
        <div class="cap-card">
            <div class="cap-icon">📑</div>
            <div class="cap-title">Document RAG</div>
            <div class="cap-desc">
                Upload PDFs or TXT files — chunks, indexes, and answers from your documents.
            </div>
        </div>
        <div class="cap-card">
            <div class="cap-icon">🔒</div>
            <div class="cap-title">Security</div>
            <div class="cap-desc">
                Built-in content safety layer with multi-stage validation pipeline.
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)



# Display chat history (with timestamps — Feature 1)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show timestamp if available
        if msg.get("timestamp"):
            st.markdown(f"<div class='msg-timestamp'>{msg['timestamp']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message SPARK AI..."):
    if not _backend_alive():
        st.error("❌ Backend is offline. Please start the server or reload.")
    else:
        timestamp = datetime.now().strftime("%I:%M %p")

        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(f"<div class='msg-timestamp'>{timestamp}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        
        # Pre-append empty assistant message
        st.session_state.messages.append({"role": "assistant", "content": "", "timestamp": "", "sources": []})

        # Build history for API (last 10 messages, excluding the empty one)
        history_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-11:-1]
        ]

        payload = {
            "message": prompt,
            "history": history_for_api,
        }
        base = api_base()
        bot_response = ""

        if use_stream:
            with st.chat_message("assistant"):
                # Pipeline stepper placeholder (Feature 3)
                stepper_container = st.empty()
                # Response text placeholder
                stream_out = st.empty()
                completed_nodes = set()
                active_node = None
                    
                acc = {}
                try:
                    with requests.post(
                        f"{base}/chat/stream",
                        json=payload,
                        stream=True,
                        timeout=(5, 360),
                    ) as resp:
                        if resp.status_code != 200:
                            bot_response = (
                                "Session expired or backend unavailable. "
                                "Please **Clear Session** or reload the page to continue."
                            )
                            stream_out.markdown(bot_response)
                        else:
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                try:
                                    chunk = json.loads(raw_line)
                                except (json.JSONDecodeError, ValueError):
                                    continue

                                node = chunk.get("node", "")
                                update = chunk.get("update", {})
                                acc.update(update)

                                # Update pipeline stepper (Feature 3)
                                if active_node and active_node != node:
                                    completed_nodes.add(active_node)
                                active_node = node
                                stepper_html = _render_pipeline_stepper(completed_nodes, active_node)
                                stepper_container.markdown(stepper_html, unsafe_allow_html=True)

                                # Show live preview of draft
                                preview = (
                                    (acc.get("final_response") or "").strip()
                                    or (acc.get("draft") or "").strip()
                                )
                                if preview:
                                    stream_out.markdown(preview)
                                    bot_response = preview
                                    st.session_state.messages[-1]["content"] = bot_response

                            # Final update
                            bot_response = (
                                (acc.get("final_response") or "").strip()
                                or (acc.get("draft") or "").strip()
                            )
                            if not bot_response:
                                bot_response = "I couldn't generate a response. Please try again."

                            # Mark final stepper as complete
                            if active_node:
                                completed_nodes.add(active_node)
                            stepper_html = _render_pipeline_stepper(completed_nodes)
                            stepper_container.markdown(stepper_html, unsafe_allow_html=True)

                            # Store sources if available
                            sources = acc.get("sources", [])
                            if sources:
                                st.session_state.messages[-1]["sources"] = sources

                except requests.exceptions.ReadTimeout:
                    bot_response = (
                        "The request timed out. "
                        "Please **Clear Session** or reload the page to continue."
                    )
                    if stream_out:
                        stream_out.markdown(bot_response)

                # Final render
                stream_out.markdown(bot_response)

                # Show response time
                elapsed = acc.get("_elapsed_ms", 0)
                if elapsed:
                    resp_timestamp = datetime.now().strftime("%I:%M %p")
                    st.markdown(
                        f"<div class='msg-timestamp'>{resp_timestamp} · {elapsed/1000:.1f}s</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.messages[-1]["timestamp"] = resp_timestamp


        else:
            with st.spinner("Processing..."):
                try:
                    resp = requests.post(
                        f"{base}/chat", json=payload, timeout=(5, 360)
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        bot_response = data.get("response", "")
                        sources = data.get("sources", [])
                        response_time = data.get("response_time_ms", 0)
                        st.session_state.messages[-1]["sources"] = sources
                    else:
                        bot_response = (
                            "Session expired or backend unavailable. "
                            "Please **Clear Session** or reload the page to continue."
                        )
                except Exception:
                    bot_response = (
                        "The request timed out or failed. "
                        "Please **Clear Session** or reload the page to continue."
                    )

        if not bot_response:
            bot_response = "I couldn't generate a response. Please try again."

        st.session_state.messages[-1]["content"] = bot_response
        st.session_state.messages[-1]["timestamp"] = datetime.now().strftime("%I:%M %p")
        st.rerun()
