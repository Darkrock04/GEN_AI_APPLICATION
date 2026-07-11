# ⚡ SPARK AI — Multi-Agent RAG Application

A production-grade, multi-agent AI assistant powered by a **Multi-Cloud Architecture** (Nvidia NIM, Google Gemini, and Cerebras). Features intelligent task routing, document RAG, and quality validation — optimized for speed and free-tier deployment.

**🔗 Live Demo:** [https://darkrock04-spark.hf.space/](https://darkrock04-spark.hf.space/)

---

## 📖 Project Overview

SPARK AI is a robust Generative AI web application providing an advanced conversational interface. Unlike simple chatbots that rely on a single LLM, SPARK AI routes each request through a sophisticated, multi-agent pipeline powered by a **Multi-Cloud Architecture**. From intelligent task routing and creative generation to document-grounded RAG (Retrieval-Augmented Generation) and content safety validation, every agent is optimized for its specific role to deliver fast, highly accurate, and reliable responses.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Chat & Reason** | Multi-agent pipeline with automated planning, dynamic routing, and validation |
| 📑 **Document RAG** | Upload PDFs/TXT — adaptive chunking, embedding, and intelligent retrieval |
| 🔒 **Content Safety** | Two-stage security gate (keyword pre-filter + LLM fallback) |
| ✅ **Quality Validation** | Consolidated relevance + factuality + coherence check |
| ⚡ **Specialized Workers** | Different routing for coding, creative, and general tasks |
| 🔄 **Session Memory** | Remembers your conversation within the current session |
| 📊 **Pipeline Streaming** | Real-time visibility into each processing stage |

---

<img width="1024" alt="arch" src="docs/images/architecture_v2.png" />




---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install
pip install -r requirements.txt

# 3. Configure API Keys
# Create a .env file and set the following keys:
# NVIDIA_API_KEY=your_key_here
# CEREBRAS_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here

# 4. Start backend
uvicorn backend.main:app --reload

# 5. Start frontend (new terminal)
streamlit run frontend/app.py
```

---

## 🤖 Models

| Agent | Provider | Model | Purpose |
|---|---|---|---|
| Security | **Nvidia NIM** | `meta/llama-3.1-8b-instruct` | Fast SAFE/UNSAFE classification |
| Planner | **Google** | `gemini-3.1-flash-lite` | Task decomposition |
| Router | **Cerebras** | `gemma-4-31b` | Classify: coding/creative/general |
| Worker (General) | **Cerebras** | `gpt-oss-120b` | General generation |
| Worker (Creative) | **Cerebras** | `gpt-oss-120b` | Creative writing |
| Worker (Coding) | **Google** | `gemini-3.5-flash` | Code generation |
| Validator | **Google** | `gemini-3.1-flash-lite` | Quality check |
| Evaluator | **Nvidia NIM** | `meta/llama-3.1-70b-instruct` | Polish & format |
| Embeddings | **Nvidia NIM** | `nvidia/nv-embedqa-e5-v5` | Document RAG vectors |

All models accessed via their respective free-tier/trial APIs.

---

## 📁 Project Structure

```
SPARK-AI/
├── app.py                  # HF Spaces launcher (FastAPI + Streamlit)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (gitignored)
├── backend/
│   ├── api_models.py       # Pydantic request/response schemas
│   ├── llm_factory.py      # Model registry & LLM initialization
│   ├── graph_agent.py      # LangGraph multi-agent pipeline (core)
│   ├── vector_store.py     # ChromaDB RAG engine with adaptive chunking
│   └── main.py             # FastAPI server & endpoints
├── frontend/
│   └── app.py              # Streamlit chat UI
├── deploy/
│   └── HUGGINGFACE_README_SNIPPET.md
├── docs/                   # Detailed documentation
└── chroma_db/              # Vector database (gitignored)
```

---



## 📡 API Reference

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Backend health check |
| `POST` | `/chat` | Synchronous chat (returns full response) |
| `POST` | `/chat/stream` | Streaming chat (NDJSON, one event per pipeline node) |
| `POST` | `/upload_doc` | Upload PDF/TXT for RAG |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{filename}` | Delete specific document |
| `POST` | `/clear_session` | Wipe all data & start fresh |

---


## 🔧 Session & Memory Behavior

- **Current session:** AI remembers your entire conversation (last 10 messages sent as context)
- **Page reload / tab close:** Everything is cleared — no persistent storage
- **Clear Session button:** Wipes chat history, uploaded documents, and vector store

---

## 📄 Document RAG Pipeline

![Document RAG Pipeline](docs/images/rag_pipeline.png)

| Document Size | Chunk Size | Overlap |
|---|---|---|
| ≤ 3 pages | 400 chars | 100 |
| 4–10 pages | 600 chars | 150 |
| 11–30 pages | 1000 chars | 200 |
| 30+ pages | 1500 chars | 300 |

---

## 📚 Documentation

See [`docs/`](docs/) for detailed documentation:

1. [Project Overview](docs/01_Project_Overview.md)
2. [Architecture & Workflow](docs/02_Architecture_and_Workflow.md)
3. [Models & Agents](docs/03_Models_and_Agents.md)
4. [Technical Modules](docs/04_Technical_Modules.md)
5. [RAG Concepts](docs/05_RAG_Concepts.md)
6. [Features Deep Dive](docs/06_Features_Deep_Dive.md)
7. [API Reference](docs/07_API_Reference.md)


