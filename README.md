# ⚡ SPARK AI — Multi-Agent RAG Application

A production-grade, multi-agent AI assistant powered by **SiliconFlow Models**. Features intelligent task routing, document RAG, and quality validation — optimized for speed and free-tier deployment.

**🔗 Live Demo:** [https://darkrock04-spark.hf.space/](https://darkrock04-spark.hf.space/)

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

## 🏗️ Architecture

<img width="1024" height="1536" alt="arch" src="https://github.com/user-attachments/assets/fce89528-ece8-4ff0-aed0-661073dd334b" />


### LLM Calls Per Request

| Request Type | LLM Calls | Path |
|---|---|---|
| Simple greeting ("hello") | **1** | Security (keyword) → Quick Response |
| Complex query (all pass) | **5** | Security + Planner + Router + Worker + Validator |
| Complex + evaluator | **5–6** | Above + Evaluator (only for responses > 500 chars) |
| Complex + 1 retry | **7** | Above + Worker retry + Validator |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install
pip install -r requirements.txt

# 3. Configure
# Create a .env file and set SILICONFLOW_API_KEY

# 4. Start backend
uvicorn backend.main:app --reload

# 5. Start frontend (new terminal)
streamlit run frontend/app.py
```

---

## 🤖 Models

| Agent | Model | Purpose |
|---|---|---|
| Security | `Qwen/Qwen2.5-7B-Instruct` | Fast SAFE/UNSAFE classification |
| Planner | `Qwen/Qwen2.5-7B-Instruct` | Task decomposition |
| Router | `Qwen/Qwen2.5-7B-Instruct` | Classify: coding/creative/general |
| Workers (×3) | `deepseek-ai/DeepSeek-V3` | Generate response |
| Validator | `Qwen/Qwen2.5-7B-Instruct` | Quality check |
| Evaluator | `deepseek-ai/DeepSeek-V3` | Polish & format |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | Document RAG vectors |

All models accessed via SiliconFlow API.

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

## 🌐 Environment Variables

Create a `.env` file and configure the following variables:

| Variable | When to set |
|----------|-------------|
| `LLM_PROVIDER` | `siliconflow` (default), `nvidia`, `openai`, `openai_compatible`, or `google_genai` |
| `SILICONFLOW_API_KEY` | Required when `LLM_PROVIDER=siliconflow` |
| `NVIDIA_API_KEY` | Required when `LLM_PROVIDER=nvidia` |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |
| `CUSTOM_LLM_BASE_URL` | Required when `LLM_PROVIDER=openai_compatible` |
| `GOOGLE_API_KEY` | Required when `LLM_PROVIDER=google_genai` |
| `BACKEND_URL` | Optional: backend base URL (default `http://localhost:8000`) |

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

## 🚀 Deploy to Hugging Face Spaces

1. Create a new **Streamlit** Space on [huggingface.co/spaces](https://huggingface.co/new-space)
2. Connect your GitHub repo (or upload files directly)
3. Copy the YAML header from [`deploy/HUGGINGFACE_README_SNIPPET.md`](deploy/HUGGINGFACE_README_SNIPPET.md) to the Space's README
4. In Space **Settings → Secrets**, add `NVIDIA_API_KEY`
5. The root `app.py` auto-starts FastAPI + Streamlit — no extra config needed

**Hardware:** CPU Basic is sufficient — all LLM inference is remote via NVIDIA NIM API.

---

## 🔧 Session & Memory Behavior

- **Current session:** AI remembers your entire conversation (last 10 messages sent as context)
- **Page reload / tab close:** Everything is cleared — no persistent storage
- **Clear Session button:** Wipes chat history, uploaded documents, and vector store

---

## 📄 Document RAG Pipeline

```
Upload → Load (PyPDF/TextLoader) → Adaptive Chunk → Embed → Store (ChromaDB)
                                         ↑
                              Dynamic sizing based on:
                              • Page count
                              • Text density
                              • Semantic separators
```

| Document Size | Chunk Size | Overlap |
|---|---|---|
| ≤ 3 pages | 400 chars | 100 |
| 4–10 pages | 600 chars | 150 |
| 11–30 pages | 1000 chars | 200 |
| 30+ pages | 1500 chars | 300 |

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Backend: Offline" | Start backend: `uvicorn backend.main:app --reload` |
| Timeout on HF Spaces | The free tier may sleep after inactivity — reload the page |
| "Session expired" | Click **Clear Session** or reload the page |
| Upload fails | Max file size is 10MB. Scanned image PDFs are not supported. |
| NVIDIA API errors | Check your API key at [build.nvidia.com](https://build.nvidia.com) |

---

## 🔮 Performance Notes (Free Tier)

- All LLM calls have a **45-second timeout** — no infinite hangs
- Security gate uses **keyword pre-filter** (catches 95%+ without LLM call)
- Validation is **1 consolidated LLM call** instead of 3 separate ones
- Evaluator is **skipped** for short responses (< 500 chars)
- Simple greetings take **1 LLM call** total
- ChromaDB runs in-memory with disk persistence — no database server needed

---

## 📚 Documentation

See [`docs/`](docs/) for detailed documentation:

1. [Project Overview](docs/01_Project_Overview.md)
2. [Architecture & Workflow](docs/02_Architecture_and_Workflow.md)
3. [Models & Agents](docs/03_NVIDIA_Models_and_Agents.md)
4. [Technical Modules](docs/04_Technical_Modules.md)
5. [RAG Concepts](docs/05_RAG_Concepts.md)
6. [Features Deep Dive](docs/06_Features_Deep_Dive.md)
7. [API Reference](docs/07_API_Reference.md)

---

## 📄 License

[MIT](LICENSE)
