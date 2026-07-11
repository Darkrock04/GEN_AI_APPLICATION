# 01 — Project Overview

## What is SPARK AI?

SPARK AI is a production-grade, multi-agent Generative AI web application that provides an advanced conversational interface powered by a **Multi-Cloud Architecture** (Nvidia NIM, Google Gemini, and Cerebras). Unlike simple chatbots that use a single LLM for everything, this system routes each request through a sophisticated pipeline of specialized agents — each optimized for its specific role.

## Key Capabilities

| Capability | Description |
|---|---|
| **🧠 Chat & Reason** | Multi-agent pipeline with planning, routing, validation, and evaluation |
| **📑 Document RAG** | Upload PDFs/TXT files and ask questions grounded in your documents |
| **🔒 Content Safety** | Two-stage security gate filters harmful requests |
| **✅ Quality Validation** | Consolidated relevance + factuality + coherence check |

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (Python) |
| **Backend API** | FastAPI, LangGraph, LangChain |
| **LLM Provider** | Multi-Cloud (Nvidia NIM, Google Gemini, Cerebras) |
| **Embeddings** | Nvidia (`nvidia/nv-embedqa-e5-v5`) |
| **Vector Database** | ChromaDB (local, ephemeral per session) |

## Project Structure

```
SPARK-AI/
├── app.py                  # HF Spaces launcher
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (gitignored)
├── backend/
│   ├── api_models.py       # Pydantic request/response schemas
│   ├── llm_factory.py      # Model registry & LLM initialization
│   ├── graph_agent.py      # LangGraph multi-agent pipeline
│   ├── vector_store.py     # ChromaDB RAG engine
│   └── main.py             # FastAPI endpoints
├── frontend/
│   └── app.py              # Streamlit chat UI
├── docs/                   # Documentation
└── deploy/                 # Hugging Face deployment config
```
