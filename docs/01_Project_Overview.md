# 01 — Project Overview

## What is SPARK AI?

SPARK AI is a production-grade, multi-agent Generative AI web application that provides an advanced conversational interface powered by a **Multi-Cloud Architecture** (Nvidia NIM, Google Gemini, and Ollama Cloud). Unlike simple chatbots that use a single LLM for everything, this system routes each request through a sophisticated pipeline of specialized agents — each optimized for its specific role.

## Key Capabilities

| Capability | Description |
|---|---|
| **🧠 Chat & Reason** | Multi-agent pipeline with planning, routing, validation, and evaluation |
| **📑 Document RAG** | Upload PDFs/TXT files and ask questions grounded in your documents |
| **🔒 Content Safety** | Two-stage security gate filters harmful requests |
| **✅ Quality Validation** | Consolidated relevance + factuality + coherence check |
| **🔭 Observability & Prompts** | Langfuse central prompt registry, generation linking, latency/token tracing, and automated validation scoring |

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (Python) |
| **Backend API** | FastAPI, LangGraph, LangChain |
| **LLM Provider** | Multi-Cloud (Google Gemini, Ollama Cloud, Nvidia NIM) |
| **Embeddings** | Google Gemini (`models/gemini-embedding-001`) |
| **Vector Database** | ChromaDB (local, ephemeral per session) |
| **Web Search** | SearXNG (self-hosted meta-search engine) |
| **Observability & Prompt Control** | Langfuse (Cloud/Self-hosted, prompt versioning, trace scoring) |

## Project Structure

```
SPARK-AI/
├── app.py                      # HF Spaces launcher (FastAPI + Streamlit)
├── requirements.txt            # Python dependencies (LangChain, LangGraph, Langfuse, etc.)
├── .env                        # Environment variables (gitignored)
├── backend/
│   ├── api_models.py           # Pydantic request/response schemas
│   ├── llm_factory.py          # Model registry & multi-cloud LLM initialization
│   ├── graph_agent.py          # LangGraph multi-agent pipeline (core)
│   ├── langfuse_prompt_manager.py # Prompt registry, auto-seeding, 300s TTL cache & scoring
│   ├── vector_store.py         # ChromaDB RAG engine with adaptive chunking
│   ├── tools.py                # SearXNG live web search integration
│   └── main.py                 # FastAPI server, startup seeding & endpoints
├── frontend/
│   └── app.py                  # Streamlit chat UI
├── docs/                       # Documentation (01 through 07)
└── deploy/                     # Hugging Face deployment config
```
