# 01 — Project Overview

## What is SPARK AI?

SPARK AI is a production-grade, enterprise **Dual-Process (System 1 + System 2) Multi-Agent Generative AI Application**. It combines an ultra-fast, non-autoregressive **System 1 Decision Engine** (powered by **Laya** running on ModernBERT-large and mmBERT-base with RLCD) with a multi-cloud ensemble of **System 2 Generative Frontier LLMs** (Google Gemini, Ollama Cloud, and Nvidia NIM).

Unlike conventional GenAI chatbots that wake up a heavy 31B–120B parameter autoregressive model for every trivial greeting or classification, SPARK AI routes requests through a cognitive dual-brain harness:
* **System 1 (Reflex Arc — Laya Engine):** Sub-second, mathematically calibrated decisions for safety filtering, multilingual greeting triage, CRAG document relevance scoring, model routing, and Universal Verification (hallucination checks).
* **System 2 (Deep Reasoning — Generative Ensemble):** Deep planning, query decomposition, code synthesis (Gemini 3.5 Flash), creative generation (GPT-OSS 120B), and knowledge synthesis (GPT-OSS 120B / Gemini 3.1 Flash-Lite).

---

## Key Capabilities

| Capability | Category | Description |
|---|---|---|
| **⚡ Dual-Brain Engine** | System 1 + System 2 | Non-autoregressive RLCD decision heads paired with generative frontier LLMs. |
| **🛡️ Multilingual Triage** | System 1 (Laya) | Sub-second safety evaluation & automatic language routing for greetings across 100+ languages. |
| **📑 Corrective RAG (CRAG)** | Hybrid Retrieval | ChromaDB semantic search + BM25 keyword search, graded by Laya relevance scoring with SearXNG web search fallback. |
| **🔬 Universal Verification** | Self-RAG | Instant non-autoregressive factual verification and hallucination detection before output delivery. |
| **🧠 Deep Reason & Code** | System 2 (Multi-Agent) | Structured planning, dynamic intent dispatching, iterative self-correction, and synthesis. |
| **🔭 Observability & Telemetry** | Control Plane | Langfuse central prompt registry, generation linking, latency/token tracing, and dual-brain decision telemetry. |

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Streamlit (Python) | Interactive chat UI, pipeline stepper, dual-brain telemetry badges |
| **Backend API** | FastAPI, LangGraph, LangChain | Asynchronous REST server, stateful graph orchestrator |
| **System 1 Engine** | **Laya** (ModernBERT-large & mmBERT-base) | Self-hosted non-autoregressive RLCD decision engine (`/v1/systemone`) |
| **System 2 Providers** | Multi-Cloud (Google Gemini, Ollama Cloud, Nvidia NIM) | Deep planning, code generation, creative writing, factuality verification |
| **Embeddings** | Google Gemini (`models/gemini-embedding-001`) | High-density semantic vector representations |
| **Vector Database** | ChromaDB (local, ephemeral per session) | Hybrid dense vector + sparse keyword retrieval |
| **Web Search** | SearXNG (self-hosted meta-search engine) | Live internet grounding when documents are irrelevant or cutoff occurs |
| **Prompt Management** | Langfuse (Cloud/Self-hosted) | Managed prompt registry, automated seeding, 300s TTL cache & quality scoring |

---

## Project Structure

```
SPARK-AI/
├── app.py                      # Root application supervisor (FastAPI + Streamlit)
├── requirements.txt            # Python dependencies (LangChain, LangGraph, Langfuse, httpx, etc.)
├── .env                        # Environment variables & API credentials (gitignored)
├── .env.example                # Configuration template with Laya endpoint settings
├── backend/
│   ├── api_models.py           # Pydantic request/response schemas (with dual-brain telemetry)
│   ├── laya_client.py          # Laya System One Decision Client (circuit breaker & typed primitives)
│   ├── llm_factory.py          # Model registry & multi-cloud LLM initialization
│   ├── graph_agent.py          # LangGraph multi-agent pipeline (10 nodes, Dual-Process orchestration)
│   ├── langfuse_prompt_manager.py # Prompt registry, auto-seeding, 300s TTL cache & trace scoring
│   ├── vector_store.py         # ChromaDB RAG engine with adaptive chunking & fallback
│   ├── tools.py                # SearXNG live web search integration & query extraction
│   └── main.py                 # FastAPI server, health endpoints & dual-brain API routes
├── frontend/
│   └── app.py                  # Streamlit chat UI with dual-brain stepper & badges
└── docs/                       # Comprehensive documentation (01 through 07)
    └── images/                 # Architecture & RAG lifecycle visual diagrams
```
