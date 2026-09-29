# ⚡ SPARK AI — Dual-Brain Multi-Agent RAG Application

A production-grade, enterprise **Dual-Process (System 1 + System 2) Multi-Agent AI Assistant** combining an ultra-fast, non-autoregressive **System 1 Decision Engine** (Laya with RLCD) with a **System 2 Generative Multi-Cloud Ensemble** (Google Gemini, Ollama Cloud, and Nvidia NIM). Features sub-second intent triage, calibrated Corrective RAG (CRAG), Universal Verification (hallucination detection), and stateful circuit-breaker resilience.



---

## 📖 The Dual-Brain Paradigm (System 1 + System 2)

Traditional multi-agent systems suffer from a heavy LLM monoculture: every simple greeting, routing check, or document relevance classification invokes a 30B–120B parameter autoregressive model, wasting 2–4 seconds and exhausting free-tier token quotas.

SPARK AI breaks this bottleneck with a **Dual-Process Cognitive Architecture** inspired by Daniel Kahneman:
* **System 1 (Reflex Arc — Laya Engine):** Non-autoregressive decision heads powered by **ModernBERT-large** (English) and **mmBERT-base** (100+ languages) fine-tuned via **Reinforcement Learning for Calibrated Decisions (RLCD)**. Operates in single-pass forward evaluations (~100–250ms), producing mathematically calibrated probability distributions without token generation.
* **System 2 (Prefrontal Cortex — Generative Frontier Ensemble):** Deep, deliberative frontier models (Google Gemini 3.5 Flash, Gemini 3.1 Flash-Lite, Ollama Cloud GPT-OSS 120B, Mistral Small 24B, Llama 3.3 70B, and Nvidia Nemotron) for date-aware planning, complex code synthesis, creative writing, and multi-source document synthesis.

---

## ✨ Key Features

| Feature | Category | Description |
|---|---|---|
| ⚡ **Dual-Brain Engine** | System 1 + 2 | Non-autoregressive RLCD decision heads paired with generative frontier LLMs. |
| 🛡️ **Multilingual Triage** | System 1 (Laya) | Sub-second safety evaluation & automatic language routing for greetings across 100+ languages. |
| 📑 **Corrective RAG (CRAG)** | Hybrid Retrieval | ChromaDB semantic search + BM25 keyword search, graded by Laya relevance scoring with SearXNG web search fallback. |
| 🔬 **Universal Verification** | Self-RAG | Instant non-autoregressive factual verification and hallucination detection before output delivery. |
| 🧠 **Deep Reason & Code** | System 2 (Multi-Agent) | Structured planning, dynamic intent dispatching, iterative self-correction, and synthesis. |
| 🔒 **Circuit-Breaker Resilience** | High Availability | Stateful circuit breaker (5 failures / 15s cooldown) ensuring zero downtime and instant fallback to System 2 LLMs. |
| 🔭 **Observability & Telemetry** | Control Plane | Langfuse central prompt registry, generation linking, latency/token tracing, and dual-brain decision telemetry. |

---

<img width="1024" alt="arch" src="docs/images/architecture_v3.png" />

---

## 🤖 Models & Decision Primitives

| Agent / Role | Cognitive System | Provider / Host | Model / Engine | Prompt / Primitive | Strategic Purpose |
|---|---|---|---|---|---|
| **Security Gate** | System 2 (Safety Guardrail) | Ollama Cloud | `nemotron-3-nano:30b` | `security_gate_prompt` | High-precision prompt injection and adversarial query audit. |
| **Greeting Reflex** | System 1 (Reflex Arc) | Laya Self-Host | mmBERT-base / ModernBERT | `is_greeting: noul` | Instant sub-second greeting detection across 100+ languages. |
| **Quick Greeter** | System 2 (Conversational) | Ollama Cloud | `gpt-oss:120b` | `simple_answer_prompt` | Natural, warm conversational greeting synthesis. |
| **Planner** | System 2 (Reasoning) | Google (Gemini) | `gemini-3.1-flash-lite` | `planner_prompt` | Date-aware query decomposition & web search trigger. |
| **Document Grader** | System 1 (Reflex Arc) | Laya Self-Host | ModernBERT-large | `doc_relevance: choice` + `score` | Corrective RAG (CRAG) calibrated document relevance grading. |
| **Router Reflex** | System 1 (Reflex Arc) | Laya Self-Host | ModernBERT-large | `task_type: choice` | Calibrated non-autoregressive intent dispatching. |
| **Worker (Coding)** | System 2 (Generative) | Google (Gemini) | `gemini-3.5-flash` | `worker_coding_prompt` | Production-grade code synthesis with strict execution tests. |
| **Worker (Creative)** | System 2 (Generative) | Ollama Cloud | `gpt-oss:120b` | `worker_creative_prompt` | Unconstrained creative writing and brainstorming. |
| **Worker (General)** | System 2 (Generative) | Ollama Cloud | `gpt-oss:120b` | `worker_general_prompt` | Comprehensive multi-source document synthesis. |
| **Universal Verification** | System 1 (Reflex Arc) | Laya Self-Host | ModernBERT-large | `faithfulness: choice` | Instant non-autoregressive factual faithfulness verification. |
| **Validator** | System 2 (Reasoning) | Google (Gemini) | `gemini-3.1-flash-lite` | `validator_prompt` | Multi-dimensional quality, relevance, and cutoff audit. |
| **Evaluator** | System 2 (Generative) | Ollama Cloud | `nemotron-3-super` | `evaluator_prompt` | Final polish, math rendering, and LaTeX formatting. |
| **Embeddings** | System 1 (Vector RAG) | Google (Gemini) | `gemini-embedding-001` | *(Vector Pipeline)* | 768-dimensional dense vector embeddings in ChromaDB. |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Darkrock04/GEN_AI_APPLICATION.git
cd GEN_AI_APPLICATION

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
Populate your `.env` file with the required keys:
```env
# System 1: Laya Decision Engine
LAYA_BASE_URL=your_self_hosted_laya_url
ENABLE_LAYA=true
LAYA_TIMEOUT_SECONDS=3.5

# System 2: Multi-Cloud Providers
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_BASE_URL=https://ollama.com/v1
NVIDIA_API_KEY=your_nvidia_api_key

# Live Web Search & Observability
SEARXNG_URL=your_self_hosted_searxng_url
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Launch Services
```bash
# Terminal 1: Start FastAPI Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Start Streamlit Frontend
streamlit run frontend/app.py
```

---

## 📁 Repository Structure

```
SPARK-AI/
├── app.py                      # Root launcher (FastAPI + Streamlit subprocesses)
├── requirements.txt            # Python dependencies (LangGraph, Langfuse, ChromaDB, etc.)
├── .env                        # Environment variables & API credentials (gitignored)
├── .env.example                # Configuration template with Laya endpoint settings
├── backend/
│   ├── api_models.py           # Pydantic schemas (with dual-brain telemetry)
│   ├── laya_client.py          # Laya System 1 Decision Client (circuit breaker & typed primitives)
│   ├── llm_factory.py          # Multi-Cloud LLM factory (Gemini, Ollama Cloud, Nvidia NIM)
│   ├── graph_agent.py          # LangGraph multi-agent pipeline (10 nodes, Dual-Brain orchestration)
│   ├── langfuse_prompt_manager.py # Prompt registry, auto-seeding, 300s TTL cache & score logger
│   ├── vector_store.py         # ChromaDB RAG engine with adaptive chunking & Gemini embeddings
│   ├── tools.py                # SearXNG live web search client & query extraction
│   └── main.py                 # FastAPI server, startup seeding & health endpoints
├── frontend/
│   └── app.py                  # Streamlit chat UI with dual-brain stepper & badges
├── docs/                       # Complete architectural manuals (01 through 07)
│   └── images/                 # Architectural & RAG lifecycle visual diagrams
└──                      # local environment 
```

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Backend health check (reports Laya and Multi-Cloud status) |
| `POST` | `/chat` | Synchronous conversation with full dual-brain telemetry |
| `POST` | `/chat/stream` | Streaming NDJSON updates broadcasting node-by-node execution |
| `POST` | `/upload_doc` | Upload PDF/TXT for adaptive chunking and vector embedding |
| `GET` | `/documents` | List uploaded documents and total chunk counts |
| `DELETE` | `/documents/{filename}` | Delete a specific document and its vector embeddings |
| `POST` | `/clear_session` | Rotate ChromaDB collection and wipe session state |
| `GET` | `/analytics` | Cumulative session usage and latency metrics |
| `POST` | `/export_chat` | Export conversation history to Markdown |

---

## 📚 Documentation Reference

For comprehensive deep dives into each subsystem, refer to the [`docs/`](docs/) directory:

1. [01 — Project Overview](docs/01_Project_Overview.md)
2. [02 — Architecture & Multi-Agent Workflow](docs/02_Architecture_and_Workflow.md)
3. [03 — Models & Agent Roles](docs/03_Models_and_Agents.md)
4. [04 — Technical Modules Explained](docs/04_Technical_Modules.md)
5. [05 — Dual-Brain RAG & CRAG](docs/05_RAG_Concepts.md)
6. [06 — Features Deep Dive & Enterprise Innovations](docs/06_Features_Deep_Dive.md)
7. [07 — API Reference](docs/07_API_Reference.md)
