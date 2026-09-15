# 04 — Technical Modules Explained

## Backend Modules

### 1. `llm_factory.py` — Multi-Cloud Model Factory
**Purpose:** Central registry mapping specialized agent roles to multi-cloud LLM providers (Ollama Cloud, Google Gemini, Nvidia NIM).

- Contains the `AGENT_CONFIG` dictionary mapping roles to cloud providers and models
- Initializes `ChatOpenAI` for Ollama Cloud / Nvidia NIM and `ChatGoogleGenerativeAI` for Google Gemini
- Configures default max token limits per role (e.g., 16 tokens for router, 64 for security, 2048 for workers)
- Includes strict timeouts and retry logic for high resilience

### 2. `langfuse_prompt_manager.py` — Prompt Registry & Quality Scoring
**Purpose:** Central prompt management plane and evaluation logger for Langfuse.

- **`DEFAULT_PROMPTS`:** Catalog of all 10 system prompts in Langfuse `{{variable}}` mustache format
- **`ensure_prompts_seeded()`:** Application boot utility that creates any missing prompts in Langfuse labeled `production` and tagged `spark-ai`
- **`get_managed_prompt(name)`:** High-performance prompt fetcher with a 300-second TTL cache, returning both the LangChain `{var}` template string and the Langfuse prompt object for trace linking
- **`log_validation_score()`:** Logs automated numeric scores (`quality_validation` = `1.0` / `0.0`) with validator feedback into Langfuse traces

### 3. `graph_agent.py` — Multi-Agent Pipeline (Core)
**Purpose:** The brain of the application. Defines the LangGraph StateGraph connecting all agent nodes.

**Key components:**
- `GraphState` — TypedDict tracking request, history, safety, plan, context, sources, task type, draft, validation status, and node timings
- `_safe_llm_call()` — Resilient wrapper that fetches managed prompts from Langfuse, binds dynamic parameters, attaches prompt linking metadata, and handles rate-limit backoffs
- `stress_test_node()` — Two-stage security check (instant keyword check + LLM fallback using `security_gate_prompt`)
- `simple_answer_node()` — Conversational greeting fast path using `simple_answer_prompt`
- `planner_node()` — Task decomposition and search intent detection using `planner_prompt`
- `web_search_node()` — SearXNG live web retrieval
- `retrieve_context_node()` — ChromaDB RAG retrieval with source citation metadata
- `router_node()` — Task classification into coding/creative/general using `router_prompt`
- `worker_agent_node()` — Specialized generation using `worker_general_prompt`, `worker_coding_prompt`, or `worker_creative_prompt`
- `validation_node()` — Consolidated quality evaluation using `validator_prompt` with automated Langfuse score logging
- `evaluation_node()` — Final polish and LaTeX/Markdown formatting using `evaluator_prompt`
- `_build_history_str()` — Semantic memory compression using `history_summarizer_prompt`
- `process_chat()` / `stream_graph_updates()` — Public entry points injecting Langfuse `CallbackHandler` with session IDs, trace names, and tags

### 4. `tools.py` — Web Search Integration
**Purpose:** Interfaces with a live SearXNG instance for real-time internet data retrieval.

- `perform_web_search(query)`: Queries SearXNG REST API, extracts page snippets and URLs, and formats them into context blocks for workers

### 5. `vector_store.py` — RAG Engine
**Purpose:** Document ingestion, embedding, storage, and retrieval using ChromaDB.

**Key features:**
- Google Gemini embeddings via `models/gemini-embedding-001`
- Adaptive chunking based on page count and text density (400–1500 chars)
- Semantic separators (`\n\n`, `\n`, `. `) for natural boundary splits
- Source citation metadata enrichment (filename, page, content preview, relevance rank)
- Duplicate detection (prevents re-embedding existing files)
- Ephemeral collection rotation on session reset (Windows file-lock safe)

### 6. `main.py` — FastAPI Server
**Purpose:** HTTP API layer connecting the frontend to the pipeline.

**Key features:**
- `@app.on_event("startup")` hook to auto-seed Langfuse prompts on boot
- Synchronous (`/chat`) and streaming NDJSON (`/chat/stream`) endpoints
- Document upload, retrieval, and deletion endpoints
- Session analytics tracking and Markdown chat export

### 7. `api_models.py` — Pydantic Schemas
**Purpose:** Type-safe request/response models for the API.

- `ChatRequest` — message + history + dynamic inference controls
- `ChatResponse` — response text + source citations + timings + status
- `DocumentUploadResponse` — message + chunks_added
- `DocumentInfo` / `SessionStatusResponse` — document listing
- `SourceChunk` — citation metadata (filename, page, preview, rank)

## Frontend

### `frontend/app.py` — Streamlit Chat UI
**Purpose:** Modern chat interface with SPARK AI branding.

**Features:**
- Dark glassmorphism theme with Outfit font
- Welcome screen with capability cards
- Auto-upload document processing (no manual button)
- Document list with individual delete buttons
- Pipeline stage streaming (human-readable node names)
- Session clear on page reload
- Graceful timeout/error handling with user-friendly messages
