# 04 — Technical Modules Explained

## Backend Modules

### 1. `llm_factory.py` — Model Factory
**Purpose:** Central registry that maps agent roles to SiliconFlow models.

- Contains the `SILICONFLOW_MODELS` dictionary with model assignments
- `get_llm(agent_type)` returns a `ChatOpenAI` instance configured for that role
- All instances include `timeout=45` and `max_retries=1` for resilience

### 2. `graph_agent.py` — Multi-Agent Pipeline (Core)
**Purpose:** The brain of the application. Defines the LangGraph StateGraph with all agent nodes.

**Key components:**
- `GraphState` — TypedDict holding all state across the pipeline
- `_safe_llm_call()` — Retry wrapper that never crashes the pipeline
- `stress_test_node()` — Two-stage security (keyword pre-filter + LLM)
- `simple_answer_node()` — Fast path for greetings (1 LLM call)
- `planner_node()` — Task decomposition into 2-4 steps
- `retrieve_context_node()` — ChromaDB RAG retrieval
- `router_node()` — Classifies: coding / creative / general
- `worker_agent_node()` — Routes to specialized worker LLM + prompt
- `validation_node()` — Consolidated single-call quality check
- `evaluation_node()` — Final polish (skipped for short responses)
- `process_chat()` / `stream_graph_updates()` — Public entry points

### 3. `vector_store.py` — RAG Engine
**Purpose:** Document ingestion, embedding, storage, and retrieval using ChromaDB.

**Key features:**
- SiliconFlow embeddings via `Qwen/Qwen3-Embedding-0.6B`
- Adaptive chunking based on page count and text density
- Semantic separators (`\n\n`, `\n`, `. `) for natural boundary splits
- Metadata enrichment (source_file, chunk_index, total_chunks)
- Duplicate detection (won't re-embed the same file twice)
- Document management (list, delete individual documents)
- Windows-safe session cleanup via collection rotation

### 4. `main.py` — FastAPI Server
**Purpose:** HTTP API layer connecting the frontend to the pipeline.

**Endpoints:**
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/chat` | Main conversation |
| `POST` | `/chat/stream` | Streaming NDJSON responses |
| `POST` | `/upload_doc` | Upload PDF/TXT for RAG |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{filename}` | Delete specific document |
| `POST` | `/clear_session` | Wipe all data |

### 5. `api_models.py` — Pydantic Schemas
**Purpose:** Type-safe request/response models for the API.

- `ChatRequest` — message + history
- `ChatResponse` — response text + status
- `DocumentUploadResponse` — message + chunks_added
- `DocumentInfo` / `SessionStatusResponse` — document listing

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
