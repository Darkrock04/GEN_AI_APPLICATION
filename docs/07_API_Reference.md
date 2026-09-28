# 07 — API Reference

## Base URL
```
http://localhost:8000
```

---

## `GET /health`
Health check endpoint reporting backend uptime, vector store health, and dual-brain engine status.

**Response:**
```json
{
    "status": "ok",
    "llm_provider": "multi-cloud",
    "system_one_engine": "laya",
    "laya_status": "online",
    "uptime": "1h 15m 30s",
    "uptime_seconds": 4530,
    "vector_store": "healthy",
    "total_chunks_indexed": 42,
    "total_documents": 1,
    "version": "5.0"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Health indicator (`"ok"`) |
| `llm_provider` | string | Active System 2 provider ensemble (`"multi-cloud"`) |
| `system_one_engine` | string | Active System 1 non-autoregressive decision engine (`"laya"`) |
| `laya_status` | string | Laya connectivity status (`"online"`, `"disabled"`, or `"offline"`) |
| `uptime` | string | Human-readable service uptime |
| `uptime_seconds` | int | Total seconds since application launch |
| `vector_store` | string | ChromaDB connection state |
| `total_chunks_indexed`| int | Total document chunks embedded in active collection |
| `total_documents` | int | Total unique documents ingested |
| `version` | string | Semantic application version |

---

## `POST /chat`
Main conversation endpoint. Executes synchronous dual-brain pipeline and returns grounded response with complete System 1 telemetry.

**Request Body:**
```json
{
    "message": "Write a Python script to compute Fibonacci numbers with caching.",
    "history": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "retrieval_k": 10
}
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `message` | string | ✅ | — | User's message prompt (max 8,000 characters) |
| `history` | list | ❌ | `[]` | List of previous conversation turns `[{"role": str, "content": str}]` |
| `temperature` | float | ❌ | `0.7` | Temperature override for System 2 worker models |
| `top_p` | float | ❌ | `0.9` | Top-p nucleus sampling override |
| `max_tokens` | int | ❌ | `2048` | Maximum completion tokens for generative output |
| `retrieval_k` | int | ❌ | `10` | Top-K vector chunks to retrieve from ChromaDB |

**Response:**
```json
{
    "response": "Here is an optimized Python implementation using functools.lru_cache:\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n```",
    "status": "success",
    "sources": [
        {
            "filename": "python_algorithms.pdf",
            "page": 14,
            "content_preview": "Memoization allows recursive Fibonacci implementations...",
            "relevance_rank": 1
        }
    ],
    "system_one_decisions": {
        "greeting": {
            "is_greeting": false,
            "confidence": 0.941,
            "method": "laya"
        },
        "router": {
            "task_type": "coding",
            "confidence": 0.861,
            "method": "laya"
        },
        "crag": {
            "is_relevant": true,
            "score": 1.36,
            "confidence": 0.802,
            "method": "laya"
        },
        "universal_verification": {
            "is_faithful": true,
            "confidence": 0.983,
            "method": "laya"
        }
    },
    "token_usage": null,
    "response_time_ms": 1845
}
```

---

## `POST /chat/stream`
Streaming conversation endpoint broadcasting NDJSON (Newline Delimited JSON) events in real time as each pipeline node completes execution.

**Request Body:** Same schema as `POST /chat`.

**Response Stream (NDJSON):**
```json
{"node": "stress_test", "update": {"is_safe": true, "system_one_decisions": {"greeting": {"is_greeting": false, "confidence": 0.94, "method": "laya"}}, "_elapsed_ms": 145}}
{"node": "planner", "update": {"plan": "1. Analyze Fibonacci recursion...", "_elapsed_ms": 320}}
{"node": "retrieve", "update": {"sources": [...], "_elapsed_ms": 210}}
{"node": "grade_documents", "update": {"doc_relevance": "RELEVANT", "system_one_decisions": {"crag": {"is_relevant": true, "score": 1.36, "confidence": 0.80, "method": "laya"}}, "_elapsed_ms": 180}}
{"node": "router", "update": {"task_type": "coding", "system_one_decisions": {"router": {"task_type": "coding", "confidence": 0.86, "method": "laya"}}, "_elapsed_ms": 140}}
{"node": "worker", "update": {"draft": "```python\n...", "_elapsed_ms": 1250}}
{"node": "validation", "update": {"validation_pass": true, "system_one_decisions": {"universal_verification": {"is_faithful": true, "confidence": 0.98, "method": "laya"}}, "_elapsed_ms": 200}}
{"node": "evaluation", "update": {"final_response": "...", "_elapsed_ms": 310}}
```

---

## `POST /upload_doc`
Upload a PDF or TXT document for adaptive chunking and vector embedding into the current session.

**Request:** `multipart/form-data` with `file` binary payload.
- Supported file types: `.pdf`, `.txt`
- Maximum file size: 10 MB

**Response:**
```json
{
    "message": "Ingested 'python_algorithms.pdf'",
    "chunks_added": 42
}
```

---

## `GET /documents`
List all ingested documents and their chunk counts in the active session.

**Response:**
```json
{
    "documents": [
        {"filename": "python_algorithms.pdf", "chunks": 42},
        {"filename": "notes.txt", "chunks": 8}
    ],
    "total_chunks": 50
}
```

---

## `DELETE /documents/{filename}`
Deletes a specific document and its vector embeddings from the active ChromaDB collection.

**Response:**
```json
{
    "status": "success",
    "message": "Deleted 'notes.txt'"
}
```

---

## `POST /clear_session`
Resets the entire session, creating a new ephemeral ChromaDB collection and purging disk caches.

**Response:**
```json
{
    "status": "success",
    "message": "Session cleared."
}
```

---

## `GET /analytics`
Retrieves cumulative session analytics.

**Response:**
```json
{
    "messages_sent": 12,
    "messages_received": 12,
    "documents_uploaded": 2,
    "total_tokens_used": 0,
    "avg_response_time_ms": 2150
}
```

---

## `POST /export_chat`
Exports the conversation history as formatted Markdown.

**Request Body:**
```json
{
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]
}
```

**Response:**
```json
{
    "content": "# SPARK AI Chat Export\n\n**User (10:00 AM)**\nHello!\n\n**SPARK AI (10:00 AM)**\nHello! How can I assist you today?"
}
```
