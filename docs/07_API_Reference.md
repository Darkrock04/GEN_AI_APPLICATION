# 07 — API Reference

## Base URL
```
http://localhost:8000
```

---

## `GET /health`
Health check to verify backend is running.

**Response:**
```json
{
    "status": "ok",
    "llm_provider": "multi-cloud",
    "uptime": "1h 15m 30s",
    "uptime_seconds": 4530,
    "vector_store": "healthy",
    "total_chunks_indexed": 42,
    "total_documents": 1,
    "version": "5.0"
}
```

---

## `POST /chat`
Main conversation endpoint. Synchronous, returns the full response.

**Request Body:**
```json
{
    "message": "What is machine learning?",
    "history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ | User's message (max 8000 chars) |
| `history` | list | ❌ | Previous conversation messages |

**Response:**
```json
{
    "response": "Machine learning is a subset of AI that...",
    "status": "success",
    "sources": [],
    "token_usage": null,
    "response_time_ms": 1542
}
```

---

## `POST /chat/stream`
Streaming conversation endpoint. Returns NDJSON with real-time pipeline state updates.

**Request Body:** Same as `/chat`

**Response (NDJSON stream):**
```json
{"node": "stress_test", "update": {"is_safe": true}}
{"node": "planner", "update": {"plan": "Step 1: ..."}}
{"node": "worker", "update": {"draft": "Here is the response..."}}
...
```

---

## `POST /upload_doc`
Upload a document for RAG.

**Request:** `multipart/form-data` with `file` field

**Supported types:** `.pdf`, `.txt`  
**Max size:** 10MB

**Response:**
```json
{
    "message": "Ingested 'report.pdf'",
    "chunks_added": 42
}
```

---

## `GET /documents`
List all uploaded documents and their chunk counts.

**Response:**
```json
{
    "documents": [
        {"filename": "report.pdf", "chunks": 42},
        {"filename": "notes.txt", "chunks": 8}
    ],
    "total_chunks": 50
}
```

---

## `DELETE /documents/{filename}`
Delete a specific document and its vector embeddings.

**Response:**
```json
{"status": "success", "message": "Deleted 'report.pdf'"}
```

---

## `POST /clear_session`
Clear all documents, vectors, and uploaded files. Automatically called when a new session starts.

**Response:**
```json
{"status": "success", "message": "Session cleared."}
```

---

## `GET /analytics`
Retrieve session-level analytics.

**Response:**
```json
{
    "messages_sent": 5,
    "messages_received": 5,
    "documents_uploaded": 1,
    "total_tokens_used": 0,
    "avg_response_time_ms": 12500
}
```

---

## `POST /export_chat`
Export the current chat history as a formatted Markdown string.

**Request Body:**
```json
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
}
```

**Response:**
```json
{
    "content": "# SPARK AI Chat Export\n\n**User (10:00 AM)**\nHello\n\n..."
}
```
