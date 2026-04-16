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
{"status": "ok", "llm_provider": "siliconflow"}
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
    "status": "success"
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
