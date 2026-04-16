import os
import json
import time
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from backend.api_models import (
    ChatRequest, ChatResponse, DocumentUploadResponse,
    DocumentInfo, SessionStatusResponse, SessionAnalytics,
    SourceChunk, TokenUsage,
)
from backend.graph_agent import process_chat, stream_graph_updates

from backend.vector_store import vector_store_manager, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SPARK AI",
    description="High-performance multi-agent RAG pipeline — ⚡ SPARK AI",
    version="5.0",
)

_cors_raw = os.getenv("CORS_ORIGINS", "*").strip()
if not _cors_raw or _cors_raw == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SESSION-LEVEL ANALYTICS TRACKER (Feature 12)
class _Analytics:
    """Simple in-memory session analytics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.documents_uploaded = 0
        self.total_tokens_used = 0
        self.response_times = []

    @property
    def avg_response_time_ms(self):
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

analytics = _Analytics()


# GLOBAL EXCEPTION HANDLER
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)[:200]}"}
    )


# HEALTH CHECK (Feature 13 — Enhanced)
_start_time = time.time()

@app.get("/health")
async def health_check():
    uptime_seconds = int(time.time() - _start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Check vector store health
    vs_status = "healthy"
    try:
        docs = vector_store_manager.get_document_list()
        total_chunks = sum(d["chunks"] for d in docs)
    except Exception:
        vs_status = "degraded"
        total_chunks = 0

    return {
        "status": "ok",
        "llm_provider": "nvidia",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": uptime_seconds,
        "vector_store": vs_status,
        "total_chunks_indexed": total_chunks,
        "total_documents": len(docs) if vs_status == "healthy" else 0,
        "version": "5.0",
    }


# CHAT (synchronous)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message or not request.message.strip():
        return ChatResponse(response="Please type a message.", status="error")

    user_message = request.message.strip()
    if len(user_message) > 8000:
        logger.warning(f"Message truncated from {len(user_message)} to 8000 chars.")
        user_message = user_message[:8000] + "... [truncated]"

    history_dicts = []
    for m in (request.history or []):
        history_dicts.append({"role": m.role, "content": m.content[:1000]})

    analytics.messages_sent += 1

    try:
        result = process_chat(request=user_message, history=history_dicts)
        response_text = result.get("response", "")
        if not response_text or not response_text.strip():
            response_text = "I couldn't generate a response. Please try rephrasing."

        # Build source citations
        sources = [
            SourceChunk(
                filename=s.get("filename", "unknown"),
                page=s.get("page"),
                content_preview=s.get("content_preview", ""),
                relevance_rank=s.get("relevance_rank", 0),
            )
            for s in result.get("sources", [])
        ]

        total_ms = result.get("total_ms", 0)
        analytics.messages_received += 1
        analytics.response_times.append(total_ms)

        return ChatResponse(
            response=response_text,
            status="success",
            sources=sources,
            response_time_ms=total_ms,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            response="Something went wrong while processing your request. Please try again.",
            status="error"
        )


# CHAT (streaming NDJSON)
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """NDJSON stream: one JSON object per line with keys `node` and `update`."""
    if not request.message or not request.message.strip():
        def err1():
            yield json.dumps({"node": "error", "update": {"final_response": "Please type a message."}}) + "\n"
        return StreamingResponse(err1(), media_type="application/x-ndjson")

    user_message = request.message.strip()
    if len(user_message) > 8000:
        logger.warning(f"Message truncated from {len(user_message)} to 8000 chars.")
        user_message = user_message[:8000] + "... [truncated]"

    history_dicts = []
    for m in (request.history or []):
        history_dicts.append({"role": m.role, "content": m.content[:1000]})

    analytics.messages_sent += 1

    def ndjson_iter():
        last_elapsed = 0
        try:
            for evt in stream_graph_updates(user_message, history_dicts):
                if "update" in evt and "_elapsed_ms" in evt["update"]:
                    last_elapsed = evt["update"]["_elapsed_ms"]
                yield json.dumps(evt, default=str) + "\n"
            analytics.messages_received += 1
            if last_elapsed > 0:
                analytics.response_times.append(last_elapsed)
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield json.dumps({
                "node": "error",
                "update": {"final_response": "Something went wrong while processing your request."},
            }, default=str) + "\n"

    return StreamingResponse(ndjson_iter(), media_type="application/x-ndjson")


# DOCUMENT UPLOAD
@app.post("/upload_doc", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        return DocumentUploadResponse(
            message=f"Unsupported type '{file_ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            chunks_added=0
        )

    # Sanitize filename
    safe_filename = "".join(
        c if c.isalnum() or c in (".", "_", "-") else "_"
        for c in os.path.basename(filename)
    )
    if not safe_filename or safe_filename.startswith("."):
        safe_filename = f"upload_{safe_filename}"

    file_content = await file.read()

    if len(file_content) == 0:
        return DocumentUploadResponse(message="File is empty.", chunks_added=0)
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_content) / (1024 * 1024)
        return DocumentUploadResponse(message=f"File too large ({size_mb:.1f}MB). Max is 10MB.", chunks_added=0)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    file_path = os.path.join(data_dir, safe_filename)

    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)
        chunks = vector_store_manager.ingest_document(file_path)
        analytics.documents_uploaded += 1
        return DocumentUploadResponse(message=f"Ingested '{safe_filename}'", chunks_added=chunks)
    except (ValueError, FileNotFoundError) as e:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        return DocumentUploadResponse(message=str(e), chunks_added=0)
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        return DocumentUploadResponse(message=f"System error: {str(e)[:200]}", chunks_added=0)


# DOCUMENT MANAGEMENT
@app.get("/documents")
async def list_documents():
    try:
        docs = vector_store_manager.get_document_list()
        total = sum(d["chunks"] for d in docs)
        return SessionStatusResponse(
            documents=[DocumentInfo(**d) for d in docs],
            total_chunks=total
        )
    except Exception:
        return SessionStatusResponse(documents=[], total_chunks=0)


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    safe_name = os.path.basename(filename)
    success = vector_store_manager.delete_document(safe_name)
    if success:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        try:
            fpath = os.path.join(data_dir, safe_name)
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception:
            pass
        return {"status": "success", "message": f"Deleted '{safe_name}'"}
    return {"status": "error", "message": f"'{safe_name}' not found."}


# SESSION MANAGEMENT
@app.post("/clear_session")
async def clear_session():
    errors = []
    try:
        vector_store_manager.clear_session()
    except Exception as e:
        errors.append(str(e))

    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                try:
                    os.remove(os.path.join(data_dir, f))
                except Exception:
                    pass
    except Exception as e:
        errors.append(str(e))

    analytics.reset()

    if errors:
        return {"status": "partial", "message": f"Errors: {'; '.join(errors)}"}
    return {"status": "success", "message": "Session cleared."}


# SESSION ANALYTICS (Feature 12)
@app.get("/analytics")
async def get_analytics():
    try:
        docs = vector_store_manager.get_document_list()
        total_chunks = sum(d["chunks"] for d in docs)
    except Exception:
        total_chunks = 0

    return SessionAnalytics(
        messages_sent=analytics.messages_sent,
        messages_received=analytics.messages_received,
        documents_uploaded=analytics.documents_uploaded,
        total_tokens_used=analytics.total_tokens_used,
        total_chunks_indexed=total_chunks,
        avg_response_time_ms=round(analytics.avg_response_time_ms, 1),
    )


# CHAT EXPORT (Feature 1)
@app.post("/export_chat")
async def export_chat(request: Request):
    """Export chat history as Markdown."""
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return {"content": "# SPARK AI Chat\n\n_No messages to export._"}

    lines = ["# ⚡ SPARK AI — Chat Export\n"]
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        if role == "user":
            lines.append(f"## 👤 User{' — ' + timestamp if timestamp else ''}\n")
        else:
            lines.append(f"## 🤖 SPARK AI{' — ' + timestamp if timestamp else ''}\n")
        lines.append(content)
        lines.append("\n---\n")

    return {"content": "\n".join(lines)}
