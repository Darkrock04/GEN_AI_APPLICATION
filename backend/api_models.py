from pydantic import BaseModel
from typing import Optional, List


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class SourceChunk(BaseModel):
    """A single source citation from retrieved documents."""
    filename: str
    page: Optional[int] = None
    content_preview: str
    relevance_rank: int


class TokenUsage(BaseModel):
    """Token usage for a single request."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    response: str
    status: str
    sources: Optional[List[SourceChunk]] = []
    token_usage: Optional[TokenUsage] = None
    response_time_ms: Optional[int] = None


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_added: int


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class SessionStatusResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int


class SessionAnalytics(BaseModel):
    """Session-level analytics for the enterprise dashboard."""
    messages_sent: int = 0
    messages_received: int = 0
    documents_uploaded: int = 0
    total_tokens_used: int = 0
    total_chunks_indexed: int = 0
    avg_response_time_ms: float = 0.0
