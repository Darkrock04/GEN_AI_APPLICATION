import os
import uuid
import time
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Disable ChromaDB telemetry before import
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.llm_factory import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB (HF free tier friendly)


def _get_embeddings():
    """Embedding model mapped directly to SiliconFlow."""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("SILICONFLOW_API_KEY not set in .env.")
    
    embed_model = os.getenv("SILICONFLOW_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    return OpenAIEmbeddings(
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        model=embed_model,
        check_embedding_ctx_length=False,
    )


class VectorStoreManager:
    def __init__(self):
        """Initialize embeddings and connect to ChromaDB."""
        try:
            self.embedding_function = _get_embeddings()
            os.makedirs(DB_DIR, exist_ok=True)
            self.db = self._connect_chromadb("spark_session")
            logger.info("VectorStoreManager initialized using SiliconFlow embeddings.")
        except Exception as e:
            logger.error(f"CRITICAL: VectorStoreManager init failed: {e}")
            raise RuntimeError(f"Could not initialize vector store: {e}")

    def _connect_chromadb(self, collection_name: str):
        """Connect to ChromaDB with stale data recovery."""
        try:
            return Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.embedding_function,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.warning(f"ChromaDB connection failed ({e}), trying fresh collection...")
            fresh_name = f"fresh_{uuid.uuid4().hex[:8]}"
            try:
                return Chroma(
                    persist_directory=DB_DIR,
                    embedding_function=self.embedding_function,
                    collection_name=fresh_name,
                )
            except Exception as e2:
                raise RuntimeError(f"ChromaDB completely broken: {e2}")

    def ingest_document(self, file_path: str) -> int:
        """
        Load → Adaptive Chunk → Embed → Store.
        Returns number of chunks added.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("The uploaded file is empty.")

        # Load
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                except Exception:
                    loader = TextLoader(file_path, encoding="latin-1")
            raw_documents = loader.load()
        except Exception as e:
            raise ValueError(f"Could not read '{os.path.basename(file_path)}': {e}")

        if not raw_documents or all(not doc.page_content.strip() for doc in raw_documents):
            raise ValueError("No readable text found. The file may be a scanned image PDF.")

        # Adaptive Chunking
        num_pages = len(raw_documents)
        total_chars = sum(len(doc.page_content) for doc in raw_documents)
        avg_chars_per_page = total_chars / max(num_pages, 1)

        # Dynamic chunk sizing based on document characteristics
        if num_pages <= 3:
            # Very small document: fine-grained chunks for precise retrieval
            chunk_size = 400
            chunk_overlap = 100
        elif num_pages <= 10:
            # Small-medium document
            chunk_size = 600
            chunk_overlap = 150
        elif num_pages <= 30:
            # Medium document
            chunk_size = 1000
            chunk_overlap = 200
        else:
            # Large document: bigger chunks to keep total count manageable
            chunk_size = 1500
            chunk_overlap = 300

        # Adjust for text density: sparse pages (e.g., slides) get smaller chunks
        if avg_chars_per_page < 200:
            chunk_size = min(chunk_size, 400)
            chunk_overlap = min(chunk_overlap, 100)

        # Use semantic separators for natural boundary splitting
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        docs = splitter.split_documents(raw_documents)
        if not docs:
            raise ValueError("Chunking produced zero chunks.")

        logger.info(
            f"Chunking '{os.path.basename(file_path)}': "
            f"{num_pages} pages, {total_chars} chars, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap} → {len(docs)} chunks"
        )

        # Metadata
        source_name = os.path.basename(file_path)
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["source_file"] = source_name
            doc.metadata["total_chunks"] = len(docs)

        # Duplicate check
        try:
            existing = self.db.get(where={"source_file": source_name})
            if existing and existing.get("ids"):
                logger.warning(f"'{source_name}' already ingested. Skipping.")
                return len(existing["ids"])
        except Exception as e:
            logger.warning(f"Duplicate check failed (proceeding anyway): {e}")

        # Store with retry
        for attempt in range(3):
            try:
                self.db.add_documents(docs)
                logger.info(f"Ingested '{source_name}' ({len(docs)} chunks).")
                return len(docs)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in ("429", "rate", "too many"))
                if is_rate_limit and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Embedding rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Failed to save to ChromaDB: {e}")

        raise RuntimeError("Failed to embed document after 3 attempts.")

    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve top-k most relevant chunks."""
        try:
            results = self.db.similarity_search(query, k=k)
            if not results:
                return ""
            return "\n\n---\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return ""

    def retrieve_with_metadata(self, query: str, k: int = 10) -> list:
        """Retrieve top-k chunks with full metadata for source citations.
        Returns list of Document objects with .page_content and .metadata."""
        try:
            results = self.db.similarity_search(query, k=k)
            return results if results else []
        except Exception as e:
            logger.warning(f"Retrieval with metadata failed: {e}")
            return []

    def get_document_list(self) -> list:
        """Returns list of uploaded documents and their chunk counts."""
        try:
            all_data = self.db.get()
            if not all_data or not all_data.get("metadatas"):
                return []
            doc_map = {}
            for meta in all_data["metadatas"]:
                if meta:
                    fname = meta.get("source_file", "unknown")
                    doc_map[fname] = doc_map.get(fname, 0) + 1
            return [{"filename": k, "chunks": v} for k, v in doc_map.items()]
        except Exception as e:
            logger.warning(f"get_document_list failed: {e}")
            return []

    def delete_document(self, filename: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            existing = self.db.get(where={"source_file": filename})
            if existing and existing.get("ids"):
                ids = existing["ids"]
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    self.db.delete(ids=ids[i:i + batch_size])
                logger.info(f"Deleted '{filename}' ({len(ids)} chunks).")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete '{filename}': {e}")
            return False

    def clear_session(self):
        """Windows-safe session cleanup via collection rotation."""
        try:
            all_data = self.db.get()
            all_ids = all_data.get("ids", [])
            if all_ids:
                batch_size = 100
                for i in range(0, len(all_ids), batch_size):
                    self.db.delete(ids=all_ids[i:i + batch_size])
            logger.info(f"Cleared {len(all_ids)} chunks.")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")

        new_name = f"session_{uuid.uuid4().hex[:8]}"
        self.db = self._connect_chromadb(new_name)
        logger.info(f"New session: '{new_name}'")


# Singleton instance
vector_store_manager = VectorStoreManager()
