import os
import logging
import re
import uuid
import time
import hashlib
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from backend.llm_factory import get_llm
from backend.vector_store import vector_store_manager
from backend.tools import perform_web_search
from backend.langfuse_prompt_manager import get_managed_prompt, log_validation_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max worker→validation cycles (initial draft + 1 retry)
MAX_DRAFT_ATTEMPTS = 2


# STATE
class GraphState(TypedDict):
    request_id: str
    request: str
    history: str
    is_safe: bool
    is_simple: bool
    needs_web_search: bool
    plan: str
    context: str
    sources: list            # Source citation metadata
    task_type: str
    draft: str
    validation_pass: bool
    feedback: str
    final_response: str
    draft_generation_count: int
    # Dynamic inference parameters (set per-request from UI)
    temperature: float
    top_p: float
    max_tokens: int
    retrieval_k: int
    # Timing
    node_timings: dict       # {node_name: elapsed_ms}


# INITIALIZE SPECIALIZED LLMs (lightweight token limits)
try:
    security_llm    = get_llm("security", max_tokens=64)
    planner_llm     = get_llm("planner", max_tokens=256)
    router_llm      = get_llm("router", max_tokens=16)
    worker_general  = get_llm("worker_general", max_tokens=2048)
    worker_coding   = get_llm("worker_coding", max_tokens=2048)
    worker_creative = get_llm("worker_creative", max_tokens=2048)
    validator_llm   = get_llm("validator", max_tokens=16)
    evaluator_llm   = get_llm("evaluator", max_tokens=1024)
    logger.info("All 8 specialized LLMs initialized.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize LLMs: {e}")
    raise


# SAFE LLM CALL WRAPPER
# Simple greeting cache (Feature 15: Performance)
_greeting_cache = {}

def _safe_llm_call(
    prompt_template: str = "",
    llm = None,
    variables: dict = None,
    fallback: str = "",
    retries: int = 1,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    prompt_name: str = None,
    langfuse_prompt: Any = None,
) -> str:
    """
    Wraps every LLM call with retry + rate-limit handling.
    Supports dynamic temperature, top_p, max_tokens overrides.
    Seamlessly fetches managed prompts from Langfuse and attaches prompt metadata for trace linking.
    Returns fallback on any failure — never crashes the pipeline.
    """
    if variables is None:
        variables = {}

    # If prompt_name is provided, fetch from Langfuse Prompt Registry (or fallback/cache)
    if prompt_name:
        managed_template, lf_obj = get_managed_prompt(prompt_name)
        if managed_template:
            prompt_template = managed_template
        if lf_obj is not None:
            langfuse_prompt = lf_obj

    if not prompt_template:
        logger.error(f"No prompt template found for prompt_name='{prompt_name}'")
        return fallback

    # Apply dynamic overrides if provided
    active_llm = llm
    bind_kwargs = {}
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    if top_p is not None:
        bind_kwargs["top_p"] = top_p
    if max_tokens is not None:
        if "google" in str(type(llm)).lower() or "gemini" in str(type(llm)).lower():
            bind_kwargs["max_output_tokens"] = max_tokens
        else:
            bind_kwargs["max_tokens"] = max_tokens
    if bind_kwargs:
        active_llm = llm.bind(**bind_kwargs)

    for attempt in range(retries + 1):
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            call_config = {}
            if langfuse_prompt is not None:
                call_config["metadata"] = {"langfuse_prompt": langfuse_prompt}

            result = (prompt | active_llm).invoke(variables, config=call_config if call_config else None)
            content = result.content if hasattr(result, "content") else str(result)
            # Gemini 3.x models return content as a list of blocks instead of a string
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts)
            if content and isinstance(content, str) and content.strip():
                return content.strip()
            return fallback
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ("429", "rate", "quota", "too many"))
            if is_rate_limit and attempt < retries:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited (attempt {attempt+1}), waiting {wait}s...")
                time.sleep(wait)
                continue
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            return fallback
    return fallback


def _timed_node(func):
    """Decorator to measure node execution time."""
    def wrapper(state: GraphState) -> GraphState:
        start = time.time()
        result = func(state)
        elapsed_ms = int((time.time() - start) * 1000)
        timings = dict(state.get("node_timings") or {})
        timings[func.__name__.replace("_node", "")] = elapsed_ms
        result["node_timings"] = timings
        return result
    wrapper.__name__ = func.__name__
    return wrapper


# SECURITY GATE
SIMPLE_PATTERNS = frozenset({
    "hello", "hi", "hey", "thanks", "bye", "okay", 
    "good", "morning", "afternoon", "evening",
    "nice", "meet", "how", "are", "you", "name",
    "what's", "up", "doing"
})


@_timed_node
def stress_test_node(state: GraphState) -> GraphState:   
    request = state["request"].strip()
    words = set(request.lower().split())

    # Unconditional LLM security check (Instantaneous on Nvidia NIM)
    safety_decision = _safe_llm_call(
        llm=security_llm,
        variables={"request": request},
        fallback="SAFE",  # Default to safe on error
        prompt_name="security_gate_prompt",
    )
    
    decision_upper = safety_decision.upper()
    is_safe = ("SAFE" in decision_upper) and ("UNSAFE" not in decision_upper)

    # Increase limit to 8 to catch "hello my name is rock" as simple chat.
    is_simple = bool(words & SIMPLE_PATTERNS) and len(request.split()) <= 8

    return {
        "is_safe": is_safe,
        "is_simple": is_simple,
        "final_response": "" if is_safe else "I'm sorry, I can't help with that request.",
        "draft_generation_count": 0,
    }


# SIMPLE GREETING
@_timed_node
def simple_answer_node(state: GraphState) -> GraphState:
    result = _safe_llm_call(
        llm=worker_general, 
        variables={"request": state["request"], "history": state.get("history", "")},
        fallback="I'm sorry, my language model is currently unreachable due to an API error. Please try again.",
        temperature=state.get("temperature", 0.7),
        prompt_name="simple_answer_prompt",
    )

    return {"final_response": result}


# PLANNER
@_timed_node
def planner_node(state: GraphState) -> GraphState:
    response = _safe_llm_call(
        llm=planner_llm,
        variables={"request": state["request"], "history": state.get("history") or "None"},
        fallback="Respond to the user.",
        prompt_name="planner_prompt",
    )
    
    needs_web = "[NEEDS_WEB_SEARCH]" in response
    clean_plan = response.replace("[NEEDS_WEB_SEARCH]", "").strip()
    
    return {"plan": clean_plan, "needs_web_search": needs_web}


# WEB SEARCH
@_timed_node
def web_search_node(state: GraphState) -> GraphState:
    results = perform_web_search(state["request"])
    
    sources = list(state.get("sources", []))
    if results and "No recent internet information" not in results:
        sources.append({
            "filename": "SearXNG Web Search",
            "page": None,
            "content_preview": "Live web results retrieved from search engine.",
            "relevance_rank": 0
        })
        
    return {"context": results, "sources": sources}


# RETRIEVAL with SOURCE CITATIONS (Feature 7)
@_timed_node
def retrieve_context_node(state: GraphState) -> GraphState:
    # Use dynamic k from user controls (default 10)
    k = state.get("retrieval_k", 10)

    # existing context from web_search_node
    existing_context = state.get("context", "")
    sources = list(state.get("sources", []))

    # Get documents with metadata for citations
    results = vector_store_manager.retrieve_with_metadata(state["request"], k=k)

    if not results:
        return {"context": existing_context, "sources": sources}

    # Build context string
    context_parts = [existing_context] if existing_context else []
    
    for i, doc in enumerate(results):
        context_parts.append(doc.page_content)
        source_info = {
            "filename": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", None),
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "relevance_rank": i + 1,
        }
        sources.append(source_info)

    context = "\n\n---\n\n".join(filter(None, context_parts))
    return {"context": context, "sources": sources}


# ROUTER
@_timed_node
def router_node(state: GraphState) -> GraphState:
    raw = _safe_llm_call(
        llm=router_llm,
        variables={"request": state["request"][:2000]},
        fallback="general",
        prompt_name="router_prompt",
    ).lower()

    task_type = "general"
    for valid in ("coding", "creative"):
        if valid in raw:
            task_type = valid
            break

    # Automate temperature and top_p based on task_type
    temp = 0.7
    top_p = 0.9
    if task_type == "coding":
        temp = 0.1
        top_p = 0.95
    elif task_type == "creative":
        temp = 1.1
        top_p = 0.9

    return {"task_type": task_type, "temperature": temp, "top_p": top_p}


# SPECIALIZED WORKERS
def _build_worker_context(state: GraphState) -> dict:
    """Build clean variable dict for worker prompts."""
    ctx = state.get("context") or ""
    history = state.get("history") or ""

    return {
        "history": history if history.strip() else "(No previous conversation)",
        "plan": state.get("plan") or "Answer the request directly.",
        "context": ctx if ctx.strip() else "(No documents uploaded — answering from knowledge)",
        "request": state["request"],
        "feedback": state.get("feedback") or "None",
    }


WORKER_PROMPT_NAMES = {
    "general": "worker_general_prompt",
    "coding": "worker_coding_prompt",
    "creative": "worker_creative_prompt",
}

WORKER_LLMS = {
    "general": worker_general,
    "coding": worker_coding,
    "creative": worker_creative,
}


@_timed_node
def worker_agent_node(state: GraphState) -> GraphState:
    task_type = state.get("task_type", "general")
    llm = WORKER_LLMS.get(task_type, worker_general)
    prompt_name = WORKER_PROMPT_NAMES.get(task_type, "worker_general_prompt")
    variables = _build_worker_context(state)
    draft = _safe_llm_call(
        llm=llm,
        variables=variables,
        fallback="I couldn't generate a response. Please try rephrasing.",
        temperature=state.get("temperature"),
        top_p=state.get("top_p"),
        max_tokens=state.get("max_tokens"),
        prompt_name=prompt_name,
    )
    next_count = state.get("draft_generation_count", 0) + 1
    return {"draft": draft, "draft_generation_count": next_count}


# CONSOLIDATED VALIDATION (single LLM call instead of 3)
@_timed_node
def validation_node(state: GraphState) -> GraphState:
    draft = state.get("draft", "")
    if not draft.strip():
        return {"validation_pass": True, "feedback": "APPROVED"}

    result = _safe_llm_call(
        llm=validator_llm,
        variables={
            "request": state["request"][:2000],
            "context": (state.get("context") or "")[:1500],
            "draft": draft[:2000],
        },
        fallback="PASS",
        prompt_name="validator_prompt",
    )

    passed = "PASS" in result.upper() and "FAIL" not in result.upper()
    feedback = "APPROVED" if passed else result

    # Log validation quality score to Langfuse
    try:
        log_validation_score(
            passed=passed,
            feedback=feedback,
            session_id=state.get("request_id"),
        )
    except Exception as e:
        logger.warning(f"Failed to log validation score to Langfuse: {e}")

    return {"validation_pass": passed, "feedback": feedback}


# EVALUATOR (skipped for short responses)
@_timed_node
def evaluation_node(state: GraphState) -> GraphState:
    draft = state.get("draft", "")
    if not draft.strip():
        return {"final_response": "I couldn't generate a response. Please try again."}

    # Skip polishing for short/simple responses — saves an LLM call
    if len(draft) < 500:
        return {"final_response": draft}

    polished = _safe_llm_call(
        llm=evaluator_llm,
        variables={"request": state["request"][:2000], "draft": draft},
        fallback=draft,
        prompt_name="evaluator_prompt",
    )
    return {"final_response": polished}


# ROUTING FUNCTIONS
def route_after_stress_test(state: GraphState) -> str:
    if not state.get("is_safe", True):
        return "end"
    if state.get("is_simple", False):
        return "simple_answer"
    return "planner"


def route_after_planner(state: GraphState) -> str:
    if state.get("needs_web_search", False):
        return "web_search"
    return "retrieve"


def route_after_validation(state: GraphState) -> str:
    if state.get("validation_pass", True):
        return "evaluation"
    if state.get("draft_generation_count", 0) >= MAX_DRAFT_ATTEMPTS:
        return "evaluation"
    return "worker"


# BUILD GRAPH
workflow = StateGraph(GraphState)
workflow.add_node("stress_test", stress_test_node)
workflow.add_node("simple_answer", simple_answer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("retrieve", retrieve_context_node)
workflow.add_node("router", router_node)
workflow.add_node("worker", worker_agent_node)
workflow.add_node("validation", validation_node)
workflow.add_node("evaluation", evaluation_node)

workflow.set_entry_point("stress_test")
workflow.add_conditional_edges(
    "stress_test", route_after_stress_test,
    {"end": END, "simple_answer": "simple_answer", "planner": "planner"},
)
workflow.add_edge("simple_answer", END)
workflow.add_conditional_edges("planner", route_after_planner, {"web_search": "web_search", "retrieve": "retrieve"})
workflow.add_edge("web_search", "retrieve")
workflow.add_edge("retrieve", "router")
workflow.add_edge("router", "worker")
workflow.add_edge("worker", "validation")
workflow.add_conditional_edges(
    "validation", route_after_validation,
    {"evaluation": "evaluation", "worker": "worker"},
)
workflow.add_edge("evaluation", END)

graph_app = workflow.compile()


# CONVERSATION MEMORY (Feature 5)
def _build_history_str(history: list) -> str:
    """Convert history list to a compact string for LLM context.
    Uses an LLM to actively summarize older messages to maintain context efficiently."""
    if not history:
        return ""

    # Recent messages (last 6) — keep full detail
    recent = history[-6:]

    # Older messages (before last 6) — summarize using LLM
    older = history[:-6] if len(history) > 6 else []
    older_summary = ""
    
    if older:
        old_snippets = []
        for msg in older:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            old_snippets.append(f"{role}: {content}")
            
        raw_older = "\n".join(old_snippets)
        
        # Use fast router LLM to summarize (override max_tokens since router defaults to 16)
        summary = _safe_llm_call(
            llm=router_llm, 
            variables={"raw_older": raw_older[:4000]}, # Cap length to avoid context limits
            fallback="Past conversation summary unavailable.",
            max_tokens=256,
            prompt_name="history_summarizer_prompt",
        )
        older_summary = f"[Earlier conversation summary]\n{summary}\n\n[Recent messages]\n"

    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")

    if not lines:
        return "No previous history."
        
    return older_summary + "\n".join(lines)


# LANGFUSE OBSERVABILITY & TRACING
def _get_langfuse_callback(session_id: str = None):
    """Initializes Langfuse CallbackHandler if credentials exist in environment."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler()
        except Exception as e:
            logger.warning(f"Could not initialize Langfuse callback: {e}")
            return None
    return None


# PUBLIC ENTRY POINTS
def _build_initial_state(request: str, history: list) -> dict:
    """Build the initial graph state from request."""
    return {
        "request": request,
        "history": _build_history_str(history),
        "draft_generation_count": 0,
        "needs_web_search": False,
        "request_id": uuid.uuid4().hex[:8],
        "temperature": 0.7,      # Default, overridden by router
        "top_p": 0.9,          # Default, overridden by router
        "max_tokens": 2048,      # Lowered for optimal Multi-Cloud performance
        "retrieval_k": 15,       # Hardcoded comprehensive retrieval context
        "sources": [],
        "node_timings": {},
    }


def process_chat(request: str, history: list = None) -> dict:
    """Main entry point. Returns dict with response, sources, and timings."""
    initial = _build_initial_state(request, history or [])

    callbacks = []
    lf_cb = _get_langfuse_callback(session_id=initial.get("request_id"))
    if lf_cb:
        callbacks.append(lf_cb)
    config = {
        "callbacks": callbacks,
        "metadata": {
            "langfuse_session_id": initial.get("request_id"),
            "langfuse_trace_name": "SPARK-AI-Workflow",
            "langfuse_tags": ["production", "agentic-rag"],
        },
    } if callbacks else {}

    start_time = time.time()
    result = graph_app.invoke(initial, config=config)
    total_ms = int((time.time() - start_time) * 1000)

    return {
        "response": result.get("final_response") or "I couldn't generate a response. Please try again.",
        "sources": result.get("sources") or [],
        "node_timings": result.get("node_timings") or {},
        "total_ms": total_ms,
    }


def stream_graph_updates(request: str, history: list = None):
    """Yields merged state fragments after each node completes (for SSE/NDJSON)."""
    initial = _build_initial_state(request, history or [])

    callbacks = []
    lf_cb = _get_langfuse_callback(session_id=initial.get("request_id"))
    if lf_cb:
        callbacks.append(lf_cb)
    config = {
        "callbacks": callbacks,
        "metadata": {
            "langfuse_session_id": initial.get("request_id"),
            "langfuse_trace_name": "SPARK-AI-Workflow",
            "langfuse_tags": ["production", "agentic-rag"],
        },
    } if callbacks else {}

    start_time = time.time()
    for chunk in graph_app.stream(initial, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            # Include timing info in each update
            elapsed_ms = int((time.time() - start_time) * 1000)
            update["_elapsed_ms"] = elapsed_ms
            yield {"node": node_name, "update": update}
