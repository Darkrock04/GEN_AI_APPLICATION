import logging
import re
import uuid
import time
import hashlib
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from backend.llm_factory import get_llm
from backend.vector_store import vector_store_manager

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
    planner_llm     = get_llm("planner", max_tokens=512)
    router_llm      = get_llm("router", max_tokens=32)
    worker_general  = get_llm("worker_general", max_tokens=16384)
    worker_coding   = get_llm("worker_coding", max_tokens=16384)
    worker_creative = get_llm("worker_creative", max_tokens=16384)
    validator_llm   = get_llm("validator", max_tokens=16)
    evaluator_llm   = get_llm("evaluator", max_tokens=16384)
    logger.info("All 8 specialized LLMs initialized.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize LLMs: {e}")
    raise


# SAFE LLM CALL WRAPPER
# Simple greeting cache (Feature 15: Performance)
_greeting_cache = {}

def _safe_llm_call(
    prompt_template: str,
    llm,
    variables: dict,
    fallback: str,
    retries: int = 1,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
) -> str:
    """
    Wraps every LLM call with retry + rate-limit handling.
    Supports dynamic temperature, top_p, max_tokens overrides.
    Returns fallback on any failure — never crashes the pipeline.
    """
    # Apply dynamic overrides if provided
    active_llm = llm
    bind_kwargs = {}
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    if top_p is not None:
        bind_kwargs["top_p"] = top_p
    if max_tokens is not None:
        bind_kwargs["max_tokens"] = max_tokens
    if bind_kwargs:
        active_llm = llm.bind(**bind_kwargs)

    for attempt in range(retries + 1):
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            result = (prompt | active_llm).invoke(variables)
            content = result.content if hasattr(result, "content") else str(result)
            if content and content.strip():
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
SAFE_KEYWORDS = frozenset({
    "summarize", "summary", "pdf", "document", "task", "explain",
    "what", "how", "why", "tell", "show", "list", "write", "code",
    "create", "help", "describe", "find", "analyze", "review",
    "generate", "give", "make", "build", "compare", "translate",
    "hello", "hi", "hey", "thanks", "bye", "okay", "yes", "no",
    "my", "i", "name", "am", "is", "are", "you", "who", "your",
    "please", "can", "could", "would", "should", "do", "does",
    "solve", "answer", "calculate", "define", "convert",
    "python", "java", "javascript", "html", "css", "react", "sql",
    "story", "poem", "essay", "blog", "email", "letter", "report",
})
SIMPLE_PATTERNS = frozenset({
    "hello", "hi", "hey", "thanks", "bye", "okay", 
    "my", "name", "is", "who", "what", "are", "you", 
    "good", "morning", "afternoon", "evening", "how", "doing"
})


@_timed_node
def stress_test_node(state: GraphState) -> GraphState:
    request = state["request"].strip()
    words = set(request.lower().split())

    # Stage 1: fast keyword check (covers 95%+ of queries)
    is_safe = bool(words & SAFE_KEYWORDS) or len(request.split()) <= 6

    # Stage 2: LLM only for truly ambiguous queries
    if not is_safe:
        result = _safe_llm_call(
            "Content safety check. Reply ONE word: SAFE or UNSAFE.\n"
            "UNSAFE = ONLY hacking, malware, illegal activity, explicit harm.\n"
            "SAFE = everything else.\nQuery: {request}",
            security_llm, {"request": request}, "SAFE"
        ).upper()
        is_safe = "UNSAFE" not in result

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
        "You are SPARK AI, a helpful conversational assistant.\n\n"
        "--- PAST CONVERSATION HISTORY ---\n{history}\n---------------------------------\n\n"
        "--- CURRENT USER MESSAGE ---\n{request}\n----------------------------\n\n"
        "Reply to the CURRENT USER MESSAGE briefly and naturally in 1-3 sentences. Be friendly. You MUST use the past conversation history if the user refers to past context (e.g., their name).",
        worker_general, 
        {"request": state["request"], "history": state.get("history", "")},
        "Hello! How can I help you today?",
        temperature=state.get("temperature", 0.7),
    )

    return {"final_response": result}


# PLANNER
@_timed_node
def planner_node(state: GraphState) -> GraphState:
    plan = _safe_llm_call(
        "You are a task planner. Given the user's request and conversation history:\n"
        "Break down this request into 1-3 clear steps.\n"
        "If it is just a conversational statement (e.g. 'My name is X', 'Hello'), reply ONLY with: 'Acknowledge and respond naturally.'\n"
        "History:\n{history}\n\nUser request: {request}\n\nPlan:",
        planner_llm,
        {"request": state["request"], "history": state.get("history") or "None"},
        "Respond to the user.",
    )
    return {"plan": plan}


# RETRIEVAL with SOURCE CITATIONS (Feature 7)
@_timed_node
def retrieve_context_node(state: GraphState) -> GraphState:
    # Use dynamic k from user controls (default 10)
    k = state.get("retrieval_k", 10)

    # Get documents with metadata for citations
    results = vector_store_manager.retrieve_with_metadata(state["request"], k=k)

    if not results:
        return {"context": "", "sources": []}

    # Build context string
    context_parts = []
    sources = []
    for i, doc in enumerate(results):
        context_parts.append(doc.page_content)
        source_info = {
            "filename": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", None),
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "relevance_rank": i + 1,
        }
        sources.append(source_info)

    context = "\n\n---\n\n".join(context_parts)
    return {"context": context, "sources": sources}


# ROUTER
@_timed_node
def router_node(state: GraphState) -> GraphState:
    raw = _safe_llm_call(
        "Classify into ONE word: 'coding', 'creative', or 'general'.\n"
        "coding = ONLY for requests explicitly asking to write software code, debug, or build programs. Do NOT use for basic conversation.\n"
        "creative = stories, poems, essays, marketing\n"
        "general = simple chat, greetings, introductions, facts, everything else (e.g. 'my name is rock')\n\n"
        "CRITICAL: If the user is just saying hello, introducing themselves, or asking general questions, classify as general.\n\n"
        "Request:\n{request}\nCategory:",
        router_llm, {"request": state["request"][:2000]}, "general"
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


WORKER_PROMPTS = {
    "general": (
        "You are SPARK AI, a highly capable but conversational assistant.\n\n"
        "Conversation History:\n{history}\n\n"
        "Plan:\n{plan}\n\n"
        "Document Context:\n{context}\n\n"
        "User: {request}\n\n"
        "Previous Feedback: {feedback}\n\n"
        "Instructions:\n"
        "- Respond naturally to the user's input.\n"
        "- If the user is just chatting or greeting you, keep the response conversational and proportional to their input. Do NOT hallucinate unsolicited technical topics, code, or math.\n"
        "- If the user asks a substantive question, answer directly and helpfully.\n"
        "- If document context is provided and relevant to their question, use it to form your answer.\n"
        "Answer:"
    ),
    "coding": (
        "You are SPARK AI, an expert software engineer.\n\n"
        "History:\n{history}\n\nPlan:\n{plan}\n\n"
        "Context:\n{context}\n\n"
        "User: {request}\n\nFeedback: {feedback}\n\n"
        "Instructions:\n"
        "- Write clean, production-ready code in markdown code blocks.\n"
        "- Add comments for non-obvious logic.\n"
        "- Include error handling and edge cases.\n"
        "- For math expressions, wrap inline math in $...$ and block equations in $$...$$.\n\n"
        "Answer:"
    ),
    "creative": (
        "You are SPARK AI, a talented creative writer.\n\n"
        "History:\n{history}\n\nPlan:\n{plan}\n\n"
        "Context:\n{context}\n\n"
        "User: {request}\n\nFeedback: {feedback}\n\n"
        "Instructions:\n"
        "- Write with vivid language and strong structure.\n"
        "- Match the style and tone requested by the user.\n"
        "- Do NOT include unsolicited technical examples or math.\n\n"
        "Answer:"
    ),
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
    prompt = WORKER_PROMPTS.get(task_type, WORKER_PROMPTS["general"])
    variables = _build_worker_context(state)
    draft = _safe_llm_call(
        prompt, llm, variables,
        "I couldn't generate a response. Please try rephrasing.",
        temperature=state.get("temperature"),
        top_p=state.get("top_p"),
        max_tokens=state.get("max_tokens"),
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
        "You are a quality reviewer. Check this draft against the request.\n\n"
        "Request: {request}\n"
        "Context: {context}\n"
        "Draft: {draft}\n\n"
        "Check ALL three criteria:\n"
        "1. RELEVANT — Does it answer the user's question?\n"
        "2. FACTUAL — Does it avoid making up information not in the context?\n"
        "3. COHERENT — Is it well-structured and clear?\n\n"
        "CRITICAL INSTRUCTION: DO NOT output any explanations or reasoning. "
        "Reply with EXACTLY the word 'PASS' if all three pass. "
        "If any fail, reply with 'FAIL: <brief reason>'.",
        validator_llm,
        {
            "request": state["request"][:2000],
            "context": (state.get("context") or "")[:1500],
            "draft": draft[:2000],
        },
        "PASS",
    )

    passed = "PASS" in result.upper() and "FAIL" not in result.upper()
    feedback = "APPROVED" if passed else result

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
        "You are a quality editor. Polish this draft for the user:\n"
        "1. Fix formatting (proper markdown, code blocks, lists).\n"
        "2. Remove any meta-commentary about internal processes.\n"
        "3. Ensure the response matches the conversational tone of the request (e.g. if it's a simple greeting, keep the reply simple; do not add weird examples).\n"
        "4. IF the draft contains math, ensure it uses dollar sign delimiters: inline $x^2$, block $$\\frac{{a}}{{b}}$$. (Do not add math if there is none).\n"
        "\nCRITICAL: Output ONLY the final polished text. Do NOT add any conversational intro (like 'Here is the polished version') or editor notes at the end. Your entire output will be shown directly to the user as the final answer.\n\n"
        "Request: {request}\nDraft:\n{draft}\n\nFinal Polished Text:",
        evaluator_llm,
        {"request": state["request"][:2000], "draft": draft},
        draft,
    )
    return {"final_response": polished}


# ROUTING FUNCTIONS
def route_after_stress_test(state: GraphState) -> str:
    if not state.get("is_safe", True):
        return "end"
    if state.get("is_simple", False):
        return "simple_answer"
    return "planner"


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
workflow.add_edge("planner", "retrieve")
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
    Summarizes older messages to keep context window efficient."""
    if not history:
        return ""

    # Recent messages (last 10) — keep full detail
    recent = history[-10:]

    # Older messages (before last 10) — compress into summary
    older = history[:-10] if len(history) > 10 else []
    older_summary = ""
    if older:
        old_snippets = []
        for msg in older[-20:]:  # Cap at 20 older messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            old_snippets.append(f"{role}: {content}")
        older_summary = "[Earlier conversation summary]\n" + "\n".join(old_snippets) + "\n\n[Recent messages]\n"

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


# PUBLIC ENTRY POINTS
def _build_initial_state(request: str, history: list) -> dict:
    """Build the initial graph state from request."""
    return {
        "request": request,
        "history": _build_history_str(history),
        "draft_generation_count": 0,
        "request_id": uuid.uuid4().hex[:8],
        "temperature": 0.7,      # Default, overridden by router
        "top_p": 0.9,          # Default, overridden by router
        "max_tokens": 8192,      # Hardcoded robust max length
        "retrieval_k": 15,       # Hardcoded comprehensive retrieval context
        "sources": [],
        "node_timings": {},
    }


def process_chat(request: str, history: list = None) -> dict:
    """Main entry point. Returns dict with response, sources, and timings."""
    initial = _build_initial_state(request, history or [])

    start_time = time.time()
    result = graph_app.invoke(initial)
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

    start_time = time.time()
    for chunk in graph_app.stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            # Include timing info in each update
            elapsed_ms = int((time.time() - start_time) * 1000)
            update["_elapsed_ms"] = elapsed_ms
            yield {"node": node_name, "update": update}
