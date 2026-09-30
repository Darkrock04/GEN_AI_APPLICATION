import os
import logging
import re
import uuid
import time
import hashlib
from datetime import datetime
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from backend.llm_factory import get_llm
from backend.vector_store import vector_store_manager
from backend.tools import perform_web_search, extract_search_query, clean_search_query
from backend.langfuse_prompt_manager import get_managed_prompt, log_validation_score
from backend.laya_client import laya_client

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
    web_search_completed: bool
    retrieval_completed: bool
    doc_relevance: str       # "relevant" | "irrelevant" | "none"
    search_query: str
    current_date: str
    plan: str
    context: str
    sources: list            # Source citation metadata
    task_type: str
    draft: str
    validation_pass: bool
    feedback: str
    final_response: str
    draft_generation_count: int
    search_retry_count: int
    # Dynamic inference parameters (set per-request from UI)
    temperature: float
    top_p: float
    max_tokens: int
    retrieval_k: int
    # Timing & Dual-Brain Telemetry
    node_timings: dict       # {node_name: elapsed_ms}
    system_one_decisions: dict # {node_name: {decision_metadata}}


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
    fallback_llm: Any = None,
) -> str:
    """
    Wraps every LLM call with retry + rate-limit handling and cross-cloud failover.
    Supports dynamic temperature, top_p, max_tokens overrides.
    Seamlessly fetches managed prompts from Langfuse and attaches prompt metadata for trace linking.
    Falls back to cross-cloud fallback_llm if primary fails, then returns fallback string — never crashes the pipeline.
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
            if attempt < retries:
                wait = 2 ** (attempt + 1) if is_rate_limit else 1
                logger.warning(f"LLM call failed (attempt {attempt+1}/{retries+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            logger.warning(f"Primary LLM call failed after {retries+1} attempts: {e}")
            if fallback_llm is not None:
                try:
                    logger.info(f"Invoking cross-cloud fallback LLM for prompt '{prompt_name}'...")
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    call_config = {}
                    if langfuse_prompt is not None:
                        call_config["metadata"] = {"langfuse_prompt": langfuse_prompt}
                    fb_result = (prompt | fallback_llm).invoke(variables, config=call_config if call_config else None)
                    fb_content = fb_result.content if hasattr(fb_result, "content") else str(fb_result)
                    if isinstance(fb_content, list):
                        parts = [b if isinstance(b, str) else b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in fb_content]
                        fb_content = "\n".join(parts)
                    if fb_content and isinstance(fb_content, str) and fb_content.strip():
                        logger.info(f"Cross-cloud fallback LLM succeeded for prompt '{prompt_name}'.")
                        return fb_content.strip()
                except Exception as fb_err:
                    logger.warning(f"Cross-cloud fallback LLM also failed: {fb_err}")
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


# DETERMINISTIC KNOWLEDGE CUTOFF & TEMPORAL AUDITOR
def _audit_draft_for_cutoff(draft: str, user_request: str = "", current_date: str = "") -> dict:
    """
    Deterministically audits a draft response for knowledge cutoff disclaimers,
    apologies, or outdated hallucinated events when the user asked for current/live info.
    Runs in 0ms with zero reliance on external LLM availability.
    Returns:
        {"has_cutoff": bool, "reason": str}
    """
    if not draft:
        return {"has_cutoff": False, "reason": ""}

    draft_lower = draft.lower()
    req_lower = (user_request or "").lower()
    clean_d = draft.strip()

    # 0. Check for unexecuted tool calls, JSON payloads, or search commands
    if (clean_d.startswith("{") and any(k in clean_d.lower() for k in ("query", "top_n", "source", "search"))) or "[needs_web_search" in draft_lower:
        return {
            "has_cutoff": True,
            "reason": "Draft is an unexecuted search tool call or command instead of an answer.",
        }

    # 1. Explicit cutoff indicators / apologies
    cutoff_phrases = (
        "[needs_web_search",
        "as of my knowledge cutoff",
        "knowledge cutoff of",
        "knowledge cutoff is",
        "my knowledge cutoff",
        "cutoff of 2024",
        "up to the middle of 2024",
        "up to mid-2024",
        "as of mid-2024",
        "mid-2024",
        "latest period for which i have reliable",
        "i do not have real-time",
        "i don't have real-time",
        "no real-time data",
        "cannot provide real-time",
        "do not have access to real-time",
        "don't have access to real-time",
        "do not have access to current",
        "don't have access to current",
        "as of my last update",
        "training data only goes up to",
        "training data cuts off",
        "training data cut off",
        "i cannot browse the live web",
        "cannot browse the live web",
        "as of september 2024",
        "as of late 2024",
        "september 2024 vibe",
        "june 2024 vibe",
    )
    for phrase in cutoff_phrases:
        if phrase in draft_lower:
            return {
                "has_cutoff": True,
                "reason": f"Draft contains explicit cutoff disclaimer or apology: '{phrase}'",
            }

    # 2. Temporal hallucination check:
    # If the user asked for "today", "live", "current", "this month", "latest" news/models/events,
    # but the draft provides outdated 2024 claims without referencing the current year (2026)
    temporal_signals = ("today", "live", "current", "latest", "recent", "this month", "released this month", "happened today")
    is_temporal_request = any(sig in req_lower for sig in temporal_signals)

    if is_temporal_request:
        outdated_year_patterns = [
            r"\bin\s+2024\b",
            r"\bas\s+of\s+2024\b",
            r"\bjune\s+2024\b",
            r"\bmid-2024\b",
            r"\bearly\s+2024\b",
            r"\blate\s+2024\b",
            r"\bthroughout\s+2024\b",
        ]
        if "2026" not in draft_lower and any(re.search(p, draft_lower) for p in outdated_year_patterns):
            return {
                "has_cutoff": True,
                "reason": "Draft references 2024 as current context for a live/present-day request.",
            }

    return {"has_cutoff": False, "reason": ""}


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
    s1_decisions = dict(state.get("system_one_decisions") or {})

    # 1. System-2 High-Precision Guardrail: Nemotron evaluates safety (zero false-positives on code)
    safety_decision = _safe_llm_call(
        llm=security_llm,
        variables={"request": request},
        fallback="SAFE",  # Default to safe on error
        prompt_name="security_gate_prompt",
    )
    decision_upper = safety_decision.upper()
    is_safe = ("SAFE" in decision_upper) and ("UNSAFE" not in decision_upper)

    # 2. System-1 Reflex: Non-autoregressive multilingual greeting & intent triage via Laya
    laya_greet = laya_client.check_greeting_and_intent(request)
    if laya_greet is not None:
        is_simple = laya_greet["is_greeting"] and len(request.split()) <= 12
        s1_decisions["stress_test"] = {
            "engine": "laya_system_one",
            "model": laya_greet.get("model_used", "english"),
            "intent": laya_greet.get("intent"),
            "greeting": is_simple,
            "greeting_prob": laya_greet.get("greeting_probability", 0.0),
            "latency_ms": laya_greet.get("latency_ms", 0),
        }
        logger.info(
            f"[Dual-Brain System-1] Laya greeting triage: is_greeting={is_simple} "
            f"({laya_greet.get('latency_ms')}ms, model={laya_greet.get('model_used')})"
        )
    else:
        # Fallback to keyword patterns
        is_simple = bool(words & SIMPLE_PATTERNS) and len(request.split()) <= 8
        s1_decisions["stress_test"] = {
            "engine": "system_two_fallback",
            "greeting": is_simple,
        }

    return {
        "is_safe": is_safe,
        "is_simple": is_simple,
        "final_response": "" if is_safe else "I'm sorry, I can't help with that request.",
        "draft_generation_count": 0,
        "system_one_decisions": s1_decisions,
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
    current_date = state.get("current_date") or datetime.now().strftime("%B %d, %Y")
    response = _safe_llm_call(
        llm=planner_llm,
        variables={
            "request": state["request"],
            "history": state.get("history") or "None",
            "current_date": current_date,
        },
        fallback="Respond to the user.",
        prompt_name="planner_prompt",
        fallback_llm=router_llm,
    )
    
    needs_web = "[NEEDS_WEB_SEARCH]" in response
    clean_plan = response.replace("[NEEDS_WEB_SEARCH]", "").strip()
    
    # Deterministic safety net: If query explicitly asks for real-time, current, or news data, force web search
    req_lower = state["request"].lower()
    search_keywords = (
        "latest", "news", "today", "current", "weather",
        "stock price", "price of", "recent", "who won",
        "what happened", "release date", "search the web",
        "browse the web", "search online", "this month", "live",
        "2026", "yesterday", "right now", "happening", "released"
    )
    if any(k in req_lower for k in search_keywords):
        needs_web = True
    
    # Extract or generate a clean, date-anchored search query
    clean_q = extract_search_query(response, default=state["request"])
    
    return {"plan": clean_plan, "needs_web_search": needs_web, "search_query": clean_q}


# WEB SEARCH
@_timed_node
def web_search_node(state: GraphState) -> GraphState:
    query_to_search = state.get("search_query") or state["request"]
    results = perform_web_search(query_to_search)
    
    sources = list(state.get("sources", []))
    if results and "No recent internet information" not in results:
        if not any(s.get("filename") == "SearXNG Web Search" for s in sources):
            sources.append({
                "filename": "SearXNG Web Search",
                "page": None,
                "content_preview": "Live web results retrieved from search engine.",
                "relevance_rank": 0
            })
        
    # Merge with any existing context from prior steps
    existing_context = state.get("context", "")
    combined_context = f"{existing_context}\n\n{results}".strip() if existing_context else results
    return {
        "context": combined_context,
        "sources": sources,
        "web_search_completed": True,
    }


# RETRIEVAL with SOURCE CITATIONS (Feature 7)
@_timed_node
def retrieve_context_node(state: GraphState) -> GraphState:
    # Use dynamic k from user controls (default 10)
    k = state.get("retrieval_k", 10)

    # existing context from web_search_node (if ran prior)
    existing_context = state.get("context", "")
    sources = list(state.get("sources", []))

    # Get documents with metadata for citations
    results = vector_store_manager.retrieve_with_metadata(state["request"], k=k)

    if not results:
        return {"context": existing_context, "sources": sources, "retrieval_completed": True}

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
    return {"context": context, "sources": sources, "retrieval_completed": True}


# DOCUMENT RELEVANCE GRADER (Corrective RAG / CRAG)
@_timed_node
def grade_documents_node(state: GraphState) -> GraphState:
    context = state.get("context", "").strip()
    sources = state.get("sources", [])
    s1_decisions = dict(state.get("system_one_decisions") or {})

    # If no documents are uploaded or context is empty
    if not context or not sources or context == "(No documents uploaded — answering from knowledge)":
        return {"doc_relevance": "irrelevant"}

    # If sources already contain SearXNG web search, skip grading
    if any(s.get("filename") == "SearXNG Web Search" for s in sources):
        return {"doc_relevance": "relevant"}

    # 1. System-1 Reflex: Non-autoregressive relevance scoring via Laya
    laya_grade = laya_client.grade_document_relevance(
        query=state["request"][:800],
        doc_chunk=context[:1500]
    )
    if laya_grade is not None:
        is_relevant = laya_grade["is_relevant"]
        s1_decisions["grade_documents"] = {
            "engine": "laya_system_one",
            "choice": laya_grade.get("choice"),
            "relevance_score": laya_grade.get("relevance_score", 0.0),
            "relevant": is_relevant,
            "latency_ms": laya_grade.get("latency_ms", 0),
        }
        logger.info(
            f"[Dual-Brain System-1] Laya CRAG relevance: choice='{laya_grade.get('choice')}', "
            f"score={laya_grade.get('relevance_score'):.2f} ({laya_grade.get('latency_ms')}ms)"
        )
    else:
        # 2. System-2 Fallback: Gemini 3.1 Flash-Lite
        decision = _safe_llm_call(
            llm=validator_llm,
            variables={
                "request": state["request"][:1000],
                "context": context[:1500],
            },
            fallback="RELEVANT",
            prompt_name="document_grader_prompt",
            fallback_llm=router_llm,
        )
        is_relevant = "RELEVANT" in decision.upper() and "IRRELEVANT" not in decision.upper()
        s1_decisions["grade_documents"] = {
            "engine": "system_two_fallback",
            "relevant": is_relevant,
        }

    return {
        "doc_relevance": "relevant" if is_relevant else "irrelevant",
        "system_one_decisions": s1_decisions,
    }


# ROUTER
@_timed_node
def router_node(state: GraphState) -> GraphState:
    task_type = None
    s1_decisions = dict(state.get("system_one_decisions") or {})

    # 1. System-1 Reflex: Non-autoregressive specialized worker routing via Laya
    laya_route = laya_client.route_task(state["request"])
    if laya_route is not None:
        candidate_choice = laya_route.get("task_type", "general")
        if candidate_choice in ("coding", "creative", "general"):
            task_type = candidate_choice
            s1_decisions["router"] = {
                "engine": "laya_system_one",
                "task_type": task_type,
                "confidence": laya_route.get("confidence", 0.0),
                "probabilities": laya_route.get("probabilities", {}),
                "latency_ms": laya_route.get("latency_ms", 0),
                "model": laya_route.get("model_used", "english"),
            }
            logger.info(
                f"[Dual-Brain System-1] Laya router: choice='{task_type}' "
                f"(conf: {laya_route.get('confidence', 0):.2f}, {laya_route.get('latency_ms')}ms)"
            )

    # 2. System-2 Fallback: Ollama Cloud Gemma-4 31B
    if not task_type:
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
        s1_decisions["router"] = {
            "engine": "system_two_fallback",
            "task_type": task_type,
        }

    # CRITICAL ROUTER OVERRIDE:
    # If task_type was classified as 'creative' but the query is clearly factual/news/live/informational,
    # prevent worker_creative from inventing fictional events!
    if task_type == "creative":
        req_lower = state["request"].lower()
        factual_keywords = (
            "news", "today", "live", "current", "latest", "recent",
            "what is", "release", "released", "update", "happened",
            "price", "weather", "who is", "when did", "explain", "model"
        )
        creative_explicit_keywords = ("story", "poem", "joke", "tale", "fiction", "song", "essay", "script", "creative writing", "rhyme")
        is_factual = any(k in req_lower for k in factual_keywords)
        is_explicit_creative = any(k in req_lower for k in creative_explicit_keywords)

        if is_factual and not is_explicit_creative:
            logger.info(f"Router Override: Query contains factual/news signals ('{state['request'][:50]}...'). Reassigning 'creative' -> 'general'.")
            task_type = "general"
        elif s1_decisions.get("router", {}).get("confidence", 1.0) < 0.60 and not is_explicit_creative:
            logger.info(f"Router Override: Low confidence ({s1_decisions.get('router', {}).get('confidence', 0):.2f}) for creative without explicit story cues. Reassigning to 'general'.")
            task_type = "general"

    # Guardrail for 'coding':
    # If task_type was classified as 'coding' but the user did NOT ask for code/programming/scripts,
    # prevent worker_coding from generating unprompted Python scripts!
    if task_type == "coding":
        req_lower = state["request"].lower()
        code_explicit_keywords = (
            "code", "python", "script", "program", "function", "debug", "sql", "html",
            "css", "javascript", "typescript", "c++", "java", "regex", "algorithm",
            "implement", "write a", "syntax", "compile", "error in", "traceback",
            "api call", "endpoint", "class", "method", "bug in", "fix my"
        )
        has_code_intent = any(k in req_lower for k in code_explicit_keywords) or ("```" in state["request"])
        
        factual_indicators = ("what is", "what are", "what r", "news", "today", "released", "came out", "explain", "overview", "compare", "summary", "which")
        is_factual_question = any(k in req_lower for k in factual_indicators)

        if not has_code_intent or (is_factual_question and not any(k in req_lower for k in ("write", "create", "implement", "build"))):
            logger.info(f"Router Override: Query is factual inquiry ('{state['request'][:50]}...'), not a coding request. Reassigning 'coding' -> 'general'.")
            task_type = "general"

    # Automate temperature and top_p based on task_type
    temp = 0.7
    top_p = 0.9
    if task_type == "coding":
        temp = 0.1
        top_p = 0.95
    elif task_type == "creative":
        temp = 1.1
        top_p = 0.9

    return {
        "task_type": task_type,
        "temperature": temp,
        "top_p": top_p,
        "system_one_decisions": s1_decisions,
    }


# SPECIALIZED WORKERS
def _build_worker_context(state: GraphState) -> dict:
    """Build clean variable dict for worker prompts."""
    ctx = state.get("context") or ""
    history = state.get("history") or ""
    current_date = state.get("current_date") or datetime.now().strftime("%B %d, %Y")

    return {
        "current_date": current_date,
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
        fallback_llm=worker_general if llm != worker_general else None,
    )
    next_count = state.get("draft_generation_count", 0) + 1
    return {"draft": draft, "draft_generation_count": next_count}


# CONSOLIDATED VALIDATION (Universal Verification + Deep LLM Grounding)
@_timed_node
def validation_node(state: GraphState) -> GraphState:
    draft = state.get("draft", "")
    if not draft.strip():
        return {"validation_pass": True, "feedback": "APPROVED"}

    context = (state.get("context") or "").strip()
    s1_decisions = dict(state.get("system_one_decisions") or {})
    current_date = state.get("current_date") or datetime.now().strftime("%B %d, %Y")

    # Step 0: Deterministic Code-Level Cutoff & Temporal Auditor (0ms, 100% resilient to LLM 503s)
    audit = _audit_draft_for_cutoff(draft, user_request=state["request"], current_date=current_date)
    if audit["has_cutoff"]:
        logger.warning(f"[Validation Node] Deterministic audit caught cutoff: {audit['reason']}")
        return {
            "validation_pass": False,
            "feedback": f"FAIL: CUTOFF_DETECTED ({audit['reason']})",
            "system_one_decisions": s1_decisions,
        }

    # 1. System-1 Universal Verification: Fast non-autoregressive hallucination check
    if context and context != "(No documents uploaded — answering from knowledge)":
        laya_verif = laya_client.verify_claim_faithfulness(context=context, claim=draft)
        if laya_verif is not None:
            s1_decisions["validation"] = {
                "engine": "laya_universal_verification",
                "verdict": laya_verif.get("verdict"),
                "is_faithful": laya_verif.get("is_faithful"),
                "confidence": laya_verif.get("confidence", 0.0),
                "latency_ms": laya_verif.get("latency_ms", 0),
            }
            # If high-confidence hallucination detected by System-1 (>= 85%)
            if not laya_verif.get("is_faithful", True) and laya_verif.get("confidence", 0.0) >= 0.85:
                logger.info(
                    f"[Dual-Brain System-1] Universal Verification caught high-confidence hallucination: "
                    f"{laya_verif.get('confidence', 0.0):.2f} ({laya_verif.get('latency_ms')}ms)"
                )
                return {
                    "validation_pass": False,
                    "feedback": "Universal Verification: Factual contradiction or hallucination detected against source documents.",
                    "system_one_decisions": s1_decisions,
                }
            else:
                logger.info(
                    f"[Dual-Brain System-1] Universal Verification: Draft is faithful "
                    f"({laya_verif.get('confidence', 0.0):.2f}, {laya_verif.get('latency_ms')}ms)"
                )

    # 2. System-2 Deep Grounding & Cutoff Detection (Gemini 3.1 Flash-Lite with multi-cloud fallback)
    result = _safe_llm_call(
        llm=validator_llm,
        variables={
            "request": state["request"][:2000],
            "context": context[:1500],
            "draft": draft[:2000],
            "current_date": current_date,
        },
        fallback="PASS",
        prompt_name="validator_prompt",
        fallback_llm=router_llm,
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

    return {
        "validation_pass": passed,
        "feedback": feedback,
        "system_one_decisions": s1_decisions,
    }


# EVALUATOR (skipped for short responses)
@_timed_node
def evaluation_node(state: GraphState) -> GraphState:
    draft = state.get("draft", "")
    if not draft.strip():
        return {"final_response": "I couldn't generate a response. Please try again."}

    clean_d = draft.strip()
    # Absolute safety filter: Never let raw JSON tool calls, queries, or bracket commands leak to the user
    if (clean_d.startswith("{") and any(k in clean_d.lower() for k in ("query", "top_n", "source", "search"))) or "[needs_web_search" in clean_d.lower():
        logger.warning("[Evaluator] Caught unexecuted tool call leaking to user. Replacing with natural response.")
        return {
            "final_response": f"I checked for breaking updates on '{state['request']}' as of {state.get('current_date')}, but live feeds are currently updating. Please try asking again in a moment."
        }

    # Skip polishing for short/simple responses — saves an LLM call
    if len(draft) < 500:
        return {"final_response": draft}

    polished = _safe_llm_call(
        llm=evaluator_llm,
        variables={"request": state["request"][:2000], "draft": draft},
        fallback=draft,
        prompt_name="evaluator_prompt",
        fallback_llm=worker_general,
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


def route_after_web_search(state: GraphState) -> str:
    # If triggered from worker self-correction loop, route directly back to worker
    if state.get("draft_generation_count", 0) > 0:
        return "worker"
    # If triggered before retrieval, route to retrieve
    if not state.get("retrieval_completed", False):
        return "retrieve"
    return "router"


def route_after_grade_documents(state: GraphState) -> str:
    # If documents are irrelevant/missing and web search hasn't run yet, trigger web search
    if state.get("doc_relevance") == "irrelevant" and not state.get("web_search_completed"):
        logger.info("CRAG: Retrieved context is irrelevant/missing. Routing to web_search for grounding.")
        return "web_search"
    return "router"


def route_after_worker(state: GraphState) -> str:
    draft = state.get("draft", "")
    current_date = state.get("current_date") or datetime.now().strftime("%B %d, %Y")

    # Check if worker triggered web search or admitted cutoff or hallucinated old years
    audit = _audit_draft_for_cutoff(draft, user_request=state["request"], current_date=current_date)
    retry_count = state.get("search_retry_count", 0)
    if audit["has_cutoff"] and retry_count < 1:
        extracted_q = extract_search_query(draft, state["request"])
        logger.info(f"Self-Correction: Worker hit cutoff/knowledge gap ({audit['reason']}). Triggering web search retry #{retry_count+1} with query: '{extracted_q}'")
        state["search_query"] = extracted_q
        state["search_retry_count"] = retry_count + 1
        return "web_search"

    return "validation"


def route_after_validation(state: GraphState) -> str:
    feedback = state.get("feedback", "")
    retry_count = state.get("search_retry_count", 0)
    # If validator caught a cutoff disclaimer and search hasn't run retry yet
    if "CUTOFF_DETECTED" in feedback and retry_count < 1:
        extracted_q = extract_search_query(feedback, state["request"])
        logger.info(f"Validator detected knowledge cutoff ({feedback}). Triggering web search retry #{retry_count+1} with query: '{extracted_q}'.")
        state["search_query"] = extracted_q
        state["search_retry_count"] = retry_count + 1
        return "web_search"

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
workflow.add_node("grade_documents", grade_documents_node)
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
workflow.add_conditional_edges("web_search", route_after_web_search, {"retrieve": "retrieve", "router": "router", "worker": "worker"})

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grade_documents, {"web_search": "web_search", "router": "router"})

workflow.add_edge("router", "worker")

workflow.add_conditional_edges("worker", route_after_worker, {"web_search": "web_search", "validation": "validation"})

workflow.add_conditional_edges(
    "validation", route_after_validation,
    {"evaluation": "evaluation", "worker": "worker", "web_search": "web_search"},
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
        "search_retry_count": 0,
        "needs_web_search": False,
        "web_search_completed": False,
        "retrieval_completed": False,
        "doc_relevance": "none",
        "search_query": "",
        "current_date": datetime.now().strftime("%B %d, %Y"),
        "request_id": uuid.uuid4().hex[:8],
        "temperature": 0.7,      # Default, overridden by router
        "top_p": 0.9,          # Default, overridden by router
        "max_tokens": 2048,      # Lowered for optimal Multi-Cloud performance
        "retrieval_k": 15,       # Hardcoded comprehensive retrieval context
        "sources": [],
        "node_timings": {},
        "system_one_decisions": {},
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
        "system_one_decisions": result.get("system_one_decisions") or {},
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
