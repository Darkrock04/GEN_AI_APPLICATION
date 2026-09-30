import os
import logging
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)

# DEFAULT PROMPT REGISTRY (Standard Langfuse {{variable}} format)
DEFAULT_PROMPTS = {
    "security_gate_prompt": (
        "You are an AI safety filter. Evaluate the following user request:\n"
        "\"{{request}}\"\n"
        "If the request is safe, reply ONLY with \"SAFE\".\n"
        "If the request contains harmful, illegal, or unethical content, reply ONLY with \"UNSAFE\".\n"
        "Decision:"
    ),
    "simple_answer_prompt": (
        "You are SPARK AI, a helpful conversational assistant.\n\n"
        "--- PAST CONVERSATION HISTORY ---\n{{history}}\n---------------------------------\n\n"
        "--- CURRENT USER MESSAGE ---\n{{request}}\n----------------------------\n\n"
        "Reply to the CURRENT USER MESSAGE briefly and naturally in 1-3 sentences. Be friendly. You MUST use the past conversation history if the user refers to past context (e.g., their name)."
    ),
    "planner_prompt": (
        "You are a task planner for SPARK AI. Today's date is {{current_date}}.\n"
        "Your base knowledge cutoff is mid-2024.\n"
        "Given the user's request and conversation history:\n"
        "1. Break down this request into 1-3 clear steps.\n"
        "2. MANDATORY TEMPORAL RULE: Today is {{current_date}}. If the user request asks for live, current, recent, or today's information (such as 'news', 'today', 'latest', 'live', 'this month', 'recent releases', 'weather', 'stock price', or post-2024 facts), you MUST append the exact string [NEEDS_WEB_SEARCH] at the end of your plan.\n"
        "If it is just a conversational statement (e.g. 'My name is X', 'Hello'), reply ONLY with: 'Acknowledge and respond naturally.'\n"
        "History:\n{{history}}\n\nUser request: {{request}}\n\nPlan:"
    ),
    "router_prompt": (
        "Classify this request into exactly ONE category:\n"
        "- \"coding\": if the user asks for code, programming, debugging, algorithms, SQL, or technical implementation.\n"
        "- \"creative\": if the user asks for stories, poems, roleplay, creative writing, brainstorming, or open-ended essays.\n"
        "- \"general\": for all other queries, facts, QA, analysis, explanations, news, current events, or math.\n\n"
        "Request: \"{{request}}\"\n\n"
        "Reply with ONLY the category name in lowercase (\"coding\", \"creative\", or \"general\")."
    ),
    "document_grader_prompt": (
        "You are a document relevance grader. Evaluate whether the following retrieved context is relevant to the user request.\n\n"
        "User Request: {{request}}\n\n"
        "Retrieved Context:\n{{context}}\n\n"
        "Criteria: Does the retrieved context contain information relevant to answering the user request?\n"
        "Reply with ONLY \"RELEVANT\" or \"IRRELEVANT\"."
    ),
    "worker_general_prompt": (
        "You are SPARK AI, an enterprise-grade AI assistant. Today's date is {{current_date}}.\n\n"
        "--- PAST CONVERSATION HISTORY ---\n{{history}}\n---------------------------------\n\n"
        "--- PLAN ---\n{{plan}}\n------------\n\n"
        "--- CONTEXT ---\n{{context}}\n---------------\n\n"
        "{{feedback}}\n\n"
        "CURRENT USER REQUEST: {{request}}\n\n"
        "Strict Enterprise Grounding Instructions:\n"
        "1. If CONTEXT is provided (such as live web search results or documents), synthesize and present that factual information clearly, thoroughly, and directly to answer the user's question.\n"
        "2. Structure your response with clear headings, bullet points, and key takeaways where appropriate.\n"
        "3. If Context is empty or uninformative and the user asks for live breaking news, summarize the latest known developments up to today ({{current_date}}) and politely note that real-time feeds are updating.\n"
        "4. ABSOLUTE FORBIDDEN: Never output raw JSON, internal search syntax, tool parameters (e.g. 'top_n', 'source'), or bracket commands to the user. Always write in polished, natural conversational English.\n"
        "Answer:"
    ),
    "worker_coding_prompt": (
        "You are SPARK AI, an expert software engineer and technical assistant. Today's date is {{current_date}}.\n\n"
        "--- PAST CONVERSATION HISTORY ---\n{{history}}\n---------------------------------\n\n"
        "--- PLAN ---\n{{plan}}\n------------\n\n"
        "--- CONTEXT ---\n{{context}}\n---------------\n\n"
        "{{feedback}}\n\n"
        "CURRENT USER REQUEST: {{request}}\n\n"
        "Strict Enterprise Coding Instructions:\n"
        "1. If the user's request is conceptual, factual, or informational (such as asking about new models, algorithms, or technical news), answer directly with clear technical explanations rather than writing an unprompted script.\n"
        "2. When the user explicitly requests code, provide complete, working code in proper markdown code blocks with language tags.\n"
        "3. If Context is provided, follow modern syntax and library conventions strictly.\n"
        "4. Briefly explain how the code works and highlight design choices.\n"
        "5. ABSOLUTE FORBIDDEN: Never output raw JSON, tool parameters, or search commands to the user.\n"
        "Answer:"
    ),
    "worker_creative_prompt": (
        "You are SPARK AI, a creative writer and ideation partner. Today's date is {{current_date}}.\n"
        "Write engaging, vivid, and original content tailored to the user's request.\n\n"
        "--- PAST CONVERSATION HISTORY ---\n{{history}}\n---------------------------------\n\n"
        "--- PLAN ---\n{{plan}}\n------------\n\n"
        "--- CONTEXT ---\n{{context}}\n---------------\n\n"
        "{{feedback}}\n\n"
        "CURRENT USER REQUEST: {{request}}\n\n"
        "Instructions:\n"
        "- Write with flair, rich vocabulary, and appropriate tone/style.\n"
        "- If the user refers to earlier themes or ideas, build on the PAST CONVERSATION HISTORY.\n"
        "- If Context is provided, incorporate relevant details creatively.\n"
        "- If the user is asking for real-world factual news, state verified facts accurately.\n"
        "- ABSOLUTE FORBIDDEN: Never output raw JSON, tool parameters, or search commands to the user.\n"
        "Answer:"
    ),
    "validator_prompt": (
        "You are a quality and hallucination validator for an enterprise AI assistant.\n"
        "Today's date is {{current_date}}.\n"
        "Evaluate whether the following DRAFT response is grounded, truthful, and helpful.\n\n"
        "User Request: {{request}}\n"
        "Context Available: {{context}}\n"
        "Draft Response: {{draft}}\n\n"
        "Check these strict criteria:\n"
        "1. Cutoff Apology: Does the draft contain knowledge cutoff apologies (e.g., 'as of my 2024 cutoff', 'up to mid-2024', 'I do not have real-time data') instead of answering? If so, FAIL with 'FAIL: CUTOFF_DETECTED'.\n"
        "2. Temporal Accuracy: Today is {{current_date}}. If the user asks for live, today's, or current news/releases, does the draft provide outdated or fabricated historical events from 2024 instead of verified current facts? If so, FAIL with 'FAIL: CUTOFF_DETECTED'.\n"
        "3. Grounding: Did the draft hallucinate facts contrary to or unsupported by the Context?\n"
        "4. Completeness: Did the draft answer what the user asked?\n\n"
        "Reply with ONLY one line:\n"
        "- If the draft passes all criteria: \"PASS\"\n"
        "- If the draft failed: \"FAIL: <brief 1-sentence explanation of what is wrong>\"\n"
        "Your decision:"
    ),
    "evaluator_prompt": (
        "You are a quality editor. Polish this draft for the user:\n"
        "1. Fix formatting (proper markdown, code blocks, lists).\n"
        "2. Remove any meta-commentary about internal processes.\n"
        "3. Ensure the response matches the conversational tone of the request and stays directly focused on the user's question (remove any unsolicited company plugs or unprompted commercial product tangents).\n"
        "4. IF the draft contains math, ensure it uses dollar sign delimiters: inline $x^2$, block $$x = y$$. (Do not add math if there is none).\n\n"
        "CRITICAL: Output ONLY the final polished text. Do NOT add any conversational intro (like 'Here is the polished version') or editor notes at the end. Your entire output will be shown directly to the user as the final answer.\n\n"
        "Request: {{request}}\nDraft:\n{{draft}}\n\nFinal Polished Text:"
    ),
    "history_summarizer_prompt": (
        "Summarize the following past conversation strictly focusing on key facts, user preferences, names, and unresolved issues. Be extremely concise.\n\n"
        "Conversation:\n{{raw_older}}\n\nSummary:"
    ),
}

_langfuse_client = None


def get_langfuse_client():
    """Singleton getter for Langfuse client."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse()
            return _langfuse_client
        except Exception as e:
            logger.warning(f"Could not initialize Langfuse client: {e}")
            return None
    return None


def ensure_prompts_seeded():
    """
    On application startup, checks Langfuse. If any prompt does not exist,
    creates it in the Langfuse Prompt Registry with 'production' label and 'spark-ai' tag.
    """
    client = get_langfuse_client()
    if not client:
        return

    logger.info("Verifying and seeding Langfuse Prompt Registry...")
    for name, template in DEFAULT_PROMPTS.items():
        try:
            # Check if prompt exists
            client.get_prompt(name, label="production", max_retries=0, fetch_timeout_seconds=2)
        except Exception:
            try:
                # Create prompt in Langfuse
                client.create_prompt(
                    name=name,
                    prompt=template,
                    labels=["production"],
                    tags=["spark-ai"],
                    type="text",
                    commit_message="Initial SPARK AI production prompt auto-seeded"
                )
                logger.info(f"Seeded prompt '{name}' into Langfuse Prompt Registry.")
            except Exception as e:
                logger.warning(f"Could not seed prompt '{name}' in Langfuse: {e}")


def get_managed_prompt(name: str) -> Tuple[str, Optional[Any]]:
    """
    Fetches the prompt from Langfuse Prompt Registry (with 300s TTL cache).
    Returns (langchain_compatible_template_string, langfuse_prompt_client_object).
    Falls back gracefully to DEFAULT_PROMPTS if Langfuse is unavailable.
    """
    fallback_template = DEFAULT_PROMPTS.get(name, "")
    client = get_langfuse_client()
    
    if client:
        try:
            prompt_obj = client.get_prompt(
                name,
                label="production",
                fallback=fallback_template,
                cache_ttl_seconds=300
            )
            # get_langchain_prompt converts {{var}} to {var} for LangChain
            template_str = prompt_obj.get_langchain_prompt()
            return template_str, prompt_obj
        except Exception as e:
            logger.warning(f"Error fetching prompt '{name}' from Langfuse: {e}")

    # Fallback to local default converting {{var}} to {var}
    template_str = fallback_template.replace("{{", "{").replace("}}", "}")
    return template_str, None


def log_validation_score(passed: bool, feedback: str, session_id: Optional[str] = None, trace_id: Optional[str] = None):
    """Logs an automated validation score to the Langfuse trace."""
    client = get_langfuse_client()
    if not client:
        return
    
    try:
        score_val = 1.0 if passed else 0.0
        client.create_score(
            name="quality_validation",
            value=score_val,
            data_type="NUMERIC",
            comment=feedback[:500] if feedback else ("PASSED" if passed else "FAILED"),
            session_id=session_id,
            trace_id=trace_id,
        )
    except Exception as e:
        logger.warning(f"Could not log validation score to Langfuse: {e}")
