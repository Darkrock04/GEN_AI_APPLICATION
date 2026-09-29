import os
import json
import logging
import urllib.request
import urllib.parse
import re
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_search_query(query: str) -> str:
    """
    Strips conversational filler phrases and anchors temporal queries with the current month/year.
    E.g. 'tell me what is the ai news today live i want' -> 'ai news today September 2026'
    """
    if not query:
        return ""

    clean = query.strip()
    now = datetime.now()
    month_year = now.strftime("%B %Y")

    # Strip conversational prefixes iteratively
    prefixes = [
        r"^can\s+(?:you|u)\s+(?:please\s+)?tell\s+me\s+(?:about\s+)?(?:the\s+)?",
        r"^could\s+you\s+(?:please\s+)?tell\s+me\s+(?:about\s+)?(?:the\s+)?",
        r"^tell\s+me\s+(?:about\s+)?(?:the\s+)?",
        r"^what\s+(?:is|are|r)\s+(?:the\s+)?",
        r"^show\s+me\s+(?:the\s+)?",
        r"^give\s+me\s+(?:the\s+)?",
        r"^i\s+want\s+(?:to\s+(?:know|see|find|get)\s+)?",
        r"^find\s+me\s+(?:the\s+)?",
        r"^search\s+(?:the\s+web\s+)?(?:for\s+)?",
        r"^lookup\s+(?:the\s+)?",
        r"^please\s+",
    ]
    for p in prefixes:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    # Strip trailing conversational fluff
    suffixes = [
        r"\s+live\s+i\s+want$",
        r"\s+i\s+want$",
        r"\s+can\s+(?:you|u)\s+tell\s+me$",
        r"\s+can\s+you$",
        r"\s+please$",
        r"\s+right\s+now$",
    ]
    for s in suffixes:
        clean = re.sub(s, "", clean, flags=re.IGNORECASE).strip()

    # If the user asked for current/live/news/recent info, anchor with current calendar date
    temporal_signals = ("today", "this month", "current", "latest", "recent", "news", "released")
    if any(k in query.lower() for k in temporal_signals):
        if str(now.year) not in clean:
            clean = f"{clean} {month_year}"

    # Domain prioritization: If query is about AI news, ensure search engines prioritize AI technology
    # rather than general news headlines (e.g. weather/politics)
    clean_lower = clean.lower()
    if re.search(r'\bai\b', clean_lower) and "artificial intelligence" not in clean_lower:
        clean = re.sub(r'\bai\b', "AI artificial intelligence", clean, flags=re.IGNORECASE)

    return clean.strip() or query.strip()


def extract_search_query(text: str, default: str = "") -> str:
    """
    Extracts a focused search query from model output containing:
    1. JSON function/tool-calls: e.g. {"query": "...", "top_n": 10}
    2. Bracket notation: [NEEDS_WEB_SEARCH: query]
    3. Or falls back to clean_search_query of the default prompt.
    """
    if not text:
        return clean_search_query(default)
    
    clean_text = text.strip()
    
    # 1. Parse JSON tool calls (when LLMs revert to native function calling)
    if clean_text.startswith("{") and clean_text.endswith("}"):
        try:
            d = json.loads(clean_text)
            for k in ("query", "q", "search_query", "search", "keyword"):
                if k in d and isinstance(d[k], str) and d[k].strip():
                    return clean_search_query(d[k].strip())
        except Exception:
            pass
            
    # Check for embedded JSON tool call in text
    json_match = re.search(r'\{[^{}]*"query"\s*:\s*"([^"]+)"[^{}]*\}', clean_text)
    if json_match:
        return clean_search_query(json_match.group(1).strip())
        
    # 2. Check for bracket notation: [NEEDS_WEB_SEARCH: query]
    match = re.search(r'\[NEEDS_WEB_SEARCH:\s*(.+?)\]', text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return clean_search_query(extracted)
            
    return clean_search_query(default)


def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Queries the self-hosted SearXNG instance and returns a formatted markdown string of results.
    """
    searxng_url = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not searxng_url:
        logger.warning("SEARXNG_URL is not set. Web search is disabled.")
        return ""

    clean_q = clean_search_query(query)
    logger.info(f"[Web Search] Cleaned search query: '{clean_q}' (original: '{query}')")

    try:
        # Build URL with params
        params = urllib.parse.urlencode({
            "q": clean_q,
            "format": "json"
        })
        url = f"{searxng_url}/search?{params}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'SparkAI-Backend/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get("results", [])
                
                if not results:
                    logger.info(f"[Web Search] Zero results returned for query: '{clean_q}'")
                    return f"No recent internet information found for: {clean_q}"

                formatted_results = [f"### Web Search Results ({datetime.now().strftime('%B %Y')}):"]
                for i, res in enumerate(results[:max_results]):
                    title = res.get("title", "No Title")
                    content = res.get("content", "No Description")
                    link = res.get("url", "#")
                    formatted_results.append(f"**{title}**\n{content}\nSource: {link}\n")
                
                logger.info(f"[Web Search] Successfully fetched {len(results[:max_results])} results from SearXNG.")
                return "\n".join(formatted_results)
            else:
                logger.error(f"SearXNG returned status code {response.status}")
                return "Failed to fetch live web results."

    except Exception as e:
        logger.error(f"Error calling SearXNG API: {e}")
        return f"Failed to perform web search due to an internal error."
