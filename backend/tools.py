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
    E.g. 'what is the new decision model came this month' -> 'AI new decision model September 2026'
    'then what is this jev and laya what kind of model there r' -> 'jev and laya AI models'
    """
    if not query:
        return ""

    clean = query.strip()
    now = datetime.now()
    month_year = now.strftime("%B %Y")

    # Strip conversational prefixes iteratively
    prefixes = [
        r"^(?:so|then|and|ok|okay)?\s*(?:can\s+(?:you|u)\s+(?:please\s+)?tell\s+me\s+(?:about\s+)?(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:could\s+you\s+(?:please\s+)?tell\s+me\s+(?:about\s+)?(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:tell\s+me\s+(?:about\s+)?(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:what\s+(?:is|are|r)\s+(?:the\s+|this\s+|these\s+|those\s+|a\s+|an\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:show\s+me\s+(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:give\s+me\s+(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:i\s+want\s+(?:to\s+(?:know|see|find|get)\s+)?(?:about\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:find\s+me\s+(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:search\s+(?:the\s+web\s+)?(?:for\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*(?:lookup\s+(?:the\s+|this\s+|these\s+)?)",
        r"^(?:so|then|and|ok|okay)?\s*please\s+",
        r"^(?:so|then|and|ok|okay)\s+",
    ]
    for p in prefixes:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    # Strip trailing conversational fluff and filler clauses
    suffixes = [
        r"\s+(?:what\s+kind\s+of\s+models?\s+(?:there\s+r|are\s+there|there\s+are|they\s+are|they\s+r))$",
        r"\s+(?:what\s+(?:are|r)\s+(?:they|these))$",
        r"\s+(?:which\s+)?came\s+(?:out\s+)?(?:this\s+month|today|this\s+week|recently|this\s+year)$",
        r"\s+(?:which\s+)?was\s+released\s+(?:this\s+month|today|this\s+week|recently|this\s+year)$",
        r"\s+live\s+i\s+want$",
        r"\s+i\s+want$",
        r"\s+can\s+(?:you|u)\s+tell\s+me$",
        r"\s+can\s+you$",
        r"\s+please$",
        r"\s+right\s+now$",
    ]
    for s in suffixes:
        clean = re.sub(s, "", clean, flags=re.IGNORECASE).strip()

    # Strip embedded conversational temporal verbs like 'came this month', 'came out today'
    clean = re.sub(r"\b(?:which\s+)?came\s+(?:out\s+)?(?:this\s+month|today|this\s+week|recently)\b", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\b(?:which\s+)?was\s+released\s+(?:this\s+month|today|this\s+week|recently)\b", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+", " ", clean).strip()

    # If the user asked for current/live/news/recent info, anchor with current calendar date
    temporal_signals = ("today", "this month", "current", "latest", "recent", "news", "released", "came out")
    if any(k in query.lower() for k in temporal_signals):
        if str(now.year) not in clean:
            clean = f"{clean} {month_year}"

    # Domain prioritization: If query is about AI / decision / reasoning models, ensure search engine targets AI
    clean_lower = clean.lower()
    if re.search(r'\bai\b', clean_lower) and "artificial intelligence" not in clean_lower:
        clean = re.sub(r'\bai\b', "AI artificial intelligence", clean, flags=re.IGNORECASE)
    elif any(k in clean_lower for k in ("decision model", "reasoning model", "language model", "llm")) and "ai" not in clean_lower:
        clean = f"AI {clean}"

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


# Domains that represent noise or uninformative homepages
EXCLUDED_DOMAINS = (
    "amazon.", "ebay.", "walmart.", "aliexpress.", "etsy.", "target.", "bestbuy.",
    "shopping.google.com"
)


def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Queries the self-hosted SearXNG instance and returns a formatted markdown string of results.
    Filters noisy e-commerce results and sanitizes non-standard unicode characters.
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
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode('utf-8', errors='ignore'))
                raw_results = data.get("results", [])
                
                # Filter out noisy e-commerce domains and bare homepages
                filtered_results = []
                for res in raw_results:
                    link = res.get("url", "").lower()
                    if any(bad in link for bad in EXCLUDED_DOMAINS):
                        continue
                    filtered_results.append(res)
                
                results = filtered_results if filtered_results else raw_results

                if not results:
                    logger.info(f"[Web Search] Zero results returned for query: '{clean_q}'")
                    return f"No recent internet information found for: {clean_q}"

                formatted_results = [f"### Web Search Results ({datetime.now().strftime('%B %Y')}):"]
                for i, res in enumerate(results[:max_results]):
                    title = res.get("title", "No Title")
                    content = res.get("content", "No Description")
                    link = res.get("url", "#")
                    # Sanitize unicode control/directional characters
                    clean_title = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', title).strip()
                    clean_content = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', content).strip()
                    formatted_results.append(f"**{clean_title}**\n{clean_content}\nSource: {link}\n")
                
                logger.info(f"[Web Search] Successfully fetched {len(results[:max_results])} results from SearXNG.")
                return "\n".join(formatted_results)
            else:
                logger.error(f"SearXNG returned status code {response.status}")
                return "Failed to fetch live web results."

    except Exception as e:
        logger.error(f"Error calling SearXNG API: {e}")
        return f"Failed to perform web search due to an internal error."
