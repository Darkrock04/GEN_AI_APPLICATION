import os
import json
import logging
import urllib.request
import urllib.parse
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_search_query(text: str, default: str = "") -> str:
    """
    Extracts a focused search query from model output containing [NEEDS_WEB_SEARCH: query]
    or falls back to conversational cleaning of the default prompt.
    """
    if not text:
        return default
    
    match = re.search(r'\[NEEDS_WEB_SEARCH:\s*(.+?)\]', text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
            
    return default

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Queries the self-hosted SearXNG instance and returns a formatted markdown string of results.
    """
    searxng_url = os.getenv("SEARXNG_URL", "https://site0230-local.hf.space").rstrip("/")
    if not searxng_url:
        logger.warning("SEARXNG_URL is not set. Web search is disabled.")
        return ""

    try:
        # Strip common conversational prefixes for cleaner search results
        clean_q = query.strip()
        prefixes = [
            "can you tell me the ", "can you tell me ", "can u tell me the ", "can u tell me ",
            "tell me about the ", "tell me about ", "tell me the ", "tell me ",
            "what is the ", "what is ", "what are the ", "what are ",
            "give me the ", "give me ", "search the web for ", "search for ",
            "what's the ", "what's "
        ]
        for prefix in prefixes:
            if clean_q.lower().startswith(prefix):
                clean_q = clean_q[len(prefix):].strip()
                break

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
                    return "No recent internet information found on this topic."

                formatted_results = ["### Web Search Results:"]
                for i, res in enumerate(results[:max_results]):
                    title = res.get("title", "No Title")
                    content = res.get("content", "No Description")
                    link = res.get("url", "#")
                    formatted_results.append(f"**{title}**\n{content}\nSource: {link}\n")
                
                return "\n".join(formatted_results)
            else:
                logger.error(f"SearXNG returned status code {response.status}")
                return "Failed to fetch live web results."

    except Exception as e:
        logger.error(f"Error calling SearXNG API: {e}")
        return f"Failed to perform web search due to an internal error."
