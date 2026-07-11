import os
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Queries the self-hosted SearXNG instance and returns a formatted markdown string of results.
    """
    searxng_url = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not searxng_url:
        logger.warning("SEARXNG_URL is not set. Web search is disabled.")
        return ""

    try:
        # Build URL with params
        params = urllib.parse.urlencode({
            "q": query,
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
