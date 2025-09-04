import os
import requests
from dotenv import load_dotenv
from typing import List
from langchain.tools import StructuredTool
from shared.components.agentic_rag_states import WebSearchInput

load_dotenv()
serp_key = os.getenv("SERPAPI_API_KEY")

def _web_search(query: str, num: int = 2) -> str:
    if not serp_key:
        return "SerpAPI error: SERPAPI_API_KEY not set."
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": serp_key, "num": num},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic_results", [])
        if not results:
            return f"No search results for: {query}"
        lines: List[str] = []
        for r in results[: num]:
            title = r.get("title") or ""
            snippet = r.get("snippet") or r.get("snippet_highlighted_words") or ""
            link = r.get("link") or r.get("displayed_link") or ""
            lines.append(f"- {title}\n  {snippet}\n  {link}")
        return "Search results:\n" + "\n".join(lines)
    except Exception as e:
        return f"SerpAPI error: {e}"
    
serp_search = StructuredTool.from_function(
    func=_web_search,
    name="web_search",
    description=(
        "Search the web using Google via SerpAPI. Use for fresh facts, news, or URLs."
    ),
    args_schema=WebSearchInput,
)





