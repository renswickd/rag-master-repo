import os
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
# from shared.components.agentic_rag_states import AgentState, RelevanceGrade
from shared.configs.static import GROQ_MODEL as DEFAULT_GROQ_MODEL
from shared.utils.document_utils import format_docs
from shared.tools.web_search_tool import serp_search
from shared.tools.currency_converter_tool import exchangerate_converter
from shared.components.agentic_rag_nodes import agent, grade_documents, rewrite, generate

load_dotenv()

class AgenticRAGReActPipeline:

    def __init__(self, tools: List[Any], groq_model: Optional[str] = None, temperature: float = 0.2):
        model_name = groq_model or DEFAULT_GROQ_MODEL
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        # Build built-in tools: web search and currency converter
        serp_key = os.getenv("SERPAPI_API_KEY")

        # Final tool list: user retriever tools + our two built-ins
        self.tools = list(tools) + [serp_search, exchangerate_converter]
        self.graph = self._build_graph()

    

        