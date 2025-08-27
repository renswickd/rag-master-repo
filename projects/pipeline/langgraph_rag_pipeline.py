import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from projects.retriever.langgraph_retriever import LangGraphRetriever
from projects.prompts.langgraph_prompts import LANGGRAPH_RAG_PROMPT
from langgraph.graph import StateGraph, END
from shared.configs.static import PERSIST_DIR, LG_RAG_TYPE, GROQ_MODEL

load_dotenv()

class LangGraphRAGPipeline:
    def __init__(self, data_dir, persist_directory=PERSIST_DIR, groq_model=GROQ_MODEL):
        self.rag_type = LG_RAG_TYPE
        self.retriever = LangGraphRetriever(data_dir, persist_directory, self.rag_type)
        self.llm = ChatGroq(
            temperature=0.2,
            model=groq_model,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        # Build graph: question -> retrieve -> generate
        self.graph = self._build_graph()

    def _build_graph(self):
        def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
            q = state.get("question", "")
            top_k = state.get("top_k", 4)
            contexts = self.retriever.retrieve(q, top_k=top_k)
            # IMPORTANT (STATE): carry forward the question (and any other needed keys)
            return {"context": "\n".join(contexts), "question": q, "top_k": top_k}

        def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
            context = state.get("context", "")
            question = state.get("question", "")
            prompt = LANGGRAPH_RAG_PROMPT.format(context=context, question=question)
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            return {"answer": content}

        g = StateGraph(dict)
        g.add_node("retrieve", retrieve_node)
        g.add_node("generate", generate_node)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    def answer(self, query: str, top_k: int = 4) -> str:
        result = self.graph.invoke({"question": query, "top_k": top_k})
        return result.get("answer", "")

    def get_pipeline_info(self):
        return self.retriever.get_collection_info()
