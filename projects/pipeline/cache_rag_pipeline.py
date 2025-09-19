import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from projects.retriever.cache_rag_retriever import CacheRAGRetriever
from projects.prompts.prompts import BASIC_RAG_PROMPT
from shared.configs.static import PERSIST_DIR, GROQ_MODEL, CACHE_RAG_TYPE, CACHE_SIMILARITY_THRESHOLD, TOP_K

load_dotenv()

class CacheRAGPipeline:
    def __init__(self, data_dir, persist_directory=PERSIST_DIR, groq_model=GROQ_MODEL):
        self.rag_type = CACHE_RAG_TYPE
        self.retriever = CacheRAGRetriever(data_dir, persist_directory, self.rag_type)
        self.llm = ChatGroq(
            temperature=0,
            model=groq_model,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        def check_cache(state: Dict[str, Any]) -> Dict[str, Any]:
            q = state.get("question", "")
            similarity_threshold = state.get("similarity_threshold", CACHE_SIMILARITY_THRESHOLD)
            hits = self.retriever.cache_search(q, top_k=1, similarity_threshold=similarity_threshold)
            if hits:
                print("DEBUG: Cache hit! Returning cached answer.")
                return {"cache_hit": True, "answer": hits[0].page_content, "question": q}
            print("DEBUG: Cache miss. Proceeding to RAG retrieval.")
            return {"cache_hit": False, "question": q}

        def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
            q = state.get("question", "")
            top_k = state.get("top_k", TOP_K)
            contexts = self.retriever.retrieve(q, top_k=top_k)
            return {"context": "\n".join(contexts), "question": q, "top_k": top_k}

        def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
            context = state.get("context", "")
            question = state.get("question", "")
            prompt = BASIC_RAG_PROMPT.format(context=context, question=question)
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            return {"answer": content, "question": question, "context": context}

        def write_cache(state: Dict[str, Any]) -> Dict[str, Any]:
            q = state.get("question", "")
            a = state.get("answer", "")
            
            if q and a and "no related contents" not in a.lower():
                print("DEBUG: Caching valid answer...")
                self.retriever.cache_upsert(q, a)
            else:
                print("Skipping cache (No valid content to cache)")
            return {"answer": a}

        g = StateGraph(dict)
        g.add_node("check_cache", check_cache)
        g.add_node("retrieve", retrieve_node)
        g.add_node("generate", generate_node)
        g.add_node("write_cache", write_cache)

        g.set_entry_point("check_cache")
        # Branch logic: 
        ## if -->> cache_hit == True -> END; 
        ## else -->> retrieve -> generate -> write_cache -> END
        def route_on_cache(state: Dict[str, Any]):
            return "END" if state.get("cache_hit") else "retrieve"

        g.add_conditional_edges(
            "check_cache",
            route_on_cache,
            {"retrieve": "retrieve", "END": END}
        )
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "write_cache")
        g.add_edge("write_cache", END)
        return g.compile()

    def answer(self, query: str, top_k: int = TOP_K, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD) -> str:
        result = self.graph.invoke({
            "question": query, 
            "top_k": top_k, 
            "similarity_threshold": similarity_threshold
        })
        return result.get("answer", "")

    def get_pipeline_info(self):
        return self.retriever.get_collection_info()