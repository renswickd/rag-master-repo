import os
from typing import List, Dict, Any
from shared.configs.retriever_configs import get_retriever_config
from shared.configs.static import HYDE_RAG_TYPE, PERSIST_DIR, EMBEDDING_MODEL, TOP_K
from langchain_chroma import Chroma

class HyDERetriever:
    def __init__(self, data_dir: str, rag_type: str = HYDE_RAG_TYPE):
        self.data_dir = data_dir
        self.rag_type = rag_type

        if get_retriever_config:
            try:
                cfg = get_retriever_config(rag_type)
                self.embedding = cfg["embedding"]
                self.text_splitter = cfg["text_splitter"]
                self.collection_name = cfg["collection_name"]
                self.persist_directory = cfg["persist_directory"]
                self.vectorstore = cfg["vectorstore"]
            except Exception:
                self._fallback_init()
        else:
            self._fallback_init()

        self._vs = None

    def _fallback_init(self):
        self.embedding = None
        self.text_splitter = None
        self.collection_name = "hyde_collection"
        self.persist_directory = PERSIST_DIR
        self.vectorstore = None

    def index_pdfs(self):
        # TODO: implement indexing if needed
        print("HyDE retriever: index_pdfs() not implemented in fallback. Use project retriever config to enable indexing.")

    def _ensure_vs(self):
        if self._vs is None:
            if self.vectorstore is not None:
                self._vs = self.vectorstore
            elif Chroma is not None and self.embedding is not None:
                self._vs = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding,
                    collection_name=self.collection_name,
                )
            else:
                raise RuntimeError("No vectorstore available for HyDE retriever (missing Chroma/embedding or project config).")

    def retrieve_by_hypdoc(self, hyp_doc: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Given a hypothetical document text, embed it and perform vector search to return top-k docs.
        Normalizes results to list of dicts: {'content': str, 'metadata': dict}
        """
        self._ensure_vs()
        embed_fn = getattr(self.embedding, "embed_text", None) or getattr(self.embedding, "embed", None)
        if embed_fn is None:
            try:
                docs = self._vs.similarity_search(hyp_doc, k=top_k)
            except Exception as e:
                raise RuntimeError(f"HyDE retrieval failed: no embedder and text similarity not available: {e}")
        else:
            vector = embed_fn(hyp_doc)
            if hasattr(self._vs, "similarity_search_by_vector"):
                docs = self._vs.similarity_search_by_vector(vector, k=top_k)
            elif hasattr(self._vs, "search_by_vector"):
                docs = self._vs.search_by_vector(vector, top_k=top_k)
            else:
                docs = self._vs.similarity_search(hyp_doc, k=top_k)

        normalized = []
        for r in docs:
            if isinstance(r, dict):
                normalized.append(r)
                continue
            content = getattr(r, "page_content", None) or getattr(r, "content", None) or getattr(r, "text", None) or str(r)
            metadata = getattr(r, "metadata", None) or {}
            normalized.append({"content": content, "metadata": metadata})
        return normalized

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        """
        Default retriever interface returning page_content strings for the given query.
        HyDE typical flow will use retrieve_by_hypdoc instead.
        """
        self._ensure_vs()
        docs = self._vs.similarity_search(query, k=top_k)
        return [getattr(d, "page_content", str(d)) for d in docs]

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            self._ensure_vs()
            count = self._vs._collection.count() if hasattr(self._vs, "_collection") else 0
            return {"collection_name": self.collection_name, "document_count": count, "rag_type": self.rag_type}
        except Exception as e:
            return {"error": str(e), "rag_type": self.rag_type}