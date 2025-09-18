from langchain_chroma import Chroma
from dotenv import load_dotenv
from shared.utils.pdf_utils import load_pdfs_from_folder
from shared.configs.retriever_configs import get_retriever_config
from shared.configs.static import CACHE_RAG_TYPE, TOP_K

load_dotenv()

class CacheRAGRetriever:
    def __init__(self, data_dir, rag_type=CACHE_RAG_TYPE):
        self.data_dir = data_dir
        self.rag_type = rag_type
        self.config = get_retriever_config(rag_type)

        self.embedding = self.config["embedding"]
        self.text_splitter = self.config["text_splitter"]
        self.retriever_collection = self.config["collection_name"]
        self.persist_directory = self.config["persist_directory"]
        self.vectorstore = self.config["vectorstore"]
        
        self.cache_collection = "cache_rag_cache_collection"

        # Vectorstores
        self.retriever_vs = None
        self.cache_vs = None

    def _ensure_retriever_vs(self):
        if self.retriever_vs is None:
            self.retriever_vs = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.retriever_collection
            )

    def _ensure_cache_vs(self):
        if self.cache_vs is None:
            self.cache_vs = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.cache_collection
            )

    def index_pdfs(self):
        print(f"Indexing PDFs for collection: {self.retriever_collection}")
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []
        metadatas = []
        for text in pdf_texts:
            chunks = self.text_splitter.create_documents([text])
            docs.extend([c.page_content for c in chunks])
            metadatas.extend([{"source": "cache-rag", "type": "retriever"} for _ in chunks])

        if not docs:
            print("No documents to index for Cache-RAG.")
            return

        self.retriever_vs = Chroma.from_texts(
            texts=docs,
            embedding=self.embedding,
            metadatas=metadatas,
            persist_directory=self.persist_directory,
            collection_name=self.retriever_collection
        )
        print(f"Successfully indexed {len(docs)} chunks in {self.retriever_collection}")

    # ---------- Cache operations ----------
    def cache_search(self, question: str, top_k: int = 1, similarity_threshold: float = 0.5):
        self._ensure_cache_vs()
        try:
            # Use similarity_search_with_score to get similarity scores
            results = self.cache_vs.similarity_search_with_score(
                question, 
                k=top_k, 
                filter={"type": {"$eq": "cache"}}
            )
            
            # Filter results based on similarity threshold
            # ChromaDB returns squared euclidean distance, convert to similarity
            filtered_results = []
            for doc, distance in results:
                # For euclidean distance, convert to similarity score
                # Similarity = 1 / (1 + distance)
                similarity = 1 / (1 + distance)
                print(f"Cache similarity check: {similarity:.3f} (distance: {distance:.3f}, threshold: {similarity_threshold})")
                if similarity >= similarity_threshold:
                    print(f"Cache hit! Similarity: {similarity:.3f}")
                    filtered_results.append(doc)
                else:
                    print(f"Cache miss - below threshold. Similarity: {similarity:.3f}")
            
            return filtered_results
        except Exception as e:
            print(f"Cache search error: {e}")
            return []

    def cache_upsert(self, question: str, answer: str):
        self._ensure_cache_vs()
        try:
            self.cache_vs.add_texts(
                texts=[answer],
                metadatas=[{"type": "cache", "question": question}]
            )
        except Exception as e:
            print(f"Cache upsert error: {e}")

    # ---------- Retrieval ----------
    def retrieve(self, query, top_k=TOP_K):
        self._ensure_retriever_vs()
        docs = self.retriever_vs.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

    def clear_cache(self):
        """Clear all cache entries from the cache collection."""
        try:
            self._ensure_cache_vs()
            # Get all cache entries
            all_cache = self.cache_vs.get(where={"type": {"$eq": "cache"}})
            if all_cache["ids"]:
                self.cache_vs.delete(ids=all_cache["ids"])
                print(f"Cleared {len(all_cache['ids'])} cache entries")
            else:
                print("Cache is already empty")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def get_collection_info(self):
        try:
            self._ensure_retriever_vs()
            self._ensure_cache_vs()
            retriever_count = self.retriever_vs._collection.count()
            cache_count = self.cache_vs._collection.count()
            return {
                "retriever_collection": self.retriever_collection,
                "cache_collection": self.cache_collection,
                "retriever_count": retriever_count,
                "cache_count": cache_count,
                "rag_type": self.rag_type
            }
        except Exception as e:
            return {"error": str(e), "rag_type": self.rag_type}