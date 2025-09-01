import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from shared.utils.pdf_utils import load_pdfs_from_folder
from shared.utils.chroma_utils import get_collection_name_for_rag_type
from shared.configs.static import PERSIST_DIR, EMBEDDING_MODEL, CACHE_RAG_TYPE

load_dotenv()

class CacheRAGRetriever:
    def __init__(self, data_dir, persist_directory=PERSIST_DIR, rag_type=CACHE_RAG_TYPE):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.rag_type = rag_type

        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # Collections
        self.retriever_collection = get_collection_name_for_rag_type(rag_type)  # "cache_rag_collection"
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
    def cache_search(self, question: str, top_k: int = 1):
        self._ensure_cache_vs()
        try:
            results = self.cache_vs.similarity_search(question, k=top_k, filter={"type": {"$eq": "cache"}})
            return results or []
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
    def retrieve(self, query, top_k=3):
        self._ensure_retriever_vs()
        docs = self.retriever_vs.similarity_search(query, k=top_k)#, filter={"type": {"$eq": "retriever"}})
        return [doc.page_content for doc in docs]

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