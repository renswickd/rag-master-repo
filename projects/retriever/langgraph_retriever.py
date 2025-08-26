from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shared.utils.pdf_utils import load_pdfs_from_folder
from shared.utils.chroma_utils import get_collection_name_for_rag_type
import os

class LangGraphRetriever:
    def __init__(self, data_dir, persist_directory="chroma_db", rag_type="langgraph"):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.rag_type = rag_type
        self.collection_name = get_collection_name_for_rag_type(rag_type)
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = None

    def index_pdfs(self):
        if not os.path.exists(self.data_dir):
            print(f"Error: data dir '{self.data_dir}' not found")
            return
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []
        for text in pdf_texts:
            docs.extend(self.text_splitter.create_documents([text]))
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        print(f"Indexed {len(docs)} chunks into collection: {self.collection_name}")

    def _ensure_store(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.collection_name,
            )


    def retrieve(self, query, top_k=4):
        self._ensure_store()
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [d.page_content for d in docs]

    def get_collection_info(self):
        self._ensure_store()
        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = 0
        return {
            "rag_type": self.rag_type,
            "collection_name": self.collection_name,
            "document_count": count,
            "data_directory": self.data_dir,
        }