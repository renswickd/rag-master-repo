import os
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from shared.utils.pdf_utils import load_pdfs_from_folder
# from shared.utils.chroma_utils import get_collection_name_for_rag_type
from shared.configs.static import AGENTIC_RAG_TYPE, TOP_K
from shared.configs.retriever_configs import get_retriever_config

load_dotenv()

class AgenticRAGRetriever:
    def __init__(self, data_dir, rag_type=AGENTIC_RAG_TYPE):
        self.data_dir = data_dir
        self.rag_type = rag_type
        self.config = get_retriever_config(rag_type)
        
        self.embedding = self.config["embedding"]
        self.text_splitter = self.config["text_splitter"]
        self.collection_name = self.config["collection_name"]
        self.persist_directory = self.config["persist_directory"]
        self.vectorstore = self.config["vectorstore"]
    
    def _ensure_store(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.collection_name,
            )
    
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
    
    def retrieve(self, query, top_k=TOP_K):
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

if __name__ == "__main__":
    retriever = AgenticRAGRetriever(data_dir="data/source_data/basic-rag", rag_type="agentic-rag")
    retriever.index_pdfs()
    print(retriever.get_collection_info())