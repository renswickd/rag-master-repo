import os
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shared.utils.pdf_utils import load_pdfs_from_folder
from shared.utils.chroma_utils import get_collection_name_for_rag_type
from dotenv import load_dotenv

load_dotenv()

class BasicRAGRetriever:
    def __init__(self, data_dir, persist_directory="chroma_db", rag_type="basic-rag"):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.rag_type = rag_type
        self.collection_name = get_collection_name_for_rag_type(rag_type)
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = None
        print(f"\n\n{self.data_dir}\n\n")

    def index_pdfs(self):
        print(f"Indexing PDFs for collection: {self.collection_name}")
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []
        for text in pdf_texts:
            docs.extend(self.text_splitter.create_documents([text]))
        
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"Successfully indexed {len(docs)} documents in collection: {self.collection_name}")

    def retrieve(self, query, top_k=3):
        if self.vectorstore is None:
            print(f"Loading existing vector store for collection: {self.collection_name}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.collection_name
            )
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

    def get_collection_info(self):
        """Get information about the current collection."""
        if self.vectorstore is None:
            self.retrieve("test", top_k=1)  # Initialize vectorstore
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "rag_type": self.rag_type
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "rag_type": self.rag_type,
                "error": str(e)
            }

if __name__ == "__main__":
    retriever = BasicRAGRetriever(data_dir="data/source_data/basic-rag/", rag_type="basic-rag")
    retriever.index_pdfs()
    print(retriever.get_collection_info())