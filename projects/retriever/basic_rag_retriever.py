import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shared.utils.pdf_utils import load_pdfs_from_folder
from dotenv import load_dotenv

load_dotenv()

class BasicRAGRetriever:
    def __init__(self, data_dir, persist_directory="chroma_db"):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = None

    def index_pdfs(self):
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []
        for text in pdf_texts:
            docs.extend(self.text_splitter.create_documents([text]))
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory=self.persist_directory,
            collection_name="basic_rag_collection"
        )
        self.vectorstore.persist()

    def retrieve(self, query, top_k=3):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name="basic_rag_collection"
            )
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]