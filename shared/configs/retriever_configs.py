from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shared.utils.chroma_utils import get_collection_name_for_rag_type
from shared.configs.static import PERSIST_DIR, EMBEDDING_MODEL

def get_retriever_config(rag_type: str):
        return {
            "embedding": HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            "text_splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
            "collection_name": get_collection_name_for_rag_type(rag_type),
            "persist_directory": PERSIST_DIR,
            "vectorstore": None,
        }

    # if rag_type == "basic-rag":
    #     return BasicRAGRetrieverConfig()
    # elif rag_type == "multi-modal":
    #     return MultiModalRetrieverConfig()
    # elif rag_type == "langgraph":
    #     return LangGraphRetrieverConfig()
    # elif rag_type == "rag-ubac":
    #     return RAGUBACRetrieverConfig()
    # elif rag_type == "cache-rag":
    #     return CacheRAGRetrieverConfig()