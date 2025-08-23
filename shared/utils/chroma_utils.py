import chromadb
from chromadb.config import Settings

def get_persistent_chroma_collection(
    collection_name: str = "basic_rag_collection",
    persist_directory: str = "chroma_db"
):
    """Create or get a persistent ChromaDB collection."""
    client = chromadb.Client(Settings(
        persist_directory=persist_directory,  # Persistent storage
        chroma_db_impl="duckdb+parquet"
    ))
    return client.get_or_create_collection(collection_name)
