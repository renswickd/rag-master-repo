import chromadb
from chromadb.config import Settings
import os

allowed_collection_types = ["basic_rag_collection", "advanced_rag_collection"]

def get_persistent_chroma_collection(
    collection_name: str = "basic_rag_collection",
    persist_directory: str = "chroma_db"
):
    """Create or get a persistent ChromaDB collection."""
    if collection_name not in allowed_collection_types:
        raise ValueError(f"Invalid collection type: {collection_name}. Allowed types are: {allowed_collection_types}")

    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.Client(Settings(
        persist_directory=persist_directory,  # Persistent storage
        chroma_db_impl="duckdb+parquet"
    ))
    return client.get_or_create_collection(collection_name)

def get_collection_name_for_rag_type(rag_type: str) -> str:
    """Generate collection name based on RAG type."""
    return f"{rag_type.replace('-', '_')}_collection"

def list_existing_collections(persist_directory: str = "chroma_db") -> list:
    """List all existing collections in the persist directory."""
    try:
        client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        return [col.name for col in client.list_collections()]
    except Exception:
        return []

def delete_collection(collection_name: str, persist_directory: str = "chroma_db"):
    """Delete a specific collection."""
    try:
        client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
