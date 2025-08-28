import chromadb
from chromadb.config import Settings
import os

allowed_collection_types = ["basic_rag_collection", "multi_modal_collection", "langgraph_collection"]

def get_persistent_chroma_collection(
    collection_name: str = "basic_rag_collection",
    persist_directory: str = "chroma_db"
):
    """Create or get a persistent ChromaDB collection."""
    if collection_name not in allowed_collection_types:
        raise ValueError(f"Invalid collection type: {collection_name}. Allowed types are: {allowed_collection_types}")

    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # New Chroma client configuration (Chroma 0.4.x+)
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(collection_name)

def get_collection_name_for_rag_type(rag_type: str) -> str:
    """Generate collection name based on RAG type."""
    return f"{rag_type.replace('-', '_')}_collection"

def list_existing_collections(persist_directory: str = "chroma_db") -> list:
    """List all existing collections in the persist directory."""
    try:
        # New Chroma client configuration
        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def delete_collection(collection_name: str, persist_directory: str = "chroma_db"):
    """Delete a specific collection."""
    try:
        # New Chroma client configuration
        client = chromadb.PersistentClient(path=persist_directory)
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")

def get_collection_info(collection_name: str, persist_directory: str = "chroma_db"):
    """Get information about a specific collection."""
    try:
        # New Chroma client configuration
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_collection(collection_name)
        count = collection.count()
        return {
            "name": collection_name,
            "document_count": count,
            "metadata": collection.metadata
        }
    except Exception as e:
        print(f"Error getting collection info for {collection_name}: {e}")
        return None

def migrate_old_chroma_data(old_persist_directory: str = "chroma_db"):
    """Helper function to check if migration is needed."""
    print("If you have old Chroma data that needs migration:")
    print("1. Install migration tool: pip install chroma-migrate")
    print("2. Run migration: chroma-migrate")
    print("3. See: https://docs.trychroma.com/deployment/migration")
