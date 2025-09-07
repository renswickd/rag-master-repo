import chromadb
from shared.configs.static import ALLOWED_COLLECTIONS

def get_collection_name_for_rag_type(rag_type: str) -> str:
    """Generate collection name based on RAG type."""
    collection_name = f"{rag_type.replace('-', '_')}_collection"
    if collection_name not in ALLOWED_COLLECTIONS:
        raise ValueError(f"Invalid RAG type: {rag_type}")
    return collection_name

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
