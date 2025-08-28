# Command line arguments
RAG_TYPES = ["basic-rag", "multi-modal", "langgraph", "rag-ubac"]
# Data dir per type
DATA_DIR_MAP = {
    "basic-rag": "data/source_data/basic-rag",
    "multi-modal": "data/source_data/multi-modal",
    "langgraph": "data/source_data/langgraph",
    "rag-ubac": "data/source_data/rag-ubac",
}

# Vector Database
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM
GROQ_MODEL = "openai/gpt-oss-20b"

# Multi Modal RAG
RAG_TYPE = "multi-modal"
## retriever
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_PROCESSOR = "openai/clip-vit-base-patch32"

# Langgraph
LG_RAG_TYPE = "langgraph"