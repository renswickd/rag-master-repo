# Command line arguments
RAG_TYPES = ["basic-rag", "multi-modal", "langgraph", "rag-ubac", "cache-rag", "agentic-rag"]
ALLOWED_COLLECTIONS = [
    "basic_rag_collection", 
    "multi_modal_collection", 
    "langgraph_collection", 
    "rag_ubac_collection",
    "cache_rag_collection",
    "cache_rag_cache_collection",
    "agentic_rag_collection"
]

# Data dir per type
DATA_DIR_MAP = {
    "basic-rag": "data/source_data/basic-rag",
    "multi-modal": "data/source_data/multi-modal",
    "langgraph": "data/source_data/langgraph",
    "rag-ubac": "data/source_data/rag-ubac",
    "cache-rag": "data/source_data/basic-rag",
    "agentic-rag": "data/source_data/agentic-rag"
}

# Vector Database
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# LLM
GROQ_MODEL = "openai/gpt-oss-20b"

# Basic RAG
B_RAG_TYPE = "basic-rag"

# Multi Modal RAG
MM_RAG_TYPE = "multi-modal"
## retriever
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_PROCESSOR = "openai/clip-vit-base-patch32"

# Langgraph
LG_RAG_TYPE = "langgraph"

# RAG-UBAC
VALID_ROLES = {"executive", "hr", "junior"}
FILE_ACCESS_METADATA = {
    "Executive-Strategy.pdf":"executive", 
    "HR-Policies-and-Benefits.pdf":"hr", 
    "Onboarding-Guide-Junior.pdf":"junior"
}
RAG_UBAC_TYPE = "rag-ubac"

# Cache-RAG
CACHE_RAG_TYPE = "cache-rag"
CACHE_SIMILARITY_THRESHOLD = 0.5

# Agentic-RAG
AGENTIC_RAG_TYPE = "agentic-rag"
