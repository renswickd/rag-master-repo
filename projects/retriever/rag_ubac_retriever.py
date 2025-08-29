import os
import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from shared.utils.chroma_utils import get_collection_name_for_rag_type
from shared.configs.static import FILE_ACCESS_METADATA, VALID_ROLES

load_dotenv()

class RAGUBACRetriever:
    def __init__(self, data_dir, persist_directory="chroma_db", rag_type="rag-ubac"):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.rag_type = rag_type
        self.collection_name = get_collection_name_for_rag_type(rag_type)
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = None

    def _get_access_levels_for_role(self, role: str):
        """Determine which documents a role can access based on hierarchy."""
        if role == "executive":
            # Executive can access all documents
            return list(FILE_ACCESS_METADATA.keys())
        elif role == "hr":
            # HR can access HR policies and onboarding (but not executive strategy)
            return [filename for filename, access_level in FILE_ACCESS_METADATA.items() 
                   if access_level in ["hr", "junior"]]
        elif role == "junior":
            # Junior can only access onboarding guide
            return [filename for filename, access_level in FILE_ACCESS_METADATA.items() 
                   if access_level == "junior"]
        else:
            return []

    def _allowed_roles_for_file(self, filename: str):
        """Get all roles that can access a specific file."""
        base_access = FILE_ACCESS_METADATA.get(filename, "executive")
        if base_access == "executive":
            return ["executive"]
        elif base_access == "hr":
            return ["executive", "hr"]
        elif base_access == "junior":
            return ["executive", "hr", "junior"]
        return ["executive"]  # Default fallback

    def index_pdfs(self):
        """Index PDFs with metadata based on FILE_ACCESS_METADATA."""
        print(f"Indexing PDFs for UBAC collection: {self.collection_name}")
        docs = []
        
        for filename in os.listdir(self.data_dir):
            if not filename.lower().endswith(".pdf"):
                continue
                
            pdf_path = os.path.join(self.data_dir, filename)
            
            # Check if file is in our access metadata
            if filename not in FILE_ACCESS_METADATA:
                print(f"Warning: {filename} not found in FILE_ACCESS_METADATA, defaulting to executive-only access")
                base_access = "executive"
            else:
                base_access = FILE_ACCESS_METADATA[filename]
            
            try:
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
            except Exception as e:
                print(f"Failed to read {pdf_path}: {e}")
                continue

            # Create metadata for each chunk
            allowed_roles = self._allowed_roles_for_file(filename)
            metadatas = [{
                "source": filename, 
                "base_access_level": base_access,
                "allowed_roles": allowed_roles,
                "file_type": "pdf"
            }]
            
            # Split text into chunks with metadata
            chunks = self.text_splitter.create_documents([text], metadatas=metadatas)
            docs.extend(chunks)

        if not docs:
            print("No documents to index for UBAC.")
            return

        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"Successfully indexed {len(docs)} chunks in UBAC collection: {self.collection_name}")
        print(f"Access levels: {FILE_ACCESS_METADATA}")

    def _ensure_store(self):
        """Ensure vectorstore is loaded."""
        if self.vectorstore is None:
            print(f"Loading existing vector store for UBAC collection: {self.collection_name}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_name=self.collection_name
            )

    def retrieve(self, query, role: str, top_k=3):
        """Retrieve documents based on role-based access control."""
        self._ensure_store()
        role = (role or "").lower().strip()
        
        if role not in VALID_ROLES:
            print(f"Unknown role '{role}'. Valid roles are: {VALID_ROLES}")
            return []
        
        # Get documents this role can access
        accessible_files = self._get_access_levels_for_role(role)
        if not accessible_files:
            print(f"Role '{role}' has no access to any documents.")
            return []
        
        # Create filter for ChromaDB based on allowed roles
        chroma_filter = {"allowed_roles": {"$in": [role]}}
        
        try:
            docs = self.vectorstore.similarity_search(query, k=top_k, filter=chroma_filter)
            retrieved_sources = set(doc.metadata.get("source", "") for doc in docs)
            print(f"Retrieved from sources: {retrieved_sources}")
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def get_collection_info(self):
        """Get information about the current collection."""
        self._ensure_store()
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "rag_type": self.rag_type,
                "file_access_metadata": FILE_ACCESS_METADATA,
                "valid_roles": list(VALID_ROLES)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "rag_type": self.rag_type,
                "error": str(e),
                "file_access_metadata": FILE_ACCESS_METADATA,
                "valid_roles": list(VALID_ROLES)
            }

    def get_role_access_info(self, role: str):
        """Get information about what documents a specific role can access."""
        if role not in VALID_ROLES:
            return {"error": f"Invalid role: {role}"}
        
        accessible_files = self._get_access_levels_for_role(role)
        return {
            "role": role,
            "accessible_files": accessible_files,
            "total_files": len(FILE_ACCESS_METADATA),
            "access_level": "full" if role == "executive" else "restricted"
        }

