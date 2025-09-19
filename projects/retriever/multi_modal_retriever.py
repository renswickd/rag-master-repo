import fitz, os , io, base64
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_community.vectorstores import FAISS
from shared.configs.retriever_configs import get_retriever_config
from dotenv import load_dotenv
from shared.configs.static import MM_RAG_TYPE, CLIP_MODEL, CLIP_PROCESSOR

load_dotenv()

class MultiModalRetriever:
    def __init__(self, data_dir, rag_type=MM_RAG_TYPE):
        self.data_dir = data_dir
        self.rag_type = rag_type
        self.config = get_retriever_config(rag_type)
        self.collection_name = self.config["collection_name"]
        self.persist_directory = self.config["persist_directory"]
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR)
        self.clip_model.eval()
        
        # Storage for documents and embeddings
        self.all_docs = []
        self.all_embeddings = []
        self.image_data_store = {}
        self.vectorstore = self.config["vectorstore"]
        self.splitter = self.config["text_splitter"]
        
        print(f"Initialized MultiModal Retriever for collection: {self.collection_name}")
        print(f"Data directory: {self.data_dir}")

    def embed_image(self, image_data):
        """Embed image using CLIP"""
        if isinstance(image_data, str):  
            image = Image.open(image_data).convert("RGB")
        else:  
            image = image_data
        
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()
    
    def embed_text(self, text):
        """Embed text using CLIP."""
        inputs = self.clip_processor(
            text=text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77  
        )
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    def index_pdfs(self):
        """Process PDFs and create embeddings for both text and images"""
        print(f"Indexing multi-modal PDFs for collection: {self.collection_name}")
        
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory {self.data_dir} does not exist!")
            return
        
        
        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            return
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_dir, pdf_file)
            print(f"Processing PDF: {pdf_file}")
            self._process_single_pdf(pdf_path)
        
        
        if self.all_docs and self.all_embeddings:
            embeddings_array = np.array(self.all_embeddings)
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=[(doc.page_content, emb) for doc, emb in zip(self.all_docs, embeddings_array)],
                embedding=None,  
                metadatas=[doc.metadata for doc in self.all_docs]
            )
            print(f"Successfully indexed {len(self.all_docs)} documents (text + images) in collection: {self.collection_name}")
        else:
            print("No documents to index")

    def _process_single_pdf(self, pdf_path):
        """Process a single PDF file for text and images"""
        try:
            doc = fitz.open(pdf_path)
            
            for i, page in enumerate(doc):
                # Process text
                text = page.get_text()
                if text.strip():
                    temp_doc = Document(page_content=text, metadata={"page": i, "type": "text", "source": pdf_path})
                    text_chunks = self.splitter.split_documents([temp_doc])
                    
                    for chunk in text_chunks:
                        embedding = self.embed_text(chunk.page_content)
                        self.all_embeddings.append(embedding)
                        self.all_docs.append(chunk)
                
                # Process images
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        
                        # Create unique identifier
                        image_id = f"page_{i}_img_{img_index}"
                        
                        # Store image as base64 for later use with GPT-4V
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        self.image_data_store[image_id] = img_base64
                        
                        # Embed image using CLIP
                        embedding = self.embed_image(pil_image)
                        self.all_embeddings.append(embedding)
                        
                        # Create document for image
                        image_doc = Document(
                            page_content=f"[Image: {image_id}]",
                            metadata={"page": i, "type": "image", "image_id": image_id, "source": pdf_path}
                        )
                        self.all_docs.append(image_doc)
                        
                    except Exception as e:
                        print(f"Error processing image {img_index} on page {i}: {e}")
                        continue
            
            doc.close()
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    def retrieve(self, query, top_k=5):
        """Unified retrieval using CLIP embeddings for both text and images."""
        if self.vector_store is None:
            print("Vector store not initialized. Please run index_pdfs() first.")
            return []
        
        
        query_embedding = self.embed_text(query)
        
        
        results = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k
        )
        
        return results

    def get_collection_info(self):
        """Get information about the current collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.all_docs),
            "text_documents": len([doc for doc in self.all_docs if doc.metadata.get("type") == "text"]),
            "image_documents": len([doc for doc in self.all_docs if doc.metadata.get("type") == "image"]),
            "rag_type": self.rag_type,
            "vector_store_initialized": self.vector_store is not None,
            "data_directory": self.data_dir
        }

    def get_image_data(self, image_id):
        """Get base64 image data for a specific image ID."""
        return self.image_data_store.get(image_id)

if __name__ == "__main__":
    retriever = MultiModalRetriever(data_dir="data/source_data/multi-modal/", rag_type=MM_RAG_TYPE)
    retriever.index_pdfs()
    print(retriever.get_collection_info())
