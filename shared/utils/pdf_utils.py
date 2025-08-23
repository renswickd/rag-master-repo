import os
import fitz  # PyMuPDF

def load_pdfs_from_folder(folder_path: str):
    """Load and concatenate text from all PDFs in a folder using PyMuPDF."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            doc = fitz.open(os.path.join(folder_path, filename))
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(text)
    return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Chunk text into overlapping segments."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
