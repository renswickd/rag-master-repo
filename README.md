# rag-master-repo

A modular Retrieval-Augmented Generation (RAG) repository.  
Currently, the `basic-rag` pipeline is implemented using LangChain, ChromaDB, and GROQ LLM.

## Features

- PDF ingestion and chunking (PyMuPDF)
- Embedding and vector storage (LangChain + ChromaDB, persistent)
- Modular retriever and pipeline structure
- GROQ LLM for answer generation
- Dedicated prompt file for easy customization

## Usage

1. Place your PDFs in `data/source_data/`.
2. Set your GROQ API key in a `.env` file:  
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```
   python main.py basic-rag
   ```
   To force re-vectorization:
   ```
   python main.py basic-rag -v
   ```

## Structure

- `projects/retriever/`: Retriever logic
- `projects/pipeline/`: RAG pipeline and prompts
- `shared/utils/`: PDF and ChromaDB utilities
- `data/source_data/`: Your PDF files
- `docs/tutorials/`: Tutorials and guides

See `docs/tutorials/basic-rag-tutorial.md` for a step-by-step tutorial.
