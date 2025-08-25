# rag-master-repo

A modular Retrieval-Augmented Generation (RAG) repository with swappable pipelines and persistent vector stores (ChromaDB). Supports basic text RAG and multi-modal (text + images) RAG.

## Features

- PDF ingestion and chunking (PyMuPDF)
- Persistent vector storage (ChromaDB PersistentClient)
- Modular retrievers and pipelines per RAG type
- GROQ LLM for basic RAG; OpenAI GPT-4.1 for multi-modal
- Dedicated prompt files per pipeline
- CLI to vectorize, query, inspect, list, and delete collections

## Quickstart

1. Data layout:
   - Place your PDFs under:
     - `data/source_data/basic-rag/`
     - `data/source_data/multi-modal/`

2. Environment:
   - `.env`:
     - For basic RAG (GROQ): `GROQ_API_KEY=your_key_here`
     - For multi-modal (OpenAI): `OPENAI_API_KEY=your_key_here`

3. Install:
   ```
   pip install -r requirements.txt
   ```

## CLI

- Basic RAG:
  ```
  python main.py --rag_type basic-rag
  python main.py --rag_type basic-rag -v         # force (re-)vectorization
  python main.py --rag_type basic-rag --info     # show collection info
  ```
- Multi-modal RAG:
  ```
  python main.py --rag_type multi-modal
  python main.py --rag_type multi-modal -v
  python main.py --rag_type multi-modal --info
  ```
- Manage collections (ChromaDB):
  ```
  python main.py --rag_type basic-rag --list-collections
  python main.py --rag_type basic-rag --delete-collection
  ```
- Interactive session quit: type `/exit` or `/quit`.

Data directory is inferred from `--rag_type`:
- `data/source_data/{rag_type}`

## Structure

- `projects/retriever/`
  - `basic_rag_retriever.py`
  - `multi_modal_retriever.py`
- `projects/pipeline/`
  - `basic_rag_pipeline.py`
  - `multi_modal_rag_pipeline.py`
- `projects/prompts/`
  - `prompts.py` (basic)
  - `multi_modal_prompts.py`
- `shared/utils/`
  - `pdf_utils.py` (PyMuPDF)
  - `chroma_utils.py` (PersistentClient, per-type collections)
- `data/source_data/{basic-rag|multi-modal}/` (PDFs)

## Notes

- ChromaDB uses `PersistentClient(path='chroma_db')` and one collection per RAG type:
  - `basic_rag_collection`
  - `multi_modal_collection`
- Use `-v` to (re-)build the vector store after adding PDFs.
- Prompts are customizable in `projects/prompts/`.

See `docs/tutorials/basic-rag-tutorial.md` for a step-by-step tutorial.
