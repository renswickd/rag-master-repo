# rag-master-repo

A modular Retrieval-Augmented Generation (RAG) repository with swappable pipelines and persistent vector stores (ChromaDB). Supports:
- Basic text RAG (GROQ)
- Multi-modal RAG (text + images via CLIP + GPT‑4.1)
- LangGraph RAG (two-node graph: retrieve → generate)
- rag-ubac: Role-based access control (UBAC) RAG

Answers are grounded: the system answers ONLY using retrieved context; otherwise it replies:
"I am a helpful assitant for you to assist with the internal knowledge base; No related contents retrived for the provided query - Try modifying your query for assistance."

## Features

- PDF ingestion and chunking (PyMuPDF)
- Persistent vector storage (Chroma PersistentClient) with per-type collections:
  - `basic_rag_collection`, `multi_modal_collection`, `langgraph_collection`
- Modular retrievers, prompts, and pipelines
- GROQ LLM for basic RAG; OpenAI GPT‑4.1 for multi‑modal; GROQ for LangGraph
- CLI for vectorizing, querying, inspecting, listing, and deleting collections
- Grounded prompts to reduce hallucinations

## Setup

1) Data layout
- Place your PDFs in:
  - `data/source_data/basic-rag/`
  - `data/source_data/multi-modal/`
  - `data/source_data/langgraph/`
  - `data/source_data/rag-ubac/`

2) Environment (.env)
- GROQ (basic, langgraph):
  ```
  GROQ_API_KEY=your_groq_key_here
  ```
- OpenAI (multi‑modal):
  ```
  OPENAI_API_KEY=your_openai_key_here
  ```
- Optional (silence tokenizers warning):
  ```
  TOKENIZERS_PARALLELISM=false
  ```

3) Install
```
pip install -r requirements.txt
```

## CLI

- Basic RAG
  ```
  python main.py --rag_type basic-rag
  python main.py --rag_type basic-rag -v         # (re-)vectorize
  python main.py --rag_type basic-rag --info     # collection info
  ```

- Multi‑Modal RAG
  ```
  python main.py --rag_type multi-modal
  python main.py --rag_type multi-modal -v
  python main.py --rag_type multi-modal --info
  ```

- LangGraph RAG
  ```
  python main.py --rag_type langgraph
  python main.py --rag_type langgraph -v
  python main.py --rag_type langgraph --info
  ```

- rag-ubac
  You will be prompted to enter your role (executive/hr/junior). Answers are restricted by role-based access. UBAC uses metadata filters; re-run vectorization after updating FILE_ACCESS_METADATA.
  ```
  python main.py --rag_type rag-ubac --vectorize
  python main.py --rag_type rag-ubac
  python main.py --rag_type rag-ubac --info
  ```

- Manage collections (ChromaDB)
  ```
  python main.py --rag_type basic-rag --list-collections
  python main.py --rag_type basic-rag --delete-collection
  ```

- Interactive session: type `/exit` or `/quit` to finish.

Data directory is inferred from the RAG type:
```
data/source_data/{basic-rag | multi-modal | langgraph | rag-ubac}
```

## Grounded Prompts

- Prompts enforce context-only answers. If no relevant context is retrieved, the system replies:
  "I am a helpful assitant for you to assist with the internal knowledge base; No related contents retrived for the provided query - Try modifying your query for assistance."

## Project Structure (key paths)

- `projects/retriever/`
  - `basic_rag_retriever.py`
  - `multi_modal_retriever.py`
  - `langgraph_retriever.py`
  - `rag_ubac_retriever.py`
- `projects/pipeline/`
  - `basic_rag_pipeline.py`
  - `multi_modal_rag_pipeline.py`
  - `langgraph_rag_pipeline.py`
  - `rag_ubac_pipeline.py`
- `projects/prompts/`
  - `prompts.py` (basic)
  - `multi_modal_prompts.py`
  - `langgraph_prompts.py`
- `shared/utils/`
  - `pdf_utils.py` (PyMuPDF)
  - `chroma_utils.py` (PersistentClient, collection helpers)
  - `rag_ubac_scripts.py`
- `shared/configs/`
  - `static.py` (FILE_ACCESS_METADATA, VALID_ROLES, RAG_UBAC_TYPE)

## Tutorials

- Basic RAG: `docs/tutorials/basic-rag-tutorial.md`
- Multi‑Modal RAG: `docs/tutorials/multi-modal-rag.md`
- RAG using Langgraph: `docs/tutorials/langgraph-rag.md`
- RAG-UBAC tutorial: see `docs/tutorials/rag-ubac-tutorial.md`
