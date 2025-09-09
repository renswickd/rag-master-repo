## Tutorial: Basic RAG Pipeline (Technical)

This guide explains the Basic RAG pipeline components, how data is ingested and vectorized, how retrieval works, and how the final answer is generated.

### Overview

The Basic RAG flow:
1) Ingest PDFs → chunk into passages
2) Embed chunks → store in persistent ChromaDB collection
3) Retrieve top‑k chunks for a query
4) Grounded generation with a strict prompt using GROQ LLM

### Key Components (files)

- Pipeline: `projects/pipeline/basic_rag_pipeline.py`
  - Creates `BasicRAGRetriever`
  - Wraps the LLM (`ChatGroq`) and composes the final prompt
- Retriever: `projects/retriever/basic_rag_retriever.py`
  - Uses `Chroma` for storage and `HuggingFaceEmbeddings` (from config) for dense embeddings
  - `index_pdfs()` builds the collection; `retrieve()` issues similarity search
- Prompt: `projects/prompts/prompts.py` → `BASIC_RAG_PROMPT`
  - Enforces “answer strictly from context” and a fallback message when context is missing
- Config: `shared/configs/static.py`
  - `GROQ_MODEL`, `EMBEDDING_MODEL`, `TOP_K`, and collection naming helpers
- Utilities: `shared/utils/pdf_utils.py` (PDF text extraction)

### Data Preparation

Place your PDFs in:

```
data/source_data/basic-rag/
```

### Environment

Create `.env` in the repo root with your GROQ key:

```
GROQ_API_KEY=your_key_here
```

Optional: run the root setup script to create a venv and install in editable mode:

```
bash ./setup.sh
source .venv/bin/activate
```

Or install dependencies manually:

```
pip install -r requirements.txt
```

### Ingestion and Vectorization

Command:

```
python main.py --rag_type basic-rag -v
```

What happens:
- `BasicRAGRetriever.index_pdfs()` reads PDFs via `load_pdfs_from_folder()`.
- Chunks are created using `RecursiveCharacterTextSplitter` (from `shared/configs/retriever_configs.py`) with `chunk_size=500` and `chunk_overlap=50`.
- Embeddings are computed with `HuggingFaceEmbeddings` using `EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`).
- Chunks + embeddings are persisted in Chroma under `chroma_db/` using a collection named from the rag type (`basic_rag_collection`).

Inspect the collection:

```
python main.py --rag_type basic-rag --info
python main.py --rag_type basic-rag --list-collections
```

### Retrieval

At query time, the retriever loads (or reuses) the Chroma collection and performs `similarity_search(query, k=TOP_K)`. The top‑k chunks are concatenated into a single context block passed to the LLM prompt.

### Grounded Generation

`BasicRAGPipeline.answer()` builds a prompt from `BASIC_RAG_PROMPT`:
- Strictly answers using the provided context
- If context is insufficient, returns the fixed fallback message defined in the prompt
GROQ model used defaults to `GROQ_MODEL` from config (e.g., `openai/gpt-oss-20b`).

### Run the Pipeline (interactive)

```
python main.py --rag_type basic-rag
```

Type questions at the prompt. Use `/exit` or `/quit` to leave.

### Troubleshooting

- No answers / empty retrieval:
  - Ensure PDFs exist under `data/source_data/basic-rag/`
  - Re‑vectorize with `-v`
  - Check collection info with `--info`
- GROQ auth errors: verify `GROQ_API_KEY` in `.env`
- Chroma persistence:
  - Collections live in `chroma_db/`
  - Manage via `--list-collections` and `--delete-collection`

### Extensibility

- Adjust chunk size/overlap in `shared/configs/retriever_configs.py`.
- Swap the embedder by changing `EMBEDDING_MODEL` in `shared/configs/static.py`.
- Customize the grounding prompt in `projects/prompts/prompts.py`.
