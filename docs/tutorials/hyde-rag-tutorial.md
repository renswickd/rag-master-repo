## HyDE RAG Tutorial

This guide explains the HyDE (Hypothetical Document Embedding) pipeline: how it generates a hypothetical document for a query, uses that document's embedding to retrieve relevant context, and then produces a grounded final answer.

### Overview

HyDE flow:
1) Generate a concise hypothetical document that directly answers the user's question (the HyDE step)
2) Embed the hypothetical document and retrieve top‑k relevant chunks from the persistent vector store
3) Compose a final grounded answer using the retrieved context and a final LLM

HyDE is useful when direct query embeddings retrieve weak context — the generated hypothetic document often produces embeddings that better match latent, relevant content.

### Key Components (files)

- Pipeline: `projects/pipeline/hyde_rag_pipeline.py`
  - Builds the HyDE pipeline: generator LLM + retriever + final LLM
- Retriever: `projects/retriever/hyde_rag_retriever.py`
  - Indexing (`index_pdfs`) and retrieval (`retrieve_by_hypdoc`) using Chroma and configured embedder
- Prompts: `projects/prompts/hyde_prompts.py`
  - `HYDE_GENERATION_PROMPT` for creating the hypothetical document
  - `HYDE_RAG_PROMPT` for the final grounded generation
- Config: `shared/configs/static.py`
  - `HYDE_RAG_TYPE`, `HYDE_COLLECTION`, `PERSIST_DIR`, `TOP_K`
- Utilities: `shared/utils/pdf_utils.py` (PDF text extraction)

### Data Preparation

Place your PDFs in:

```
data/source_data/hyde/
```

Vectorization will create or update the persistent Chroma collection `hyde_collection` under `chroma_db/`.

### Environment

HyDE uses the same LLM config pattern as other pipelines. Ensure the following environment variables are set (in `.env` at repo root):

```
GROQ_API_KEY=your_groq_key_here
```

Install dependencies if needed:

```
pip install -r requirements.txt
```

### Indexing / Vectorization

To (re-)vectorize HyDE documents (index PDF chunks into `hyde_collection`):

```
python main.py --rag_type hyde -v
```

What happens:
- `HyDERetriever.index_pdfs()` reads PDFs via `shared/utils/pdf_utils.py`.
- Text is split into chunks (configurable via `shared/configs/retriever_configs.py` if present).
- Embeddings are computed with the configured embedding model and persisted to Chroma under `chroma_db/` in `hyde_collection`.

Inspect collection info:

```
python main.py --rag_type hyde --info
python main.py --rag_type basic-rag --list-collections
```

### HyDE Retrieval and Answering Flow

1) HyDE generation: `HyDERAGPipeline` uses a generator LLM (low temperature) to produce a focused hypothetical document from the user query (`HYDE_GENERATION_PROMPT`).
2) HyDE retrieval: the hypothetical document is embedded and used to perform a vector search via `HyDERetriever.retrieve_by_hypdoc()` returning top‑k documents.
3) Final generation: the retrieved context is concatenated and passed into `HYDE_RAG_PROMPT` with the original user query; a final LLM produces the grounded answer.

This two-stage process often retrieves documents that are semantically closer to the ideal answer than the original query alone.

### Run the Pipeline (interactive)

```
python main.py --rag_type hyde
```

Type questions at the prompt. Use `/exit` or `/quit` to leave.

### Troubleshooting

- No retrieval results / poor answers:
  - Ensure PDFs exist under `data/source_data/hyde/` and re-run with `-v`.
  - Confirm `hyde_collection` exists via `--list-collections` and check `--info`.
- Embedding or Chroma errors:
  - Verify embedding configuration in `shared/configs/retriever_configs.py` or `shared/configs/static.py`.
- LLM generation issues:
  - Check `GROQ_API_KEY` in `.env` and ensure network access for the model API.

### Extensibility

- Swap the generator or final LLM by providing model overrides when instantiating `HyDERAGPipeline` in code.
- Tune text splitter chunk size/overlap in `shared/configs/retriever_configs.py` to change retrieval granularity.
- Adjust `TOP_K` in `shared/configs/static.py` to retrieve more or fewer documents for the final prompt.

### References

- Pipeline implementation: `projects/pipeline/hyde_rag_pipeline.py`
- Retriever implementation: `projects/retriever/hyde_rag_retriever.py`
- Prompts: `projects/prompts/hyde_prompts.py`

