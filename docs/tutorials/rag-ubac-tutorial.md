## RAG-UBAC Tutorial

This tutorial shows how to use the role-based access control (UBAC) RAG pipeline.

### 1. What is RAG-UBAC?

RAG-UBAC enforces role-based access during retrieval by attaching access metadata to chunks and filtering results according to a user’s role.

- Roles: `executive`, `hr`, `junior`
- Documents (example set in `data/source_data/rag-ubac/`):
  - `Executive-Strategy.pdf` → executive only
  - `HR-Policies-and-Benefits.pdf` → executive, hr
  - `Onboarding-Guide-Junior.pdf` → executive, hr, junior

### 2. Configuration

Edit `shared/configs/static.py`:
- `FILE_ACCESS_METADATA`: maps filename → base access level
- `VALID_ROLES`: allowed user roles
- `RAG_UBAC_TYPE`: `"rag-ubac"`
- `PERSIST_DIR`, `EMBEDDING_MODEL`, `GROQ_MODEL`: runtime configs

Example:
```python
FILE_ACCESS_METADATA = {
    "Executive-Strategy.pdf": "executive",
    "HR-Policies-and-Benefits.pdf": "hr",
    "Onboarding-Guide-Junior.pdf": "junior"
}
```

### 3. Index your data

Place PDFs into `data/source_data/rag-ubac/`, then run:
```bash
python main.py --rag_type rag-ubac --vectorize
```

This:
- Reads PDFs with PyMuPDF
- Splits into chunks
- Writes to Chroma with per-chunk metadata

### 4. Ask questions

```bash
python main.py --rag_type rag-ubac
```

You’ll be prompted for your role:
- `executive`: can access all
- `hr`: can access HR + onboarding
- `junior`: can access onboarding only

Examples:
- As `junior`, asking about HR policy should return the “no related contents” guardrail.
- As `hr`, asking about executive strategy should be restricted.

### 5. Internals

- Pipeline: `projects/pipeline/rag_ubac_pipeline.py`
- Retriever: `projects/retriever/rag_ubac_retriever.py`
  - Indexing attaches `access_role` per chunk (duplicated per allowed role)
  - Retrieval uses Chroma filter: `{"access_role": {"$eq": "<role>"}}`
- Role prompt: `shared/components/rag_ubac_scripts.py`

### 6. Maintenance tips

- If you add more documents or roles, update `FILE_ACCESS_METADATA` and re-vectorize.
- To reset or remove the collection:
```bash
python main.py --delete-collection --rag_type rag-ubac
```
- List all collections:
```bash
python main.py --list-collections
```

### 7. Troubleshooting

- “No related contents” for allowed role: ensure the file is correctly mapped in `FILE_ACCESS_METADATA` and re-vectorize.
- Metadata filter errors: confirm metadata fields are simple types (strings) and operator is `$eq`.