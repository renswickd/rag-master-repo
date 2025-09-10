## Tutorial: LangGraph RAG

This tutorial explains the LangGraph-based RAG pipeline in this repo from a conceptual and technical perspective. We use a compact two‑node graph (retrieve → generate) to make the core ideas clear, and highlight how to extend it to richer, agentic flows.

### LangGraph

LangGraph lets you model LLM applications as stateful graphs:
- Nodes are functions that read and update a shared state.
- Edges define execution order or conditional routing.
- The graph runtime merges each node’s returned partial state into the global state before the next node runs.

In our simple pipeline, the state is a plain Python `dict` with keys like `question`, `context`, and `answer`. For dicts, LangGraph uses last‑write‑wins semantics on overlapping keys.

### Data Preparation

Place PDFs here:

```
data/source_data/langgraph/
```

### Environment

Install and configure:

```
pip install -r requirements.txt
```

Create `.env` with your GROQ key:

```
GROQ_API_KEY=your_groq_key_here
```

Optional:

```
TOKENIZERS_PARALLELISM=false
```

### Vectorization (index build)

Build or update the `langgraph_collection` in Chroma:

```
python main.py --rag_type langgraph -v
```

Under the hood, the retriever:
- Loads PDFs, chunks text (recursive splitter), embeds with HuggingFace, and persists to Chroma persistent DB.
- File: `projects/retriever/langgraph_retriever.py`

### Pipeline Structure (StateGraph)

File: `projects/pipeline/langgraph_rag_pipeline.py`

```python
g = StateGraph(dict)
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
graph = g.compile()
```

- State type: `dict` (simple, explicit keys)
- Entry: `retrieve` runs first; its output is merged into state
- Then `generate` uses that state and writes `answer`, then `END`

Retrieve node (preserves question in state):

```python
def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("question", "")
    top_k = state.get("top_k", TOP_K)
    contexts = self.retriever.retrieve(q, top_k=top_k)
    return {"context": "\n".join(contexts), "question": q, "top_k": top_k}
```

Generate node (grounded prompt):

```python
def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    context = state.get("context", "")
    question = state.get("question", "")
    prompt = LANGGRAPH_RAG_PROMPT.format(context=context, question=question)
    resp = self.llm.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"answer": content}
```

Prompt lives at:

```
projects/prompts/langgraph_prompts.py
```

### Why a Graph for “Simple” RAG?

Even a basic two‑step flow benefits from explicit state and edges:
- Clear separation of concerns (retrieval vs. generation)
- Easy to add nodes (grading, rewriting) and conditional edges later
- Streaming and tracing support via LangGraph runners

### Running the Pipeline

Start the CLI:

```
python main.py --rag_type langgraph
```

Type questions at the prompt. Quit with `/exit` or `/quit`.

Inspect collections and info:

```
python main.py --rag_type langgraph --info
python main.py --rag_type langgraph --list-collections
python main.py --rag_type langgraph --delete-collection
```

### Extending Toward Agentic Graphs

LangGraph shines when you add control flow:
- Conditional routing with grading: e.g., `retrieve → grade → (generate | rewrite → retrieve)`
- Tools via `ToolNode` and agent policies
- Message‑oriented state (`add_messages`) for multi‑turn conversations

See the agentic implementation in this repo for examples of:
- Prebuilt `ToolNode` and `tools_condition` for deciding when to execute tools
- Structured grading to gate generation
- Query rewriting loops to improve recall

Files to explore:
- `shared/components/agentic_rag_nodes.py`
- `projects/pipeline/agentic_rag_pipeline.py`

### Retriever Details (Indexing)

File: `projects/retriever/langgraph_retriever.py`
- Uses `Chroma` with `HuggingFaceEmbeddings`
- Recursive splitter (`chunk_size=500`, `overlap=50`)
- `retrieve(query, k)` executes vector similarity search and returns top passages

Vector DB note: Chroma is great for local development. For production‑scale indices (millions of chunks, concurrent queries), consider a managed vector database with ANN indexing, metadata filtering, and hybrid search (e.g., Pinecone, Weaviate, Qdrant, Milvus, OpenSearch/Elasticsearch, pgvector).

### Summary

This pipeline demonstrates a minimal LangGraph: explicit state, two nodes, grounded prompting. It’s simple to run, easy to reason about, and ready to grow into more sophisticated, tool‑using workflows without rewriting from scratch.

- [RAG Series: Part 2 - RAG using LangGraph](https://medium.com/@renswick.d/rag-series-part-2-rag-with-langgraph-1f5f2e669518)
