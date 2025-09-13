## Agentic-RAG Tutorial

This tutorial explains the Agentic RAG pipeline which combines LangGraph orchestration with tool use (ReAct style). The agent can call:
- `resume_retriever` to query your resume knowledge base (Chroma collection: `agentic_rag_collection`)
- `web_search` (SerpAPI-backed) for fresh information
- `currency_convert` for currency conversions

### 1) Data Preparation

Place your resume or profile PDFs under:

```
data/source_data/agentic_rag/
```

Vectorization uses HuggingFace sentence embeddings (`all-MiniLM-L6-v2`) and stores chunks in ChromaDB under the collection name derived from the rag type: `agentic_rag_collection`.

### 2) Environment

- GROQ LLM (used by the agent):
```
GROQ_API_KEY=your_groq_key
```
- SerpAPI (for `web_search` tool):
```
SERPAPI_API_KEY=your_serpapi_key
```

Install dependencies:

```
pip install -r requirements.txt
```

### 3) Vectorize

Run vectorization to (re)build the `agentic_rag_collection` from your PDFs:

```
python main.py --rag_type agentic-rag -v
```

Under the hood, this calls `AgenticRAGRetriever.index_pdfs()`:
- Loads PDFs (`shared/utils/pdf_utils.py`)
- Splits with `RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)`
- Embeds with `HuggingFaceEmbeddings`
- Persists to Chroma (`chroma_db/`) as `agentic_rag_collection`

You can inspect the collection:

```
python main.py --rag_type agentic-rag --info
python main.py --rag_type basic-rag --list-collections
```

### 4) Run the Agent

Start an interactive session:

```
python main.py --rag_type agentic-rag
```

Ask questions like:
- "Summarize the candidate's AWS experience"
- "What programming languages does the candidate use most?"
- "What is 1 EUR in NZD?" (currency tool)
- "Latest news about AWS Summit in NZ" (web search)

The agent uses a strict system prompt to favor tool use. If the model tries to answer directly without tools, the graph routes to a restricted response that reminds supported capabilities.

### 5) Pipeline Overview

Files involved:
- Pipeline: `projects/pipeline/agentic_rag_pipeline.py`
- Nodes: `shared/components/agentic_rag_nodes.py`
- Retriever: `projects/retriever/agentic_rag_retriever.py`
- Tools: `shared/tools/agentic_retriever_tool.py`, `shared/tools/web_search_tool.py`, `shared/tools/currency_converter_tool.py`
- State: `shared/components/agentic_rag_states.py`

Graph (LangGraph) structure:
- `agent`: LLM step with tools bound (`resume_retriever`, `web_search`, `currency_convert`). Decides next action.
- `retrieve`: `ToolNode` executes tool calls emitted by the agent.
- `grade_documents`: Uses structured output (`RelevanceGrade`) to decide if retrieved content is relevant.
  - "yes" → `generate`
  - "no" → `rewrite`
- `generate`: Composes a final answer grounded in retrieved context.
- `rewrite`: Rewrites the query to improve retrieval and loops back to `agent`.
- `restricted`: Fallback for attempts to bypass tool use.

Routing:
- `agent` → conditional:
  - If tools were called → `retrieve`
  - If answered directly → `restricted`
- `retrieve` → `grade_documents` → `generate` or `rewrite`
- `generate` → END; `rewrite` → `agent`

### 6) Tools In Depth

- Resume retriever (`shared/tools/agentic_retriever_tool.py`): wraps your resume vector store as a `StructuredTool`.

  Args schema (`shared/components/agentic_rag_states.py`):
  ```python
  class AgenticRetrieverInput(BaseModel):
      query: constr(min_length=1)
      top_k: conint(ge=1, le=20) = 5
  ```

  Tool factory:
  ```python
  def make_agentic_retriever_tool(retriever: AgenticRAGRetriever) -> StructuredTool:
      def _retrieve(query: str, top_k: int = 5) -> str:
          docs = retriever.retrieve(query, top_k=top_k)
          if not docs:
              return "No relevant results found in resume collection."
          return "\n".join(f"[{i+1}] {d[:1200]}" for i, d in enumerate(docs))
      return StructuredTool.from_function(
          func=_retrieve,
          name="resume_retriever",
          description="Retrieve relevant chunks from the resume vector store.",
          args_schema=AgenticRetrieverInput,
      )
  ```

- Web search (`shared/tools/web_search_tool.py`): SerpAPI-backed Google search for fresh info.

  Args schema:
  ```python
  class WebSearchInput(BaseModel):
      query: constr(min_length=1)
      num: conint(ge=1, le=10) = 5
  ```

  Tool:
  ```python
  serp_search = StructuredTool.from_function(
      func=_web_search,
      name="web_search",
      description="Search the web using Google via SerpAPI.",
      args_schema=WebSearchInput,
  )
  ```

- Currency convert (`shared/tools/currency_converter_tool.py`): live FX conversion.

  Args schema:
  ```python
  class CurrencyConvertInput(BaseModel):
      amount: confloat(gt=0)
      from_currency: constr(min_length=3, max_length=3)
      to_currency: constr(min_length=3, max_length=3)
  ```

  Tool:
  ```python
  exchangerate_converter = StructuredTool.from_function(
      func=_currency_convert,
      name="currency_convert",
      description="Convert currency amounts using live foreign exchange rates.",
      args_schema=CurrencyConvertInput,
  )
  ```

Binding and execution in the graph (`projects/pipeline/agentic_rag_pipeline.py`):
```python
self.retriever = AgenticRAGRetriever(data_dir=data_dir, rag_type=rag_type)
resume_tool = make_agentic_retriever_tool(self.retriever)
self.tools = [resume_tool, serp_search, exchangerate_converter]
workflow.add_node("retrieve", ToolNode(self.tools))
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: "restricted"})
```

### 7) Troubleshooting

- Empty or irrelevant answers:
  - Ensure PDFs are in `data/source_data/agentic_rag/`
  - Re-vectorize with `-v`
  - Check collection info with `--info`
- Web search failures:
  - Set `SERPAPI_API_KEY`; without it, the tool returns a clear error string
- GROQ issues:
  - Verify `GROQ_API_KEY`
- Chroma persistence:
  - Database lives under `chroma_db/`; use `--list-collections` / `--delete-collection` for maintenance

### 8) Extensibility

You can add more tools and pass them into the pipeline via `extra_tools` if you instantiate `AgenticRAGReActPipeline` directly in code. Tools created with LangChain `StructuredTool` are easiest to slot into the existing `ToolNode`.
