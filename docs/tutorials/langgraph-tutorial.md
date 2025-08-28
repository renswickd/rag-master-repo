# Tutorial: LangGraph RAG Pipeline

This tutorial walks you through using the LangGraph-based RAG pipeline (retriever → generation).

## 1. Prepare Your Data
- Place your PDFs in:
  ```
  data/source_data/langgraph/
  ```

## 2. Set Up Your Environment
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Create a `.env` file with your GROQ API key:
  ```
  GROQ_API_KEY=your_groq_key_here
  ```
- Optional (silence tokenizer warning):
  ```
  TOKENIZERS_PARALLELISM=false
  ```

## 3. Vectorize (optional)
- Build/update the `langgraph_collection` in Chroma:
  ```
  python main.py --rag_type langgraph -v
  ```

## 4. Run the Pipeline
```
python main.py --rag_type langgraph
```


## 5. Ask Questions
- Type your question in the prompt.
- To quit, type:
  ```
  /exit
  ```
  or
  ```
  /quit
  ```

## 6. Inspect or Manage Collections
- List collections:
  ```
  python main.py --rag_type langgraph --list-collections
  ```
- Show pipeline info:
  ```
  python main.py --rag_type langgraph --info
  ```
- Delete the collection:
  ```
  python main.py --rag_type langgraph --delete-collection
  ```

## 7. Customize the Prompt
- Edit:
  ```
  projects/prompts/langgraph_prompts.py
  ```

## 8. Implementation Notes
- Retriever: `projects/retriever/langgraph_retriever.py`
  - Chroma persistent vector store (per-type collection)
  - HF embeddings + recursive text splitting
- Pipeline: `projects/pipeline/langgraph_rag_pipeline.py`
  - Two-node LangGraph: retrieve → generate
  - GROQ model via `ChatGroq`