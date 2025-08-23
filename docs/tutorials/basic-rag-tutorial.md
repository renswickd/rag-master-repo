# Tutorial: Basic RAG Pipeline

This tutorial walks you through using the basic RAG pipeline in this repository.

## 1. Prepare Your Data

- Place your PDF files in the `data/source_data/` directory.

## 2. Set Up Your Environment

- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Create a `.env` file in the root directory with your GROQ API key:
  ```
  GROQ_API_KEY=your_key_here
  ```

## 3. Run the Pipeline

- To start the basic RAG pipeline:
  ```
  python main.py basic-rag
  ```
- To force re-vectorization of your data:
  ```
  python main.py basic-rag -v
  ```

## 4. Ask Questions

- Type your question at the prompt.
- The system will retrieve relevant context from your PDFs and generate an answer using the GROQ LLM.

---

**You can customize the prompt in `projects/pipeline/prompts.py` to change the assistant's behavior.**
