# Tutorial: Multi-Modal RAG Pipeline

This tutorial walks you through using the multi-modal (text + images) RAG pipeline in this repository.

## 1. Prepare Your Data

- Place your PDF files (containing text and/or images) in:
  ```
  data/source_data/multi-modal/
  ```

## 2. Set Up Your Environment

- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Create a `.env` file in the root directory with your OpenAI API key (for GPT-4.1):
  ```
  OPENAI_API_KEY=your_openai_key_here
  ```

## 3. Vectorize Your Data (optional but recommended after adding new PDFs)

- To index multi-modal PDFs (text + images) into the vector store:
  ```
  python main.py --rag_type multi-modal -v
  ```

## 4. Run the Pipeline

- To start the multi-modal RAG pipeline:
  ```
  python main.py --rag_type multi-modal
  ```

## 5. Ask Questions

- Type your question at the prompt.
- The system will retrieve both text chunks and relevant images (embedded via CLIP) from your PDFs and generate an answer using GPT-4.1.
- To quit the interactive mode, type:
  ```
  /exit
  ```
  or
  ```
  /quit
  ```

## 6. Inspect or Manage Collections

- List existing collections (ChromaDB):
  ```
  python main.py --rag_type multi-modal --list-collections
  ```
- Show pipeline info:
  ```
  python main.py --rag_type multi-modal --info
  ```
- Delete the multi-modal collection:
  ```
  python main.py --rag_type multi-modal --delete-collection
  ```

## 7. Customize the Prompt

- Edit the multi-modal prompt in:
  ```
  projects/prompts/multi_modal_prompts.py
  ```
- The pipeline that consumes the prompt:
  ```
  projects/pipeline/multi_modal_rag_pipeline.py
  ```

## 8. Implementation Notes

- Retriever: `projects/retriever/multi_modal_retriever.py`
  - Uses CLIP (`openai/clip-vit-base-patch32`) to embed both text and images.
  - Splits text with `RecursiveCharacterTextSplitter`.
  - Builds a unified FAISS vector index from precomputed embeddings.
- Pipeline: `projects/pipeline/multi_modal_rag_pipeline.py`
  - Builds a multimodal message (text + base64 images) and queries GPT-4.1.
- Utilities:
  - PDF parsing via PyMuPDF (`fitz`).
  - Chroma utils for collection naming (shared across RAG types).

---
