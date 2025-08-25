import os
from projects.retriever.multi_modal_retriever import MultiModalRetriever
from projects.prompts.multi_modal_prompts import MULTIMODAL_RAG_PROMPT
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class MultiModalRAGPipeline:
    def __init__(
        self,
        data_dir,
        persist_directory="chroma_db",
        groq_model="mixtral-8x7b-32768"
    ):
        self.rag_type = "multi-modal"
        self.retriever = MultiModalRetriever(data_dir, persist_directory, self.rag_type)
        
        # Initialize GPT-4 Vision model
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = init_chat_model("openai:gpt-4.1")

    def answer(self, query, top_k=5):
        """Main pipeline for multimodal RAG."""
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(query, k=top_k)
        
        # Create multimodal message
        message = self._create_multimodal_message(query, context_docs)
        
        # Get response from GPT-4V
        response = self.llm.invoke([message])
        
        # Print retrieved context info
        self._print_retrieved_info(context_docs)
        
        return response.content

    def _create_multimodal_message(self, query, retrieved_docs):
        """Create a message with both text and images for GPT-4V."""
        content = []
        
        # Add the query
        content.append({
            "type": "text",
            "text": f"Question: {query}\n\nContext:\n"
        })
        
        # Separate text and image documents
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        
        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page']}]: {doc.page_content}"
                for doc in text_docs
            ])
            content.append({
                "type": "text",
                "text": f"Text excerpts:\n{text_context}\n"
            })
        
        # Add images
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id:
                image_data = self.retriever.get_image_data(image_id)
                if image_data:
                    content.append({
                        "type": "text",
                        "text": f"\n[Image from page {doc.metadata['page']}]:\n"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
        
        # Add instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease answer the question based on the provided text and images."
        })
        
        return HumanMessage(content=content)

    def _print_retrieved_info(self, context_docs):
        """Print information about retrieved documents."""
        print(f"\nRetrieved {len(context_docs)} documents:")
        for doc in context_docs:
            doc_type = doc.metadata.get("type", "unknown")
            page = doc.metadata.get("page", "?")
            if doc_type == "text":
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  - Text from page {page}: {preview}")
            else:
                print(f"  - Image from page {page}")
        print("\n")

    def get_pipeline_info(self):
        """Get information about the current pipeline and collection."""
        return self.retriever.get_collection_info()
