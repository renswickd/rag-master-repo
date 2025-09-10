import os
from projects.retriever.basic_rag_retriever import BasicRAGRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from projects.prompts.prompts import BASIC_RAG_PROMPT
from shared.configs.static import GROQ_MODEL, B_RAG_TYPE

load_dotenv()

class BasicRAGPipeline:
    def __init__(
        self,
        data_dir,
        groq_model=GROQ_MODEL,
        rag_type=B_RAG_TYPE
    ):
        self.rag_type = rag_type
        self.retriever = BasicRAGRetriever(data_dir, rag_type=rag_type)
        self.llm = ChatGroq(
            temperature=0.2,
            model=groq_model,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def answer(self, query, top_k=3):
        contexts = self.retriever.retrieve(query, top_k=top_k)
        context = "\n".join(contexts)
        prompt = BASIC_RAG_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response

    def get_pipeline_info(self):
        """Get information about the current pipeline and collection."""
        return self.retriever.get_collection_info()
