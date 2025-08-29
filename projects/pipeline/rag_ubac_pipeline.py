import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from projects.retriever.rag_ubac_retriever import RAGUBACRetriever
from projects.prompts.prompts import BASIC_RAG_PROMPT

load_dotenv()

class RAGUBACPipeline:
    def __init__(
        self,
        data_dir,
        persist_directory="chroma_db",
        groq_model="openai/gpt-oss-20b",
        rag_type="rag-ubac",
        role="executive"
    ):
        self.rag_type = rag_type
        self.role = role
        self.retriever = RAGUBACRetriever(data_dir, persist_directory, rag_type)
        self.llm = ChatGroq(
            temperature=0.2,
            model=groq_model,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def answer(self, query, top_k=3):
        contexts = self.retriever.retrieve(query, role=self.role, top_k=top_k)
        context = "\n".join(contexts)
        if not context.strip():
            return 'I am a helpful assitant for you to assist with the internal knowledge base; No related contents retrived for the provided query - Try modifying your query for assistance.'
        prompt = BASIC_RAG_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response

    def get_pipeline_info(self):
        info = self.retriever.get_collection_info()
        info["role"] = self.role
        return info