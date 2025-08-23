import os
from projects.retriever.basic_rag_retriever import BasicRAGRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from projects.prompts.prompts import BASIC_RAG_PROMPT

load_dotenv()

class BasicRAGPipeline:
    def __init__(
        self,
        data_dir,
        persist_directory="chroma_db",
        groq_model="mixtral-8x7b-32768"
    ):
        self.retriever = BasicRAGRetriever(data_dir, persist_directory)
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
