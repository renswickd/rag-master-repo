import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from shared.configs.static import GROQ_MODEL, TOP_K
from projects.retriever.hyde_rag_retriever import HyDERetriever
from projects.prompts.hyde_prompts import HYDE_GENERATION_PROMPT, HYDE_RAG_PROMPT

load_dotenv()

class HyDERAGPipeline:
    """
    HyDE pipeline: generate a hypothetical document, retrieve using its embedding,
    then generate a final answer using retrieved context.
    """
    def __init__(self, data_dir: str, generator_model: str = None, final_model: str = None):
        self.data_dir = data_dir
        self.retriever = HyDERetriever(data_dir)
        gen_model = generator_model or GROQ_MODEL
        final_m = final_model or GROQ_MODEL

        self.generator_llm = ChatGroq(temperature=0.0, model=gen_model, api_key=os.getenv("GROQ_API_KEY"))
        self.final_llm = ChatGroq(temperature=0.0, model=final_m, api_key=os.getenv("GROQ_API_KEY"))

    def answer(self, query: str, top_k: int = TOP_K) -> str:
        # 1) generate hypothetical document
        gen_prompt = HYDE_GENERATION_PROMPT.format(query=query)
        gen_resp = self.generator_llm.invoke(gen_prompt)
        hyde_doc = getattr(gen_resp, "content", str(gen_resp)).strip()

        # 2) retrieve using the hypothetical doc embedding
        retrieved = self.retriever.retrieve_by_hypdoc(hyde_doc, top_k=top_k)
        context_texts = [d.get("content", "") for d in retrieved]
        context_block = "\n\n".join(context_texts).strip() or "No relevant retrieved context."

        # 3) final prompt and generation
        rag_prompt = HYDE_RAG_PROMPT.format(context=context_block, query=query)
        final_resp = self.final_llm.invoke(rag_prompt)
        final_answer = getattr(final_resp, "content", str(final_resp))
        return final_answer

    def get_pipeline_info(self) -> Dict[str, Any]:
        return self.retriever.get_collection_info()