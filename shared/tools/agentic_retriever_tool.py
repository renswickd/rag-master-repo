from typing import List
from langchain.tools import StructuredTool
from projects.retriever.agentic_rag_retriever import AgenticRAGRetriever
from shared.components.agentic_rag_states import AgenticRetrieverInput
from shared.configs.static import TOP_K


def make_agentic_retriever_tool(retriever: AgenticRAGRetriever) -> StructuredTool:
    """Creates a StructuredTool for the Agentic RAG retriever.

    The tool searches your resume knowledge base stored in ChromaDB under
    the collection configured for the agentic RAG type (agentic_rag_collection).
    """

    def _retrieve(query: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant chunks from the resume knowledge base.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of results to retrieve. Defaults to TOP_K.

        Returns:
            str: The retrieved documents, formatted as a single string with each document on a new line.
    """
        try:
            docs: List[str] = retriever.retrieve(query, top_k=top_k)
            if not docs:
                return "No relevant results found in resume collection."

            lines: List[str] = []
            for i, d in enumerate(docs, 1):
                snippet = d.strip()
                lines.append(f"[{i}] {snippet}")
            return "\n".join(lines)
        except Exception as e:
            return f"Retriever error: {e}"

    return StructuredTool.from_function(
        func=_retrieve,
        name="resume_retriever",
        description=(
            "Retrieve relevant chunks from the resume vector store. "
            "Use this for questions about the candidate's background, skills, or experience."
        ),
        args_schema=AgenticRetrieverInput,
    )

