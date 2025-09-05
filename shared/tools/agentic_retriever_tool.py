from typing import List#, Callable
from langchain.tools import StructuredTool
from projects.retriever.agentic_rag_retriever import AgenticRAGRetriever
from shared.components.agentic_rag_states import AgenticRetrieverInput


def make_agentic_retriever_tool(retriever: AgenticRAGRetriever) -> StructuredTool:
    """Factory to create a StructuredTool for the Agentic RAG retriever.

    The tool searches your resume knowledge base stored in ChromaDB under
    the collection configured for the agentic RAG type (agentic_rag_collection).
    """

    def _retrieve(query: str, top_k: int = 5) -> str:
        try:
            docs: List[str] = retriever.retrieve(query, top_k=top_k)
            if not docs:
                return "No relevant results found in resume collection."
            # Join top docs into a simple plaintext context block
            lines: List[str] = []
            for i, d in enumerate(docs, 1):
                snippet = d.strip()
                # # Keep snippets modest to avoid overly long messages
                # if len(snippet) > 1200:
                #     snippet = snippet[:1200] + "..."
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

