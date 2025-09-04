from typing import Any

def format_docs(docs: Any) -> str:
    """Best-effort conversion of tool outputs to a printable context string."""
    try:
        # If it's a list of Document-like objects
        from langchain_core.documents import Document  # type: ignore

        if isinstance(docs, list) and all(hasattr(d, "page_content") for d in docs):
            return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    except Exception:
        return str(docs)