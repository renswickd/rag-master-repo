HYDE_GENERATION_PROMPT = """Create a concise hypothetical document that directly answers the user's question.
Write in a factual, knowledgeable tone. Keep the document focused and include short relevant facts or steps.

Question: {query}

Hypothetical document:"""

HYDE_RAG_PROMPT = """You are an assistant. Use the provided retrieved context to answer the question.
If the context does not contain the answer, reply succinctly with "No relevant content found."

Context:
{context}

Question:
{query}

Answer:"""