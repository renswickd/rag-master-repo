from typing import Any, Dict, Literal
from shared.components.agentic_rag_states import AgentState, RelevanceGrade
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from shared.utils.document_utils import format_docs
from langchain_core.output_parsers import StrOutputParser

def agent(self, state: AgentState) -> Dict[str, Any]:
    """Decide next action using the model; binds tools for ReAct."""
    messages = state["messages"]

    # Strict system instructions to compel tool usage
    sys = SystemMessage(content=(
        "You are restricted to three capabilities only:\n"
        "1) retriever tools provided, 2) web_search, 3) currency_convert.\n"
        "- Always use one of these tools to act.\n"
        "- For currency tasks, call currency_convert with: amount (float), from_currency (3 letters), to_currency (3 letters).\n"
        "- For factual lookup, call web_search. For corpus knowledge, call a retriever tool."
    ))

    model = self.llm.bind_tools(self.tools)

    # If supported by ChatGroq in your env, force tool use:
    try:
        model = self.llm.bind_tools(self.tools)#, tool_choice="required")
    except TypeError:
        # If not supported, the strict system prompt above will still push the model to use tools.
        pass

    response = model.invoke([sys, *messages])

    if getattr(self, "debug", False):
        try:
            print("[agent debug] tool_calls:", getattr(response, "tool_calls", None))
            print("[agent debug] content sample:", str(getattr(response, "content", "")))
        except Exception:
            pass

    return {"messages": [response]}


def grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
    """Check if retrieved docs are relevant to the question using Pydantic-validated output."""
    print("--- _grade_documents ---")

    # def _latest_human_text(messages):
    #     for m in reversed(messages):
    #         if isinstance(m, HumanMessage):
    #             return m.content
    #     return ""
    
    # def _latest_tool_result(messages):
    #     for m in reversed(messages):
    #         if isinstance(m, ToolMessage):
    #             return m.content
    #     return ""
    
    # Build structured grader
    grader = self.llm.with_structured_output(RelevanceGrade)

    # Simple prompt (avoid external hub dependency)
    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Here is the retrieved document: \n\n{context}\n\n"
            "Here is the user question: {question}\n"
            "If the document contains keyword(s) or semantic meaning related to the user question,"
            " grade it as relevant. Give a binary score 'yes' or 'no'."
        ),
        input_variables=["context", "question"],
    )

    print(f"_grade documents prompt: {prompt}")

    chain = prompt | grader

    messages = state["messages"]
    question = messages[0].content if messages else ""
    print(f"_grade documents question: {question}")
    last_message = messages[-1] if messages else None
    print(f"_grade documents last message: {last_message}")
    docs_content = getattr(last_message, "content", "") if last_message else ""
    print(f"_grade documents content: {docs_content}")


    # In tool flows, the last message after ToolNode can be a tool result or AI content
    # context_text = format_docs(docs_content)

    print(f"Grading question: {question}")
    print(f"Context sample: {docs_content}")

    scored = chain.invoke({"question": question, "context": docs_content})
    score = (scored.binary_score or "").strip().lower()
    print(f"_grade documents score: {score}")
    return "generate" if score == "yes" else "rewrite"

    # question = _latest_human_text(messages)
    # docs_content = _latest_tool_result(messages)

    # context_text = format_docs(docs_content)

    # print(f"Grading question: {question}")
    # print(f"Context sample: {context_text}")

    # scored = chain.invoke({"question": question, "context": context_text})
    # score = (scored.binary_score or "").strip().lower()
    # print(f"_grade documents score: {score}")

    # return "generate" if score == "yes" else "rewrite"

def generate(self, state: AgentState) -> Dict[str, Any]:
    print("--- _generate ---")
    """RAG answer generation from docs and question."""
    messages = state["messages"]
    question = messages[0].content if messages else ""
    docs_content = getattr(messages[-1], "content", "") if messages else ""

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "Be concise and cite sources with links when available.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )
    # formatted_context = format_docs(docs_content)
    chain = prompt | self.llm | StrOutputParser()
    response = chain.invoke({"context": docs_content, "question": question})
    return {"messages": [AIMessage(content=response)]}

def rewrite(self, state: AgentState) -> Dict[str, Any]:
    print("--- _rewrite ---")
    """Rewrite the question to improve retrieval."""
    messages = state["messages"]
    question = messages[0].content if messages else ""

    rewrite_prompt = (
        "Look at the input and reason about the underlying semantic intent/meaning.\n"
        "Here is the initial question:\n"
        f"{question}\n\n"
        "Formulate an improved question:"
    )

    response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
    return {"messages": [response]}
