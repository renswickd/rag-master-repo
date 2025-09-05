import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from shared.configs.static import (
    GROQ_MODEL as DEFAULT_GROQ_MODEL,
    AGENTIC_RAG_TYPE,
)
from shared.tools.web_search_tool import serp_search
from shared.tools.currency_converter_tool import exchangerate_converter
from shared.tools.agentic_retriever_tool import make_agentic_retriever_tool
from projects.retriever.agentic_rag_retriever import AgenticRAGRetriever
from shared.components.agentic_rag_nodes import agent, grade_documents, rewrite, generate
from shared.components.agentic_rag_states import AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

class AgenticRAGReActPipeline:

    def __init__(
        self,
        data_dir: str,
        groq_model: Optional[str] = None,
        temperature: float = 0.2,
        debug: bool = False,
        rag_type: str = AGENTIC_RAG_TYPE,
        extra_tools: Optional[List[Any]] = None,
    ):
        model_name = groq_model or DEFAULT_GROQ_MODEL
        print(f"Using model: {model_name}")
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.debug = debug

        # Initialize the resume retriever and corresponding tool
        self.retriever = AgenticRAGRetriever(data_dir=data_dir, rag_type=rag_type)
        resume_tool = make_agentic_retriever_tool(self.retriever)

        # Final tool list: resume retriever + built-ins + optional extras
        self.tools = [resume_tool, serp_search, exchangerate_converter]
        if extra_tools:
            self.tools.extend(list(extra_tools))

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Bind node functions to this instance via closures
        workflow.add_node("agent", lambda state: agent(self, state))
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("rewrite", lambda state: rewrite(self, state))
        workflow.add_node("generate", lambda state: generate(self, state))
        # Node to handle conversations that try to bypass tools
        def _restricted(_: AgentState) -> Dict[str, Any]:
            print("--- _restricted ---")
            msg = (
                "This app only supports: document retrieval, web search, and currency conversion. "
                "Your request appears outside this scope. Please use one of the supported capabilities."
            )
            return {"messages": [AIMessage(content=msg)]}

        workflow.add_node("restricted", _restricted)

        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                # If the agent called any tool, execute it
                "tools": "retrieve",
                # If the agent tried to answer directly, treat as out of scope
                END: "restricted",
            },
        )

        workflow.add_conditional_edges("retrieve", lambda state: grade_documents(self, state))
        workflow.add_edge("generate", END)
        workflow.add_edge("restricted", END)
        workflow.add_edge("rewrite", "agent")

        return workflow.compile()

    def answer(self, question: str) -> str:
        result = self.graph.invoke({"messages": [HumanMessage(content=question)]})
        # Extract last assistant message content
        msgs = result.get("messages", [])
        if not msgs:
            return ""
        last = msgs[-1]
        try:
            return getattr(last, "content", str(last))
        except Exception:
            return str(last)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        return self.retriever.get_collection_info()
        
if __name__ == "__main__":

    ## Test the flow (ensure data exists at data/source_data/agentic_rag)
    graph = AgenticRAGReActPipeline(data_dir="data/source_data/agentic-rag", debug=True)

    # answer = graph.answer("What is 1 EUR in INR")
    # answer = graph.answer("generate a cover letter for my personal resume")
    answer = graph.answer("Summarize the candidate's AWS experience")
    print("\n\nAnswer: ", answer)


    

        
