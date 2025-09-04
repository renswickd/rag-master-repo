from typing_extensions import TypedDict
from typing import Annotated, Sequence, Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field, confloat, conint, constr
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Conversation state; add_messages appends rather than replaces
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RelevanceGrade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class WebSearchInput(BaseModel):
    query: constr(min_length=1)  # type: ignore
    num: conint(ge=1, le=10) = 5  # type: ignore


class CurrencyConvertInput(BaseModel):
    amount: confloat(gt=0)  # type: ignore
    from_currency: constr(min_length=3, max_length=3)  # type: ignore
    to_currency: constr(min_length=3, max_length=3)  # type: ignore
