from typing import List, TypedDict, Literal
from langchain_core.pydantic_v1 import BaseModel, Field


class AgentState(TypedDict):
    messages: List[str]
    next: Literal["Researcher", "Coder", "Finished"]


class SupervisorResponse(BaseModel):
    """Supervisor的返回值"""
    next: Literal["Researcher", "Coder", "Finished"]

