from typing import List, TypedDict
from langchain_core.messages import BaseMessage


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List[BaseMessage]
    results: dict
    result: str
