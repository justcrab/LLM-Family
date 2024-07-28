from langchain_core.pydantic_v1 import BaseModel
from typing import TypedDict


class TranslationState(TypedDict):
    task: str


class TranslationResponse(BaseModel):
    """TranslationResponse"""
    response: str
