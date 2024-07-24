import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    ai_name: str = os.environ.get("ai_name")
    ai_role: str = os.environ.get("ai_role")
    ai_goals: str = os.environ.get("ai_goals")

    api_key: str = os.environ.get("api_key")
    base_url: str = os.environ.get("base_url")

