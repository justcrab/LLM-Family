from configs.config import Config
from typing import List
from commands.command_registry import CommandRegistry


class AIConfig:
    def __init__(self, ai_name: str = "", ai_role: str = "", ai_goals: List[str] = ""):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.command_registry = None

    @classmethod
    def load(self, config: Config) -> "AIConfig":
        ai_name = config.ai_name
        ai_role = config.ai_role
        ai_goals = config.ai_goals.strip().split("\n")
        return AIConfig(ai_name, ai_role, ai_goals)


