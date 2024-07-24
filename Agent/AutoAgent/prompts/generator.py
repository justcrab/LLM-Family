import json
from dataclasses import dataclass
from typing import Optional, Callable, TypedDict, Dict, Any, List
from configs.config import Config
from configs.ai_config import AIConfig
from schema.llm import llm_response_schema
from prompts.prompt import constraints, resources, performance_evaluations, DEFAULT_PROMPT_START
from commands.command_registry import CommandRegistry


class PromptGenerator:
    class Command(TypedDict):
        label: str
        name: str
        params: dict[str, str]
        function: Optional[Callable]

    name: str
    role: str
    goals: list[str]

    constraints: list[str]
    commands: list[Command]
    resources: list[str]
    performance_evaluation: list[str]
    command_registry: CommandRegistry

    def __init__(self):
        self.constraints = []
        self.commands = []
        self.resources = []
        self.performance_evaluation = []
        self.command_registry = None
        self.name = "Bob"
        self.role = "AI"
        self.goals = []

    def _generate_command_string(self, command: Dict[str, Any]) -> str:
        params_string = ", ".join(
            f'"{key}": "{value}"' for key, value in command["params"].items()
        )
        return f'{command["label"]}: "{command["name"]}", params: {params_string}'

    def _generate_numbered_list(self, items: List[Any], item_type="list") -> str:
        if item_type == "commands":
            command_strings = []
            if self.command_registry:
                command_strings += [
                    str(item)
                    for item in self.command_registry.commands.values()
                    if item.enabled
                ]
            # terminate commands is added manually
            command_strings += [self._generate_command_string(item) for item in items]
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(command_strings))
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def _generate_commands(self, config: Config) -> str:
        return (
            "Commands:\n"
            f"{self._generate_numbered_list(self.commands, item_type='commands')}\n\n"
        )

    def add_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)

    def add_command(self, command_label: str, command_name: str,
            params: dict[str, str] = {}, function: Optional[Callable] = None,
    ) -> None:
        command_params = {name: type for name, type in params.items()}
        command: PromptGenerator.Command = {
            "label": command_label,
            "name": command_name,
            "params": command_params,
            "function": function,
        }
        self.commands.append(command)

    def add_resource(self, resource: str) -> None:
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation: str) -> None:
        self.performance_evaluation.append(evaluation)

    @classmethod
    def build_default_generator(self, config: Config):
        prompt_generator = PromptGenerator()
        for constraint in constraints:
            prompt_generator.add_constraint(constraint)

        for resource in resources:
            prompt_generator.add_resource(resource)

        for performance_evaluation in performance_evaluations:
            prompt_generator.add_performance_evaluation(performance_evaluation)
        return prompt_generator

    def generate_prompt(self, config: Config) -> str:
        return (
            f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
            f"{self._generate_commands(config)}"
            f"Resources:\n{self._generate_numbered_list(self.resources)}\n\n"
            "Performance Evaluation:\n"
            f"{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            "Respond with only valid JSON conforming to the following schema: \n"
            f"{json.dumps(llm_response_schema(config))}\n"
        )


def construct_full_prompt(config: Config, ai_config: AIConfig, prompt_generator: PromptGenerator = None) -> str:
    if prompt_generator is None:
        prompt_generator = PromptGenerator.build_default_generator(config)

    prompt_generator.name, prompt_generator.role = ai_config.ai_name, ai_config.ai_role
    prompt_generator.goals, prompt_generator.command_registry = ai_config.ai_goals, ai_config.command_registry

    full_prompt = f"You are {prompt_generator.name}, {prompt_generator.role}\n{DEFAULT_PROMPT_START}\n\nGOALS:\n"
    for i, goal in enumerate(prompt_generator.goals):
        full_prompt += f"{i + 1}. {goal}\n"
    full_prompt += f"\n\n{prompt_generator.generate_prompt(config)}"
    return full_prompt
