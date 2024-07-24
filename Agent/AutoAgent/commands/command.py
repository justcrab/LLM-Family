import importlib
from typing import Callable, List, Optional, Any, TypedDict
from dataclasses import dataclass


class CommandParameterSpec(TypedDict):
    type: str
    description: str
    required: bool


@dataclass
class CommandParameter:
    name: str
    type: str
    description: str
    required: bool

    def __repr__(self):
        return f"CommandParameter('{self.name}', '{self.type}', '{self.description}' , '{self.required}"


class Command:
    def __init__(self, name: str, description: str, method: Callable,
                 parameters: List[CommandParameter], enabled: bool, disabled_reason: Optional[str] = None):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason

    def __call__(self, *args, **kwargs) -> Any:
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.type if param.required else f'Optional[{param.type}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description}, params: ({', '.join(params)})"
