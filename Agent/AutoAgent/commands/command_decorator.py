import functools
from configs.config import Config
from commands.command import Command, CommandParameter, CommandParameterSpec
from typing import Any, Callable, Optional, TypedDict

COMMAND_IDENTIFIER = "Command"


def command(
    name: str,
    description: str,
    parameters: dict[str, CommandParameterSpec],
    enabled: bool | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""

    def decorator(func: Callable[..., Any]) -> Command:
        typed_parameters = [
            CommandParameter(
                name=param_name,
                description=parameter.get("description"),
                type=parameter.get("type", "string"),
                required=parameter.get("required", False),
            )
            for param_name, parameter in parameters.items()
        ]
        cmd = Command(
            name=name,
            description=description,
            method=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        wrapper.command = cmd

        setattr(wrapper, COMMAND_IDENTIFIER, True)

        return wrapper

    return decorator