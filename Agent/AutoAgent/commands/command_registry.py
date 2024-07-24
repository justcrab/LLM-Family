import importlib
from typing import Callable, List, Optional, Any
from commands.command import Command
from commands.command_decorator import COMMAND_IDENTIFIER


class CommandRegistry:
    commands: dict[str, Command] = {}

    def get_command(self, name: str) -> Command | None:
        if name in self.commands:
            return self.commands[name]

    def register_command(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd

    def import_command(self, name):
        module = importlib.import_module(name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, COMMAND_IDENTIFIER) and getattr(attr, COMMAND_IDENTIFIER):
                self.register_command(attr.command)
