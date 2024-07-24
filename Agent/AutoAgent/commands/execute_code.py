from commands.command_decorator import command


@command(
    name="calculate",
    description="a calculate to calculate",
    parameters={
        "a": {"type": "int", "description": "first element", "required": True},
        "b": {"type": "int", "description": "second element", "required": True}
    }
)
def calculate(a, b):
    return (a + b)
