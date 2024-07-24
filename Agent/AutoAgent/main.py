from configs.config import Config
from configs.ai_config import AIConfig
from prompts.generator import construct_full_prompt
from commands.command_registry import CommandRegistry

COMMAND_CATEGORIES = [
    "commands.execute_code"
]


if __name__ == "__main__":
    # 1 配置config和ai_config
    config = Config()
    ai_config = AIConfig.load(config)
    # 2 配置工具命令仓库
    command_registry = CommandRegistry()
    for command_category in COMMAND_CATEGORIES:
        command_registry.import_command(command_category)
    ai_config.command_registry = command_registry
    # 3 构建Agent Prompt
    full_prompt = construct_full_prompt(config, ai_config)
    # 4 开启Agent WorkFlow

    print(full_prompt)