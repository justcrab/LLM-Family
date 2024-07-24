import os
import json
from configs.config import Config
from typing import Any
LLM_DEFAULT_RESPONSE_FORMAT = "llm_response"


def llm_response_schema(
    config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        json_schema = json.load(f)
    return json_schema
