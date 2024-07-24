from base import BaseLLM
from openai import OpenAI


class DeepSeek(BaseLLM):
    def __init__(self, config):
        super().__init__()
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages, config):
        response = self.client.chat.completions.create(
            model="deepseek-chat", messages=messages
        )
        content = response.choices[0].message.content
        return content

