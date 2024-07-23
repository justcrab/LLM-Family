from openai import OpenAI


class DeepSeek:
    def __init__(self):
        self.client = OpenAI(
            api_key="your api_key",
            base_url="your base_url",
        )

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stop=["Observation:", "Observation:\n"],
        )
        content = response.choices[0].message.content
        return content


if __name__ == "__main__":
    task = """hello?"""
    llm = DeepSeek()
    message = [{"role": "user", "content": task}]
    result = llm.generate(message)
    print(result)
