import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("api_key")
base_url = os.environ.get("base_url")
llm = os.environ.get("llm")

if llm in ["deepseek-chat"]:
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
elif llm in ["glm-4"]:
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)


def get_completion(system_message, prompt, model=llm, temperature=1.0):
    response = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
