DEEPSEEK_API_KEY = "your api_key"
BASE_URL = "https://api.deepseek.com"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
