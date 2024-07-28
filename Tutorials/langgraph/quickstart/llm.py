from langchain_openai import ChatOpenAI
DEEPSEEK_API_KEY = "your api_key"
BASE_URL = "https://api.deepseek.com"

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
