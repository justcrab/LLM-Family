DEEPSEEK_API_KEY="your api_key"
BASE_URL="https://api.deepseek.com"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)


res = llm.invoke("how can langsmith help with testing?")
print(res.content)