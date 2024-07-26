DEEPSEEK_API_KEY="your api_key"
BASE_URL="https://api.deepseek.com"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

res = chain.invoke({"input": "how can langsmith help with testing?"})
print(res)