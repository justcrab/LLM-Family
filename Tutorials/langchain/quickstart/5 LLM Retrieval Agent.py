DEEPSEEK_API_KEY="your api_key"
BASE_URL="https://api.deepseek.com"
TAVILY_API_KEY="your tavily_api_key"
EMBEDDING_MODEL_NAME=""r"F:\Bert"

import os
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# data load / split
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents_split = text_splitter.split_documents(documents)
# embedding / embedding db / retriever
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector = FAISS.from_documents(documents_split, embeddings)
retriever = vector.as_retriever()

# tool
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
retriever_tool = create_retriever_tool(retriever, "langsmith_search", "搜索与LangSmith相关的信息。有关LangSmith的任何问题，您必须使用此工具！")
search_tool = TavilySearchResults(max_results=3)
tools = [retriever_tool, search_tool]

# agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
res = agent_executor.invoke({"input": "langsmith如何帮助测试？"})
print(res)