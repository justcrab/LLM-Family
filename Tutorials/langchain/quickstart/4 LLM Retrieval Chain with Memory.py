DEEPSEEK_API_KEY="your api_key"
BASE_URL="https://api.deepseek.com"

import warnings
warnings.filterwarnings(action="ignore")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "根据上面的对话，生成一个搜索查询来获取与对话相关的信息")
])

# data source and split
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents_split = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=r"F:\Bert")
vector = FAISS.from_documents(documents_split, embeddings)
retriever = vector.as_retriever()

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
prompt = ChatPromptTemplate.from_messages([
    ("system", "根据下面的上下文回答用户的问题：\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})
print(response)
