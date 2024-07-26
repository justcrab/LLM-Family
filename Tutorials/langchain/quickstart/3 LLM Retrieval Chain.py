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

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# data source and split
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents_split = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=r"F:\Bert")
vector = FAISS.from_documents(documents_split, embeddings)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector.as_retriever(), document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
