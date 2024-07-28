from langchain.prompts import ChatPromptTemplate


translation_first_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a great translator, and please translate chinese into english."),
    ("user", "{task}")
])

translation_second_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a great translator, and please translate english into chinese."),
    ("user", "{task}")
])