import os
from openai import OpenAI
from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

api_key = "your api_key"
base_url = "https://api.deepseek.com"
os.environ["http.proxy"] = "http://127.0.0.1:33210"
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_API_KEY"] = "your api_key"
client = Client()


# 1 create a task
prompt = ChatPromptTemplate.from_messages([
  ("system", "Please review the user query below and determine if it contains any form of toxic behavior, such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, and 'Not toxic' if it doesn't."),
  ("user", "{text}")
])
chat_model = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url=base_url)
output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser


# 2 create a dataset
# already create a dataset, so here do not do that
# examples = [
#     ("Shut up, idiot", "Toxic"),
#     ("You're a wonderful person", "Not toxic"),
#     ("This is the worst thing ever", "Toxic"),
#     ("I had a great day today", "Not toxic"),
#     ("Nobody likes you", "Toxic"),
#     ("This is unacceptable. I want to speak to the manager.", "Not toxic"),
# ]
dataset_name = "Toxic Queries"
# dataset = client.create_dataset(dataset_name=dataset_name)
# inputs, outputs = zip(
#     *[({"text": text}, {"label": label}) for text, label in examples]
# )
# client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)


# 3 create a evaluator
def evaluate_text(run: Run, example: Example) -> dict:
    score = run.outputs.get("output") == example.outputs.get("label")
    return {"score": score, "key": "correct_label"}


# 4 evaluate
results = evaluate(
    # 4.1 task inputs and outputs
    chain.invoke,
    # 4.2 dataset name
    data=dataset_name,
    # 4.3 evaluate
    evaluators=[evaluate_text],
    experiment_prefix="Toxic Queries",
    description="Testing the baseline system.",  # optional
)
