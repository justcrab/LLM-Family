docs: https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_llm_application#run-an-evaluation

At a high-level, the evaluation process involves the following steps:

1. Define your LLM application or target task.
2. Creating or selecting a dataset to evaluate your LLM application. Your evaluation criteria may or may not require expected outputs in the dataset.
3. Configuring evaluators to score the outputs of your LLM application, sometimes against expected outputs.
4. Running the evaluation and viewing the results.

**看完本章的示例我们可以很清楚的看到LangSmith的Evaluation，重点在于Dataset的构建；Task的构建；Evaluator的构建过程。**

本章节使用判别是否有毒的例子来展示LangSmith的Evaluation过程。

## 1 Run an evaluation

使用原生的OPENAI的llm调用形式。

### 1.1: Define your target task

```python
from langsmith import traceable, wrappers
from openai import Client

openai = wrappers.wrap_openai(Client())

@traceable
def label_text(text):
    messages = [
        {
            "role": "system",
            "content": "Please review the user query below and determine if it contains any form of toxic behavior, such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, and 'Not toxic' if it doesn't.",
        },
        {"role": "user", "content": text},
    ]
    result = openai.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo", temperature=0
    )
    return result.choices[0].message.content
```

### 1.2 Create or select a dataset

```python
from langsmith import Client

client = Client()

# Create a dataset
examples = [
    ("Shut up, idiot", "Toxic"),
    ("You're a wonderful person", "Not toxic"),
    ("This is the worst thing ever", "Toxic"),
    ("I had a great day today", "Not toxic"),
    ("Nobody likes you", "Toxic"),
    ("This is unacceptable. I want to speak to the manager.", "Not toxic"),
]

dataset_name = "Toxic Queries"
dataset = client.create_dataset(dataset_name=dataset_name)
inputs, outputs = zip(
    *[({"text": text}, {"label": label}) for text, label in examples]
)
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
```

### 1.3 Configure evaluators to score the outputs

correct_label(root_run: Run, example: Example)前者root_run为task的输出，example是数据集的ground truth。

```python
from langsmith.schemas import Example, Run

def correct_label(root_run: Run, example: Example) -> dict:
    score = root_run.outputs.get("output") == example.outputs.get("label")
    return {"score": int(score), "key": "correct_label"}
```

### 1.4 Run the evaluation and view the results

请注意此处lambda inputs: label_text(inputs["text"])中的输入输出是task的输入和输出。

```python
from langsmith.evaluation import evaluate

dataset_name = "Toxic Queries"

results = evaluate(
    lambda inputs: label_text(inputs["text"]),
    data=dataset_name,
    evaluators=[correct_label],
    experiment_prefix="Toxic Queries",
    description="Testing the baseline system.",  # optional
)
```

### 1.5 Result

View the evaluation results for experiment: 'Toxic Queries-f6c25e04' at:
https://smith.langchain.com/o/680c5cba-08d6-5fdb-b6c2-281af9f74ed1/datasets/75a837ba-f4c7-40b4-b57c-48cd5427d674/compare?selectedSessions=89164a94-43c1-4dc3-95e9-9c6856a0ed63

## 2 Evaluation a langchain runnable 

使用Langchain的调用形式。

### 2.1: Define your target task

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
  ("system", "Please review the user query below and determine if it contains any form of toxic behavior, such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, and 'Not toxic' if it doesn't."),
  ("user", "{text}")
])
chat_model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser
```

### 2.2 Create or select a dataset

```python
from langsmith import Client

client = Client()

# Create a dataset
examples = [
    ("Shut up, idiot", "Toxic"),
    ("You're a wonderful person", "Not toxic"),
    ("This is the worst thing ever", "Toxic"),
    ("I had a great day today", "Not toxic"),
    ("Nobody likes you", "Toxic"),
    ("This is unacceptable. I want to speak to the manager.", "Not toxic"),
]

dataset_name = "Toxic Queries"
dataset = client.create_dataset(dataset_name=dataset_name)
inputs, outputs = zip(
    *[({"text": text}, {"label": label}) for text, label in examples]
)
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
```

### 2.3 Configure evaluators to score the outputs

correct_label(root_run: Run, example: Example)前者root_run为task的输出，example是数据集的ground truth。

```python
from langsmith.schemas import Example, Run

def correct_label(root_run: Run, example: Example) -> dict:
    score = root_run.outputs.get("output") == example.outputs.get("label")
    return {"score": int(score), "key": "correct_label"}
```

### 2.3 Run the evaluation and view the results

请注意此处lambda inputs: label_text(inputs["text"])中的输入输出是task的输入和输出。

```python
from langsmith.evaluation import evaluate

results = evaluate(
    chain.invoke,
    data=dataset_name,
    evaluators=[correct_label],
    experiment_prefix="Toxic Queries",
)
```

### 2.5 Result

View the evaluation results for experiment: 'Toxic Queries-2e6d0212' at:
https://smith.langchain.com/o/680c5cba-08d6-5fdb-b6c2-281af9f74ed1/datasets/75a837ba-f4c7-40b4-b57c-48cd5427d674/compare?selectedSessions=0ae71b4c-2fa2-4375-9558-4d6e29892989



至于如何使用特定版本、子集、split的数据集，如何使用summary evaluator请查看https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_llm_application#run-an-evaluation