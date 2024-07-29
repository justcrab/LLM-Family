import os
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_API_KEY"] = "your api_key"

example_inputs = [
  ("What is the largest mammal?", "The blue whale"),
  ("What do mammals and birds have in common?", "They are both warm-blooded"),
  ("What are reptiles known for?", "Having scales"),
  ("What's the main characteristic of amphibians?", "They live both in water and on land"),
]

client = Client()
dataset_name = "Elementary Animal Questions"

dataset = client.create_dataset(dataset_name, description="Questions and answers about animal phylogenetics.")
for input, output in example_inputs:
    client.create_example(
        inputs={"question": input},
        outputs={"answer": output},
        metadata={"source": "Wikidata"},
        dataset_id=dataset.id
    )
