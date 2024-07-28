LangGraph 是一个用于构建具有状态的、多参与者应用程序的库，它利用大型语言模型（LLMs）来创建代理和多代理工作流程。与其他LLM框架相比，它提供了以下核心优势：循环性、可控性和持久性。LangGraph 允许您定义涉及循环的流程，这对于大多数代理架构至关重要，使其区别于基于DAG（有向无环图）的解决方案。作为一个非常底层的框架，它提供了对应用程序流程和状态的精细控制，这对于创建可靠的代理至关重要。此外，LangGraph 内置了持久性功能，支持高级的人机交互和记忆特性。

LangGraph比起Langchain的chain的实现，LangGraph将Langchian中顺序或者路由的chain变成Graph实现，更加优雅和方便调试。

LangGraph示例的构建也非常的简单：

* 1 Prompt | Model | Output_parser
* 2 StateSchema
* 3 Graph Defination
* 4 Graph Node and Edge
* 5 Graph Compile
* 6 Graph Execute

按照顺序本人定义了一个模版，非常适合上述的构建过程：

* 1 llm.py -> LLM的调用定义
* 2 prompt.py -> 各个LLM的Prompt的定义
* 3 schema.py -> Graph的State定义，以及各个LLM的output response的定义
* 4 node.py -> 将各个LLM封装为Graph中的node
* 5 graph.py -> Graph的定义；Node和Edge的定义；Graph编译；Grpah执行

接下来我们介绍一个例子来展开将LangGraph的使用，中间会穿插需要注意的细节：

需求：我们将创建两个顺序的LLM，第一个LLM的功能是将中文翻译为英文，第二个LLM的功能是将英文翻译为中文。

### 1 llm.py

本教程全程使用DeepSeek API（与OpenAI的调用方式基本一致）

```python
from langchain_openai import ChatOpenAI
DEEPSEEK_API_KEY = "your api_key"
BASE_URL = "https://api.deepseek.com"

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

```

### 2 prompt.py

由于node使用LLMChain，故Prompt需要使用ChatPrompt

```python
from langchain.prompts import ChatPromptTemplate


translation_first_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a great translator, and please translate chinese into english."),
    ("user", "{task}")
])

translation_second_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a great translator, and please translate english into chinese."),
    ("user", "{task}")
])
```

### 3 schema.py

GraphState的定义是必备的，Graph在node之间的信息是通过GraphState传递。

* GraphState的定义：定义了Graph中node的信息流动，经典实现一般为task和result，此处由于输入输出均为1个，则省略了result。
* LLM Response的定义：规范LLM返回数据的数据格式

```python
from langchain_core.pydantic_v1 import BaseModel
from typing import TypedDict


class TranslationState(TypedDict):
    task: str


class TranslationResponse(BaseModel):
    """TranslationResponse"""
    response: str

```

### 4 node.py

该文件定义了两个翻译LLM Chain，同时将其包装为node函数。

```python
from llm import llm
from prompt import translation_first_prompt, translation_second_prompt
from schema import TranslationState, TranslationResponse


translation_first_chain = translation_first_prompt | llm.with_structured_output(TranslationResponse)
translation_second_chain = translation_second_prompt | llm.with_structured_output(TranslationResponse)


def translation_first_node(state: TranslationState):
    llm_response = translation_first_chain.invoke(state["task"])
    return {"task": llm_response.response}


def translation_second_node(state: TranslationState):
    llm_response = translation_second_chain.invoke(state["task"])
    return {"task": llm_response.response}

```

### 5 graph.py

Graph的定义；Node和Edge的定义；Graph编译；Grpah执行

```python
from langgraph.graph import START, StateGraph, END
from schema import TranslationState
from node import translation_first_node, translation_second_node


graph = StateGraph(TranslationState)

graph.add_node("translation_first", translation_first_node)
graph.add_node("translation_second", translation_second_node)

graph.add_edge(START, "translation_first")
graph.add_edge("translation_first", "translation_second")
graph.add_edge("translation_second", END)

app = graph.compile()

for state in app.stream({"task": "举头望明月，低头思故乡"}):
    print(state)
    print("---")
```

**最终结果：**

```python
{'translation_first': {'task': 'Lifting my head to gaze at the bright moon, lowering it to think of my hometown.'}}
---
{'translation_second': {'task': '抬头望明月，低头思故乡。'}}
---
```

