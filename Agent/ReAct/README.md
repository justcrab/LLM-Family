这是一个基于ReACT框架的最小Agent实现。

ReACT Agent的必备模块：

* 1 Model I/O
  * 1.1 llm
  * 1.2 Prompt
  * 1.3 Parser
* 2 WorkFlow -> Mrkl System
* 3 Tools

### 1 Model I/O

#### 1.1 LLM

使用了Deepseek-chat的API。

#### 1.2 Prompt

经典的ReACT框架的Prompt，

* 1 先告知Tools的name和Tools parameter，
* 2 Thought, Action, Action Input和observation的重复迭代直至finnal answer.

```
system_prompt = """Answer the following questions as best you can. You have access to the following tools: `
<tool_list>`\n\n"""

instruct_prompt = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [<tool_names>]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Notes:
    1 Use tools only when necessary, Do not use tools if the content and tool description do not match
    2 Please reply in Chinese
    3 A question may require the use of multiple tools or multiple uses of one tool to obtain an answer 
        eg:请告诉我小米su7和特斯拉model 3哪个配置和性价比更高？
        补充：该问题需要使用search工具，需要同时搜索小米su7和特斯拉model 3两个词条

History: <history>

"""
```

#### 1.3 Parser

从LLM的回答中解析Acton和Action Input来获取tool name和tool parameter。

```
Action: calculator
Action Input: {"a": "12312231", "b": "873862984", "operator": "+"}
```

### 2 WorkFlow

while循环下，组织Agent按照如下WorkFlow进行工作：

* 1 构造Prompt，LLM回复llm_result；
* 2 从LLM回复llm_result中解析出Action和Action Input后的内容得到tool name和tool 参数；
* 3 tool根据参数执行得到答案，作为Observation的答案
* 4 构造Prompt，重复1-3步骤，直到在第2步骤中解析出final answer，结束循环，得到答案。

### 3 Tools

提供3部分有关tool的信息：

* 1 tool name
* 2 tool parameter
* 3 tool defination

eg:

```
tool_names = ["calculator"]
tools_list = [
	{
        "name": "calculator",
        "description": "a calculator to calculate math problem of two element",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "the first element input"},
                "b": {"type": "string", "description": "the second element input"},
                "operator": {
                    "type": "string",
                    "description": "the operator to calculate the first element and second element",
                },
            },
        },
    },
]
def calculator(a, b, operator):
    if operator == "+":
        return int(a) + int(b)
    elif operator == "-":
        return int(a) - int(b)
    elif operator == "*":
        return int(a) * int(b)
    elif operator == "/":
        return int(a) / int(b)
```

### 4 使用说明

* 1 命令运行：python agent.py
* 2 前端运行：python app.py