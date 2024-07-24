AutoGPT大火已经一年多了，其版本更新迭代也很快，由于疯狂的版本迭代，对于我们去洞悉AutoGPT的原理实在是不利。

本项目从AutoGPT V0.4.4源码出发进行魔改，魔改的原则和目的如下：

* 1 原则：保持AutoGPT的核心功能不变
* 2 目的：
  * 2.1 重新调整项目架构，包括代码和文件位置，更加方便大家看懂源码，同时更符合直觉。
  * 2.2 加入对国产大模型的支持

### 1 项目简析

从仓库的另外Agent项目ReACT和TranslationAgent我们已经说过，理解Agent主要是需要理解Agent Prompt和Agent WorkFlow, Prompt和构建和WorkFlow的定义决定了Agent的下限。

#### 1.1 Agent Prompt

AutoGPT的Agent Prompt的大概组织如下：

* 1 AIAgent profile
* 2 GOALS
* 3 Constraints
* 4 Commands
* 5 Resources
* 6 Performance Evaluation
* 7 Response Constraint

```
You are CrabGPT, A coder helper
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:
1. I want to get a code about GNN with pyg for node classification


Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed below e.g. command_name

Commands:
1. calculate: a calculate to calculate, params: (a: int, b: int)

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every commands has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

Respond with only valid JSON conforming to the following schema: 
{"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {"thoughts": {"type": "object", "properties": {"text": {"type": "string", "description": "thoughts"}, "reasoning": {"type": "string"}, "plan": {"type": "string", "description": "- short bulleted\n- list that conveys\n- long-term plan"}, "criticism": {"type": "string", "description": "constructive self-criticism"}, "speak": {"type": "string", "description": "thoughts summary to say to user"}}, "required": ["text", "reasoning", "plan", "criticism", "speak"], "additionalProperties": false}, "command": {"type": "object", "properties": {"name": {"type": "string"}, "args": {"type": "object"}}, "required": ["name", "args"], "additionalProperties": false}}, "required": ["thoughts", "command"], "additionalProperties": false}

```

#### 1.2 Agent WorkFlow

wating ...

### 2 项目结构

本项目从更方便理解源码的角度入手，将项目分为如下：

* agent
  * 主要定义Agent WorkFlow
* commands
  * 主要对Tools的定义
* configs
  * 基本配置config和Agent 配置ai_config
* llm
  * 国产大模型调用
* log
  * 日志打印
* memory
  * Agent历史记忆
* prompts
  * Agent Prompt构造
* schema
  * llm的回复规范
* main.py
  * 项目启动文件

### 3 具体细节

#### 3.1 Prompts

prompt.py: 存放AutoGPT的default prompt

generator.py: Prompt 产生器，基于prompt.py中的default prompt结合以下信息构造init prompt

* 1 configs/ai_config.py -> ai_config中的Agent profile(ai_name, ai_role, ai_goals);
* 2 commands/command_registry.py -> commad registry中的command资源;
* 3 prompts/prompt.py -> default prompt中的Constraints, Commands, Resources, Performance Evaluation;
* 4 schema/llm.py -> llm response schema;

#### 3.2 llm

base.py: 定义一个base类，自定义一个抽象函数chat(llm交互接口)

llm.py: 国产大模型API接口

#### 3.3 configs

config.py: 项目基本配置（包括api_key, base_url等），从.env环境变量中读取。

ai_config.py: 自定义Agent的配置

#### 3.4 commands

command.py: 定义了Agent Command的函数参数CommandParameter以及Command。

command_decorator.py: 存放Agent Command的装饰器，功能为将函数包装为Command。

command_registry.py: 将全部的Agent Command集中于registry进行管理。

其他均为实际Command函数。

#### 3.5 schema

llm.py: 定义了llm回复模版



wating ...