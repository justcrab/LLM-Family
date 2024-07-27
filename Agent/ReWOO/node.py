import os, re
from llm import llm
from schema import ReWOO
from langchain.prompts import ChatPromptTemplate
from prompt import plan_prompt, solve_prompt
from langchain.tools.tavily_search import TavilySearchResults

TAVILY_API_KEY = "your api_key"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
search = TavilySearchResults(max_results=3)

# 1 planer
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", plan_prompt)])
planner = prompt_template | llm

# 2 model
model = llm


def plan_step(state: ReWOO):
    task = state["task"]
    result = planner.invoke({"task": task})
    # plan #En tool tool_parameter
    matches = re.findall(regex_pattern, result.content)
    return {"steps": matches, "plan_string": result.content}


def _get_current_task(state: ReWOO):
    if state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_step(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = state["results"] or {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        result = search.invoke(tool_input)
    elif tool == "LLM":
        result = model.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


def solve_step(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state["results"] or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    return {"result": result.content}


def end_step(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"
