import os
from llm import llm
from typing import Literal
from schema import Plan, Replan, Response, PlanExecute
from langgraph.prebuilt import create_react_agent
from langchain.tools.tavily_search import TavilySearchResults
from prompt import agent_prompt, planner_prompt, replanner_prompt

TAVILY_API_KEY = "your api_key"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tools = [TavilySearchResults(max_results=3)]

planer = planner_prompt | llm.with_structured_output(Plan)
agent = create_react_agent(llm, tools, messages_modifier=agent_prompt)
replaner = replanner_prompt | llm.with_structured_output(Replan)


async def plan_step(state: PlanExecute):
    print(f"===================plan=======================")
    plan = await planer.ainvoke({"message": [("user", state["input"])]})
    return {"plan": plan.steps}


async def agent_step(state: PlanExecute):
    print(f"===================agent=======================")
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent.ainvoke({"messages": [("user", task_formatted)]})
    return {"past_steps": (task, agent_response["messages"][-1].content)}


async def replan_step(state: PlanExecute):
    print(f"===================replan=======================")
    replan = await replaner.ainvoke(state)
    if isinstance(replan.plans, Response):
        return {"response": replan.plans.response}
    else:
        return {"plan": replan.plans.steps}


def end_step(state: PlanExecute) -> Literal["__end__", "agent"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"
