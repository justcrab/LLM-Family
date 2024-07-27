import os
from llm import llm
from prompt import supervisor_prompt, researcher_prompt, coder_prompt
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from schema import SupervisorResponse, AgentState
from langchain_core.messages import HumanMessage
from langgraph.graph import END


TAVILY_API_KEY = "your api_key"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
researcher_tools = [TavilySearchResults()]
coder_tools = [PythonREPLTool()]
supervisor_chain = supervisor_prompt | llm.with_structured_output(SupervisorResponse)
researcher_agent = create_openai_tools_agent(
    llm=llm, prompt=researcher_prompt, tools=researcher_tools
)
coder_agent = create_openai_tools_agent(llm=llm, prompt=coder_prompt, tools=coder_tools)
researcher_agent = AgentExecutor(agent=researcher_agent, tools=researcher_tools)
coder_agent = AgentExecutor(agent=coder_agent, tools=coder_tools)


def supervisor_node(state: AgentState):
    print("==========================supervisor_node==========================")
    print(state)
    response = supervisor_chain.invoke(state["messages"])
    return {"next": response.next}


def researcher_node(state: AgentState):
    print("==========================researcher_node==========================")
    response = researcher_agent.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response["output"])]}


def coder_node(state: AgentState):
    print("==========================coder_node==========================")
    response = coder_agent.invoke({"messages": state["messages"]})
    print(f"response is {response}")
    return {"messages": [HumanMessage(content=response["output"])]}


def transfer_state_function(state: AgentState):
    if state["next"] == "Researcher":
        return "researcher"
    elif state["next"] == "Coder":
        return "coder"
    elif state["next"] == "Finished":
        return END
