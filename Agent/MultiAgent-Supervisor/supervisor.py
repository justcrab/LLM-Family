from schema import AgentState
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from node import supervisor_node, researcher_node, coder_node, transfer_state_function

graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", transfer_state_function)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

app = graph.compile()

for state in app.stream({"messages": [HumanMessage(content="Write a brief research report on pikas.")]}):
    if "__end__" not in state:
        print(state)