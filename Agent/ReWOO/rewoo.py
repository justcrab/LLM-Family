from node import _get_current_task, plan_step, tool_step, solve_step, end_step
from langgraph.graph import START, StateGraph, END
from schema import ReWOO


graph = StateGraph(ReWOO)
graph.add_node("plan", plan_step)
graph.add_node("tool", tool_step)
graph.add_node("solve", solve_step)
graph.add_edge(START, "plan")
graph.add_edge("plan", "tool")
graph.add_conditional_edges("tool", end_step)
graph.add_edge("solve", END)


app = graph.compile()

for s in app.stream({"task": "雷军几几年创办了小米"}):
    print(s)
    print("---")
