"""
1 Initialize the model and tools.
2 Initialize graph with state.
3 Define graph nodes.
4 Define entry point and graph edges.
5 Compile the graph.
6 Execute the graph.

"""
import asyncio
from schema import PlanExecute
from node import plan_step, agent_step, replan_step, end_step
from langgraph.graph import StateGraph, START


# 1 Initialize the tools and models.


# 2 Initialize graph with state.
workflow = StateGraph(PlanExecute)

# 3 Define graph nodes.
workflow.add_node("planer", plan_step)
workflow.add_node("agent", agent_step)
workflow.add_node("replaner", replan_step)

# 4 Define entry point and graph edges.
workflow.add_edge(START, "planer")
workflow.add_edge("planer", "agent")
workflow.add_edge("agent", "replaner")
workflow.add_conditional_edges("replaner", end_step)

# 5 Compile the graph.
app = workflow.compile()

# 6 Execute the graph.
config = {"recursion_limit": 50}
inputs = {"input": "请告诉我雷军成为金山办公CEO的年份?"}


async def main():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    asyncio.run(main())
