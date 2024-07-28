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