import time
import gradio as gr
from agent import ReAct

agent = ReAct()


def slow_echo(message, history):
    return_message = ""
    result = agent.run(message)
    for res in result:
        return_message = return_message + res
        yield return_message


gr.ChatInterface(slow_echo).launch()
# 请问23232*323232等于多少？同时告诉我今天浙江杭州西湖区的天气怎么样？
