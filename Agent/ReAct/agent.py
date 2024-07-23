from prompt import MrklPrompt
from llm import DeepSeek
from tools import *
from outparser import ToolParser


class ReAct:
    def __init__(self):
        self.prompt_generator = MrklPrompt()
        self.llm = DeepSeek()
        self.parser = ToolParser()
        self.tool_names = self.get_tool_names()
        self.tool_list = self.get_tool_list()
        self.history = []

    def run(self, task):
        messages = []
        system_prompt = self.prompt_generator.init_prompt(task, self.tool_names)
        user_prompt = {"role": "user", "content": "Question: " + task + "\n"}
        messages.append(system_prompt)
        messages.append(user_prompt)
        llm_result, answer_result, exec_result = "", "", ""
        idx = 0
        while True:
            idx += 1
            messages = self.prompt_generator.generate(
                answer_result, exec_result, messages
            )
            output = self.print_messages(messages)
            print(f"idx is {idx}\n{output}")
            # 1 prompt / LLM Generate
            try:
                llm_result = self.llm.generate(messages)
            except RuntimeError as e:
                return [{"exec_result": e}]
            # 2 Answer? / Action / Action Input
            # 第一次不會解析Answer，第二次即之後都會解析得到Answer
            try:
                action, action_args, answer_result = self.parser.parse_response(
                    llm_result
                )
            except ValueError as e:
                return [{"exec_result": e}]
            yield answer_result
            if action is None:
                messages = self.prompt_generator.generate(answer_result, "", messages)
                output = self.print_messages(messages)
                print(f"idx is {idx+1}\n{output}")
                break
            # 3 Observation
            if action in self.tool_names:
                exec_result = str(eval(action)(**action_args))

    def get_tool_names(self):
        return tool_names

    def get_tool_list(self):
        return tools_list

    def print_messages(self, messages):
        prompt = ""
        prompt += messages[0]["content"]
        prompt += messages[-1]["content"]
        return prompt


if __name__ == "__main__":
    task = (
        "請問12182342312+12491771781等於多少?同時我想問一下今天杭州西湖區的天氣怎麼樣？"
    )
    agent = ReAct()
    print(f"agent.run(task) is {agent.run(task)}")
