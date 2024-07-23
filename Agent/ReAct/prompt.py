system_prompt = """Answer the following questions as best you can. You have access to the following tools: `
<tool_list>`\n\n"""

instruct_prompt = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [<tool_names>]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Notes:
    1 Use tools only when necessary, Do not use tools if the content and tool description do not match
    2 Please reply in Chinese
    3 A question may require the use of multiple tools or multiple uses of one tool to obtain an answer 
        eg:请告诉我小米su7和特斯拉model 3哪个配置和性价比更高？
        补充：该问题需要使用search工具，需要同时搜索小米su7和特斯拉model 3两个词条

History: <history>

"""

user_template = """Question: <user_input>\n"""

exec_template = """Observation: <exec_result>\n"""

assistant_template = ""

from tools import tools_list


class MrklPrompt:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.prompt = ""

    def get_tool_list(self, tool_names):
        return tools_list

    def get_history(self):
        history_str = ""
        for i in range(len(self.history)):
            history_str += self.history[i]["content"] + "\n"
        return history_str

    def _init_prompt(self, task, tool_names):
        # system_prompt + instruct_prompt + knowledge_prompt + history prompt + user/assistant prompt
        # system_prompt + instruct_prompt + knowledge_prompt + history prompt
        self.prompt += system_prompt + instruct_prompt
        tools_list = self.get_tool_list(tool_names)
        tool_names = ",".join([f"'{tool}'" for tool in tool_names])
        # user/assistant prompt
        self.prompt = self.prompt.replace("<tool_names>", tool_names).replace(
            "<tool_list>", str(tools_list)
        )

    def init_prompt(self, task, tool_names):
        self._init_prompt(task, tool_names)
        system_message = {"role": "system", "content": f"{self.prompt}"}
        return system_message

    def generate(self, answer_result, exec_result, messages):
        if len(answer_result) != 0:
            messages[-1]["content"] += answer_result
        if len(exec_result) != 0:
            exec_result = exec_template.replace("<exec_result>", str(exec_result))
            messages[-1]["content"] += exec_result
        return messages


if __name__ == "__main__":
    prompt = MrklPrompt()
    print(prompt.init_prompt("hello", ["we", "good"]))
    # print(prompt.get_history())
