import re
import json


class ToolParser:
    def __init__(self):
        self.tool = ""

    def parse_response(self, llm_result):
        """
        Thought: To find the sum of the two numbers, I need to use the calculator tool.
        Action: calculator
        Action Input: {"a": "12312231", "b": "873862984", "operator": "+"}
        Observation: The calculator returns the result of the addition operation.
        Thought: I now know the final answer.
        Final Answer: 12312231 + 873862984 = 886175215
        """
        # 匹配正则表达式
        action, action_args, answer = None, "", ""
        match_action = re.search(r"Action: (.*)\sAction Input: (\{[^{}]*\})", llm_result)
        match_answer = re.search(r"(.*)Observation:", llm_result, re.S)
        if match_action and match_answer:
            # 提取捕获组
            action = match_action.group(1)
            action_args = json.loads(match_action.group(2))
            answer = match_answer.group(1)
        else:
            answer = llm_result
        return action, action_args, answer


if __name__ == "__main__":
    tool = ToolParser()
    text = """
    Thought: To find the sum of the two numbers, I need to use the calculator tool.
    Action: calculator\nAction Input: {"a": "12312231", "b": "873862984", "operator": "+"}\nObservation: The calculator returns the result of the addition operation.
    Thought: I now know the final answer.
    Final Answer: 12312231 + 873862984 = 886175215
    """
    tool.parse_response(text)

