from abc import abstractmethod


class BaseLLM(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def chat(self, messages, config):
        pass
