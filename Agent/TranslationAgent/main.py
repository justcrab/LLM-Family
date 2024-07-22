from .translate import translate
from schema import TranslationSingleInput


if __name__ == "__main__":
    input = TranslationSingleInput()
    input.text = "Mamba作为时序状态空间类模型的扛把子之作，一开始就是打着取代Transformer的架构的目的而来的，我们来看一下到底Mamba的魅力在哪里，又是如何能取代Transformer架构呢？"
    output = translate(input, mode="improve")
    print(output.improve)
