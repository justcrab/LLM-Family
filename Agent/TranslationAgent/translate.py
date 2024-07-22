import os
from typing import Union
from dotenv import load_dotenv
from complex_translate import split_chunk_text, multi_chunk_translate
from simple_translate import calculate_num_tokens, one_chunk_translate
from schema import TranslationSingleInput, TranslationMultiInput


load_dotenv()
max_token_per_chunk = int(os.environ.get("max_token_per_chunk"))
llm_name = os.environ.get("llm")


def translate(input: Union[TranslationSingleInput, TranslationMultiInput], mode: str) -> Union[TranslationSingleInput, TranslationMultiInput]:
    num_tokens = calculate_num_tokens(input, llm_name)
    output = ""
    if num_tokens <= max_token_per_chunk:
        print(f"translating text as a single chunk")
        output = one_chunk_translate(input)
    else:
        print(f"translating text as multi chunk")
        new_input = TranslationMultiInput(texts=input.text)
        new_input.texts = split_chunk_text(new_input.texts, max_token_per_chunk, llm_name)
        output = multi_chunk_translate(new_input)
        if mode == "init":
            output = "".join(output.translations)
        elif mode == "reflect":
            output = "".join(output.reflections)
        elif mode == "improve":
            output = "".join(output.improves)
    return output


if __name__ == '__main__':
    input = TranslationSingleInput()
    input.text = "Mamba作为时序状态空间类模型的扛把子之作，一开始就是打着取代Transformer的架构的目的而来的，我们来看一下到底Mamba的魅力在哪里，又是如何能取代Transformer架构呢？"
    output = translate(input, mode="improve")
    print(output)

