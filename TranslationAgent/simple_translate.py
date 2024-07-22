from get_completion import get_completion
from schema import TranslationSingleInput


def one_chunk_initial_translation(input: TranslationSingleInput) -> TranslationSingleInput:
    source_language, target_language, text = input.source_language, input.target_language, input.text
    system_message = f"if you are an expert linguist, specializing in translation from {source_language} to {target_language}."
    translation_message = f"""This is an {source_language} to {target_language} translation, please provide the {target_language}
     for this text. Do not provide any explanations or text apart from the translation.
     {source_language}: {text}
     
     {target_language}:"""
    input.translation = get_completion(system_message, translation_message)
    return input


def one_chunk_reflect_translation(input: TranslationSingleInput) -> TranslationSingleInput:
    source_language, target_language, text, translation, country, field = (input.source_language,
                                           input.target_language, input.text, input.translation, input.country, input.field)
    system_message = (f"If you are an expert linguist, specializing in translation from {source_language} to {target_language}."
                      f"You will be provided with a source text and its translation and your goal is to improve the translation.")

    reflection_message = f"""You task is to carefully read a source text and a translation from {source_language} to {target_language}.,
    and then give constructive criticism and helpful suggestions to improve the translation. You final style an tone of the translation 
    should match the style of {target_language} and colloquially spoken in {country} and field in {field}.
    
    The source text and initial translation delimited by XLM tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:
    
    <SOURCE_TEXT>
    {text}
    </SOURCE_TEXT>
    
    <TRANSLATION>
    {translation}
    </TRANSLATION>
    
    When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
    (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
    (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_language}).\n\
    
    Write a list of specific, helpful and constructive suggestions for improving the translation.
    Each suggestion should address one specific part of the translation.
    Output only the suggestions and nothing else."""
    input.reflection = get_completion(system_message, reflection_message)
    return input


def one_chunk_improve_translation(input: TranslationSingleInput) -> TranslationSingleInput:
    source_language, target_language, text, translation, reflection = (input.source_language,
                                                   input.target_language, input.text, input.translation, input.reflection)
    system_message = f"You are an expert linguist, specializing in translation editing from {source_language} to {target_language}."

    improve_message = f"""Your task is to carefully read, then edit, a translation from {source_language} to {target_language}, taking into
    account a list of expert suggestions and constructive criticisms.

    The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
    as follows:

    <SOURCE_TEXT>
    {text}
    </SOURCE_TEXT>

    <TRANSLATION>
    {translation}
    </TRANSLATION>

    <EXPERT_SUGGESTIONS>
    {reflection}
    </EXPERT_SUGGESTIONS>

    Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
    (ii) fluency (by applying {source_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
    (iii) style (by ensuring the translations reflect the style of the source text)
    (iv) terminology (inappropriate for context, inconsistent use), or
    (v) other errors.

    Output only the new translation and nothing else."""

    input.improve = get_completion(system_message, improve_message)
    return input


def one_chunk_translate(input: TranslationSingleInput) -> TranslationSingleInput:
    input = one_chunk_initial_translation(input)
    input = one_chunk_reflect_translation(input)
    input = one_chunk_improve_translation(input)
    return input


def calculate_num_tokens(input: TranslationSingleInput, llm) -> int:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"tokenizer/{llm}", local_files_only=True)
    input_ids = tokenizer.encode(input.text)
    return len(input_ids)


if __name__ == '__main__':
    input = TranslationSingleInput()
    input.text = "Mamba作为时序状态空间类模型的扛把子之作，一开始就是打着取代Transformer的架构的目的而来的，我们来看一下到底Mamba的魅力在哪里，又是如何能取代Transformer架构呢？"
    # output = one_chunk_translate(input)
    output = calculate_num_tokens(input)
    print(output)