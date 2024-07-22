from get_completion import get_completion
from schema import TranslationSingleInput, TranslationMultiInput


def multi_chunk_initial_translation(input: TranslationMultiInput) -> TranslationMultiInput:
    source_language, target_language, texts = input.source_language, input.target_language, input.texts
    system_message = f"You are an expert linguist, specializing in translation from {source_language} to {target_language}."

    translation_prompt = """Your task is to provide a professional translation from {source_language} to {target_language} of PART of a text.

    The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
    delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
    of the other text. Do not output anything other than the translation of the indicated part of the text.

    <SOURCE_TEXT>
    {tagged_text}
    </SOURCE_TEXT>

    To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
    <TRANSLATE_THIS>
    {chunk_to_translate}
    </TRANSLATE_THIS>

    Output only the translation of the portion you are asked to translate, and nothing else."""
    translations = []
    for id, text in enumerate(texts):
        tagged_text = "".join(texts[0:id]) + "<TRANSLATE_THIS>" + text + "</TRANSLATE_THIS>" + "".join(texts[id+1:])
        translation_message = translation_prompt.format(source_language=source_language,
                             target_language=target_language, tagged_text=tagged_text, chunk_to_translate=text)
        translation = get_completion(system_message, translation_message)
        translations.append(translation)
    input.translations = translations
    return input


def multi_chunk_reflect_translation(input: TranslationMultiInput) -> TranslationMultiInput:
    source_language, target_language, texts, translations, country, field = (input.source_language,
                               input.target_language, input.texts, input.translations, input.country, input.field)
    system_message = f"You are an expert linguist specializing in translation from {source_language} to {target_language}. \
    You will be provided with a source text and its translation and your goal is to improve the translation."

    reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_language} to {target_language}, \
    and then give constructive criticism and helpful suggestions for improving the translation.
    The final style and tone of the translation should match the style of {target_language} colloquially spoken in {country} and field in {field}.

    The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
    is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
    as context for critiquing the translated part.

    <SOURCE_TEXT>
    {tagged_text}
    </SOURCE_TEXT>

    To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
    <TRANSLATE_THIS>
    {chunk_to_translate}
    </TRANSLATE_THIS>

    The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
    <TRANSLATION>
    {translation_chunk}
    </TRANSLATION>

    When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
    (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
    (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_language}).\n\

    Write a list of specific, helpful and constructive suggestions for improving the translation.
    Each suggestion should address one specific part of the translation.
    Output only the suggestions and nothing else."""

    reflections = []
    for id, text in enumerate(texts):
        tagged_text = "".join(texts[0:id]) + "<TRANSLATE_THIS>" + text + "</TRANSLATE_THIS>" + "".join(texts[id + 1:])
        translation = translations[id]
        reflection_message = reflection_prompt.format(source_language=source_language, target_language=target_language,
               tagged_text=tagged_text, chunk_to_translate=text, translation_chunk=translation, country=country, field=field)
        reflection = get_completion(system_message, reflection_message)
        reflections.append(reflection)
    input.reflections = reflections
    return input


def multi_chunk_improve_translation(input: TranslationMultiInput) -> TranslationMultiInput:
    source_language, target_language, texts, translations, reflections, country, field = (input.source_language,
                         input.target_language, input.texts, input.translations, input.reflections ,input.country, input.field)
    system_message = f"You are an expert linguist, specializing in translation editing from {source_language} to {target_language}."

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_language} to {target_language}, taking into \
    account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

    The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
    is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
    as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

    <SOURCE_TEXT>
    {tagged_text}
    </SOURCE_TEXT>

    To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
    <TRANSLATE_THIS>
    {chunk_to_translate}
    </TRANSLATE_THIS>

    The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
    <TRANSLATION>
    {translation_chunk}
    </TRANSLATION>

    The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, are as follows:
    <EXPERT_SUGGESTIONS>
    {reflection_chunk}
    </EXPERT_SUGGESTIONS>

    Taking into account the expert suggestions rewrite the translation to improve it, paying attention
    to whether there are ways to improve the translation's

    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
    (iii) style (by ensuring the translations reflect the style of the source text)
    (iv) terminology (inappropriate for context, inconsistent use), or
    (v) other errors.

    Output only the new translation of the indicated part and nothing else."""

    improves = []
    for id, text in enumerate(texts):
        tagged_text = "".join(texts[0:id]) + "<TRANSLATE_THIS>" + text + "</TRANSLATE_THIS>" + "".join(texts[id + 1:])
        translation, reflection = translations[id], reflections[id]
        improvement_message = improvement_prompt.format(source_language=source_language, target_language=target_language,
            tagged_text=tagged_text, chunk_to_translate=text, translation_chunk=translation, reflection_chunk=reflection, country=country, field=field)
        improve = get_completion(system_message, improvement_message)
        improves.append(improve)
    input.improves = improves
    return input


def multi_chunk_translate(input: TranslationMultiInput) -> TranslationMultiInput:
    input = multi_chunk_initial_translation(input)
    input = multi_chunk_reflect_translation(input)
    input = multi_chunk_improve_translation(input)
    return input


def split_chunk_text(texts, max_token_per_chunk, llm):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"tokenizer/{llm}", local_files_only=True)
    input_ids = tokenizer.encode(texts)
    return_texts = []
    for i in range(len(input_ids) // max_token_per_chunk):
        text_ids = input_ids[i*max_token_per_chunk: (i+1)*max_token_per_chunk]
        text = tokenizer.decode(text_ids)
        return_texts.append(text)
    return return_texts
