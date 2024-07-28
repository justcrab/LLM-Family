from llm import llm
from prompt import translation_first_prompt, translation_second_prompt
from schema import TranslationState, TranslationResponse


translation_first_chain = translation_first_prompt | llm.with_structured_output(TranslationResponse)
translation_second_chain = translation_second_prompt | llm.with_structured_output(TranslationResponse)


def translation_first_node(state: TranslationState):
    llm_response = translation_first_chain.invoke(state["task"])
    return {"task": llm_response.response}


def translation_second_node(state: TranslationState):
    llm_response = translation_second_chain.invoke(state["task"])
    return {"task": llm_response.response}
