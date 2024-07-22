from dataclasses import dataclass


@dataclass
class TranslationSingleInput:
    field: str = "AI Academic"
    country: str = "england"
    source_language: str = "chinese"
    target_language: str = "english"
    text: str = None
    translation: str = None
    reflection: str = None
    improve: str = None


@dataclass
class TranslationMultiInput:
    field: str = "AI Academic"
    country: str = "england"
    source_language: str = "chinese"
    target_language: str = "english"
    texts: str = None
    translations: str = None
    reflections: str = None
    improves: str = None