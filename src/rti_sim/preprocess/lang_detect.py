from __future__ import annotations
from typing import Optional
from lingua import Language, LanguageDetectorBuilder

_LANGS = [Language.ENGLISH, Language.FRENCH]
_DETECTOR = LanguageDetectorBuilder.from_languages(*_LANGS).with_preloaded_language_models().build()

def detect_lang(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    lang = _DETECTOR.detect_language_of(text)
    if lang is None:
        return None
    if lang == Language.ENGLISH:
        return "en"
    if lang == Language.FRENCH:
        return "fr"
    return None

def majority_lang(*codes: Optional[str]) -> Optional[str]:
    counts = {}
    for c in codes:
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
