"""Stage 1 NER tool — spaCy pipeline (injectable), verifier = span/label schema validation."""

from __future__ import annotations

import json
from typing import Any, Callable

from router.types import ToolResult


class NerTool:
    def __init__(self, nlp: Callable[[str], Any] | None):
        self.nlp = nlp

    @classmethod
    def load(cls, model: str = "en_core_web_sm") -> "NerTool":
        try:
            import spacy

            return cls(spacy.load(model))
        except Exception:
            return cls(None)

    def extract(self, text: str) -> ToolResult:
        if self.nlp is None:
            return ToolResult(None, verified=False, reason="spaCy pipeline unavailable")
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            start, end = ent.start_char, ent.end_char
            if not ent.label_ or text[start:end] != ent.text:
                return ToolResult(None, verified=False,
                                  reason=f"schema validation failed for span {ent.text!r}")
            entities.append({"text": ent.text, "label": ent.label_, "start": start, "end": end})
        if not entities:
            return ToolResult(None, verified=False, reason="no entities found; escalating")
        return ToolResult(json.dumps(entities, ensure_ascii=False), verified=True,
                          confidence=1.0, reason=f"spaCy extracted {len(entities)} entities")
