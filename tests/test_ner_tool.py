"""Stage 1 NER tool: spaCy (injectable), verifier = span/label schema validation."""

import json

from router.tools.ner_tool import NerTool


class FakeEnt:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


def make_nlp(ents):
    return lambda text: FakeDoc(ents)


TEXT = "Barack Obama visited Paris."


def test_extracts_entities_as_json_with_valid_spans():
    nlp = make_nlp([FakeEnt("Barack Obama", "PERSON", 0, 12), FakeEnt("Paris", "GPE", 21, 26)])
    r = NerTool(nlp).extract(TEXT)
    assert r.verified
    ents = json.loads(r.answer)
    assert ents == [
        {"text": "Barack Obama", "label": "PERSON", "start": 0, "end": 12},
        {"text": "Paris", "label": "GPE", "start": 21, "end": 26},
    ]


def test_span_mismatch_fails_schema_validation():
    nlp = make_nlp([FakeEnt("Obama", "PERSON", 0, 12)])  # slice(0,12) != "Obama"
    r = NerTool(nlp).extract(TEXT)
    assert not r.verified


def test_empty_label_fails_schema_validation():
    nlp = make_nlp([FakeEnt("Paris", "", 21, 26)])
    r = NerTool(nlp).extract(TEXT)
    assert not r.verified


def test_no_entities_found_escalates():
    r = NerTool(make_nlp([])).extract(TEXT)
    assert not r.verified


def test_missing_spacy_pipeline_escalates():
    r = NerTool(None).extract(TEXT)
    assert not r.verified
    assert "unavailable" in r.reason
