"""Free verification gates for LLM-produced answers (Stages 2 and 3)."""

from router.verifiers import (
    answers_agree,
    extract_code_block,
    majority_answer,
    normalize_answer,
    summary_ok,
    valid_ner_entities,
    valid_sentiment_label,
)


def test_normalize_strips_case_punctuation_articles():
    assert normalize_answer("The Eiffel Tower.") == "eiffel tower"
    assert normalize_answer("  42  ") == "42"


def test_answers_agree_on_normalized_match():
    assert answers_agree("Paris", "paris.")
    assert answers_agree("The answer is Paris", "Paris")  # containment
    assert not answers_agree("Paris", "London")


def test_majority_answer():
    ans, ratio = majority_answer(["Paris", "paris.", "London"])
    assert normalize_answer(ans) == "paris"
    assert ratio == 2 / 3


def test_valid_sentiment_label():
    assert valid_sentiment_label("Positive") == "positive"
    assert valid_sentiment_label("The sentiment is negative.") == "negative"
    assert valid_sentiment_label("I think it's mixed") is None


def test_summary_length_bounds():
    source = " ".join(["word"] * 200)
    ok, _ = summary_ok(source, " ".join(["word"] * 30))
    assert ok
    too_long, _ = summary_ok(source, source)  # not compressed at all
    assert not too_long
    too_short, _ = summary_ok(source, "word")
    assert not too_short


def test_summary_requires_key_entity_coverage():
    source = ("NASA launched the Artemis mission from Florida. " * 5
              + "The Artemis program is led by NASA engineers in Florida. " * 5)
    good = "NASA launched the Artemis mission from Florida, a major milestone for it."
    bad = "A space agency launched a rocket somewhere, which was a major milestone for it."
    assert summary_ok(source, good)[0]
    assert not summary_ok(source, bad)[0]


def test_valid_ner_entities_parses_json_list():
    ents = valid_ner_entities('[{"text": "Paris", "label": "GPE"}]')
    assert ents == [{"text": "Paris", "label": "GPE"}]


def test_valid_ner_entities_accepts_fenced_json():
    ents = valid_ner_entities('```json\n[{"text": "Paris", "label": "GPE"}]\n```')
    assert ents is not None


def test_valid_ner_entities_rejects_bad_schema():
    assert valid_ner_entities('{"not": "a list"}') is None
    assert valid_ner_entities('[{"text": "Paris"}]') is None  # missing label
    assert valid_ner_entities("no json here") is None


def test_extract_code_block():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nEnjoy!"
    assert extract_code_block(text) == "def f():\n    return 1"
    # Raw code with no fences comes back as-is.
    assert extract_code_block("def g():\n    pass") == "def g():\n    pass"
