"""Stage 1 sentiment tool: VADER lexicon, verifier = confidence >= threshold."""

from router.tools.sentiment_tool import classify_sentiment, extract_target_text


def test_strong_positive_verified():
    r = classify_sentiment("I absolutely loved this movie, it was fantastic!", threshold=0.5)
    assert r.answer == "positive"
    assert r.verified
    assert r.confidence >= 0.5


def test_strong_negative_verified():
    r = classify_sentiment("Horrible experience. Terrible, rude staff. I hated it.", threshold=0.5)
    assert r.answer == "negative"
    assert r.verified


def test_weak_signal_not_verified():
    r = classify_sentiment("The package arrived on a Tuesday.", threshold=0.5)
    assert not r.verified


def test_extract_target_text_prefers_quoted_segment():
    prompt = "Classify the sentiment of this review: 'The food was amazing.'"
    assert extract_target_text(prompt) == "The food was amazing."


def test_extract_target_text_falls_back_to_after_colon():
    prompt = "Classify the sentiment: the food was cold and bland"
    assert extract_target_text(prompt) == "the food was cold and bland"


def test_extract_target_text_whole_prompt_when_no_structure():
    assert extract_target_text("Great product!") == "Great product!"
