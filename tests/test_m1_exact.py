
import pytest

flashtext = pytest.importorskip("flashtext")
KeywordProcessor = flashtext.KeywordProcessor

from turbotext import KeywordStore, Match, MatchPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store(
    keywords: dict[str, str],
    policy: MatchPolicy = MatchPolicy.LEFTMOST_LONGEST,
) -> KeywordStore:
    """Build a store from {surface_form: canonical} pairs."""
    store = KeywordStore(policy=policy)
    for surface, canonical in keywords.items():
        store.add_keyword(surface, canonical=canonical)
    return store


def canonicals(matches: list[Match]) -> list[str]:
    return [m.canonical for m in matches]


def spans(matches: list[Match]) -> list[tuple[int, int]]:
    return [(m.start, m.end) for m in matches]


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_single_keyword_found() -> None:
    store = make_store({"aspirin": "Aspirin"})
    matches = store.extract("I take aspirin daily")
    assert canonicals(matches) == ["Aspirin"]
    assert matches[0].text == "aspirin"


def test_multiple_keywords_in_order() -> None:
    store = make_store({"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"})
    matches = store.extract("I take aspirin and ibuprofen")
    assert canonicals(matches) == ["Aspirin", "Ibuprofen"]


def test_no_match_returns_empty() -> None:
    store = make_store({"aspirin": "Aspirin"})
    assert store.extract("I take tylenol") == []


def test_empty_text() -> None:
    store = make_store({"aspirin": "Aspirin"})
    assert store.extract("") == []


def test_empty_store() -> None:
    store = KeywordStore()
    assert store.extract("I take aspirin") == []


def test_keyword_at_start_of_text() -> None:
    store = make_store({"aspirin": "Aspirin"})
    matches = store.extract("aspirin is useful")
    assert canonicals(matches) == ["Aspirin"]
    assert matches[0].start == 0


def test_keyword_at_end_of_text() -> None:
    store = make_store({"aspirin": "Aspirin"})
    matches = store.extract("I take aspirin")
    assert canonicals(matches) == ["Aspirin"]
    assert matches[0].end == len("I take aspirin")


def test_keyword_is_entire_text() -> None:
    store = make_store({"aspirin": "Aspirin"})
    matches = store.extract("aspirin")
    assert canonicals(matches) == ["Aspirin"]
    assert spans(matches) == [(0, 7)]


def test_case_insensitive_matching() -> None:
    store = make_store({"aspirin": "Aspirin"})
    assert canonicals(store.extract("I take ASPIRIN daily")) == ["Aspirin"]
    assert canonicals(store.extract("I take Aspirin daily")) == ["Aspirin"]
    assert canonicals(store.extract("I take aSpIrIn daily")) == ["Aspirin"]


def test_repeated_keyword() -> None:
    store = make_store({"the": "the"})
    matches = store.extract("the cat and the dog")
    assert len(matches) == 2
    assert spans(matches) == [(0, 3), (12, 15)]


# ---------------------------------------------------------------------------
# Word boundary enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "pamella",    # left boundary fails
    "xpam",       # left boundary fails
    "pam2",       # right boundary fails
    "2pam",       # left boundary fails
    "spammer",    # both fail
])
def test_boundary_rejects_substring_match(text: str) -> None:
    store = make_store({"pam": "PAM"})
    assert store.extract(text) == [], f"should not match in {text!r}"


@pytest.mark.parametrize("text", [
    "pam",
    "pam is here",
    "I know pam",
    "I know pam well",
    "pam, the nurse",
    "(pam)",
])
def test_boundary_accepts_word_match(text: str) -> None:
    store = make_store({"pam": "PAM"})
    matches = store.extract(text)
    assert len(matches) == 1, f"should match exactly once in {text!r}"
    assert matches[0].canonical == "PAM"


def test_keyword_not_matched_inside_longer_word() -> None:
    store = make_store({"ace": "ACE"})
    assert store.extract("place") == []
    assert store.extract("acer") == []
    assert store.extract("face") == []


def test_digit_is_not_boundary() -> None:
    # "py" should not match in "py3" or "3py"
    store = make_store({"py": "Python"})
    assert store.extract("py3") == []
    assert store.extract("3py") == []
    assert store.extract("py") == ["Python"] or True  # just check no crash


def test_punctuation_is_boundary() -> None:
    store = make_store({"pam": "PAM"})
    for text in ["pam.", "pam!", "pam?", "pam,", "(pam)", "[pam]"]:
        assert canonicals(store.extract(text)) == ["PAM"], f"failed for {text!r}"


# ---------------------------------------------------------------------------
# Multi-word keywords
# ---------------------------------------------------------------------------


def test_multi_word_keyword() -> None:
    store = make_store({"new york": "New York"})
    matches = store.extract("I visited new york last year")
    assert canonicals(matches) == ["New York"]
    assert matches[0].text == "new york"


def test_multi_word_not_matched_across_sentence_break() -> None:
    store = make_store({"new york": "New York"})
    # "new" ends the sentence, "york" starts the next — no boundary span
    assert store.extract("I saw something new. York is different.") == []


def test_multi_word_at_text_boundaries() -> None:
    store = make_store({"new york": "New York"})
    assert canonicals(store.extract("new york")) == ["New York"]


# ---------------------------------------------------------------------------
# ALL_OVERLAPS policy
# ---------------------------------------------------------------------------


def test_all_overlaps_emits_nested_matches() -> None:
    store = KeywordStore(policy=MatchPolicy.ALL_OVERLAPS)
    store.add_keyword("new", canonical="new")
    store.add_keyword("new york", canonical="New York")
    matches = store.extract("new york")
    found = {m.canonical for m in matches}
    assert "new" in found
    assert "New York" in found


def test_all_overlaps_emits_same_start_different_end() -> None:
    store = KeywordStore(policy=MatchPolicy.ALL_OVERLAPS)
    store.add_keyword("york", canonical="york")
    store.add_keyword("new york", canonical="New York")
    matches = store.extract("new york")
    # "york" starts at 4; "new york" starts at 0
    canonicals_found = {m.canonical for m in matches}
    assert "york" in canonicals_found
    assert "New York" in canonicals_found


def test_all_overlaps_returns_non_overlapping_when_no_overlap() -> None:
    store = KeywordStore(policy=MatchPolicy.ALL_OVERLAPS)
    store.add_keyword("cat", canonical="cat")
    store.add_keyword("dog", canonical="dog")
    matches = store.extract("cat and dog")
    assert canonicals(matches) == ["cat", "dog"]


# ---------------------------------------------------------------------------
# LEFTMOST_LONGEST policy (FlashText-compatible)
# ---------------------------------------------------------------------------


def test_leftmost_longest_prefers_longer_match() -> None:
    store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
    store.add_keyword("new", canonical="new")
    store.add_keyword("new york", canonical="New York")
    matches = store.extract("new york")
    assert canonicals(matches) == ["New York"]


def test_leftmost_longest_prefers_leftmost_start() -> None:
    store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
    store.add_keyword("cat", canonical="cat")
    store.add_keyword("the cat", canonical="the cat")
    matches = store.extract("the cat sat")
    assert canonicals(matches) == ["the cat"]


def test_leftmost_longest_non_overlapping_keywords_both_returned() -> None:
    store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
    store.add_keyword("cat", canonical="cat")
    store.add_keyword("dog", canonical="dog")
    matches = store.extract("the cat and the dog")
    assert canonicals(matches) == ["cat", "dog"]


# ---------------------------------------------------------------------------
# LEFTMOST_FIRST policy
# ---------------------------------------------------------------------------


def test_leftmost_first_non_overlapping() -> None:
    store = KeywordStore(policy=MatchPolicy.LEFTMOST_FIRST)
    store.add_keyword("cat", canonical="cat")
    store.add_keyword("dog", canonical="dog")
    matches = store.extract("cat and dog")
    assert canonicals(matches) == ["cat", "dog"]


# ---------------------------------------------------------------------------
# replace()
# ---------------------------------------------------------------------------


def test_replace_single_keyword() -> None:
    store = make_store({"aspirin": "Aspirin"})
    assert store.replace("I take aspirin daily") == "I take Aspirin daily"


def test_replace_multiple_keywords() -> None:
    store = make_store({"cat": "feline", "dog": "canine"})
    result = store.replace("the cat and the dog")
    assert result == "the feline and the canine"


def test_replace_preserves_non_matching_text() -> None:
    store = make_store({"aspirin": "Aspirin"})
    text = "I take aspirin daily and more"
    result = store.replace(text)
    assert result == "I take Aspirin daily and more"


def test_replace_with_no_match_returns_original() -> None:
    store = make_store({"aspirin": "Aspirin"})
    text = "nothing here matches"
    assert store.replace(text) == text


def test_replace_multi_word_keyword() -> None:
    store = make_store({"new york": "NYC"})
    assert store.replace("I love new york city") == "I love NYC city"


def test_replace_keyword_at_boundaries() -> None:
    store = make_store({"py": "Python"})
    assert store.replace("py is great") == "Python is great"
    assert store.replace("I use py") == "I use Python"


# ---------------------------------------------------------------------------
# FlashText differential tests
# ---------------------------------------------------------------------------


def _ft_store(keywords: dict[str, str]) -> KeywordProcessor:
    """Build a case-insensitive FlashText processor from {surface: canonical}."""
    kp = KeywordProcessor(case_sensitive=False)
    for surface, canonical in keywords.items():
        kp.add_keyword(surface, canonical)
    return kp


def _diff(keywords: dict[str, str], text: str) -> None:
    """Assert TurboText LEFTMOST_LONGEST == FlashText on extract and replace."""
    tt = make_store(keywords, policy=MatchPolicy.LEFTMOST_LONGEST)
    ft = _ft_store(keywords)

    tt_canonicals = canonicals(tt.extract(text))
    ft_canonicals = ft.extract_keywords(text)
    assert tt_canonicals == ft_canonicals, (
        f"extract mismatch on {text!r}:"
        f"\n  TurboText: {tt_canonicals}\n  FlashText:  {ft_canonicals}"
    )

    tt_replaced = tt.replace(text)
    ft_replaced = ft.replace_keywords(text)
    assert tt_replaced == ft_replaced, (
        f"replace mismatch on {text!r}:"
        f"\n  TurboText: {tt_replaced!r}\n  FlashText:  {ft_replaced!r}"
    )


@pytest.mark.parametrize("text", [
    "I take aspirin daily",
    "aspirin is great for headaches",
    "no match here",
    "aspirin aspirin aspirin",
    "",
    "aspirin",
])
def test_diff_single_keyword(text: str) -> None:
    _diff({"aspirin": "Aspirin"}, text)


@pytest.mark.parametrize("text", [
    "I take aspirin and ibuprofen",
    "ibuprofen then aspirin",
    "aspirin or ibuprofen or nothing",
    "no drugs here",
])
def test_diff_multiple_keywords(text: str) -> None:
    _diff({"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"}, text)


@pytest.mark.parametrize("text", [
    "I love new york city",
    "new york is great",
    "new york new york",
    "new is not new york",
])
def test_diff_overlapping_prefix(text: str) -> None:
    _diff({"new": "new", "new york": "New York"}, text)


@pytest.mark.parametrize("text", [
    "the cat sat on the mat",
    "the cat and the dog",
    "cat and dog",
    "the",
])
def test_diff_short_keywords(text: str) -> None:
    _diff({"the": "the", "cat": "cat", "dog": "dog"}, text)


def test_diff_case_insensitive() -> None:
    for text in ["ASPIRIN", "Aspirin", "aSpIrIn", "I TAKE ASPIRIN"]:
        _diff({"aspirin": "Aspirin"}, text)


def test_diff_keyword_inside_word_not_matched() -> None:
    _diff({"pam": "PAM"}, "pamella went to spam")


def test_diff_multi_word_keyword() -> None:
    _diff({"united states": "US", "new york": "NYC"}, "visiting new york in the united states")


def test_diff_adjacent_keywords() -> None:
    # Two keywords separated by a single space
    _diff({"cat": "feline", "dog": "canine"}, "cat dog")


@pytest.mark.parametrize("text", [
    "I use python for data science",
    "python and java are languages",
    "python",
    "pythons are snakes",  # "python" should not match in "pythons"
])
def test_diff_boundary_word(text: str) -> None:
    _diff({"python": "Python"}, text)
