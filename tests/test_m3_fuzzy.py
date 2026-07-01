
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from reference_matcher import extract_all_overlaps

from turbotext import FuzzyConfig, KeywordStore, MatchPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fuzzy_store(
    keywords: list[str],
    max_edit: int = 1,
    policy: MatchPolicy = MatchPolicy.ALL_OVERLAPS,
) -> KeywordStore:
    store = KeywordStore(policy=policy, fuzzy=FuzzyConfig(max_edit_distance=max_edit))
    for kw in keywords:
        store.add_keyword(kw, canonical=kw)
    return store


# ---------------------------------------------------------------------------
# Substitution (one character replaced)
# ---------------------------------------------------------------------------


def test_substitution_matches() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I pet cbt today")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1
    assert matches[0].text == "cbt"


def test_substitution_at_start() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("bat is here")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1


def test_substitution_at_end() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I see car")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1


def test_exact_match_preferred_over_fuzzy() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("cat and cbt")
    exact = [m for m in matches if m.edit_distance == 0]
    fuzzy = [m for m in matches if m.edit_distance == 1]
    assert len(exact) == 1
    assert exact[0].text == "cat"
    assert len(fuzzy) == 1
    assert fuzzy[0].text == "cbt"


# ---------------------------------------------------------------------------
# Insertion (extra character in text)
# ---------------------------------------------------------------------------


def test_insertion_in_middle() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I see caat")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1
    assert matches[0].text == "caat"


def test_insertion_at_start() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("xcat is here")
    # "xcat" spans (0,4); left boundary passes (start of text), right boundary: ' '
    assert any(m.edit_distance == 1 and m.text == "xcat" for m in matches)


def test_insertion_at_end() -> None:
    store = fuzzy_store(["cat"])
    # "cats" at end-of-text: edit_distance=1 (one inserted 's'), right boundary is
    # end-of-text which always passes — same rule as exact matching.
    matches = store.extract("I see cats")
    assert any(m.text == "cats" and m.edit_distance == 1 for m in matches)
    # Mid-word (followed by a word char): right boundary fails.
    assert store.extract("catster") == []


# ---------------------------------------------------------------------------
# Deletion (character missing from text)
# ---------------------------------------------------------------------------


def test_deletion_in_middle() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I see ct")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1
    assert matches[0].text == "ct"


def test_deletion_at_start() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I see at")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1
    assert matches[0].text == "at"


def test_deletion_at_end() -> None:
    store = fuzzy_store(["cat"])
    matches = store.extract("I see ca")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1
    assert matches[0].text == "ca"


# ---------------------------------------------------------------------------
# Boundary enforcement still applies for fuzzy matches
# ---------------------------------------------------------------------------


def test_fuzzy_match_respects_left_boundary() -> None:
    # "cbt" in "xcbt" has no left boundary before 'c' → should not match
    store = fuzzy_store(["cat"])
    assert store.extract("xcbt") == []


def test_fuzzy_match_respects_right_boundary() -> None:
    # "cbt" in "cbts" — right boundary fails at 's'
    store = fuzzy_store(["cat"])
    assert store.extract("cbts") == []


def test_fuzzy_match_at_text_edges() -> None:
    store = fuzzy_store(["cat"])
    # Left edge: no left boundary needed
    matches = store.extract("cbt here")
    assert len(matches) == 1
    # Right edge
    matches = store.extract("I see cbt")
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# max_edit_distance=0 regression: no fuzzy matches allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I pet cbt today",   # substitution
    "I see caat",        # insertion
    "I see ct",          # deletion
    "bat is here",       # substitution at start
])
def test_no_fuzzy_at_edit_distance_zero(text: str) -> None:
    store = fuzzy_store(["cat"], max_edit=0)
    assert store.extract(text) == [], f"should not fuzzy-match in {text!r}"


def test_exact_still_works_at_edit_distance_zero() -> None:
    store = fuzzy_store(["cat"], max_edit=0)
    matches = store.extract("I see cat here")
    assert len(matches) == 1
    assert matches[0].edit_distance == 0


# ---------------------------------------------------------------------------
# max_edit_distance=2
# ---------------------------------------------------------------------------


def test_edit_distance_2_substitutions() -> None:
    store = fuzzy_store(["aspirin"], max_edit=2)
    # "aspxrxn" — 2 substitutions
    matches = store.extract("take aspxrxn daily")
    assert len(matches) == 1
    assert matches[0].edit_distance == 2


def test_edit_distance_2_finds_edit_1_as_well() -> None:
    store = fuzzy_store(["aspirin"], max_edit=2)
    matches = store.extract("take asparin daily")  # 1 substitution
    assert len(matches) == 1
    assert matches[0].edit_distance == 1


def test_too_many_edits_not_matched() -> None:
    store = fuzzy_store(["cat"], max_edit=1)
    # "xyz" requires 3 substitutions — should not match
    assert store.extract("xyz") == []


# ---------------------------------------------------------------------------
# Real-world example
# ---------------------------------------------------------------------------


def test_pharmaceutical_typo() -> None:
    store = fuzzy_store(["aspirin", "ibuprofen"], max_edit=1)
    text = "patient takes asparin and ibuprofen daily"
    matches = store.extract(text)
    canonicals = {m.canonical for m in matches}
    assert "aspirin" in canonicals  # "asparin" → "aspirin" with 1 substitution
    assert "ibuprofen" in canonicals  # exact match


def test_multi_word_fuzzy() -> None:
    store = fuzzy_store(["new york"], max_edit=1)
    # "new yort" — 1 substitution in second word
    matches = store.extract("visiting new yort")
    assert len(matches) == 1
    assert matches[0].edit_distance == 1


# ---------------------------------------------------------------------------
# No zero-length matches ever emitted
# ---------------------------------------------------------------------------


def test_no_zero_length_matches() -> None:
    store = fuzzy_store(["a", "ab", "abc"], max_edit=2)
    for text in ["x", " ", "  ", "xyz", ""]:
        for m in store.extract(text):
            assert m.start < m.end, f"zero-length match at ({m.start},{m.end}) in {text!r}"


# ---------------------------------------------------------------------------
# Property-based: TurboText ALL_OVERLAPS matches brute-force Levenshtein reference
# ---------------------------------------------------------------------------

_ALPHABET = "abcde "
_kw_st = st.text(alphabet="abcde", min_size=2, max_size=5)
_text_st = st.text(alphabet=_ALPHABET, min_size=0, max_size=20)


def _tt_match_set(
    keywords: list[str], text: str, max_edit: int
) -> set[tuple[int, int, str]]:
    store = fuzzy_store(keywords, max_edit=max_edit)
    return {(m.start, m.end, m.canonical) for m in store.extract(text)}


def _ref_match_set(
    keywords: list[str], text: str, max_edit: int
) -> set[tuple[int, int, str]]:
    kw_dict = {kw: kw for kw in keywords}
    refs = extract_all_overlaps(text, kw_dict, max_edit_distance=max_edit)
    return {(r.start, r.end, r.canonical) for r in refs}


@given(
    keywords=st.lists(_kw_st, min_size=1, max_size=5, unique=True),
    text=_text_st,
)
@settings(max_examples=400)
def test_fuzzy_matches_reference_at_edit_0(keywords: list[str], text: str) -> None:
    tt = _tt_match_set(keywords, text, max_edit=0)
    ref = _ref_match_set(keywords, text, max_edit=0)
    assert tt == ref, (
        f"edit=0 mismatch\nkeywords={keywords}\ntext={text!r}"
        f"\nTT only: {tt - ref}\nRef only: {ref - tt}"
    )


@given(
    keywords=st.lists(_kw_st, min_size=1, max_size=4, unique=True),
    text=_text_st,
)
@settings(max_examples=400)
def test_fuzzy_matches_reference_at_edit_1(keywords: list[str], text: str) -> None:
    tt = _tt_match_set(keywords, text, max_edit=1)
    ref = _ref_match_set(keywords, text, max_edit=1)
    assert tt == ref, (
        f"edit=1 mismatch\nkeywords={keywords}\ntext={text!r}"
        f"\nTT only: {tt - ref}\nRef only: {ref - tt}"
    )
