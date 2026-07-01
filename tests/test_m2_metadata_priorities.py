
from itertools import combinations
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from turbotext import KeywordStore, Match, MatchPolicy
from turbotext.resolve import RawMatch, _highest_priority, _optimal_weighted

# ---------------------------------------------------------------------------
# Brute-force oracle
# ---------------------------------------------------------------------------


def _non_overlapping(subset: list[RawMatch]) -> bool:
    s = sorted(subset, key=lambda m: m.start)
    return all(s[i].end <= s[i + 1].start for i in range(len(s) - 1))


def brute_force_optimal_weight(
    matches: list[RawMatch], priorities: dict[str, float]
) -> float:
    """Total weight of the optimal non-overlapping subset (exhaustive search)."""
    best = 0.0
    for r in range(len(matches) + 1):
        for subset in combinations(matches, r):
            if _non_overlapping(list(subset)):
                w = sum(priorities.get(m.keyword_id, 1.0) for m in subset)
                if w > best:
                    best = w
    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store(
    keywords: dict[str, tuple[str, float]],  # surface -> (canonical, priority)
    policy: MatchPolicy = MatchPolicy.HIGHEST_PRIORITY,
    **extra: Any,
) -> KeywordStore:
    store = KeywordStore(policy=policy, **extra)
    for surface, (canonical, priority) in keywords.items():
        store.add_keyword(surface, canonical=canonical, priority=priority)
    return store


def canonicals(matches: list[Match]) -> list[str]:
    return [m.canonical for m in matches]


# ---------------------------------------------------------------------------
# Metadata on Match
# ---------------------------------------------------------------------------


def test_match_exposes_category() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin", canonical="Aspirin", category="DRUG")
    matches = store.extract("take aspirin")
    assert matches[0].category == "DRUG"


def test_match_exposes_priority() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin", canonical="Aspirin", priority=7.5)
    matches = store.extract("take aspirin")
    assert matches[0].priority == 7.5


def test_match_exposes_keyword_id() -> None:
    store = KeywordStore()
    kid = store.add_keyword("aspirin", canonical="Aspirin")
    matches = store.extract("take aspirin")
    assert matches[0].keyword_id == kid


def test_match_exposes_metadata_kwargs() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin", canonical="Aspirin", source="rxnorm", code="1191")
    matches = store.extract("take aspirin")
    assert matches[0].metadata == {"source": "rxnorm", "code": "1191"}


def test_match_metadata_is_empty_when_not_provided() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin")
    matches = store.extract("take aspirin")
    assert matches[0].metadata == {}


def test_match_category_is_none_when_not_provided() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin")
    matches = store.extract("take aspirin")
    assert matches[0].category is None


def test_match_metadata_is_copy_not_reference() -> None:
    store = KeywordStore()
    store.add_keyword("aspirin", tag="drug")
    matches = store.extract("take aspirin")
    matches[0].metadata["injected"] = True  # type: ignore[index]
    # Re-extract; the stored keyword's metadata should be unaffected
    matches2 = store.extract("take aspirin")
    assert "injected" not in matches2[0].metadata


# ---------------------------------------------------------------------------
# HIGHEST_PRIORITY resolver — direct (RawMatch level)
# ---------------------------------------------------------------------------


def _raw(start: int, end: int, kid: str) -> RawMatch:
    return RawMatch(start=start, end=end, keyword_id=kid, edit_distance=0)


def test_highest_priority_picks_higher_priority_match() -> None:
    lo = _raw(0, 5, "lo")
    hi = _raw(0, 5, "hi")
    priorities = {"lo": 1.0, "hi": 10.0}
    result = _highest_priority([lo, hi], priorities)
    assert result == [hi]


def test_highest_priority_tie_broken_by_span_length() -> None:
    short = _raw(0, 3, "short")
    long_ = _raw(0, 7, "long")
    priorities = {"short": 5.0, "long": 5.0}
    result = _highest_priority([short, long_], priorities)
    assert result == [long_]


def test_highest_priority_non_overlapping_both_kept() -> None:
    a = _raw(0, 4, "a")
    b = _raw(5, 9, "b")
    priorities = {"a": 3.0, "b": 1.0}
    result = _highest_priority([a, b], priorities)
    assert set(result) == {a, b}


def test_highest_priority_result_is_sorted_by_start() -> None:
    a = _raw(10, 14, "a")
    b = _raw(0, 4, "b")
    priorities = {"a": 1.0, "b": 1.0}
    result = _highest_priority([a, b], priorities)
    assert result[0].start < result[1].start


# ---------------------------------------------------------------------------
# HIGHEST_PRIORITY through KeywordStore
# ---------------------------------------------------------------------------


def test_store_highest_priority_overlap() -> None:
    store = make_store(
        {"aspirin": ("Aspirin", 10.0), "ibuprofen": ("Ibuprofen", 5.0)},
        policy=MatchPolicy.HIGHEST_PRIORITY,
    )
    # No overlap here — both returned
    matches = store.extract("aspirin and ibuprofen")
    assert canonicals(matches) == ["Aspirin", "Ibuprofen"]


def test_store_highest_priority_selects_by_priority() -> None:
    # "new york" and "new" overlap; "york" and "new york" overlap.
    # Give highest priority to "york".
    store = KeywordStore(policy=MatchPolicy.HIGHEST_PRIORITY)
    store.add_keyword("new", canonical="new", priority=1.0)
    store.add_keyword("new york", canonical="New York", priority=2.0)
    store.add_keyword("york", canonical="York", priority=5.0)
    matches = store.extract("new york")
    # "york" (priority=5) beats "new york" (priority=2) which beats "new" (priority=1).
    # "new" and "york" don't overlap, so both should survive.
    assert "York" in canonicals(matches)
    assert "New York" not in canonicals(matches)


def test_store_highest_priority_replace() -> None:
    store = KeywordStore(policy=MatchPolicy.HIGHEST_PRIORITY)
    store.add_keyword("acetaminophen", canonical="Acetaminophen", priority=10.0)
    store.add_keyword("tylenol", canonical="Acetaminophen", priority=8.0)
    result = store.replace("take acetaminophen or tylenol")
    assert result == "take Acetaminophen or Acetaminophen"


# ---------------------------------------------------------------------------
# OPTIMAL_WEIGHTED resolver — direct (RawMatch level)
# ---------------------------------------------------------------------------


def test_optimal_weighted_beats_greedy() -> None:
    # The classic case where a greedy priority-first approach fails:
    #   A: (0, 10) weight=5  — high priority but long, blocks B and C
    #   B: (0, 4)  weight=3  — lower priority
    #   C: (5, 9)  weight=3  — lower priority, compatible with B
    # Greedy picks A (weight=5); optimal picks B+C (weight=6).
    a = _raw(0, 10, "a")
    b = _raw(0, 4, "b")
    c = _raw(5, 9, "c")
    priorities = {"a": 5.0, "b": 3.0, "c": 3.0}
    result = _optimal_weighted([a, b, c], priorities)
    total = sum(priorities[m.keyword_id] for m in result)
    assert total == 6.0
    assert set(result) == {b, c}


def test_optimal_weighted_empty_input() -> None:
    assert _optimal_weighted([], {}) == []


def test_optimal_weighted_single_match() -> None:
    m = _raw(0, 5, "x")
    assert _optimal_weighted([m], {"x": 3.0}) == [m]


def test_optimal_weighted_no_overlap_all_selected() -> None:
    a = _raw(0, 3, "a")
    b = _raw(5, 8, "b")
    c = _raw(10, 13, "c")
    priorities = {"a": 1.0, "b": 2.0, "c": 3.0}
    result = _optimal_weighted([a, b, c], priorities)
    assert set(result) == {a, b, c}


def test_optimal_weighted_result_is_non_overlapping() -> None:
    matches = [_raw(0, 5, "a"), _raw(3, 8, "b"), _raw(6, 10, "c")]
    priorities = {"a": 1.0, "b": 5.0, "c": 1.0}
    result = _optimal_weighted(matches, priorities)
    assert _non_overlapping(result)


def test_optimal_weighted_result_sorted_by_start() -> None:
    matches = [_raw(10, 14, "a"), _raw(0, 4, "b"), _raw(5, 9, "c")]
    priorities = {"a": 1.0, "b": 1.0, "c": 1.0}
    result = _optimal_weighted(matches, priorities)
    assert all(result[i].start <= result[i + 1].start for i in range(len(result) - 1))


# ---------------------------------------------------------------------------
# OPTIMAL_WEIGHTED through KeywordStore
# ---------------------------------------------------------------------------


def test_store_optimal_weighted_beats_greedy() -> None:
    # Text "ab cd": keywords "ab cd" (priority=5, spans whole phrase),
    # "ab" (priority=3, left word), "cd" (priority=3, right word).
    # Greedy by priority picks "ab cd" (5); optimal picks "ab"+"cd" (3+3=6).
    store = KeywordStore(policy=MatchPolicy.OPTIMAL_WEIGHTED)
    store.add_keyword("ab cd", canonical="LONG", priority=5.0)
    store.add_keyword("ab", canonical="SHORT1", priority=3.0)
    store.add_keyword("cd", canonical="SHORT2", priority=3.0)
    matches = store.extract("ab cd")
    total = sum(m.priority for m in matches)
    assert total == 6.0
    assert {m.canonical for m in matches} == {"SHORT1", "SHORT2"}


def test_store_optimal_weighted_replace() -> None:
    store = KeywordStore(policy=MatchPolicy.OPTIMAL_WEIGHTED)
    store.add_keyword("ab cd", canonical="LONG", priority=5.0)
    store.add_keyword("ab", canonical="SHORT1", priority=3.0)
    store.add_keyword("cd", canonical="SHORT2", priority=3.0)
    result = store.replace("ab cd")
    assert result == "SHORT1 SHORT2"


# ---------------------------------------------------------------------------
# Property-based: OPTIMAL_WEIGHTED vs brute-force oracle
# ---------------------------------------------------------------------------


# Generate a list of non-negative-width intervals with unique keyword IDs.
_interval_st = st.builds(
    lambda start, length, kid: RawMatch(start, start + length, kid, 0),
    start=st.integers(min_value=0, max_value=50),
    length=st.integers(min_value=1, max_value=10),
    kid=st.text(alphabet="abcdefghij", min_size=1, max_size=3),
)


@given(
    matches=st.lists(_interval_st, min_size=0, max_size=10),
    base_priority=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=300)
def test_optimal_weighted_achieves_oracle_weight(
    matches: list[RawMatch], base_priority: float
) -> None:
    # Assign each unique kid a random-ish priority derived from base_priority.
    kids = list({m.keyword_id for m in matches})
    priorities = {k: base_priority * (1 + i * 0.5) for i, k in enumerate(kids)}

    result = _optimal_weighted(matches, priorities)

    # Result must be non-overlapping.
    assert _non_overlapping(result), "result contains overlapping spans"

    # Total weight must equal the brute-force optimum.
    oracle = brute_force_optimal_weight(matches, priorities)
    actual = sum(priorities.get(m.keyword_id, 1.0) for m in result)
    assert abs(actual - oracle) < 1e-9, (
        f"weight {actual} != oracle {oracle}\nmatches={matches}\npriorities={priorities}"
    )


@given(
    matches=st.lists(_interval_st, min_size=0, max_size=8),
)
@settings(max_examples=200)
def test_highest_priority_result_is_always_non_overlapping(
    matches: list[RawMatch],
) -> None:
    priorities = {m.keyword_id: 1.0 for m in matches}
    result = _highest_priority(matches, priorities)
    assert _non_overlapping(result)
