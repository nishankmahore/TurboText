
import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class RawMatch:
    """An unresolved match as emitted by the search engine."""

    start: int
    end: int
    keyword_id: str
    edit_distance: int


class MatchPolicy(enum.Enum):
    """Conflict-resolution strategy applied after match collection."""

    ALL_OVERLAPS = "all_overlaps"
    LEFTMOST_LONGEST = "leftmost_longest"
    LEFTMOST_FIRST = "leftmost_first"
    HIGHEST_PRIORITY = "highest_priority"
    OPTIMAL_WEIGHTED = "optimal_weighted"


def resolve(
    matches: list[RawMatch],
    policy: MatchPolicy,
    priorities: dict[str, float] | None = None,
) -> list[RawMatch]:
    """Apply *policy* to *matches*, returning the selected subset in span order."""
    if not matches:
        return []
    if policy is MatchPolicy.ALL_OVERLAPS:
        return sorted(matches, key=lambda m: (m.start, m.end))
    if policy is MatchPolicy.LEFTMOST_LONGEST:
        return _leftmost_longest(matches)
    if policy is MatchPolicy.LEFTMOST_FIRST:
        return _leftmost_first(matches)
    if policy is MatchPolicy.HIGHEST_PRIORITY:
        return _highest_priority(matches, priorities or {})
    if policy is MatchPolicy.OPTIMAL_WEIGHTED:
        return _optimal_weighted(matches, priorities or {})
    raise ValueError(f"Unknown policy: {policy}")  # pragma: no cover


def _leftmost_longest(matches: list[RawMatch]) -> list[RawMatch]:
    """FlashText-compatible greedy: leftmost start wins; ties broken by longest span."""
    sorted_matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
    result: list[RawMatch] = []
    cursor = 0
    for m in sorted_matches:
        if m.start >= cursor:
            result.append(m)
            cursor = m.end
    return result


def _leftmost_first(matches: list[RawMatch]) -> list[RawMatch]:
    """Greedy: earliest start wins; ties broken by which match was found first."""
    sorted_matches = sorted(matches, key=lambda m: m.start)
    result: list[RawMatch] = []
    cursor = 0
    for m in sorted_matches:
        if m.start >= cursor:
            result.append(m)
            cursor = m.end
    return result


def _highest_priority(matches: list[RawMatch], priorities: dict[str, float]) -> list[RawMatch]:
    """Greedy: highest priority wins in each conflict cluster; ties by span length then position."""
    sorted_matches = sorted(
        matches,
        key=lambda m: (-(priorities.get(m.keyword_id, 1.0)), -(m.end - m.start), m.start),
    )
    result: list[RawMatch] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(m: RawMatch) -> bool:
        return any(m.start < e and m.end > s for s, e in occupied)

    for m in sorted_matches:
        if not overlaps(m):
            result.append(m)
            occupied.append((m.start, m.end))

    return sorted(result, key=lambda m: m.start)


def _optimal_weighted(matches: list[RawMatch], priorities: dict[str, float]) -> list[RawMatch]:
    """Weighted interval scheduling: maximise total priority over non-overlapping matches.

    Classic O(n log n) DP. This is the novel piece; no existing keyword matcher
    does optimal (rather than greedy) overlap resolution.
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: m.end)
    n = len(sorted_matches)

    # For each match i, find the latest j such that sorted_matches[j].end <= sorted_matches[i].start
    def latest_compatible(i: int) -> int:
        lo, hi = 0, i - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_matches[mid].end <= sorted_matches[i].start:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    weights = [priorities.get(m.keyword_id, 1.0) for m in sorted_matches]
    dp = [0.0] * (n + 1)
    for i in range(n):
        j = latest_compatible(i)
        dp[i + 1] = max(dp[i], dp[j + 1] + weights[i])

    # Backtrack to recover selected matches
    selected: list[RawMatch] = []
    i = n - 1
    while i >= 0:
        j = latest_compatible(i)
        if dp[j + 1] + weights[i] >= dp[i]:
            selected.append(sorted_matches[i])
            i = j
        else:
            i -= 1

    return sorted(selected, key=lambda m: m.start)
