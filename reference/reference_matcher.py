
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RefMatch:
    start: int
    end: int
    surface_form: str
    canonical: str
    edit_distance: int


def _levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein. O(|a|*|b|), correct by construction."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


_WORD_CHAR = re.compile(r"\w")


def _is_boundary(ch: str) -> bool:
    return _WORD_CHAR.match(ch) is None


def extract_all_overlaps(
    text: str,
    keywords: dict[str, str],  # surface_form -> canonical
    max_edit_distance: int = 0,
) -> list[RefMatch]:
    """Brute-force O(n^2 * k) search — every substring, every keyword, every distance.

    Returns all overlapping matches satisfying word-boundary constraints.
    """
    text_lower = text.lower()
    n = len(text_lower)
    results: list[RefMatch] = []

    for surface, canonical in keywords.items():
        sf_lower = surface.lower()
        sf_len = len(sf_lower)

        for start in range(n):
            # Only begin at a word boundary
            if start > 0 and not _is_boundary(text_lower[start - 1]):
                continue

            # Try all span lengths that could plausibly match within the edit budget
            for end in range(start + max(1, sf_len - max_edit_distance),
                             min(n, start + sf_len + max_edit_distance) + 1):
                # Only end at a word boundary
                if end < n and not _is_boundary(text_lower[end]):
                    continue

                span = text_lower[start:end]
                dist = _levenshtein(span, sf_lower)
                if dist <= max_edit_distance:
                    results.append(RefMatch(start, end, surface, canonical, dist))

    return sorted(results, key=lambda m: (m.start, m.end))
