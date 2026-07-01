import random
import re
import string

import pytest
from flashtext import KeywordProcessor
from fuzzywuzzy import fuzz

from turbotext import FuzzyConfig, KeywordStore, MatchPolicy


_RNG = random.Random(42)


def _random_word(length: int) -> str:
    return "".join(_RNG.choices(string.ascii_lowercase, k=length))


def _build_keywords(n: int, avg_len: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        w = _random_word(_RNG.randint(avg_len - 2, avg_len + 2))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _build_text(keywords: list[str], n_words: int, typo_rate: float = 0.0) -> str:
    filler = [_random_word(_RNG.randint(3, 8)) for _ in range(n_words)]
    tokens: list[str] = []
    for i in range(n_words):
        tokens.append(filler[i])
        kw = _RNG.choice(keywords)
        if typo_rate > 0 and _RNG.random() < typo_rate:
            pos = _RNG.randint(0, len(kw) - 1)
            sub = _RNG.choice(string.ascii_lowercase.replace(kw[pos], ""))
            kw = kw[:pos] + sub + kw[pos + 1 :]
        tokens.append(kw)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_KEYWORDS = 100
N_TEXT_WORDS = 2_000

# fuzzywuzzy ratio threshold for "1 edit away" on avg 8-char words (~87% match)
_FUZZY_THRESHOLD = 80


@pytest.fixture(scope="module")
def keywords() -> list[str]:
    return _build_keywords(N_KEYWORDS)


@pytest.fixture(scope="module")
def exact_text(keywords: list[str]) -> str:
    return _build_text(keywords, N_TEXT_WORDS, typo_rate=0.0)


@pytest.fixture(scope="module")
def typo_text(keywords: list[str]) -> str:
    return _build_text(keywords, N_TEXT_WORDS, typo_rate=0.5)


# --- TurboText stores ---

@pytest.fixture(scope="module")
def tt_exact(keywords: list[str]) -> KeywordStore:
    store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
    for kw in keywords:
        store.add_keyword(kw)
    return store


@pytest.fixture(scope="module")
def tt_fuzzy(keywords: list[str]) -> KeywordStore:
    store = KeywordStore(
        policy=MatchPolicy.LEFTMOST_LONGEST,
        fuzzy=FuzzyConfig(max_edit_distance=1),
    )
    for kw in keywords:
        store.add_keyword(kw)
    return store


# --- FlashText processor ---

@pytest.fixture(scope="module")
def ft_processor(keywords: list[str]) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for kw in keywords:
        kp.add_keyword(kw)
    return kp


# --- re: compiled alternation pattern ---

@pytest.fixture(scope="module")
def re_pattern(keywords: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(kw) for kw in keywords]
    # Sort longest first so the alternation prefers longer matches (FlashText-equivalent)
    escaped.sort(key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Exact matching benchmarks
# ---------------------------------------------------------------------------


def test_bench_exact_turbotext(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    tt_exact: KeywordStore,
    exact_text: str,
) -> None:
    result = benchmark(tt_exact.extract, exact_text)
    assert len(result) > 0


def test_bench_exact_flashtext(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    ft_processor: KeywordProcessor,
    exact_text: str,
) -> None:
    result = benchmark(ft_processor.extract_keywords, exact_text)
    assert len(result) > 0


def test_bench_exact_re(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    re_pattern: re.Pattern[str],
    exact_text: str,
) -> None:
    def run() -> list[str]:
        return re_pattern.findall(exact_text)

    result = benchmark(run)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Fuzzy matching benchmarks
# ---------------------------------------------------------------------------


def _fuzzywuzzy_scan(
    text: str, keywords: list[str], threshold: int
) -> list[tuple[str, str]]:
    """Naive O(tokens × keywords) fuzzy scan using fuzzywuzzy."""
    hits: list[tuple[str, str]] = []
    tokens = text.split()
    for token in tokens:
        for kw in keywords:
            if fuzz.ratio(token, kw) >= threshold:
                hits.append((token, kw))
                break
    return hits


def test_bench_fuzzy_turbotext(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    tt_fuzzy: KeywordStore,
    typo_text: str,
) -> None:
    result = benchmark(tt_fuzzy.extract, typo_text)
    assert len(result) > 0


def test_bench_fuzzy_fuzzywuzzy(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    keywords: list[str],
    typo_text: str,
) -> None:
    result = benchmark(_fuzzywuzzy_scan, typo_text, keywords, _FUZZY_THRESHOLD)
    assert len(result) > 0
