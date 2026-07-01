
import random
import string

import pytest

from turbotext import FuzzyConfig, KeywordStore, MatchPolicy

# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

_RNG = random.Random(42)


def _random_word(length: int) -> str:
    return "".join(_RNG.choices(string.ascii_lowercase, k=length))


def _build_keywords(n: int, avg_len: int = 8) -> list[str]:
    return [_random_word(_RNG.randint(avg_len - 2, avg_len + 2)) for _ in range(n)]


def _build_text(keywords: list[str], n_words: int, typo_rate: float = 0.0) -> str:
    """Interleave keyword hits with random filler words.

    At typo_rate > 0, inject a one-character substitution into keyword occurrences
    to exercise the fuzzy path.
    """
    filler = [_random_word(_RNG.randint(3, 8)) for _ in range(n_words)]
    tokens: list[str] = []
    for i in range(n_words):
        tokens.append(filler[i])
        kw = _RNG.choice(keywords)
        if typo_rate > 0 and _RNG.random() < typo_rate:
            # Substitute one random character
            pos = _RNG.randint(0, len(kw) - 1)
            sub = _RNG.choice(string.ascii_lowercase.replace(kw[pos], ""))
            kw = kw[:pos] + sub + kw[pos + 1 :]
        tokens.append(kw)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_KEYWORDS = 100
N_TEXT_WORDS = 2_000  # ~20 k chars


@pytest.fixture(scope="module")
def keywords() -> list[str]:
    return _build_keywords(N_KEYWORDS)


@pytest.fixture(scope="module")
def exact_text(keywords: list[str]) -> str:
    return _build_text(keywords, N_TEXT_WORDS, typo_rate=0.0)


@pytest.fixture(scope="module")
def typo_text(keywords: list[str]) -> str:
    # Half of keyword occurrences have a 1-char substitution.
    return _build_text(keywords, N_TEXT_WORDS, typo_rate=0.5)


@pytest.fixture(scope="module")
def store_exact(keywords: list[str]) -> KeywordStore:
    store = KeywordStore(policy=MatchPolicy.ALL_OVERLAPS)
    for kw in keywords:
        store.add_keyword(kw)
    return store


@pytest.fixture(scope="module")
def store_fuzzy(keywords: list[str]) -> KeywordStore:
    store = KeywordStore(
        policy=MatchPolicy.ALL_OVERLAPS,
        fuzzy=FuzzyConfig(max_edit_distance=1),
    )
    for kw in keywords:
        store.add_keyword(kw)
    return store


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def test_bench_exact_k0(  # type: ignore[type-arg]
    benchmark: pytest.fixture,
    store_exact: KeywordStore,
    exact_text: str,
) -> None:
    """Exact matching (k=0) on a clean corpus — baseline."""
    result = benchmark(store_exact.extract, exact_text)
    assert len(result) > 0


def test_bench_fuzzy_k1_clean(  # type: ignore[type-arg]
    benchmark: pytest.fixture,
    store_fuzzy: KeywordStore,
    exact_text: str,
) -> None:
    """Fuzzy matching (k=1) on a clean corpus — frontier branching overhead with no typos."""
    result = benchmark(store_fuzzy.extract, exact_text)
    assert len(result) > 0


def test_bench_fuzzy_k1_typos(  # type: ignore[type-arg]
    benchmark: pytest.fixture,
    store_fuzzy: KeywordStore,
    typo_text: str,
) -> None:
    """Fuzzy matching (k=1) on a 50%-typo corpus — the primary fuzzy use-case."""
    result = benchmark(store_fuzzy.extract, typo_text)
    assert len(result) > 0
