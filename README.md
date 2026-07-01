<p align="center">
  <img src="https://raw.githubusercontent.com/nishankmahore/TurboText/main/assets/logo.png" alt="TurboText" width="360"/>
</p>

<h1 align="center">TurboText</h1>

<p align="center">
  <b>Lightning-fast, boundary-aware keyword matching for Python</b><br/>
  Exact · Fuzzy · Multi-word · Unicode · Pluggable conflict resolution
</p>

<p align="center">
  <a href="https://pypi.org/project/turbotext/"><img src="https://img.shields.io/badge/PyPI-0.2.0-blue?logo=pypi&logoColor=white" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python&logoColor=white" alt="Python"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"/></a>
  <a href="https://github.com/nishankmahore/TurboText/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-passing-brightgreen?logo=github" alt="CI"/></a>
  <img src="https://img.shields.io/badge/typed-yes-informational" alt="Typed"/>
  <img src="https://img.shields.io/badge/Cython-accelerated-orange" alt="Cython"/>
</p>

---

> TurboText finds keywords in text the way a human editor would — enforcing real word boundaries, tolerating typos, handling multi-word phrases, and letting you declare which match wins when keywords overlap, including an **optimal** (not just greedy) weighted resolver that no other keyword library provides.

---

## Table of Contents

- [Why TurboText?](#why-turbotext)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Adding Keywords](#adding-keywords)
- [Extracting Matches](#extracting-matches)
- [Replacing Keywords](#replacing-keywords)
- [Fuzzy Matching](#fuzzy-matching)
- [Conflict Resolution Policies](#conflict-resolution-policies)
- [Per-Keyword Metadata](#per-keyword-metadata)
- [Word Boundary Rules](#word-boundary-rules)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Development](#development)
- [Author](#author)
- [License](#license)

---

## Why TurboText?

| | `re` | FlashText | RapidFuzz | **TurboText** |
|---|:---:|:---:|:---:|:---:|
| Exact keyword extraction | ✅ | ✅ | ❌ | ✅ |
| Fuzzy / typo-tolerant | ❌ | ❌ | ✅ | ✅ |
| Word boundary enforcement | ✅ | ✅ | ❌ | ✅ |
| Extracts spans from running text | ✅ | ✅ | ❌ | ✅ |
| Multi-word phrases (native) | ✅ | ✅ | ⚠️ | ✅ |
| O(text) scaling with vocab | ❌ | ✅ | ❌ | ✅ |
| Per-keyword metadata | ❌ | ❌ | ❌ | ✅ |
| Conflict resolution policies | ❌ | ❌ | ❌ | ✅ |
| Optimal weighted resolution | ❌ | ❌ | ❌ | ✅ |

---

## Features

- **Exact matching** — trie-based O(n) scan, constant in vocabulary size
- **Fuzzy matching** — Levenshtein distance up to *k* via bounded-edit frontier search
- **Word boundary enforcement** — Unicode-correct; matches `cat` but not `cats`, `scat`, or `concatenate`
- **Multi-word keywords** — `"new york"`, `"product manager"` work out of the box
- **Bulk loading** — accepts `list`, `dict[str, str]`, or `dict[str, list[str]]`
- **Five conflict policies** — `ALL_OVERLAPS`, `LEFTMOST_LONGEST`, `LEFTMOST_FIRST`, `HIGHEST_PRIORITY`, `OPTIMAL_WEIGHTED`
- **Optimal resolution** — O(n log n) weighted interval scheduling DP — picks the globally best non-overlapping set
- **Rich match objects** — canonical form, offsets, edit distance, category, priority, custom metadata
- **Cython fast-path** — compiled extension ships with every wheel; **1.4× faster than FlashText** on 1 M-word corpora

---

## Installation

```bash
pip install turbotext
```

The Cython extension is built automatically when installing from PyPI (wheels are pre-compiled for CPython 3.10–3.13 on Linux, macOS, and Windows). No extra steps needed.

TurboText falls back to pure Python automatically if the compiled extension is unavailable (e.g. unsupported platform or source install without a C compiler).

<details>
<summary>Install with uv (development)</summary>

```bash
git clone https://github.com/nishankmahore/TurboText
cd TurboText
uv sync --group dev
python setup.py build_ext --inplace
```

</details>

---

## Quick Start

```python
from turbotext import KeywordStore, MatchPolicy, FuzzyConfig

store = KeywordStore(
    policy=MatchPolicy.LEFTMOST_LONGEST,
    fuzzy=FuzzyConfig(max_edit_distance=1),
)

store.add_keywords({
    "aspirin":   ["aspirin", "asprin", "aspirin tablet"],
    "ibuprofen": ["ibuprofen", "ibuprofen tablet"],
})

text = "Patient takes asprin and ibuprofen tablet daily"

for m in store.extract(text):
    print(f"{m.canonical:12}  ed={m.edit_distance}  [{m.start}:{m.end}]  '{m.text}'")
# aspirin       ed=1  [14:20]  'asprin'
# ibuprofen     ed=0  [25:42]  'ibuprofen tablet'

print(store.replace(text))
# Patient takes aspirin and ibuprofen daily
```

---

## Adding Keywords

### Single keyword

```python
store = KeywordStore()

# minimal — surface form becomes the canonical
store.add_keyword("aspirin")

# with canonical form
store.add_keyword("aspirin", canonical="Aspirin")

# with full metadata
kid = store.add_keyword(
    "aspirin",
    canonical="Aspirin",
    category="DRUG",
    priority=10.0,
    rxnorm_code="1191",   # any extra kwargs become metadata
    source="rxnorm",
)
print(kid)  # "3f4a2b1c-..."  UUID — useful for tracking
```

### From a list

```python
store.add_keywords(["java", "python", "rust"])
# surface form = canonical for each entry
```

### From a `{surface: canonical}` dict

```python
store.add_keywords({
    "py":   "Python",
    "js":   "JavaScript",
    "k8s":  "Kubernetes",
})
```

### From a `{canonical: [surfaces]}` dict  *(most common bulk shape)*

```python
keyword_dict = {
    "java":               ["java", "java_2e", "java programing"],
    "product management": ["PM", "product manager", "prod mgmt"],
    "machine learning":   ["ML", "machine learning", "deep learning"],
}
store.add_keywords(keyword_dict)
```

### Mixed dict

String values follow `{surface: canonical}`; list values follow `{canonical: [surfaces]}`.

```python
store.add_keywords({
    "java": ["java_2e", "java programing"],  # canonical → [surfaces]
    "py":   "Python",                         # surface   → canonical
})
```

### Shared category and priority

```python
store.add_keywords(
    {
        "aspirin":   ["aspirin", "asprin"],
        "ibuprofen": ["ibuprofen", "advil"],
    },
    category="DRUG",
    priority=5.0,
)
```

---

## Extracting Matches

```python
matches = store.extract("take aspirin and ibuprofen daily")

for m in matches:
    print(m.text)           # surface text found in the input
    print(m.canonical)      # normalised keyword name
    print(m.start, m.end)   # character offsets — text[m.start:m.end]
    print(m.edit_distance)  # 0 = exact, 1 = one typo, etc.
    print(m.category)       # "DRUG"
    print(m.priority)       # 10.0
    print(m.keyword_id)     # UUID string
    print(m.metadata)       # {"rxnorm_code": "1191", ...}
```

`Match` uses `__slots__` for fast bulk creation — all fields are fixed and the object is lightweight.

---

## Replacing Keywords

```python
store = KeywordStore()
store.add_keywords({
    "aspirin":       "Aspirin",
    "tylenol":       "Acetaminophen",   # alias → same canonical
    "ibuprofen":     "Ibuprofen",
})

print(store.replace("take aspirin or tylenol twice daily"))
# take Aspirin or Acetaminophen twice daily
```

---

## Fuzzy Matching

```python
from turbotext import FuzzyConfig

store = KeywordStore(fuzzy=FuzzyConfig(max_edit_distance=1))
store.add_keyword("aspirin")

store.extract("take asprin")    # substitution  i → r  ✅
store.extract("take aspirn")    # deletion      missing i  ✅
store.extract("take aspirrin")  # insertion     extra r  ✅
```

### Boundary enforcement still applies

```python
store.extract("I see cbt")   # ✅ whole word
store.extract("I see xcbt")  # ❌ left boundary fails
store.extract("I see cbts")  # ❌ right boundary fails
```

### Choosing `max_edit_distance`

| Value | Use case |
|:---:|---|
| `0` | Exact matching only *(default)* — uses Cython fast-path when available |
| `1` | Single-character typos — medical terms, product names |
| `2` | Two-character errors — longer technical terms |

> Higher values increase recall but also false positives. Start with `1` and tune.

---

## Conflict Resolution Policies

When keywords overlap, the policy decides which match to keep.

<details>
<summary><b>ALL_OVERLAPS</b> — return everything, you decide</summary>

```python
store = KeywordStore(policy=MatchPolicy.ALL_OVERLAPS)
store.add_keywords(["new", "new york", "york"])

store.extract("new york")
# → ["new", "new york", "york"]
```

</details>

<details>
<summary><b>LEFTMOST_LONGEST</b> (default) — FlashText-compatible greedy</summary>

```python
store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
store.add_keywords(["new", "new york"])

store.extract("new york")
# → ["new york"]   longest match wins
```

</details>

<details>
<summary><b>LEFTMOST_FIRST</b> — earliest start wins</summary>

```python
store = KeywordStore(policy=MatchPolicy.LEFTMOST_FIRST)
store.add_keywords(["new", "new york"])

store.extract("new york")
# → ["new"]   first token wins
```

</details>

<details>
<summary><b>HIGHEST_PRIORITY</b> — priority-based greedy</summary>

```python
store = KeywordStore(policy=MatchPolicy.HIGHEST_PRIORITY)
store.add_keyword("new york", canonical="New York", priority=2.0)
store.add_keyword("york",     canonical="York",     priority=5.0)
store.add_keyword("new",      canonical="New",      priority=1.0)

store.extract("new york")
# → ["New", "York"]   priority 5 beats priority 2; "new" doesn't overlap "york"
```

</details>

<details>
<summary><b>OPTIMAL_WEIGHTED</b> — globally optimal, not greedy</summary>

```python
store = KeywordStore(policy=MatchPolicy.OPTIMAL_WEIGHTED)
store.add_keyword("ab cd", canonical="LONG",   priority=5.0)
store.add_keyword("ab",    canonical="SHORT1", priority=3.0)
store.add_keyword("cd",    canonical="SHORT2", priority=3.0)

store.extract("ab cd")
# Greedy picks "LONG" (5).  Optimal picks "SHORT1"+"SHORT2" (3+3=6).
# → ["SHORT1", "SHORT2"]
```

</details>

---

## Per-Keyword Metadata

```python
store.add_keyword(
    "aspirin",
    canonical="Aspirin",
    category="DRUG",
    priority=10.0,
    rxnorm_code="1191",
    source="rxnorm",
    approved=True,
)

m = store.extract("take aspirin")[0]
print(m.canonical)               # "Aspirin"
print(m.category)                # "DRUG"
print(m.priority)                # 10.0
print(m.metadata["rxnorm_code"]) # "1191"
```

`Match.metadata` is a **shallow copy** — mutating it does not affect the stored keyword.

---

## Word Boundary Rules

TurboText uses Unicode word boundaries — word characters are `[a-zA-Z0-9_]` and their Unicode equivalents.

```python
store.add_keyword("cat")

# ✅ Accepted
store.extract("cat")          # text edge
store.extract("the cat sat")  # spaces
store.extract("(cat)")        # punctuation
store.extract("cat, sat")     # comma

# ❌ Rejected
store.extract("cats")         # right boundary fails
store.extract("scat")         # left boundary fails
store.extract("cat2")         # digit is a word char
store.extract("concatenate")  # substring
```

---

## Performance

> Apple M-series · best-of-3 runs · Cython extension enabled

### 1 M-word throughput (1,000 keywords, 7.8 MB corpus)

TurboText's Aho-Corasick engine with inline lowercasing and zero-copy resolve beats FlashText on large documents.

![1M throughput](https://raw.githubusercontent.com/nishankmahore/TurboText/main/benches/1m_throughput.png)

| Library | Time (s) | Matches | vs FlashText |
|---|:---:|:---:|:---:|
| **TurboText (k=0)** | **0.61** | ~500,000 | **1.4× faster** |
| FlashText | 0.85 | ~500,000 | baseline |

---

### Exact matching — vocabulary scaling (k=0)

Both TurboText and FlashText are O(text) — scan time is flat as vocabulary grows. `re` alternation degrades linearly with term count.

![k=0 scaling](https://raw.githubusercontent.com/nishankmahore/TurboText/main/benches/k0_scaling.png)

| Library | 100 terms | 1,000 terms | 5,000 terms | 20,000 terms | Complexity |
|---|:---:|:---:|:---:|:---:|:---:|
| **TurboText** | **1.4 ms** | **1.6 ms** | **2.0 ms** | **2.9 ms** | O(text) |
| FlashText | 2.9 ms | 3.1 ms | 3.6 ms | 3.3 ms | O(text) |
| `re` | 4.8 ms | 40.3 ms | 213.5 ms | 834.8 ms | O(text × vocab) |

---

### Fuzzy matching — vocabulary scaling (k=1)

TurboText does a **single-pass trie scan** regardless of vocabulary size. RapidFuzz and FuzzyWuzzy tokenise then score every token against every keyword — O(tokens × vocab).

![k=1 scaling](https://raw.githubusercontent.com/nishankmahore/TurboText/main/benches/k1_scaling.png)

| Library | 100 terms | 500 terms | 1,000 terms | 2,000 terms | 5,000 terms | Boundary-aware | Extracts spans |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TurboText** | 93 ms | 141 ms | 172 ms | 209 ms | **237 ms** | ✅ | ✅ |
| RapidFuzz | **13 ms** | 63 ms | 122 ms | 257 ms | 607 ms | ❌ | ❌ |
| FuzzyWuzzy | 222 ms | 1,109 ms | 2,208 ms | 4,423 ms | 10,924 ms | ❌ | ❌ |

TurboText overtakes RapidFuzz at ~2,000 keywords and is **2.6× faster** at 5,000 terms. Unlike RapidFuzz (a string scorer used with `process.extractOne` per token), TurboText returns character offsets, enforces word boundaries, and handles multi-word phrases natively.

---

## API Reference

<details>
<summary><b>KeywordStore(policy, fuzzy)</b></summary>

```python
store = KeywordStore(
    policy=MatchPolicy.LEFTMOST_LONGEST,    # default
    fuzzy=FuzzyConfig(max_edit_distance=0), # default — exact only
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy` | `MatchPolicy` | `LEFTMOST_LONGEST` | Conflict-resolution strategy |
| `fuzzy` | `FuzzyConfig \| None` | `None` | Fuzzy config; `None` = exact only |

</details>

<details>
<summary><b>add_keyword(surface_form, *, canonical, category, priority, **metadata) → str</b></summary>

```python
kid = store.add_keyword(
    "aspirin",
    canonical="Aspirin",
    category="DRUG",
    priority=10.0,
    rxnorm_code="1191",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `surface_form` | `str` | required | Text to search for |
| `canonical` | `str` | `surface_form` | Normalised name returned on match |
| `category` | `str \| None` | `None` | Grouping label |
| `priority` | `float` | `1.0` | Weight for priority-based policies |
| `**metadata` | `Any` | — | Arbitrary extra fields |

Returns the keyword's UUID string.

</details>

<details>
<summary><b>add_keywords(keywords, *, category, priority) → list[str]</b></summary>

| Shape | Example |
|---|---|
| `list[str]` | `["java", "python"]` |
| `dict[str, str]` | `{"py": "Python"}` — surface → canonical |
| `dict[str, list[str]]` | `{"Python": ["py", "python3"]}` — canonical → surfaces |

`category` and `priority` apply to every keyword in the call.

</details>

<details>
<summary><b>extract(text) → list[Match]</b></summary>

Returns resolved matches in span order.

```python
matches = store.extract("Patient takes aspirin daily")
```

</details>

<details>
<summary><b>replace(text) → str</b></summary>

Replaces every matched span with its canonical form.

```python
result = store.replace("take aspirin or tylenol")
```

</details>

<details>
<summary><b>Match fields</b></summary>

`Match` uses `__slots__` — all fields are set at construction and the object is lightweight.

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Matched surface text as it appears in the input |
| `canonical` | `str` | Normalised form from `add_keyword` |
| `start` | `int` | Start character offset |
| `end` | `int` | End character offset (exclusive) |
| `edit_distance` | `int` | Levenshtein distance (0 = exact) |
| `category` | `str \| None` | User-supplied category |
| `priority` | `float` | User-supplied priority |
| `keyword_id` | `str` | UUID from `add_keyword` |
| `metadata` | `dict` | Copy of extra kwargs |

</details>

<details>
<summary><b>FuzzyConfig and MatchPolicy</b></summary>

```python
FuzzyConfig(max_edit_distance=1)  # int, default 0
```

| `MatchPolicy` | Description |
|---|---|
| `ALL_OVERLAPS` | Return every match |
| `LEFTMOST_LONGEST` | Greedy — leftmost, then longest |
| `LEFTMOST_FIRST` | Greedy — leftmost, then insertion order |
| `HIGHEST_PRIORITY` | Greedy — highest priority wins cluster |
| `OPTIMAL_WEIGHTED` | Exact — maximise total priority globally |

</details>

---

## Development

```bash
uv sync --group dev                        # install dependencies
python setup.py build_ext --inplace        # build Cython extension
uv run pytest                              # run tests
uv run ruff check src tests                # lint
uv run mypy src/turbotext                  # type-check
uv run pytest benches/ --benchmark-only    # throughput benchmarks
uv run python benches/bench_1m_words.py    # 1 M-word TurboText vs FlashText
uv run python benches/scaling_benchmark.py # regenerate scaling charts
```

<details>
<summary>Project layout</summary>

```
src/turbotext/
    __init__.py       public API exports
    trie.py           TrieNode + TrieBuilder
    frontier.py       bounded-edit frontier search (pure Python)
    _fast.pyx         Cython hot-path for exact search (k=0)
    _fast.pyi         type stub for the Cython extension
    resolve.py        conflict-resolution policies
    store.py          KeywordStore public class
reference/
    reference_matcher.py    brute-force oracle for differential testing
tests/
    test_m0_smoke.py                   API surface + add_keywords shapes
    test_m1_exact.py                   exact matching + boundary rules
    test_m2_metadata_priorities.py     metadata, HIGHEST_PRIORITY, OPTIMAL_WEIGHTED
    test_m3_fuzzy.py                   fuzzy matching + hypothesis property tests
benches/
    bench_m3.py             pytest-benchmark: k=0 vs k=1 throughput
    bench_comparison.py     pytest-benchmark: TurboText vs FlashText vs re vs RapidFuzz
    bench_1m_words.py       1 M-word throughput: TurboText vs FlashText
    scaling_benchmark.py    vocabulary-sweep scaling charts
assets/
    logo.png
```

</details>

---

## Author

<p>
  <b>Nishank Mahore</b><br/>
  <a href="mailto:nishankmahore@gmail.com">nishankmahore@gmail.com</a> ·
  <a href="https://github.com/nishankmahore">github.com/nishankmahore</a>
</p>

If TurboText is useful to you, feel free to open an issue, suggest a feature, or contribute a pull request.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE) for the full text.

---

<p align="center">
  Made with Python · Powered by Cython · Built for speed
</p>
