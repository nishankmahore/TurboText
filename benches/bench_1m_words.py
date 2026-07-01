
import random
import string
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from flashtext import KeywordProcessor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from turbotext import KeywordStore, MatchPolicy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_KEYWORDS = 1_000          # dictionary size
N_TEXT_WORDS = 1_000_000    # exactly 1 M word tokens in the corpus
N_REPS = 3                  # best-of-N timing runs
_RNG = random.Random(42)


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

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


def _build_text(keywords: list[str], n_words: int) -> str:
    tokens: list[str] = []
    for _ in range(n_words):
        if _RNG.random() < 0.5:
            tokens.append(_RNG.choice(keywords))
        else:
            tokens.append(_random_word(_RNG.randint(3, 8)))
    return " ".join(tokens)


def _timeit(fn: object, reps: int = N_REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"Building {N_KEYWORDS:,} keywords and {N_TEXT_WORDS:,}-word corpus …")
    keywords = _build_keywords(N_KEYWORDS)
    text = _build_text(keywords, N_TEXT_WORDS)

    n_chars = len(text)
    print(f"Corpus size: {n_chars:,} chars  ({n_chars / 1e6:.1f} MB)\n")
    print(text[:1000])

    # ── TurboText (k=0 exact) ────────────────────────────────────────────────
    tt = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
    for kw in keywords:
        tt.add_keyword(kw)

    t_tt = _timeit(lambda s=tt, t=text: s.extract(t))
    n_tt = len(tt.extract(text))
    print(f"TurboText  (k=0) : {t_tt:.3f}s  ({n_tt:,} matches)")

    # ── FlashText ────────────────────────────────────────────────────────────
    ft = KeywordProcessor(case_sensitive=False)
    for kw in keywords:
        ft.add_keyword(kw)

    t_ft = _timeit(lambda k=ft, t=text: k.extract_keywords(t))
    n_ft = len(ft.extract_keywords(text))
    print(f"FlashText        : {t_ft:.3f}s  ({n_ft:,} matches)")

    speedup = t_ft / t_tt
    print(f"\nTurboText is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than FlashText")

    # ── Chart ────────────────────────────────────────────────────────────────
    labels = ["TurboText\n(k=0 exact)", "FlashText"]
    times  = [t_tt, t_ft]
    colors = ["#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("white")
    ax.grid(True, axis="y", color="white", linewidth=1.2, zorder=0)

    bars = ax.bar(labels, times, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.4, zorder=3)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            f"{t:.2f}s",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    ax.set_ylabel("Time (sec)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(
        f"1 M-word corpus · {N_KEYWORDS:,} keywords  —  wall-clock time (lower is better)",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(bottom=0, top=max(times) * 1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    out = Path(__file__).parent / "1m_throughput.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    run()
