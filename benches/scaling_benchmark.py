import random
import re
import string
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from flashtext import KeywordProcessor
from fuzzywuzzy import fuzz as fw_fuzz
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from turbotext import FuzzyConfig, KeywordStore, MatchPolicy

# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

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
            kw = kw[:pos] + sub + kw[pos + 1:]
        tokens.append(kw)
    return " ".join(tokens)


N_TEXT_WORDS = 2_000   # ~20 k chars per text
N_REPS = 3             # best-of-N for stability


def _timeit(fn: object, reps: int = N_REPS) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# k=0 benchmark: TurboText vs FlashText vs re
# ---------------------------------------------------------------------------

K0_COUNTS = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000]


def bench_k0() -> dict[str, list[float | None]]:
    print("=== k=0: Exact matching ===")
    results: dict[str, list[float | None]] = {
        "TurboText": [], "FlashText": [], "re": [],
    }

    for n in K0_COUNTS:
        print(f"  n={n:>6} ...", end="  ", flush=True)
        keywords = _build_keywords(n)
        text = _build_text(keywords, N_TEXT_WORDS)

        # TurboText (Cython fast path active at k=0)
        store = KeywordStore(policy=MatchPolicy.LEFTMOST_LONGEST)
        for kw in keywords:
            store.add_keyword(kw)
        results["TurboText"].append(round(_timeit(lambda s=store, t=text: s.extract(t)), 4))

        # FlashText
        kp = KeywordProcessor(case_sensitive=False)
        for kw in keywords:
            kp.add_keyword(kw)
        results["FlashText"].append(round(_timeit(lambda k=kp, t=text: k.extract_keywords(t)), 4))

        # re — alternation pattern; sort longest-first for greedy semantics
        escaped = sorted([re.escape(kw) for kw in keywords], key=len, reverse=True)
        try:
            pat = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
            results["re"].append(round(_timeit(lambda p=pat, t=text: p.findall(t)), 4))
        except Exception:
            results["re"].append(None)

        tt = results["TurboText"][-1]
        ft = results["FlashText"][-1]
        rv = results["re"][-1]
        print(f"TurboText={tt:.4f}s  FlashText={ft:.4f}s  re={rv if rv else 'N/A'}")

    return results


# ---------------------------------------------------------------------------
# k=1 benchmark: TurboText vs RapidFuzz vs FuzzyWuzzy
# ---------------------------------------------------------------------------

K1_COUNTS = [100, 500, 1000, 2000, 5000]
_THRESHOLD = 80   # ratio score ≈ 1-edit-away for avg 8-char words


def _rapidfuzz_scan(tokens: list[str], keywords: list[str]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for token in tokens:
        m = rf_process.extractOne(
            token, keywords, scorer=rf_fuzz.ratio, score_cutoff=_THRESHOLD
        )
        if m:
            hits.append((token, m[0]))
    return hits


def _fuzzywuzzy_scan(tokens: list[str], keywords: list[str]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for token in tokens:
        for kw in keywords:
            if fw_fuzz.ratio(token, kw) >= _THRESHOLD:
                hits.append((token, kw))
                break
    return hits


def bench_k1() -> dict[str, list[float | None]]:
    print("\n=== k=1: Fuzzy matching ===")
    results: dict[str, list[float | None]] = {
        "TurboText": [], "RapidFuzz": [], "FuzzyWuzzy": [],
    }

    for n in K1_COUNTS:
        print(f"  n={n:>6} ...", end="  ", flush=True)
        keywords = _build_keywords(n)
        text = _build_text(keywords, N_TEXT_WORDS, typo_rate=0.5)
        tokens = text.split()

        # TurboText (pure Python frontier; boundary-aware, multi-word capable)
        store = KeywordStore(
            policy=MatchPolicy.LEFTMOST_LONGEST,
            fuzzy=FuzzyConfig(max_edit_distance=1),
        )
        for kw in keywords:
            store.add_keyword(kw)
        results["TurboText"].append(round(_timeit(lambda s=store, t=text: s.extract(t)), 4))

        # RapidFuzz: token scan (C-accelerated Levenshtein, no boundary enforcement)
        kw_list = list(keywords)
        results["RapidFuzz"].append(
            round(_timeit(lambda tk=tokens, kl=kw_list: _rapidfuzz_scan(tk, kl)), 4)
        )

        # FuzzyWuzzy: same token scan backed by Python Levenshtein
        results["FuzzyWuzzy"].append(
            round(_timeit(lambda tk=tokens, kl=kw_list: _fuzzywuzzy_scan(tk, kl)), 4)
        )

        tt = results["TurboText"][-1]
        rf = results["RapidFuzz"][-1]
        fw = results["FuzzyWuzzy"][-1]
        print(f"TurboText={tt:.4f}s  RapidFuzz={rf:.4f}s  FuzzyWuzzy={fw:.4f}s")

    return results


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

_STYLE: dict[str, dict[str, object]] = {
    "TurboText":  {"color": "#2ca02c", "marker": "o", "linewidth": 2.5, "zorder": 5},
    "FlashText":  {"color": "#d62728", "marker": "s", "linewidth": 2.0, "zorder": 4},
    "re":         {"color": "#1f77b4", "marker": "^", "linewidth": 2.0, "zorder": 3},
    "RapidFuzz":  {"color": "#1f77b4", "marker": "s", "linewidth": 2.0, "zorder": 4},
    "FuzzyWuzzy": {"color": "#ff7f0e", "marker": "^", "linewidth": 2.0, "zorder": 3},
}

# Names that belong in the zoomed inset (fast lines hidden in the main view)
_INSET_NAMES = {"TurboText", "FlashText", "RapidFuzz"}


def _annotate(ax: object, xs: list, ys: list, color: str, fontsize: float = 8.5) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(  # type: ignore[attr-defined]
            f"{y:.4f}" if y < 0.01 else f"{y:.2f}",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def _plot(
    counts: list[int],
    results: dict[str, list[float | None]],
    title: str,
    out_path: Path,
    inset: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)

    # ── Main plot ────────────────────────────────────────────────────────────
    for name, times in results.items():
        xs = [counts[i] for i, t in enumerate(times) if t is not None]
        ys = [t for t in times if t is not None]
        style = _STYLE.get(name, {"color": "#333333", "marker": "o"})
        ax.plot(xs, ys, label=name, markersize=7, **style)  # type: ignore[arg-type]
        _annotate(ax, xs, ys, str(style["color"]))

    ax.set_xlabel("No of Terms", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Time (Sec)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="center", pad=14)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    # ── Inset zoomed subplot ─────────────────────────────────────────────────
    if inset:
        inset_names = {n: v for n, v in results.items() if n in _INSET_NAMES}
        if inset_names:
            # Determine inset y-ceiling: max of fast lines + 20 %
            fast_max = max(
                t
                for times in inset_names.values()
                for t in times
                if t is not None
            )
            y_ceil = fast_max * 1.3

            # Place inset: left side, below legend
            axins = ax.inset_axes([0.05, 0.35, 0.42, 0.42])
            axins.set_facecolor("white")
            axins.grid(True, color="#e0e0e0", linewidth=0.6, zorder=0)

            for name, times in inset_names.items():
                xs = [counts[i] for i, t in enumerate(times) if t is not None]
                ys = [t for t in times if t is not None]
                style = _STYLE.get(name, {"color": "#333333", "marker": "o"})
                axins.plot(xs, ys, markersize=5, **style)  # type: ignore[arg-type]
                _annotate(axins, xs, ys, str(style["color"]), fontsize=7.5)

            axins.set_xlim(left=0, right=max(counts) * 1.05)
            axins.set_ylim(bottom=0, top=y_ceil)
            axins.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )
            axins.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            axins.set_xlabel("No of Terms", fontsize=8, labelpad=4)
            axins.set_ylabel("Time (Sec)", fontsize=8, labelpad=4)
            axins.set_title(
                f"Zoomed in (Y-axis: 0 to {y_ceil:.3f} sec)",
                fontsize=8.5,
                fontweight="bold",
                pad=5,
            )
            axins.tick_params(labelsize=7)
            for spine in axins.spines.values():
                spine.set_color("#aaaaaa")
                spine.set_linewidth(0.8)

            # Dashed rectangle on main axes showing zoom region
            rect_x0, rect_x1 = 0, max(counts) * 1.05
            rect_y0, rect_y1 = 0, y_ceil
            ax.add_patch(
                plt.Rectangle(  # type: ignore[attr-defined]
                    (rect_x0, rect_y0),
                    rect_x1 - rect_x0,
                    rect_y1 - rect_y0,
                    linewidth=1.2,
                    edgecolor="#888888",
                    facecolor="none",
                    linestyle="--",
                    zorder=6,
                )
            )
            # Connector lines from rectangle corners to inset
            ax.annotate(
                "",
                xy=axins.get_position().p0,  # type: ignore[attr-defined]
                xycoords="figure fraction",
                xytext=(rect_x0, rect_y1),
                textcoords=ax.transData,  # type: ignore[attr-defined]
                arrowprops=dict(arrowstyle="-", color="#888888",
                                linestyle="dashed", lw=0.9),
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent

    k0_data = bench_k0()
    _plot(
        K0_COUNTS,
        k0_data,
        "No of Terms Vs time taken (sec)  ·  Exact matching (k=0)",
        out_dir / "k0_scaling.png",
        inset=True,
    )

    k1_data = bench_k1()
    _plot(
        K1_COUNTS,
        k1_data,
        "No of Terms Vs time taken (sec)  ·  Fuzzy matching (k=1)",
        out_dir / "k1_scaling.png",
        inset=True,
    )

    print("\nAll done.")
