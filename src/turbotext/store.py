import uuid
from dataclasses import dataclass, field
from typing import Any

from turbotext.frontier import search
from turbotext.resolve import MatchPolicy, RawMatch, resolve
from turbotext.trie import TrieBuilder

try:
    from turbotext._fast import build_aho_corasick as _build_aho_corasick
    from turbotext._fast import build_ctrie as _build_ctrie
    from turbotext._fast import resolve_ll_fast as _resolve_ll_fast
    from turbotext._fast import search_ac_fast as _fast_search_ac_fast
    from turbotext._fast import search_aho_corasick as _fast_search_ac
    _HAS_FAST = True
    _HAS_AC = True
except ImportError:
    _HAS_FAST = False
    _HAS_AC = False
    _fast_search_ac_fast = None  # type: ignore[assignment]
    _resolve_ll_fast = None  # type: ignore[assignment]
    try:
        from turbotext._fast import build_ctrie as _build_ctrie
        from turbotext._fast import search_exact as _fast_search_exact
        _HAS_FAST = True
    except ImportError:
        pass


@dataclass
class FuzzyConfig:
    max_edit_distance: int = 0


@dataclass
class Keyword:
    id: str
    surface_form: str
    canonical: str
    category: str | None = None
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Match:
    """A matched keyword span.  Uses __slots__ for fast bulk creation."""

    __slots__ = (
        "text", "canonical", "start", "end", "edit_distance",
        "category", "priority", "keyword_id", "metadata",
    )

    def __init__(
        self,
        text: str,
        canonical: str,
        start: int,
        end: int,
        edit_distance: int,
        category: "str | None",
        priority: float,
        keyword_id: str,
        metadata: "dict[str, Any] | None" = None,
    ) -> None:
        self.text = text
        self.canonical = canonical
        self.start = start
        self.end = end
        self.edit_distance = edit_distance
        self.category = category
        self.priority = priority
        self.keyword_id = keyword_id
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        return (
            f"Match(text={self.text!r}, canonical={self.canonical!r}, "
            f"start={self.start}, end={self.end}, "
            f"edit_distance={self.edit_distance}, category={self.category!r}, "
            f"priority={self.priority}, keyword_id={self.keyword_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Match):
            return NotImplemented
        return (
            self.text == other.text
            and self.canonical == other.canonical
            and self.start == other.start
            and self.end == other.end
            and self.edit_distance == other.edit_distance
            and self.category == other.category
            and self.priority == other.priority
            and self.keyword_id == other.keyword_id
        )


def _is_boundary(ch: str) -> bool:
    return not (ch.isalnum() or ch == "_")


class KeywordStore:
    """Boundary-aware keyword index with configurable overlap resolution and fuzzy matching."""

    def __init__(
        self,
        policy: MatchPolicy = MatchPolicy.LEFTMOST_LONGEST,
        fuzzy: FuzzyConfig | None = None,
    ) -> None:
        self._policy = policy
        self._fuzzy = fuzzy or FuzzyConfig()
        self._keywords: dict[str, Keyword] = {}
        self._builder = TrieBuilder()
        self._ctrie: Any = None  # CNode trie built lazily when _HAS_FAST
        # Flat cache: kid → (canonical, category, priority, metadata) for fast _to_match
        self._kw_attrs: dict[str, tuple[str, str | None, float, dict[str, Any]]] = {}

    def add_keyword(
        self,
        surface_form: str,
        canonical: str | None = None,
        category: str | None = None,
        priority: float = 1.0,
        **metadata: Any,
    ) -> str:
        """Register a keyword and return its assigned id."""
        kid = str(uuid.uuid4())
        canon = canonical if canonical is not None else surface_form
        kw = Keyword(
            id=kid,
            surface_form=surface_form,
            canonical=canon,
            category=category,
            priority=priority,
            metadata=metadata,
        )
        self._keywords[kid] = kw
        self._kw_attrs[kid] = (canon, category, priority, metadata)
        self._builder.insert(surface_form.lower(), kid)
        self._ctrie = None  # invalidate so it is rebuilt on next extract()
        return kid

    def add_keywords(
        self,
        keywords: list[str] | dict[str, str | list[str]],
        category: str | None = None,
        priority: float = 1.0,
    ) -> list[str]:
        """Register multiple keywords at once and return their ids.

        Three input shapes are supported:

          list[str]
              Each string is both the surface form and the canonical.
              store.add_keywords(["java", "python"])

          dict[str, str]
              Keys are surface forms, values are canonicals.
              store.add_keywords({"java programing": "Java", "py": "Python"})

          dict[str, list[str]]
              Keys are canonicals, values are lists of surface forms / aliases.
              store.add_keywords({
                  "java": ["java_2e", "java programing"],
                  "product management": ["PM", "product manager"],
              })

        ``category`` and ``priority`` are applied to every keyword in the call.
        """
        if isinstance(keywords, list):
            return [
                self.add_keyword(kw, category=category, priority=priority)
                for kw in keywords
            ]
        ids: list[str] = []
        for key, value in keywords.items():
            if isinstance(value, list):
                # {canonical: [surface1, surface2, ...]}
                for surface in value:
                    ids.append(
                        self.add_keyword(surface, canonical=key,
                                         category=category, priority=priority)
                    )
            else:
                # {surface: canonical}
                ids.append(
                    self.add_keyword(key, canonical=value,
                                     category=category, priority=priority)
                )
        return ids

    def _ensure_actrie(self) -> None:
        if self._ctrie is None:
            self._ctrie = _build_ctrie(
                [(kw.surface_form.lower(), kid) for kid, kw in self._keywords.items()]
            )
            _build_aho_corasick(self._ctrie)

    def extract(self, text: str) -> list[Match]:
        """Return matches in *text*, resolved according to the store's policy."""
        k = self._fuzzy.max_edit_distance
        attrs = self._kw_attrs

        # Fast path: exact + leftmost-longest — inline lowercase, no RawMatch alloc,
        # no separate resolve pass.
        if k == 0 and _HAS_AC and self._policy is MatchPolicy.LEFTMOST_LONGEST:
            self._ensure_actrie()
            tuples = _fast_search_ac_fast(self._ctrie, text)
            resolved_t = _resolve_ll_fast(tuples)
            return [
                Match(
                    text=text[s:e],
                    canonical=(a := attrs[kid])[0],
                    start=s,
                    end=e,
                    edit_distance=0,
                    category=a[1],
                    priority=a[2],
                    keyword_id=kid,
                    metadata=a[3].copy() if a[3] else {},
                )
                for s, e, kid in resolved_t
            ]

        low = text.lower()
        if k == 0 and _HAS_AC:
            self._ensure_actrie()
            raw = _fast_search_ac(self._ctrie, low)
        elif k == 0 and _HAS_FAST:
            if self._ctrie is None:
                self._ctrie = _build_ctrie(
                    [(kw.surface_form.lower(), kid) for kid, kw in self._keywords.items()]
                )
            raw = _fast_search_exact(self._ctrie, low)
        else:
            raw = search(
                self._builder.root,
                low,
                k,
                _is_boundary,
            )
        resolved = resolve(
            raw,
            self._policy,
            {kid: kw.priority for kid, kw in self._keywords.items()},
        )
        return [
            Match(
                text=text[m.start:m.end],
                canonical=(a := attrs[m.keyword_id])[0],
                start=m.start,
                end=m.end,
                edit_distance=m.edit_distance,
                category=a[1],
                priority=a[2],
                keyword_id=m.keyword_id,
                metadata=a[3].copy() if a[3] else {},
            )
            for m in resolved
        ]

    def replace(self, text: str) -> str:
        """Replace matched spans with their canonical forms."""
        matches = self.extract(text)
        if not matches:
            return text
        parts: list[str] = []
        cursor = 0
        for m in matches:
            parts.append(text[cursor : m.start])
            parts.append(m.canonical)
            cursor = m.end
        parts.append(text[cursor:])
        return "".join(parts)

    def _to_match(self, raw: RawMatch, original_text: str) -> Match:
        canon, category, priority, metadata = self._kw_attrs[raw.keyword_id]
        return Match(
            text=original_text[raw.start : raw.end],
            canonical=canon,
            start=raw.start,
            end=raw.end,
            edit_distance=raw.edit_distance,
            category=category,
            priority=priority,
            keyword_id=raw.keyword_id,
            metadata=metadata.copy() if metadata else {},
        )
