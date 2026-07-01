from collections.abc import Callable

from turbotext.resolve import RawMatch
from turbotext.trie import TrieNode

# Maps node id → (node, minimum cost reaching it from the current start position).
_Frontier = dict[int, tuple[TrieNode, int]]


def search(
    root: TrieNode,
    text: str,
    max_edit_distance: int,
    is_boundary: Callable[[str], bool],
) -> list[RawMatch]:
    """Emit every (start, end, keyword_id, edit_distance) span in *text*.

    Boundary checking at emission: a candidate is only accepted if the
    characters immediately before/after the span are boundary characters
    (or the span is at a text edge).

    Returns all overlapping matches; the caller decides which to keep.
    """
    if max_edit_distance == 0:
        return _search_exact(root, text, is_boundary)
    return _search_fuzzy(root, text, max_edit_distance, is_boundary)


def _search_exact(
    root: TrieNode,
    text: str,
    is_boundary: Callable[[str], bool],
) -> list[RawMatch]:
    """Fast path for max_edit_distance=0.

    At k=0 the frontier always contains exactly one node, so the full
    dict-based frontier machinery reduces to a single pointer walk.
    """
    results: list[RawMatch] = []
    n = len(text)
    for start in range(n):
        if start > 0 and not is_boundary(text[start - 1]):
            continue
        node = root
        pos = start
        while pos < n:
            child = node.children.get(text[pos])
            if child is None:
                break
            node = child
            pos += 1
            if node.is_terminal and node.keyword_id is not None:
                if pos == n or is_boundary(text[pos]):
                    results.append(RawMatch(start, pos, node.keyword_id, 0))
    return results


def _search_fuzzy(
    root: TrieNode,
    text: str,
    max_edit_distance: int,
    is_boundary: Callable[[str], bool],
) -> list[RawMatch]:
    """Full frontier search for max_edit_distance > 0."""
    results: list[RawMatch] = []
    n = len(text)

    for start in range(n):
        if start > 0 and not is_boundary(text[start - 1]):
            continue

        frontier: _Frontier = {id(root): (root, 0)}
        _deletion_closure(frontier, max_edit_distance)

        pos = start

        while pos <= n and frontier:
            ch = text[pos] if pos < n else None

            # Guard pos > start: the deletion closure above can put terminal nodes
            # into the initial frontier (entire keyword deleted), which would produce
            # zero-length matches. We never want to emit before consuming at least
            # one text character.
            if pos > start:
                for node, cost in frontier.values():
                    if node.is_terminal and node.keyword_id is not None:
                        if pos == n or is_boundary(text[pos]):
                            results.append(RawMatch(start, pos, node.keyword_id, cost))

            if ch is None:
                break

            # Expand via character-consuming transitions (match, substitution,
            # insertion). Deletion transitions are handled separately below via
            # _deletion_closure, because they don't consume a text character and
            # must be resolved at the same text position, not the next one.
            next_frontier: _Frontier = {}
            for node, cost in frontier.values():
                if ch in node.children:
                    _update(next_frontier, node.children[ch], cost)  # match
                if cost < max_edit_distance:
                    for edge_ch, child in node.children.items():
                        if edge_ch != ch:
                            _update(next_frontier, child, cost + 1)  # substitution
                    _update(next_frontier, node, cost + 1)  # insertion (skip text char)

            if next_frontier:
                _deletion_closure(next_frontier, max_edit_distance)

            frontier = next_frontier
            pos += 1

    return results


def _deletion_closure(frontier: _Frontier, max_edit_distance: int) -> None:
    """Expand *frontier* in-place via deletion transitions (trie advance, no text consumed).

    Runs to fixed point. Converges in at most max_edit_distance rounds because
    each step costs +1 and tries are acyclic — a path can't revisit a node.
    """
    changed = True
    while changed:
        changed = False
        for _, (node, cost) in list(frontier.items()):
            if cost >= max_edit_distance:
                continue
            for child in node.children.values():
                child_id = id(child)
                new_cost = cost + 1
                existing = frontier.get(child_id)
                if existing is None or new_cost < existing[1]:
                    frontier[child_id] = (child, new_cost)
                    changed = True


def _update(frontier: _Frontier, node: TrieNode, cost: int) -> None:
    """Keep only the minimum cost per node."""
    nid = id(node)
    existing = frontier.get(nid)
    if existing is None or cost < existing[1]:
        frontier[nid] = (node, cost)
