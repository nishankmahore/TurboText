

class TrieNode:
    """Single node in the keyword trie.

    Uses __slots__ to avoid per-instance __dict__ overhead at large vocabulary sizes —
    this matters in pure Python more than it would behind a native core.
    """

    __slots__ = ("children", "keyword_id", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.keyword_id: str | None = None
        self.is_terminal: bool = False


class TrieBuilder:
    """Builds and owns a trie for efficient keyword lookup."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, surface_form: str, keyword_id: str) -> None:
        node = self.root
        for ch in surface_form:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True
        node.keyword_id = keyword_id
