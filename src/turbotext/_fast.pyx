# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uint8_t
from cpython.list cimport PyList_GET_ITEM, PyList_SET_ITEM, PyList_GET_SIZE
from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.ref cimport Py_INCREF

from collections import deque

from turbotext.resolve import RawMatch

# ---------------------------------------------------------------------------
# Boundary lookup table for ASCII (1 = boundary, 0 = word char)
# ---------------------------------------------------------------------------

cdef uint8_t _BT[128]

def _init_bt() -> None:
    cdef int i
    for i in range(128):
        # word chars: A-Z  a-z  0-9  _
        if (65 <= i <= 90) or (97 <= i <= 122) or (48 <= i <= 57) or i == 95:
            _BT[i] = 0
        else:
            _BT[i] = 1

_init_bt()

# ---------------------------------------------------------------------------
# ASCII lowercase lookup table  (_LC[c] == tolower(c))
# ---------------------------------------------------------------------------

cdef uint8_t _LC[128]

def _init_lc() -> None:
    cdef int i
    for i in range(128):
        if 65 <= i <= 90:
            _LC[i] = <uint8_t>(i + 32)
        else:
            _LC[i] = <uint8_t>i

_init_lc()


cdef inline uint8_t _boundary(Py_UCS4 cp) noexcept:
    if cp < 128:
        return _BT[cp]
    # Unicode: reuse Python's isalnum (correct for non-ASCII word chars)
    cdef str s = chr(cp)
    return 0 if (s.isalnum() or s == "_") else 1


# ---------------------------------------------------------------------------
# CNode: trie node with 128-slot list for ASCII children
# ---------------------------------------------------------------------------

cdef class CNode:
    cdef list   _ch          # 128 items: CNode | None, indexed by ord(char)
    cdef dict   _ext         # non-ASCII overflow
    cdef public bint   is_terminal
    cdef public object keyword_id    # str | None
    cdef Py_ssize_t kw_length        # keyword length at this terminal (used by AC)
    cdef CNode  _fail                # Aho-Corasick failure link
    cdef list   _output              # list of (keyword_id, kw_length) tuples

    def __cinit__(self):
        self._ch         = [None] * 128
        self._ext        = {}
        self.is_terminal = False
        self.keyword_id  = None
        self.kw_length   = 0
        self._fail       = None
        self._output     = None

    cdef CNode _get(self, Py_UCS4 cp):
        cdef object item
        if cp < 128:
            # PyList_GET_ITEM: direct C pointer dereference, no bounds check
            item = <object>PyList_GET_ITEM(self._ch, <Py_ssize_t>cp)
            if item is None:
                return None
            return <CNode>item
        return self._ext.get(cp)

    cdef void _set(self, Py_UCS4 cp, CNode child):
        if cp < 128:
            self._ch[<int>cp] = child
        else:
            self._ext[cp] = child


# ---------------------------------------------------------------------------
# build_ctrie: convert [(surface, keyword_id)] → CNode trie
# ---------------------------------------------------------------------------

def build_ctrie(list items):
    """Build a CNode trie. surface_form must already be lowercased."""
    cdef CNode root  = CNode()
    cdef CNode node, child
    cdef Py_UCS4 cp
    cdef str surface
    cdef Py_ssize_t depth

    for surface, kid in items:
        node  = root
        depth = 0
        for cp in surface:
            child = node._get(cp)
            if child is None:
                child = CNode()
                node._set(cp, child)
            node   = child
            depth += 1
        node.is_terminal = True
        node.keyword_id  = kid
        node.kw_length   = depth

    return root


def build_aho_corasick(CNode root):
    """Transform a CNode trie into a full Aho-Corasick automaton in-place.

    After this call:
    - Every _ch slot is non-None (goto function is complete).
    - Every node has a _fail link.
    - Every node with matches has an _output list of (keyword_id, kw_length).
    Calling build_ctrie then build_aho_corasick once is sufficient; don't
    add keywords afterwards without rebuilding both.
    """
    cdef CNode node, child, f_child, fail_node
    cdef int c

    # Root's failure link is itself; root's output is empty.
    root._fail   = root
    root._output = [] if not root.is_terminal else [(root.keyword_id, root.kw_length)]

    # BFS queue — start with depth-1 children.
    q = deque()
    for c in range(128):
        child = <CNode>PyList_GET_ITEM(root._ch, c)
        if child is None:
            # Goto completion at root: missing chars loop back to root.
            root._ch[c] = root
        else:
            child._fail   = root
            child._output = [(child.keyword_id, child.kw_length)] if child.is_terminal else []
            q.append(child)

    while q:
        node = q.popleft()
        fail_node = node._fail

        for c in range(128):
            child = <CNode>PyList_GET_ITEM(node._ch, c)
            if child is None:
                # Goto completion: follow failure link's goto.
                node._ch[c] = fail_node._ch[c]
            else:
                # Real child: set its failure link.
                f_child      = <CNode>PyList_GET_ITEM(fail_node._ch, c)
                child._fail  = f_child
                # Build output: own matches + failure's output.
                if child.is_terminal:
                    child._output = [(child.keyword_id, child.kw_length)] + f_child._output
                else:
                    child._output = f_child._output  # share list (read-only after build)
                q.append(child)


def search_aho_corasick(CNode root, str text):
    """O(n) exact keyword scan using the pre-built Aho-Corasick automaton.

    Requires build_aho_corasick(root) to have been called first so that the
    goto function is complete (no None slots in _ch for ASCII input).
    """
    cdef:
        list       results = []
        Py_ssize_t n       = len(text)
        Py_ssize_t pos, match_start
        Py_UCS4    cp
        CNode      node    = root
        object     item

    for pos in range(n):
        cp   = text[pos]
        if cp < 128:
            item = <object>PyList_GET_ITEM(node._ch, <Py_ssize_t>cp)
            node = <CNode>item
        else:
            node = node._ext.get(cp, root)

        if node._output is not None and PyList_GET_SIZE(node._output) > 0:
            for kid, kw_len in node._output:
                match_start = pos + 1 - kw_len
                if match_start < 0:
                    continue
                # Left boundary: start of text or preceded by a boundary char
                if match_start > 0 and not _boundary(text[match_start - 1]):
                    continue
                # Right boundary: end of text or followed by a boundary char
                if pos + 1 < n and not _boundary(text[pos + 1]):
                    continue
                results.append(RawMatch(match_start, pos + 1, kid, 0))

    return results


# ---------------------------------------------------------------------------
# search_ac_fast: AC scan with inline lowercase → list of (start, end, kid) tuples
# resolve_ll_fast: leftmost-longest in Cython (no Python-level resolve pass)
# ---------------------------------------------------------------------------

def search_ac_fast(CNode root, str text):
    """AC scan on the original (mixed-case) text.

    Lowercasing is done inline via the _LC table — no text.lower() copy needed.
    Returns list[(start, end, kid)] plain tuples; no RawMatch allocation.
    """
    cdef:
        list       results = []
        Py_ssize_t n       = len(text)
        Py_ssize_t pos, match_start
        Py_UCS4    cp
        CNode      node    = root
        object     item
        list       output

    for pos in range(n):
        cp = text[pos]
        if cp < 128:
            item = <object>PyList_GET_ITEM(node._ch, <Py_ssize_t>_LC[cp])
            node = <CNode>item
        else:
            node = node._ext.get(ord(chr(cp).lower()), root)

        output = node._output
        if output is not None and PyList_GET_SIZE(output) > 0:
            for kid, kw_len in output:
                match_start = pos + 1 - kw_len
                if match_start < 0:
                    continue
                if match_start > 0 and not _boundary(text[match_start - 1]):
                    continue
                if pos + 1 < n and not _boundary(text[pos + 1]):
                    continue
                results.append((match_start, pos + 1, kid))

    return results


def resolve_ll_fast(list raw):
    """Leftmost-longest resolution on list of (start, end, kid) tuples.

    Sorts once in C (via list.sort), then does the cursor-based selection
    in a tight Cython loop — no Python-level iterator overhead.
    Returns a new filtered list of the same (start, end, kid) tuples.
    """
    cdef:
        Py_ssize_t i, n
        Py_ssize_t cursor = 0, start, end
        object     item

    if not raw:
        return raw

    # Sort: start ascending; for equal starts, longer match first (end descending).
    raw.sort(key=lambda t: (t[0], -t[1]))
    n = PyList_GET_SIZE(raw)

    result = []
    for i in range(n):
        item  = <object>PyList_GET_ITEM(raw, i)
        start = <Py_ssize_t><object>(<object>PyTuple_GET_ITEM(<tuple>item, 0))
        end   = <Py_ssize_t><object>(<object>PyTuple_GET_ITEM(<tuple>item, 1))
        if start >= cursor:
            result.append(item)
            cursor = end
    return result


# ---------------------------------------------------------------------------
# search_exact: O(n·d) exact keyword scan (edit distance = 0)
# ---------------------------------------------------------------------------

def search_exact(CNode root, str text):
    """Return list[RawMatch] for all boundary-delimited matches.

    All loop counters are C Py_ssize_t; boundary checks hit the C table
    directly — zero Python function-call overhead on ASCII text.
    """
    cdef:
        list       results = []
        Py_ssize_t n       = len(text)
        Py_ssize_t start, pos
        Py_UCS4    cp
        CNode      node, child

    for start in range(n):
        if start > 0:
            cp = text[start - 1]
            if not _boundary(cp):
                continue

        node = root
        pos  = start

        while pos < n:
            cp    = text[pos]
            child = node._get(cp)
            if child is None:
                break
            node  = child
            pos  += 1

            if node.is_terminal and node.keyword_id is not None:
                if pos == n or _boundary(text[pos]):
                    results.append(RawMatch(start, pos, node.keyword_id, 0))

    return results
