"""Property test: the cheap pre-gate must never reject a real store match.

`select.may_mention_store` exists only to avoid running the expensive
alternation regexes. It is a *necessary condition* for `STORE_PATTERNS`, so if
it can ever reject a string one of those patterns matches, stage 1 silently
drops candidates and no output looks wrong.

This enumerates the strings each pattern can produce — expanding optional
characters and alternations both ways — and asserts the gate accepts every one.
It caught `\\bruss?o'?s\\b`, whose optional second "s" means it matches "ruso's",
which the literal "russ" would have rejected.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import pytest

from groceries.select import STORE_PATTERNS, STORES, may_mention_store

try:  # 3.11+ moved the parser; keep the public-ish fallback
    from re import _parser as regex_parser  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older interpreters
    import sre_parse as regex_parser

MAX_REPEAT_EXPANSION = 2


def expand(node: Any) -> list[str]:
    """Enumerate strings a simple pattern can produce."""
    if hasattr(node, "data"):
        parts = [expand(n) for n in node.data]
        return ["".join(c) for c in itertools.product(*parts)] or [""]
    op, av = node
    name = str(op)
    if "LITERAL" in name and "NOT" not in name:
        return [chr(av)]
    if "REPEAT" in name:
        lo, hi, sub = av
        inner = expand(sub)
        out: set[str] = set()
        for n in range(lo, min(hi, MAX_REPEAT_EXPANSION) + 1):
            for combo in itertools.product(inner, repeat=n):
                out.add("".join(combo))
        return sorted(out)
    if "SUBPATTERN" in name:
        return expand(av[3])
    if "BRANCH" in name:
        return [s for branch in av[1] for s in expand(branch)]
    if name.endswith("IN"):
        chars = []
        for sub in av:
            sub_name = str(sub[0])
            if "LITERAL" in sub_name and "NOT" not in sub_name:
                chars.append(chr(sub[1]))
            elif "CATEGORY" in sub_name:
                chars.append(" ")
        return chars or [" "]
    if "AT" in name or "ASSERT" in name:
        return [""]
    if "ANY" in name:
        return ["x"]
    return [""]


@pytest.mark.parametrize("store", sorted(STORE_PATTERNS))
def test_gate_accepts_every_string_the_pattern_matches(store: str) -> None:
    pattern = STORES[store]
    variants = [v for v in expand(regex_parser.parse(STORE_PATTERNS[store], re.I))
                if pattern.search(v)]
    assert variants, f"expansion produced nothing matching {store}"
    rejected = [v for v in variants if not may_mention_store(v.lower())]
    assert not rejected, (
        f"gate rejects {rejected!r} which the {store} pattern matches — "
        "stage 1 would silently drop these candidates"
    )
