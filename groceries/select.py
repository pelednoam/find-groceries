"""Stage 1: narrow the Reddit corpus to documents worth sending to a model.

A document qualifies when it names a grocery store *and* carries price/quality
vocabulary. That combination is about 0.9% of the corpus — the point of this
stage is that the expensive stage never sees the other 99%.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .jsonl import read_jsonl, write_atomic
from .types import Candidate, RawDoc

# Canonical store name -> match pattern. Bare abbreviations ("MB", "WF") are
# deliberately excluded: they inflate Market Basket and Whole Foods with
# megabytes and unrelated initialisms.
STORE_PATTERNS: dict[str, str] = {
    "Market Basket": r"market\s*basket",
    "Trader Joe's": r"trader\s*joe(?:'?s)?",
    "Star Market": r"star\s*market",
    "Whole Foods": r"whole\s*foods?",
    "Stop & Shop": r"stop\s*(?:&|and|'?n'?)\s*shop",
    "Shaw's": r"\bshaw'?s\b",
    "H Mart": r"\bh[\s-]?mart\b",
    "Wegmans": r"\bwegmans\b",
    "Aldi": r"\baldi\b",
    "Costco": r"\bcostco\b",
    "BJ's": r"\bbj'?s\b(?!\s*(?:restaurant|brewhouse))",
    "Haymarket": r"\bhaymarket\b",
    "Russo's": r"\bruss?o'?s\b",
    "Broadway Marketplace": r"broadway\s*marketplace",
    "Dave's Fresh Pasta": r"dave'?s\s*fresh\s*pasta",
    "Formaggio Kitchen": r"formaggio",
    "Cardullo's": r"cardullo",
    "Reliable Market": r"reliable\s*market",
    "Ebisuya": r"ebisuya",
    "Target": r"\btarget\b",
}

STORES: dict[str, re.Pattern[str]] = {
    name: re.compile(pat, re.I) for name, pat in STORE_PATTERNS.items()
}

# Names that are also ordinary English words — or Boston place names —
# need food context to count. Haymarket is an MBTA station, a garage and
# a neighbourhood before it is a produce market.
CONTEXT_GATED: frozenset[str] = frozenset({"Target", "Haymarket"})

GROCERY_CONTEXT: re.Pattern[str] = re.compile(
    r"\b(grocer|groceries|supermarket|produce|shopping|aisle|cart|checkout|"
    r"milk|eggs|bread|meat|chicken|veggies?|vegetables?|fruit|cheese|seafood|"
    r"deli|frozen|pantry|food shop)",
    re.I,
)

EVALUATIVE: re.Pattern[str] = re.compile(
    r"\b(price[sd]?|pricing|pricey|cost(?:s|ly)?|cheap(?:er|est)?|expensive|"
    r"afford(?:able)?|bargain|deal|sale|overpriced|rip[\s-]?off|value|budget|"
    r"quality|fresh(?:ness|er|est)?|stale|rotten|moldy|spoiled|wilted|"
    r"produce|selection|variety|best|worst|better|great|terrible|awful|"
    r"recommend|avoid|prefer|worth it|not worth|"
    r"sucks?|sucked|trash|garbage|crap|crappy|gross|disgusting|nasty|filthy|"
    r"dirty|horrible|mediocre|disappointing|never again|dumpster|"
    r"awesome|amazing|love|favorite|favourite|go[\s-]?to|solid|spectacular|"
    r"overrated|underrated|steal|pricier|dollars?|bucks?)\b|\$\d",
    re.I,
)

BOT_AUTHORS: frozenset[str] = frozenset(
    {"AutoModerator", "[deleted]", "B0tRank", "RemindMeBot", "sneakpeekbot"}
)

MIN_CHARS = 40
MAX_CHARS = 6000

# A necessary condition for any STORE_PATTERNS match: if none of these
# substrings is present, no store regex can fire, so the document cannot
# become a candidate no matter what else it says. Checked before the
# alternation regexes because it is roughly 3.7x cheaper than they are.
STORE_LITERALS: tuple[str, ...] = (
    "basket", "joe", "star", "whole", "shop", "shaw", "mart", "wegmans",
    "aldi", "costco", "bj", "haymarket", "rus", "marketplace", "pasta",
    "formaggio", "cardullo", "reliable", "ebisuya", "target",
)


def may_mention_store(lowered: str) -> bool:
    """Cheap pre-gate; never rejects a text that a store pattern would match."""
    return any(literal in lowered for literal in STORE_LITERALS)


@dataclass
class SelectionReport:
    """What a selection pass saw and kept."""

    scanned: int = 0
    kept: int = 0
    duplicates: int = 0
    per_store: Counter[str] = field(default_factory=Counter)
    per_subreddit: Counter[str] = field(default_factory=Counter)


def doc_text(raw: RawDoc) -> str:
    """Comment body, or post title + selftext."""
    body = raw.get("body")
    if body is not None:
        return body
    title = raw.get("title") or ""
    selftext = raw.get("selftext") or ""
    return f"{title}\n\n{selftext}".strip()


def permalink(raw: RawDoc, subreddit: str, kind: str) -> str:
    """Reconstruct a reddit.com path for a post or comment."""
    if kind == "posts":
        existing = raw.get("permalink")
        if existing:
            return existing
        return f"/r/{subreddit}/comments/{raw['id']}/"
    link = (raw.get("link_id") or "").replace("t3_", "")
    return f"/r/{subreddit}/comments/{link}/_/{raw['id']}/"


def matched_stores(text: str) -> list[str]:
    """Store names the text mentions, after context-gating generic words."""
    hits = [name for name, pat in STORES.items() if pat.search(text)]
    if not hits:
        return []
    if GROCERY_CONTEXT.search(text):
        return hits
    # No food context, so names that are also ordinary English words don't count.
    return [h for h in hits if h not in CONTEXT_GATED]


def is_evaluative(text: str) -> bool:
    return EVALUATIVE.search(text) is not None


def first_mention(text: str, stores: Iterable[str]) -> int:
    """Character offset of the earliest mention of any of `stores`."""
    starts = [
        m.start() for store in stores if (m := STORES[store].search(text)) is not None
    ]
    return min(starts, default=0)


def excerpt(text: str, stores: list[str]) -> tuple[str, bool]:
    """Trim to MAX_CHARS while keeping a store mention in view.

    Stores are matched against the full document, so a naive head-truncation
    can hand the model text that names no store at all while the prompt claims
    one was matched — which invites it to invent a claim. When the first
    mention falls past the cut, window around it instead of taking the head.
    """
    if len(text) <= MAX_CHARS:
        return text, False
    head = text[:MAX_CHARS]
    if matched_stores(head):
        return head, True
    start = max(0, first_mention(text, stores) - MAX_CHARS // 4)
    return text[start : start + MAX_CHARS], True


def make_candidate(
    raw: RawDoc, subreddit: str, kind: str, stores: list[str], text: str
) -> Candidate:
    body, truncated = excerpt(text, stores)
    # Recomputed against the text actually sent: the prompt names these as
    # pre-filter matches, and advertising a store the model cannot see is an
    # invitation to invent a claim about it.
    visible = matched_stores(body)
    return Candidate(
        id=raw["id"],
        subreddit=subreddit,
        kind=kind,
        created_utc=raw["created_utc"],
        score=raw.get("score"),
        author=raw.get("author", ""),
        permalink=permalink(raw, subreddit, kind),
        stores=visible,
        text=body,
        truncated=truncated,
    )


def is_bot(raw: RawDoc) -> bool:
    return raw.get("author", "") in BOT_AUTHORS


def evaluate(raw: RawDoc, subreddit: str, kind: str) -> Candidate | None:
    """Return a Candidate if this document qualifies, else None."""
    if is_bot(raw):
        return None
    text = doc_text(raw)
    if len(text) < MIN_CHARS:
        return None
    if not may_mention_store(text.lower()):
        return None
    if not is_evaluative(text):
        return None
    stores = matched_stores(text)
    if not stores:
        return None
    return make_candidate(raw, subreddit, kind, stores, text)


def iter_shard(path: Path) -> Iterator[tuple[RawDoc, str, str]]:
    """Yield (raw, subreddit, kind) for every record in one shard file."""
    kind = path.parent.parent.name
    subreddit = path.parent.name
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                # One corrupt line in a 1,504-shard corpus must not abort a
                # scan that takes five minutes.
                continue
            yield raw, subreddit, kind


def iter_candidates(
    shards: Iterable[Path],
    report: SelectionReport,
    limit: int | None = None,
    subreddits: Collection[str] | None = None,
) -> Iterator[Candidate]:
    """Yield qualifying candidates, tallying into `report` as it goes.

    A generator so the caller can stream straight to disk; the full corpus
    produces ~84k candidates, which is a lot to hold in memory for no reason.
    """
    seen_texts: set[str] = set()
    for shard in shards:
        for raw, subreddit, kind in iter_shard(shard):
            if subreddits is not None and subreddit not in subreddits:
                continue
            report.scanned += 1
            cand = evaluate(raw, subreddit, kind)
            if cand is None:
                continue
            # Reposted boilerplate is the single biggest contamination risk:
            # one copypasta appears 947 times and matches a store name, so a
            # claim drawn from it would enter the aggregate 947 times over.
            digest = hashlib.sha256(cand["text"].encode("utf-8")).hexdigest()
            if digest in seen_texts:
                report.duplicates += 1
                continue
            seen_texts.add(digest)
            # Check the limit before counting, so --limit 0 yields nothing.
            if limit is not None and report.kept >= limit:
                return
            report.kept += 1
            report.per_subreddit[subreddit] += 1
            for store in cand["stores"]:
                report.per_store[store] += 1
            yield cand


def select(
    shards: Iterable[Path],
    limit: int | None = None,
    subreddits: Collection[str] | None = None,
) -> tuple[list[Candidate], SelectionReport]:
    """Eager wrapper around `iter_candidates` for callers that want a list."""
    report = SelectionReport()
    return list(iter_candidates(shards, report, limit, subreddits)), report


def write_candidates(candidates: Iterable[Candidate], path: Path) -> int:
    return write_atomic(
        path, (json.dumps(c, ensure_ascii=False) + "\n" for c in candidates)
    )


def read_candidates(path: Path) -> list[Candidate]:
    rows, _unparseable = read_jsonl(path)
    return [cast(Candidate, r) for r in rows]
