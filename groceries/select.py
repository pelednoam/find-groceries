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
from typing import Final, cast

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

# Compiled WITHOUT re.IGNORECASE and matched against `text.lower()`. re.I
# applies full Unicode case-folding ('\u017f' folds to 's'), which is strictly
# wider than str.lower() -- so a gate built on str.lower() could reject text
# an re.I pattern matches. Using one equivalence on both sides removes the
# whole class of mismatch.
STORES: dict[str, re.Pattern[str]] = {
    name: re.compile(pat) for name, pat in STORE_PATTERNS.items()
}

# Names that are also ordinary English words — or Boston place names —
# need food context to count. Haymarket is an MBTA station, a garage and
# a neighbourhood before it is a produce market.
CONTEXT_GATED: frozenset[str] = frozenset({"Target", "Haymarket"})

GROCERY_CONTEXT: re.Pattern[str] = re.compile(
    r"\b(grocer|groceries|supermarket|produce|shopping|aisle|cart|checkout|"
    r"milk|eggs|bread|meat|chicken|veggies?|vegetables?|fruit|cheese|seafood|"
    r"deli|frozen|pantry|food shop)",
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
    r"overrated|underrated|steal|pricier|dollars?|bucks?|"
    # Stage 2 gained delivery_online, crowding_hours, store_lifecycle,
    # labor_ethics and parking_access, and stage 3 gained a half-life for
    # each -- but the gate here had no vocabulary for any of them, so no
    # document that only discusses one was ever selected. The categories
    # existed downstream of a filter that could not feed them.
    r"deliver(?:y|ed|ies)|instacart|curbside|pickup|shipt|doordash|"
    r"coupons?|loyalty|rewards|discounts?|clearance|markdowns?|"
    r"crowded|packed|mobbed|busy|lines|queues?|madhouse|"
    # Bare "open" and "union" are not usable here: "are they open late" is a
    # question, and Union Square is a neighbourhood in two of the three
    # subreddits. Same for bare "lot" ("a lot of") and singular "line".
    r"opened|opening|reopen(?:ed|ing)?|clos(?:ing|ed|es|ure)|"
    r"renovat(?:e|ed|ion|ing)|remodel(?:ed|ing)?|"
    r"unioniz(?:e|ed|ing|ation)|strike|wages?|employees?|workers?|staff|"
    # "parking" already covers "parking garage"; bare "garage" is a place.
    r"parking|validat(?:e|ed|ion))\b|\$\d",
)

BOT_AUTHORS: frozenset[str] = frozenset(
    {"AutoModerator", "[deleted]", "B0tRank", "RemindMeBot", "sneakpeekbot"}
)

MIN_CHARS = 40
PARENT_CONTEXT_CHARS = 1500
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


def casefold_preserving(text: str) -> str:
    """Lowercase without changing the length of the string.

    `str.lower` expands some characters -- "İ" becomes two -- which silently
    invalidates any offset computed on the result and used against the
    original. Characters that would expand are left alone: they are never
    part of a store name, so folding them buys nothing.
    """
    return "".join(c if len(c.lower()) != 1 else c.lower() for c in text)


def matched_stores(text: str) -> list[str]:
    """Store names the text mentions, after context-gating generic words.

    Case folding happens once, here, so the pre-gate and the patterns share
    one definition of equality.
    """
    lowered = text.lower()
    hits = [name for name, pat in STORES.items() if pat.search(lowered)]
    if not hits:
        return []
    if GROCERY_CONTEXT.search(lowered):
        return hits
    # No food context, so names that are also ordinary English words don't count.
    return [h for h in hits if h not in CONTEXT_GATED]


def is_evaluative(text: str) -> bool:
    return EVALUATIVE.search(text.lower()) is not None


def first_mention(text: str, stores: Iterable[str]) -> int:
    """Character offset of the earliest mention of any of `stores`.

    Offsets come from `str.casefold`, which is length-preserving for the
    characters that appear here; `str.lower` is not -- "İ".lower() is two
    characters -- and an offset taken from lowered text and applied to the
    original slides the excerpt window by one per such character.
    """
    folded = casefold_preserving(text)
    starts = [
        m.start() for store in stores if (m := STORES[store].search(folded)) is not None
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
    raw: RawDoc,
    subreddit: str,
    kind: str,
    stores: list[str],
    text: str,
    parent_body: str = "",
    inherited: bool = False,
) -> Candidate | None:
    """Assemble a Candidate. `inherited` means the store came from the parent.

    That flag has to be explicit. The previous `matched_stores(body) or stores`
    conflated two cases: an inherited referent, where the excerpt legitimately
    names no store, and a *directly* matched document whose excerpt lost its
    store to windowing -- where falling back to `stores` tells the model a
    store was matched in text it cannot see, which is exactly the invitation
    to invent a claim the recomputation exists to prevent.
    """
    body, truncated = excerpt(text, stores)
    visible = stores if inherited else matched_stores(body)
    # A context-gated store ("Target", "Haymarket") can match the full text
    # and then fail inside the window, because the food words that licensed
    # it fell outside. Send nothing rather than a document whose prompt
    # claims a store the model cannot see -- that is the invitation to
    # invent a claim that recomputing `visible` exists to prevent.
    if not visible:
        return None
    return Candidate(
        id=raw["id"],
        subreddit=subreddit,
        kind=kind,
        created_utc=raw["created_utc"],
        score=raw.get("score"),
        author=raw.get("author", ""),
        parent_body=parent_body[:PARENT_CONTEXT_CHARS],
        permalink=permalink(raw, subreddit, kind),
        stores=visible,
        text=body,
        truncated=truncated,
    )


def is_bot(raw: RawDoc) -> bool:
    return raw.get("author", "") in BOT_AUTHORS


def evaluate(
    raw: RawDoc, subreddit: str, kind: str, parent_body: str = ""
) -> Candidate | None:
    """Return a Candidate if this document qualifies, else None.

    Two ways to qualify. Either the text names a store itself, or it is
    evaluative grocery talk whose parent comment names exactly one store --
    "the parking there is at least less terrible than Everett" is a real claim
    about a store the reply never names. Requiring exactly one parent store
    keeps the referent unambiguous.
    """
    if is_bot(raw):
        return None
    text = doc_text(raw)
    if len(text) < MIN_CHARS:
        return None
    if not is_evaluative(text):
        return None

    if may_mention_store(text.lower()) and (stores := matched_stores(text)):
        # No parent context here on purpose. The reply names its own store, so
        # the parent adds nothing to resolve -- and it is a second span of
        # untrusted text in the prompt, paid for on every one of ~10,000
        # documents, to no end.
        return make_candidate(raw, subreddit, kind, stores, text)

    # Inherited referent: the reply carries the judgement, the parent names
    # the store. Require grocery context so unrelated replies don't ride along.
    if not parent_body or not GROCERY_CONTEXT.search(text.lower()):
        return None
    parent_stores = matched_stores(parent_body)
    if len(parent_stores) != 1:
        return None
    return make_candidate(
        raw, subreddit, kind, parent_stores, text, parent_body, inherited=True
    )


def parent_index(rows: list[RawDoc]) -> dict[str, str]:
    """Map comment id -> body for parent lookups within one shard."""
    return {r["id"]: r["body"] for r in rows if r.get("id") and r.get("body")}


def parent_of(raw: RawDoc, index: dict[str, str]) -> str:
    """Body of this document's parent comment, or "" if not in this shard."""
    parent = raw.get("parent_id", "")
    if not parent.startswith("t1_"):
        return ""  # t3_ parents are the post itself, not a comment
    return index.get(parent[3:], "")


def read_shard(path: Path) -> tuple[list[RawDoc], str, str]:
    """Read one shard fully, so parents can be resolved before evaluating.

    Streaming would use less memory, but a reply's parent can appear anywhere
    in the shard -- including after it -- so the index has to be complete
    before the first document is judged. The largest shard is ~40MB.
    """
    kind = path.parent.parent.name
    subreddit = path.parent.name
    rows: list[RawDoc] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # One corrupt line in a 1,504-shard corpus must not abort a
                # scan that takes five minutes.
                continue
    return rows, subreddit, kind


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
        # Filter on the path before decompressing: restricting to three of
        # four subreddits otherwise still inflates every shard of the fourth.
        if subreddits is not None and shard.parent.name not in subreddits:
            continue
        rows, subreddit, kind = read_shard(shard)
        index = parent_index(rows)
        for raw in rows:
            report.scanned += 1
            cand = evaluate(raw, subreddit, kind, parent_of(raw, index))
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


CANDIDATE_FIELDS: Final[frozenset[str]] = frozenset(Candidate.__annotations__)


def _is_candidate(row: object) -> bool:
    """Reject rows stage 2 would only fail on later, at cost.

    A working set written by an older version of stage 1 is missing fields the
    prompt builder indexes into. Blindly casting turned that into a KeyError
    per document, after the request had been paid for.
    """
    if not isinstance(row, dict) or not CANDIDATE_FIELDS <= row.keys():
        return False
    if not isinstance(row.get("created_utc"), int) or isinstance(
        row.get("created_utc"), bool
    ):
        return False
    if not isinstance(row.get("stores"), list) or not all(
        isinstance(s, str) for s in row["stores"]
    ):
        return False
    score = row.get("score")
    if score is not None and (not isinstance(score, int) or isinstance(score, bool)):
        return False
    return all(
        isinstance(row.get(f), str)
        for f in ("id", "subreddit", "kind", "author", "parent_body", "permalink", "text")
    )


def read_candidates(path: Path) -> tuple[list[Candidate], int]:
    """Read the working set. Returns (candidates, n_dropped)."""
    rows, unparseable = read_jsonl(path)
    good = [cast(Candidate, r) for r in rows if _is_candidate(r)]
    return good, len(rows) - len(good) + unparseable
