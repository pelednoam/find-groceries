"""Stage 3: roll claims up into verdicts a shopper can act on.

Design notes, each answering a specific criticism of the previous version:

* **Branch matters more than chain.** The corpus insists on it — "the Stop &
  Shop in Mission Hill is one of the best", "H Mart Burlington is cheaper than
  Central Square". Cells are keyed on (store, location, category) with a
  chain-level parent, so a Cambridge reader is not told about Westwood.
* **Items are indexed separately.** Collapsing every `specific_item` claim for
  a store into one scalar answers nothing; "is this worth buying here" needs
  its own index.
* **Price is ordinal, not categorical.** A plurality vote over
  cheap/fair/expensive picked "fair" for Whole Foods when three of five claims
  said expensive. Scored on a line instead.
* **Thin evidence is shrunk toward neutral.** One claim at +1.00 and forty at
  +0.70 were previously indistinguishable in the output.
* **Not every claim is an independent opinion.** Upvotes, per-author and
  per-document caps, and reciprocal-comparison collapsing all bear on that.
* **Reddit opinion goes stale at different rates.** Cleanliness ages faster
  than a decades-stable price gap, so the half-life is per category.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict
from functools import cache
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Any, Final, cast

from .jsonl import read_jsonl, write_atomic
from .types import SourcedClaim

DEFAULT_HALF_LIFE_YEARS: Final = 4.0
SECONDS_PER_YEAR: Final = 365.25 * 24 * 3600

# Things that change on a renovation cycle age faster than a structural price
# gap that has held for decades.
HALF_LIFE_BY_CATEGORY: Final[Mapping[str, float]] = {
    "cleanliness": 2.0,
    "service_checkout": 2.0,
    "crowding_hours": 2.0,
    "parking_access": 3.0,
    "delivery_online": 2.0,
    "deals_loyalty": 1.5,
    "selection": 3.0,
    "store_lifecycle": 1.0,
    "price_overall": 7.0,
    "quality_overall": 6.0,
}

CONFIDENCE_WEIGHT: Final[Mapping[str, float]] = {"high": 1.0, "medium": 0.6, "low": 0.3}
DEFAULT_CONFIDENCE_WEIGHT: Final = 0.6
SENTIMENT_SCORE: Final[Mapping[str, float]] = {"positive": 1.0, "negative": -1.0}
PRICE_SCORE: Final[Mapping[str, float]] = {"cheap": -1.0, "fair": 0.0, "expensive": 1.0}

# Pulls a thin cell toward 0. With k=2, one high-confidence claim reads ~+0.33
# rather than +1.00, and forty claims are barely damped.
SHRINKAGE_K: Final = 2.0
# One effusive comment produced six of a store's eight claims in calibration.
MAX_CLAIMS_PER_DOCUMENT: Final = 3
# One opinionated regular over eight years is not many independent opinions.
MAX_CLAIMS_PER_AUTHOR_CELL: Final = 2
POSITIVE_THRESHOLD: Final = 0.15
# Below this much price evidence, shrinkage dominates and every cell reads
# "fair" regardless of what it says. Emit nothing rather than a wrong label.
MIN_PRICE_WEIGHT: Final = 1.0
# Roughly one low-confidence claim from this decade. Below that a cell is one
# person's passing remark, and printing a number for it implies more.
DEFAULT_MIN_WEIGHT: Final = 0.3
MIN_TIMESTAMP: Final = 1_100_000_000  # ~2004; Reddit did not exist before this
# Placeholders the dumps use for accounts that no longer exist. Capping these
# together would treat every deleted account in the corpus as one person.
# AutoModerator is deliberately NOT here: it is exactly one prolific poster,
# which is what the cap is for. (Stage 1's BOT_AUTHORS drops it earlier
# anyway; exempting it here stated the opposite rule in the same codebase.)
ANONYMOUS_AUTHORS: Final[frozenset[str]] = frozenset({"", "[deleted]", "[removed]"})

CellKey = tuple[str, str, str]  # store, location, category

REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"store", "category", "claim", "sentiment", "confidence",
     "source_key", "created_utc", "permalink"}
)
STRING_FIELDS: Final[tuple[str, ...]] = (
    "store", "category", "claim", "sentiment", "confidence", "source_key",
    "permalink",
)
# Absent is fine (an older claims file predates them); present-but-wrong-typed
# is not. `location` reaches `.strip()` in `build` and `score` reaches a
# numeric comparison in `vote_weight`, so a bad value there is a crash at the
# trust boundary rather than a rejected row.
OPTIONAL_STRING_FIELDS: Final[tuple[str, ...]] = (
    "location", "item", "comparator_store", "author",
)
# C0/C1/DEL, bidi overrides, and zero-width formatting. Newline and tab go too:
# these render into a terminal and into one-line table cells.
UNSAFE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0000, 0x001F),  # C0 controls, including ESC, newline and tab
    (0x007F, 0x009F),  # DEL and the C1 controls
    (0x00AD, 0x00AD),  # soft hyphen
    (0x061C, 0x061C),  # Arabic letter mark
    (0x180E, 0x180E),  # Mongolian vowel separator
    (0x200B, 0x200F),  # zero-width space/joiners, LRM, RLM
    (0x2028, 0x2029),  # line and paragraph separators
    (0x202A, 0x202E),  # bidi embedding and override
    (0x2060, 0x2064),  # word joiner and the invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xD800, 0xDFFF),  # lone surrogates: unencodable, never legitimate here
    (0xFEFF, 0xFEFF),  # zero-width no-break space / BOM
    (0xFFF9, 0xFFFB),  # interlinear annotation
)
# Built from the table rather than written as a literal: the previous version
# spelled these as the invisible characters themselves, so the pattern could
# not be reviewed by reading it and a missing class -- U+2028/U+2029, exactly
# the "hide the rest of the line" case -- went unnoticed.
UNSAFE_TEXT: Final = re.compile(
    "[" + "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in UNSAFE_RANGES) + "]"
)


def scrub(value: str) -> str:
    """Remove characters that can rewrite a terminal or hide text."""
    return UNSAFE_TEXT.sub("", value)


@cache
def _non_locations() -> frozenset[str]:
    """Values that turn up in `location` but are not places.

    Read from the stage-2 vocabulary rather than hand-listed, so the two
    cannot drift apart. Imported inside the function because extract imports
    select which imports nothing from here -- keeping it lazy avoids adding a
    cycle for one constant.
    """
    from .extract import CATEGORIES

    return frozenset(normalise(c) for c in CATEGORIES)


def normalise(value: str) -> str:
    """Fold free text into a stable key.

    `location` and `item` are unconstrained model prose used as dictionary
    keys, so "Somerville", "somerville" and "Somerville " were three separate
    branches of the same store, each individually too thin to clear the
    evidence threshold.
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


# "Beacon", "Beacon St" and "Beacon Street" are one store. Measured on the
# real corpus, collapsing the street-type suffix merges 6% of branch keys.
STREET_TYPE: Final = re.compile(
    r"\s+(st|street|ave|avenue|rd|road|sq|square|blvd|boulevard|pl|plaza|"
    r"hwy|highway|ln|lane|dr|drive|pkwy|parkway|tpke|turnpike)$"
)


def branch_key(location: str, store: str = "") -> str:
    """Normalise a branch name, dropping the noise variants the model emits.

    A qualifier after a comma is *kept*: "Beacon St, Somerville" and "Beacon
    Street, Washington Square" are genuinely different Star Markets, and
    merging them would be worse than splitting them.

    Occasionally the model puts something in `location` that is not a place:
    the category name, or the store's own name. Rare (16 of 28,225 claims on
    the real corpus) but each one becomes a visible branch heading, so they
    are dropped to "" -- the claim still counts, just at chain level.
    """
    key = normalise(location)
    if key in _non_locations() or (store and key == normalise(store)):
        return ""
    # "Somerville, MA" and "the Somerville one" are the same branch.
    key = re.sub(r"^the\s+", "", key)
    key = re.sub(r"\s+(one|store|location)$", "", key)
    key = re.sub(r",?\s*(ma|mass|massachusetts)$", "", key).strip()
    # Applied per comma-separated part, so "beacon st, somerville" folds its
    # street name without losing the town that distinguishes it.
    return ", ".join(STREET_TYPE.sub("", p.strip()) for p in key.split(","))


def half_life_for(category: str) -> float:
    return HALF_LIFE_BY_CATEGORY.get(category, DEFAULT_HALF_LIFE_YEARS)


def recency_weight(created_utc: int, now: int, half_life_years: float) -> float:
    """Exponential decay by age. Future timestamps clamp to weight 1.0."""
    years = max((now - created_utc) / SECONDS_PER_YEAR, 0.0)
    return float(0.5 ** (years / half_life_years))


def vote_weight(score: int | None) -> float:
    """Community agreement, damped. A downvoted claim is discounted.

    Upvotes are a weak signal but not no signal, so this moves weight within
    roughly [0.5, 2.0] rather than letting one viral comment dominate.
    """
    if score is None:
        return 1.0
    if score < 0:
        return 0.5
    return float(min(1.0 + math.log1p(score) / 5.0, 2.0))


def claim_weight(claim: SourcedClaim, now: int) -> float:
    confidence = CONFIDENCE_WEIGHT.get(
        claim.get("confidence", ""), DEFAULT_CONFIDENCE_WEIGHT
    )
    decay = recency_weight(claim["created_utc"], now, half_life_for(claim["category"]))
    return decay * confidence * vote_weight(claim.get("score"))


@dataclass
class Cell:
    """Accumulated evidence for one (store, location, category) triple."""

    n: int = 0
    weight: float = 0.0
    score: float = 0.0
    valenced_weight: float = 0.0
    price_score: float = 0.0
    price_weight: float = 0.0
    price_counts: dict[str, int] = field(default_factory=dict)
    examples: list[tuple[float, SourcedClaim]] = field(default_factory=list)
    # Cells are keyed on a casefolded name so spelling variants merge, but
    # the reader should see "Somerville", not "somerville".
    labels: Counter[str] = field(default_factory=Counter)

    def label(self, fallback: str = "") -> str:
        common = self.labels.most_common(1)
        return common[0][0] if common else fallback

    def add(self, claim: SourcedClaim, weight: float) -> None:
        self.n += 1
        self.weight += weight
        sentiment = claim.get("sentiment", "neutral")
        if sentiment in SENTIMENT_SCORE:
            # Neutral and mixed count as evidence but must not drag the mean
            # toward zero -- "nobody had a view" is not "views cancelled out".
            self.score += weight * SENTIMENT_SCORE[sentiment]
            self.valenced_weight += weight
        signal = claim.get("price_signal", "none")
        if signal in PRICE_SCORE:
            self.price_score += weight * PRICE_SCORE[signal]
            self.price_weight += weight
            self.price_counts[signal] = self.price_counts.get(signal, 0) + 1
        self.examples.append((weight, claim))

    def sentiment(self) -> float:
        """Weighted mean, shrunk toward 0 by the evidence actually present."""
        if not self.valenced_weight:
            return 0.0
        return self.score / (self.valenced_weight + SHRINKAGE_K)

    def price_level(self) -> float | None:
        """-1 cheap .. +1 expensive, shrunk. None when cost is undiscussed."""
        if not self.price_weight:
            return None
        return self.price_score / (self.price_weight + SHRINKAGE_K)

    def price_label(self) -> str | None:
        """cheap / fair / expensive, or None when the evidence cannot say.

        "fair" must mean "the evidence says middling", never "there is not
        enough evidence to tell". Shrinkage pulls a thin cell toward 0, so
        without a floor a single "cheap" claim at weight 0.3 lands at -0.13
        and gets printed as "fair" -- the opposite of what the one person who
        commented actually said.
        """
        level = self.price_level()
        if level is None or self.price_weight < MIN_PRICE_WEIGHT:
            return None
        if level < -POSITIVE_THRESHOLD:
            return "cheap"
        if level > POSITIVE_THRESHOLD:
            return "expensive"
        return "fair"

    def top_examples(self, n: int) -> list[SourcedClaim]:
        return [c for _, c in heapq.nlargest(n, self.examples, key=itemgetter(0))]


@dataclass
class Totals:
    """Per-store rollup, derived from cells rather than accumulated twice."""

    n: int = 0
    weight: float = 0.0
    score: float = 0.0
    valenced_weight: float = 0.0

    def sentiment(self) -> float:
        if not self.valenced_weight:
            return 0.0
        return self.score / (self.valenced_weight + SHRINKAGE_K)


def month(created_utc: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(created_utc))


def _is_valid(row: Any) -> bool:
    """Validate the whole row, not just key presence."""
    if not isinstance(row, dict) or not REQUIRED_FIELDS <= row.keys():
        return False
    ts = row.get("created_utc")
    if not isinstance(ts, int) or isinstance(ts, bool):
        return False
    # An out-of-range int passes isinstance and then raises inside time.gmtime,
    # so bound it here rather than at the formatter.
    if not MIN_TIMESTAMP <= ts <= int(time.time()) + SECONDS_PER_YEAR:
        return False
    score = row.get("score")
    if score is not None and (not isinstance(score, int) or isinstance(score, bool)):
        return False
    if not isinstance(row.get("transient", False), bool):
        return False
    if any(
        f in row and not isinstance(row[f], str) for f in OPTIONAL_STRING_FIELDS
    ):
        return False
    return all(isinstance(row.get(f), str) for f in STRING_FIELDS)


def read_claims(path: Path) -> tuple[list[SourcedClaim], int]:
    """Read claims, dropping malformed rows. Returns (claims, n_dropped).

    Scrubbing happens here as well as on the write path: this is the trust
    boundary for on-disk data, and a claims file produced by an older version
    of the pipeline has not been through the stage-2 scrubber.
    """
    rows, unparseable = read_jsonl(path)
    good: list[SourcedClaim] = []
    for row in rows:
        if not _is_valid(row):
            continue
        for f in ("claim", "location", "item", "comparator_store"):
            if isinstance(row.get(f), str):
                row[f] = scrub(row[f])
        good.append(cast(SourcedClaim, row))
    return good, len(rows) - len(good) + unparseable


def dedupe(claims: Iterable[SourcedClaim]) -> list[SourcedClaim]:
    """Drop claims repeated by a crash-resume.

    `Sink.write` flushes claims before the done-key, so a process killed
    between the two re-extracts the document and appends its claims again.
    Identity is the canonical document key -- not `source_id`, since Reddit
    base-36 ids collide across the post and comment namespaces.
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[SourcedClaim] = []
    for claim in claims:
        key = (claim["source_key"], claim["store"], claim["category"], claim["claim"])
        if key in seen:
            continue
        seen.add(key)
        out.append(claim)
    return out


def collapse_reciprocals(claims: Iterable[SourcedClaim]) -> list[SourcedClaim]:
    """Mark both halves of a comparison made by one document.

    "Market Basket is cheaper than Shaw's" yields a claim about each store.
    They are one observation, not two, and counting them as two let Shaw's
    accumulate a reputation that was really the shadow of Market Basket
    enthusiasm. Marked here, half-weighted in `build`.
    """
    by_doc: dict[str, list[SourcedClaim]] = defaultdict(list)
    for claim in claims:
        by_doc[claim["source_key"]].append(claim)
    out: list[SourcedClaim] = []
    for group in by_doc.values():
        stores = {c["store"] for c in group}
        for claim in group:
            comparator = claim.get("comparator_store", "")
            if comparator and comparator in stores:
                marked = dict(claim)
                marked["_reciprocal"] = True
                out.append(cast(SourcedClaim, marked))
            else:
                out.append(claim)
    return out


def limit_per_source(claims: Iterable[SourcedClaim]) -> list[SourcedClaim]:
    """Cap how much one document, and one author per cell, can contribute.

    Highest-confidence claims are kept: the cap should discard the weakest
    part of a prolific poster's contribution, not an arbitrary part of it.
    """
    per_doc: dict[tuple[str, str], int] = defaultdict(int)
    per_author: dict[CellKey, int] = defaultdict(int)
    out: list[SourcedClaim] = []
    ordered = sorted(
        claims,
        key=lambda c: (
            -CONFIDENCE_WEIGHT.get(
                c.get("confidence", ""), DEFAULT_CONFIDENCE_WEIGHT
            ),
            # Break ties toward recent evidence rather than toward file order,
            # so the cap is deterministic and drops the stalest claims.
            -c["created_utc"],
            c["source_key"],
            c["claim"],
        ),
    )
    for claim in ordered:
        # Per (document, store), not per document. A comment comparing five
        # stores is five observations, not one; capping globally discarded
        # evidence about stores the writer had said only one thing about.
        doc = (claim["source_key"], claim["store"])
        if per_doc[doc] >= MAX_CLAIMS_PER_DOCUMENT:
            continue
        # Deleted and suppressed accounts are not one prolific poster.
        author = claim.get("author", "")
        anonymous = author in ANONYMOUS_AUTHORS
        author_key = (author, claim["store"], claim["category"])
        if not anonymous and per_author[author_key] >= MAX_CLAIMS_PER_AUTHOR_CELL:
            continue
        per_doc[doc] += 1
        if not anonymous:
            per_author[author_key] += 1
        out.append(claim)
    return out


def prepare(
    claims: Iterable[SourcedClaim], exclude_transient: bool = True
) -> list[SourcedClaim]:
    """Everything that must happen before anything is counted.

    Transient claims are dropped *first*. Filtering them inside `build`, after
    the caps, let three closing-down-sale claims fill a document's cap and
    starve the durable claim in the same comment -- the evidence was discarded
    by a rule that was then itself discarded.
    """
    rows: Iterable[SourcedClaim] = dedupe(claims)
    if exclude_transient:
        rows = [c for c in rows if not c.get("transient")]
    return limit_per_source(collapse_reciprocals(rows))


def build(
    claims: Iterable[SourcedClaim], now: int, exclude_transient: bool = True
) -> tuple[dict[CellKey, Cell], dict[str, Cell]]:
    """Accumulate branch-level cells and a per-store item index."""
    cells: dict[CellKey, Cell] = defaultdict(Cell)
    items: dict[str, Cell] = defaultdict(Cell)
    for claim in claims:
        if exclude_transient and claim.get("transient"):
            continue
        weight = claim_weight(claim, now)
        if claim.get("_reciprocal"):
            weight *= 0.5
        location = branch_key(claim.get("location", ""), claim["store"])
        cell = cells[(claim["store"], location, claim["category"])]
        cell.add(claim, weight)
        if location:
            cell.labels[claim["location"].strip()] += 1
        item = normalise(claim.get("item", ""))
        if item:
            index = items[f"{claim['store']}|{item}"]
            index.add(claim, weight)
            index.labels[claim["item"].strip()] += 1
    return dict(cells), dict(items)


def chain_rollup(cells: Mapping[CellKey, Cell]) -> dict[tuple[str, str], Cell]:
    """Fold every branch of a chain into one (store, category) cell."""
    rolled: dict[tuple[str, str], Cell] = defaultdict(Cell)
    for (store, _location, category), cell in cells.items():
        parent = rolled[(store, category)]
        parent.n += cell.n
        parent.weight += cell.weight
        parent.score += cell.score
        parent.valenced_weight += cell.valenced_weight
        parent.price_score += cell.price_score
        parent.price_weight += cell.price_weight
        for k, v in cell.price_counts.items():
            parent.price_counts[k] = parent.price_counts.get(k, 0) + v
        parent.examples.extend(cell.examples)
    return dict(rolled)


def totals_from(
    cells: Mapping[tuple[str, str], Cell], shopping_only: bool = True
) -> dict[str, Totals]:
    """Sum chain cells per store.

    `shopping_only` drops labor_ethics and store_lifecycle: both are real, but
    a headline "which store is best" number built partly on regional civic
    affection for a chain is not answering the question asked.
    """
    from .extract import NON_SHOPPING_CATEGORIES

    totals: dict[str, Totals] = defaultdict(Totals)
    for (store, category), cell in cells.items():
        if shopping_only and category in NON_SHOPPING_CATEGORIES:
            continue
        t = totals[store]
        t.n += cell.n
        t.weight += cell.weight
        t.score += cell.score
        t.valenced_weight += cell.valenced_weight
    return dict(totals)


def branch_totals_from(
    cells: Mapping[CellKey, Cell], shopping_only: bool = True
) -> dict[tuple[str, str], Totals]:
    """Sum branch cells per (store, branch).

    Same rollup as `totals_from`, one level down. A branch needs its own
    headline number before it can be compared against anything else measured
    at that branch — a per-location star rating, for instance.
    """
    from .extract import NON_SHOPPING_CATEGORIES

    totals: dict[tuple[str, str], Totals] = defaultdict(Totals)
    for (store, location, category), cell in cells.items():
        if not location:
            continue
        if shopping_only and category in NON_SHOPPING_CATEGORIES:
            continue
        t = totals[(store, location)]
        t.n += cell.n
        t.weight += cell.weight
        t.score += cell.score
        t.valenced_weight += cell.valenced_weight
    return dict(totals)


def headline_half_life(cells: Mapping[tuple[str, str], Cell]) -> float:
    """The rate at which the headline number actually ages.

    Store totals are an evidence-weighted mix of categories with half-lives
    from 1.5 to 7 years, so the mix ages at neither the default nor any one
    category's rate — it comes out near 4.7y on this corpus. Anything
    compared against that number should be decayed at the same rate, or the
    comparison quietly favours whichever side is aged more gently.
    """
    from .extract import NON_SHOPPING_CATEGORIES

    num = den = 0.0
    for (_store, category), cell in cells.items():
        if category in NON_SHOPPING_CATEGORIES:
            continue
        num += cell.weight * half_life_for(category)
        den += cell.weight
    return num / den if den else DEFAULT_HALF_LIFE_YEARS


def _cell_view(cell: Cell, max_examples: int) -> dict[str, Any]:
    level = cell.price_level()
    return {
        "n_claims": cell.n,
        "weighted_evidence": round(cell.weight, 2),
        "sentiment": round(cell.sentiment(), 3),
        "price_level": None if level is None else round(level, 3),
        "price_signal": cell.price_label(),
        "price_distribution": dict(cell.price_counts),
        "evidence": [
            {
                "claim": c["claim"],
                "date": month(c["created_utc"]),
                "permalink": c["permalink"],
                "confidence": c["confidence"],
                "location": c.get("location", ""),
            }
            for c in cell.top_examples(max_examples)
        ],
    }


def aggregate(
    claims: Iterable[SourcedClaim],
    now: int | None = None,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_examples: int = 5,
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the verdict document the app queries."""
    stamp = int(time.time()) if now is None else now
    cells, items = build(prepare(claims), stamp)
    chain = chain_rollup(cells)
    totals = totals_from(chain)

    stores: dict[str, dict[str, Any]] = {}
    for (store, category), cell in chain.items():
        if cell.weight < min_weight:
            continue
        stores.setdefault(store, {})[category] = _cell_view(cell, max_examples)

    branches: dict[str, dict[str, dict[str, Any]]] = {}
    for (store, location, category), cell in cells.items():
        if not location or cell.weight < min_weight:
            continue
        name = cell.label(location)
        branches.setdefault(store, {}).setdefault(name, {})[category] = _cell_view(
            cell, max_examples
        )

    # Branch-level headline numbers, built the same way as the chain ones so
    # a branch can be compared against anything else measured at that branch.
    # Display names come from the cells, so "somerville" reads "Somerville".
    branch_view: dict[str, dict[str, Any]] = {}
    labels: dict[tuple[str, str], str] = {}
    for (store, location, _cat), cell in cells.items():
        if location:
            labels.setdefault((store, location), cell.label(location))
    for (store, location), t in branch_totals_from(cells).items():
        branch_view.setdefault(store, {})[labels[(store, location)]] = {
            "n_claims": t.n,
            "weighted_evidence": round(t.weight, 2),
            "sentiment": round(t.sentiment(), 3),
        }

    item_index: dict[str, dict[str, Any]] = {}
    for key, cell in items.items():
        if cell.weight < min_weight:
            continue
        store, item = key.split("|", 1)
        item_index.setdefault(store, {})[cell.label(item)] = _cell_view(cell, 2)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp)),
        "method": {
            "half_life_years": dict(HALF_LIFE_BY_CATEGORY),
            "default_half_life_years": DEFAULT_HALF_LIFE_YEARS,
            "shrinkage_k": SHRINKAGE_K,
            "min_weight": min_weight,
            "max_claims_per_document": MAX_CLAIMS_PER_DOCUMENT,
            "max_claims_per_author_cell": MAX_CLAIMS_PER_AUTHOR_CELL,
            "transient_claims": "excluded",
            "note": (
                "sentiment is shrunk toward 0 by shrinkage_k, so thin cells "
                "read closer to neutral than their raw mean; neutral and mixed "
                "claims count as evidence but not toward the mean"
            ),
        },
        "corpus": dict(corpus) if corpus else None,
        "stores": stores,
        "branches": branches,
        "items": item_index,
        "headline_half_life_years": round(headline_half_life(chain), 3),
        "branch_totals": branch_view,
        "store_totals": {
            store: {
                "n_claims": t.n,
                "weighted_evidence": round(t.weight, 2),
                "sentiment": round(t.sentiment(), 3),
                # A store whose every cell was suppressed has no viewable
                # evidence; saying so beats printing a confident number.
                "insufficient_evidence": store not in stores,
            }
            for store, t in totals.items()
        },
    }


def write_verdicts(summary: Mapping[str, Any], path: Path) -> None:
    write_atomic(path, [json.dumps(summary, indent=2, ensure_ascii=False)])


def format_totals(summary: Mapping[str, Any]) -> str:
    totals: Mapping[str, Mapping[str, Any]] = summary["store_totals"]
    lines = [
        f"{'store':<22}{'claims':>8}{'evidence':>10}{'sentiment':>11}  note",
        "-" * 64,
    ]
    for store, v in sorted(
        totals.items(), key=lambda kv: -float(kv[1]["weighted_evidence"])
    ):
        note = "insufficient evidence" if v["insufficient_evidence"] else ""
        lines.append(
            f"{store:<22}{v['n_claims']:>8,}{v['weighted_evidence']:>10.1f}"
            f"{v['sentiment']:>+11.2f}  {note}"
        )
    return "\n".join(lines)


def format_store(summary: Mapping[str, Any], store: str, max_evidence: int = 3) -> str:
    categories: Mapping[str, Mapping[str, Any]] = summary["stores"].get(store, {})
    lines = [store, "=" * len(store)]
    if not categories:
        lines.append("  (no claims above the evidence threshold)")
        return "\n".join(lines)
    ordered = sorted(
        categories.items(), key=lambda kv: -float(kv[1]["weighted_evidence"])
    )
    for category, v in ordered:
        sentiment = float(v["sentiment"])
        mark = (
            "+" if sentiment > POSITIVE_THRESHOLD
            else ("-" if sentiment < -POSITIVE_THRESHOLD else "~")
        )
        lines.append(
            f"\n  {category:<18} {mark} sentiment={sentiment:+.2f}  "
            f"n={v['n_claims']}  price={v['price_signal'] or '-'}"
        )
        for e in list(v["evidence"])[:max_evidence]:
            where = f" [{e['location']}]" if e.get("location") else ""
            lines.append(f"      [{e['date']}]{where} {e['claim'][:100]}")
    branches: Mapping[str, Any] = summary.get("branches", {}).get(store, {})
    if branches:
        lines.append(
            f"\n  branches with their own evidence: {', '.join(sorted(branches))}"
        )
    items: Mapping[str, Any] = summary.get("items", {}).get(store, {})
    if items:
        top = sorted(
            items.items(), key=lambda kv: -float(kv[1]["weighted_evidence"])
        )[:5]
        lines.append(
            "  items: " + ", ".join(f"{k} ({v['sentiment']:+.2f})" for k, v in top)
        )
    return "\n".join(lines)
