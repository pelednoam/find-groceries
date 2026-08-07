"""Stage 3: roll claims up into per-store, per-category verdicts.

Reddit opinion goes stale — a 2018 verdict on a store that has since been
renovated should not outweigh a 2025 one. Claims are weighted by age with a
four-year half-life and by the extractor's own confidence.
"""

from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Any, Final, cast

from .jsonl import read_jsonl, write_atomic
from .types import SourcedClaim

HALF_LIFE_YEARS: Final = 4.0
SECONDS_PER_YEAR: Final = 365.25 * 24 * 3600

CONFIDENCE_WEIGHT: Final[Mapping[str, float]] = {
    "high": 1.0, "medium": 0.6, "low": 0.3,
}
SENTIMENT_SCORE: Final[Mapping[str, float]] = {
    "positive": 1.0, "mixed": 0.0, "neutral": 0.0, "negative": -1.0,
}
DEFAULT_CONFIDENCE_WEIGHT: Final = 0.6
POSITIVE_THRESHOLD: Final = 0.15


def recency_weight(created_utc: int, now: int, half_life_years: float = HALF_LIFE_YEARS) -> float:
    """Exponential decay by age. Future timestamps clamp to weight 1.0."""
    years = max((now - created_utc) / SECONDS_PER_YEAR, 0.0)
    return float(0.5 ** (years / half_life_years))


def claim_weight(claim: SourcedClaim, now: int) -> float:
    confidence = CONFIDENCE_WEIGHT.get(
        claim.get("confidence", ""), DEFAULT_CONFIDENCE_WEIGHT
    )
    return recency_weight(claim["created_utc"], now) * confidence


@dataclass
class Cell:
    """Accumulated evidence for one (store, category) pair."""

    n: int = 0
    weight: float = 0.0
    score: float = 0.0
    price: dict[str, float] = field(default_factory=dict)
    examples: list[tuple[float, SourcedClaim]] = field(default_factory=list)

    def add(self, claim: SourcedClaim, weight: float) -> None:
        self.n += 1
        self.weight += weight
        self.score += weight * SENTIMENT_SCORE.get(claim.get("sentiment", ""), 0.0)
        signal = claim.get("price_signal", "none")
        if signal != "none":
            self.price[signal] = self.price.get(signal, 0.0) + weight
        self.examples.append((weight, claim))

    def sentiment(self) -> float:
        return self.score / self.weight if self.weight else 0.0

    def dominant_price(self) -> str | None:
        if not self.price:
            return None
        return max(self.price, key=lambda k: self.price[k])

    def top_examples(self, n: int) -> list[SourcedClaim]:
        heaviest = heapq.nlargest(n, self.examples, key=itemgetter(0))
        return [claim for _, claim in heaviest]


def month(created_utc: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(created_utc))


def dedupe(claims: Iterable[SourcedClaim]) -> list[SourcedClaim]:
    """Drop claims repeated by a crash-resume.

    `Sink.write` flushes claims before the done-key, so a process killed
    between those two flushes re-extracts the document on the next run and
    appends its claims a second time. Identity is (source document, store,
    category, claim text) — the same document legitimately yields several
    claims, but not two identical ones.
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[SourcedClaim] = []
    for claim in claims:
        # source_key, not source_id: Reddit base-36 ids collide across the
        # post and comment namespaces, so id alone would merge a post and a
        # comment that happen to make the same claim.
        key = (
            claim["source_key"],
            claim["store"],
            claim["category"],
            claim["claim"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(claim)
    return out


def build(claims: Iterable[SourcedClaim], now: int) -> dict[tuple[str, str], Cell]:
    """Accumulate per-(store, category) evidence."""
    cells: dict[tuple[str, str], Cell] = defaultdict(Cell)
    for claim in claims:
        cells[(claim["store"], claim["category"])].add(claim, claim_weight(claim, now))
    return dict(cells)


@dataclass
class Totals:
    """Per-store rollup. Derived from the cells rather than accumulated twice."""

    n: int = 0
    weight: float = 0.0
    score: float = 0.0

    def sentiment(self) -> float:
        return self.score / self.weight if self.weight else 0.0


def totals_from(cells: Mapping[tuple[str, str], Cell]) -> dict[str, Totals]:
    """Sum cells per store. Unfiltered — totals count suppressed cells too."""
    totals: dict[str, Totals] = defaultdict(Totals)
    for (store, _category), cell in cells.items():
        t = totals[store]
        t.n += cell.n
        t.weight += cell.weight
        t.score += cell.score
    return dict(totals)


def aggregate(
    claims: Iterable[SourcedClaim],
    now: int | None = None,
    min_weight: float = 1.0,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Produce the verdict document the app queries."""
    stamp = int(time.time()) if now is None else now
    cells = build(dedupe(claims), stamp)
    totals = totals_from(cells)

    stores: dict[str, dict[str, Any]] = {}
    for (store, category), cell in cells.items():
        if cell.weight < min_weight:
            continue
        stores.setdefault(store, {})[category] = {
            "n_claims": cell.n,
            "weighted_evidence": round(cell.weight, 2),
            "sentiment": round(cell.sentiment(), 3),
            "price_signal": cell.dominant_price(),
            "evidence": [
                {
                    "claim": c["claim"],
                    "date": month(c["created_utc"]),
                    "permalink": c["permalink"],
                    "confidence": c.get("confidence", "medium"),
                }
                for c in cell.top_examples(max_examples)
            ],
        }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp)),
        "half_life_years": HALF_LIFE_YEARS,
        "stores": stores,
        "store_totals": {
            store: {
                "n_claims": t.n,
                "weighted_evidence": round(t.weight, 2),
                "sentiment": round(t.sentiment(), 3),
            }
            for store, t in totals.items()
        },
    }


REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"store", "category", "claim", "sentiment", "confidence",
     "source_key", "created_utc", "permalink"}
)


STRING_FIELDS: Final[tuple[str, ...]] = (
    "store", "category", "claim", "sentiment", "confidence", "source_key",
    "permalink",
)


def _is_valid(row: Any) -> bool:
    """Validate the whole row, not just key presence.

    Presence-only checking let `{"store": []}` reach `dedupe`, where an
    unhashable value in the identity tuple raises instead of being rejected
    at the boundary that claims to own this.
    """
    if not isinstance(row, dict) or not REQUIRED_FIELDS <= row.keys():
        return False
    if not isinstance(row.get("created_utc"), int):
        return False
    return all(isinstance(row.get(f), str) for f in STRING_FIELDS)


def read_claims(path: Path) -> tuple[list[SourcedClaim], int]:
    """Read claims, dropping malformed rows. Returns (claims, n_dropped).

    This is the pipeline's trust boundary for on-disk data: everything
    downstream may assume provenance is present, so rows that lack it are
    rejected here rather than silently defaulted deep in the scoring math.
    """
    rows, unparseable = read_jsonl(path)
    good: list[SourcedClaim] = [cast(SourcedClaim, r) for r in rows if _is_valid(r)]
    return good, len(rows) - len(good) + unparseable


def write_verdicts(summary: Mapping[str, Any], path: Path) -> None:
    write_atomic(path, [json.dumps(summary, indent=2, ensure_ascii=False)])


def format_totals(summary: Mapping[str, Any]) -> str:
    totals: Mapping[str, Mapping[str, Any]] = summary["store_totals"]
    lines = [f"{'store':<22}{'claims':>8}{'evidence':>10}{'sentiment':>11}", "-" * 51]
    for store, v in sorted(totals.items(), key=lambda kv: -float(kv[1]["weighted_evidence"])):
        lines.append(
            f"{store:<22}{v['n_claims']:>8,}{v['weighted_evidence']:>10.1f}"
            f"{v['sentiment']:>+11.2f}"
        )
    return "\n".join(lines)


def format_store(summary: Mapping[str, Any], store: str, max_evidence: int = 3) -> str:
    categories: Mapping[str, Mapping[str, Any]] = summary["stores"].get(store, {})
    lines = [store, "=" * len(store)]
    if not categories:
        lines.append("  (no claims above the evidence threshold)")
        return "\n".join(lines)
    ordered = sorted(categories.items(), key=lambda kv: -float(kv[1]["weighted_evidence"]))
    for category, v in ordered:
        sentiment = float(v["sentiment"])
        mark = "+" if sentiment > POSITIVE_THRESHOLD else ("-" if sentiment < -POSITIVE_THRESHOLD else "~")
        lines.append(
            f"\n  {category:<18} {mark} sentiment={sentiment:+.2f}  "
            f"n={v['n_claims']}  price={v['price_signal'] or '-'}"
        )
        for e in list(v["evidence"])[:max_evidence]:
            lines.append(f"      [{e['date']}] {e['claim'][:110]}")
    return "\n".join(lines)
