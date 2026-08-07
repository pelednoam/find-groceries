"""Measure whether stage 2 is actually any good.

Until now the only evidence that extraction worked was that it returned
plausible-looking JSON. That is not evidence: the failure modes that matter
here are all ones that produce well-formed output — inventing a claim about a
store the text merely names, reading an affirmative sentence as a positive
one, filing "the Wegmans is a mile away" as parking_access.

So: a hand-labelled gold set, and a scorer that runs the real extractor over
it and reports where it disagrees. Every case in the gold set exists because
it is a documented failure mode of the prompt, not because it was convenient.

Matching is on the (store, category, sentiment) triple. The free-text `claim`
is deliberately not scored -- there are many correct phrasings, and grading
prose against prose needs a judge, which is a second thing to be wrong. The
two structured flags stage 3 depends on, `transient` and `comparator_store`,
are scored separately over the claims that matched: a wrong `transient` does
not make the claim wrong, but it does put a closing-down sale into the
aggregate as a durable property of the store.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from .extract import Extractor
from .jsonl import read_jsonl
from .types import Candidate, SourcedClaim

Triple = tuple[str, str, str]  # store, category, sentiment


class GoldCase(TypedDict):
    """One labelled document.

    `expected` is the set of claims a correct extraction must produce. An
    empty list is a real and important label: most of the corpus supports no
    claim at all, and a model that finds one anyway is the main risk.
    """

    id: str
    why: str  # the failure mode this case exists to catch
    text: str
    parent_body: str
    stores: list[str]
    expected: list[dict[str, str]]


# Both the labels (plain dicts) and the extractor's output (SourcedClaim)
# flow through here, so these read the fields structurally rather than
# insisting on one of the two types.
ClaimLike = Mapping[str, Any]


def triples(claims: Iterable[ClaimLike]) -> set[Triple]:
    return {(c["store"], c["category"], c["sentiment"]) for c in claims}


Flags = tuple[bool, str, str]  # transient, comparator_store, price_signal


def flags(claims: Iterable[ClaimLike]) -> dict[Triple, Flags]:
    """Index the consequential non-triple fields by the triple they belong to.

    `price_signal` belongs here, not nowhere: it is the sole input to stage
    3's entire ordinal price index. Leaving it unscored let the evaluator
    report exact agreement on a run that would have told the reader Whole
    Foods was cheap.
    """
    return {
        (c["store"], c["category"], c["sentiment"]): (
            bool(c.get("transient", False)),
            str(c.get("comparator_store", "")),
            str(c.get("price_signal", "none")),
        )
        for c in claims
    }


def as_candidate(case: GoldCase) -> Candidate:
    """Wrap a gold case in the shape stage 2 consumes."""
    return Candidate(
        id=case["id"],
        subreddit="gold",
        kind="comments",
        created_utc=1_700_000_000,
        score=None,
        author="",
        parent_body=case["parent_body"],
        permalink=f"/gold/{case['id']}",
        stores=case["stores"],
        text=case["text"],
        truncated=False,
    )


@dataclass
class CaseResult:
    case_id: str
    why: str
    expected: set[Triple]
    got: set[Triple]
    error: str | None = None
    expected_flags: dict[Triple, Flags] = field(default_factory=dict)
    got_flags: dict[Triple, Flags] = field(default_factory=dict)

    @property
    def missed(self) -> set[Triple]:
        return self.expected - self.got

    @property
    def spurious(self) -> set[Triple]:
        return self.got - self.expected

    @property
    def matched(self) -> set[Triple]:
        return self.expected & self.got

    def flag_disagreements(self) -> list[tuple[Triple, Flags, Flags]]:
        return [
            (t, self.expected_flags[t], self.got_flags[t])
            for t in sorted(self.matched)
            if t in self.expected_flags
            and t in self.got_flags
            and self.expected_flags[t] != self.got_flags[t]
        ]

    @property
    def exact(self) -> bool:
        return (
            self.error is None
            and self.expected == self.got
            and not self.flag_disagreements()
        )


@dataclass
class Score:
    """Aggregate agreement between the extractor and the labels."""

    cases: list[CaseResult] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(len(c.expected & c.got) for c in self.cases)

    @property
    def false_positives(self) -> int:
        return sum(len(c.spurious) for c in self.cases)

    @property
    def false_negatives(self) -> int:
        return sum(len(c.missed) for c in self.cases)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases if c.error)

    def precision(self) -> float | None:
        """None, not 1.0, when the extractor emitted nothing at all.

        A run that returns zero claims has not achieved perfect precision; it
        has produced no evidence about precision. Printing 1.00 there is the
        single most flattering way to report a total failure.
        """
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else None

    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        if p is None:
            return 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def exact_match(self) -> float:
        """The metric that matters most: whole documents graded correctly."""
        return (
            sum(1 for c in self.cases if c.exact) / len(self.cases)
            if self.cases
            else 1.0
        )

    def flag_accuracy(self) -> float | None:
        """Agreement on transient / comparator_store / price_signal."""
        matched = sum(len(c.matched) for c in self.cases)
        wrong = sum(len(c.flag_disagreements()) for c in self.cases)
        return (matched - wrong) / matched if matched else None

    def silence_accuracy(self) -> float | None:
        """How often the extractor correctly finds nothing.

        Broken out because it is the single easiest metric to lose: a model
        nudged toward productivity scores well on documents that do support a
        claim and floods the aggregate from the ones that do not.

        Errored cases are excluded, not counted as silence. A case that threw
        also produced no claims, and crediting that as a correct "found
        nothing" turns a broken run into a perfect score on the one metric
        this eval exists to protect.
        """
        quiet = [c for c in self.cases if not c.expected and c.error is None]
        return sum(1 for c in quiet if not c.got) / len(quiet) if quiet else None


def read_gold(path: Path) -> list[GoldCase]:
    rows, unparseable = read_jsonl(path)
    if unparseable:
        raise ValueError(f"{path}: {unparseable} unparseable lines")
    return [row for row in rows if not str(row.get("id", "")).startswith("#")]


def score_one(extractor: Extractor, case: GoldCase) -> CaseResult:
    expected = triples(case["expected"])
    try:
        claims, _ = extractor.extract(as_candidate(case))
    except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
        return CaseResult(case["id"], case["why"], expected, set(), str(exc)[:200])
    return CaseResult(
        case["id"],
        case["why"],
        expected,
        triples(claims),
        expected_flags=flags(case["expected"]),
        got_flags=flags(claims),
    )


def evaluate(extractor: Extractor, cases: Sequence[GoldCase]) -> Score:
    return Score([score_one(extractor, case) for case in cases])


def pct(value: float | None) -> str:
    """Render a metric, distinguishing "perfect" from "no evidence"."""
    return "n/a" if value is None else f"{value:.0%}"


def format_score(score: Score) -> str:
    lines = [
        f"cases            {len(score.cases)}",
        f"exact match      {pct(score.exact_match())}",
        f"precision        {pct(score.precision())}",
        f"recall           {pct(score.recall())}",
        f"f1               {score.f1():.2f}",
        f"correct silence  {pct(score.silence_accuracy())}",
        f"flag agreement   {pct(score.flag_accuracy())}",
    ]
    if score.errors:
        lines.append(f"errors           {score.errors}")
    failures = [c for c in score.cases if not c.exact]
    if failures:
        lines += ["", "disagreements:"]
    for c in failures:
        lines.append(f"\n  {c.case_id} — {c.why}")
        if c.error:
            lines.append(f"      ERROR {c.error}")
        for t in sorted(c.missed):
            lines.append(f"      missed    {t}")
        for t in sorted(c.spurious):
            lines.append(f"      spurious  {t}")
        for t, want, got in c.flag_disagreements():
            lines.append(f"      flags     {t}: want {want}, got {got}")
    return "\n".join(lines)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def write_report(score: Score, path: Path) -> None:
    payload = {
        "exact_match": _round(score.exact_match()),
        "precision": _round(score.precision()),
        "recall": _round(score.recall()),
        "f1": round(score.f1(), 4),
        "silence_accuracy": _round(score.silence_accuracy()),
        "flag_accuracy": _round(score.flag_accuracy()),
        "errors": score.errors,
        "cases": [
            {
                "id": c.case_id,
                "why": c.why,
                "exact": c.exact,
                "expected": sorted(c.expected),
                "got": sorted(c.got),
                "flag_disagreements": [
                    {"triple": list(t), "want": list(w), "got": list(g)}
                    for t, w, g in c.flag_disagreements()
                ],
                "error": c.error,
            }
            for c in score.cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def claims_for(case: GoldCase, claims: Sequence[SourcedClaim]) -> list[SourcedClaim]:
    """Claims belonging to one case, for eyeballing what the model wrote."""
    return [c for c in claims if c["source_id"] == case["id"]]
