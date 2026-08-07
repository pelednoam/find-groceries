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


def flags(claims: Iterable[ClaimLike]) -> dict[Triple, tuple[bool, str]]:
    """Index the unscored-but-consequential fields by the triple they belong to."""
    return {
        (c["store"], c["category"], c["sentiment"]): (
            bool(c.get("transient", False)),
            str(c.get("comparator_store", "")),
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
    expected_flags: dict[Triple, tuple[bool, str]] = field(default_factory=dict)
    got_flags: dict[Triple, tuple[bool, str]] = field(default_factory=dict)

    @property
    def missed(self) -> set[Triple]:
        return self.expected - self.got

    @property
    def spurious(self) -> set[Triple]:
        return self.got - self.expected

    @property
    def matched(self) -> set[Triple]:
        return self.expected & self.got

    def flag_disagreements(self) -> list[tuple[Triple, tuple[bool, str], tuple[bool, str]]]:
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

    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0

    def exact_match(self) -> float:
        """The metric that matters most: whole documents graded correctly."""
        return (
            sum(1 for c in self.cases if c.exact) / len(self.cases)
            if self.cases
            else 1.0
        )

    def flag_accuracy(self) -> float:
        """Agreement on `transient`/`comparator_store`, over matched claims."""
        matched = sum(len(c.matched) for c in self.cases)
        wrong = sum(len(c.flag_disagreements()) for c in self.cases)
        return (matched - wrong) / matched if matched else 1.0

    def silence_accuracy(self) -> float:
        """How often the extractor correctly finds nothing.

        Broken out because it is the single easiest metric to lose: a model
        nudged toward productivity scores well on documents that do support a
        claim and floods the aggregate from the ones that do not.
        """
        quiet = [c for c in self.cases if not c.expected]
        return sum(1 for c in quiet if not c.got) / len(quiet) if quiet else 1.0


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


def format_score(score: Score) -> str:
    lines = [
        f"cases            {len(score.cases)}",
        f"exact match      {score.exact_match():.0%}",
        f"precision        {score.precision():.2f}",
        f"recall           {score.recall():.2f}",
        f"f1               {score.f1():.2f}",
        f"correct silence  {score.silence_accuracy():.0%}",
        f"flag agreement   {score.flag_accuracy():.0%}",
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


def write_report(score: Score, path: Path) -> None:
    payload = {
        "exact_match": round(score.exact_match(), 4),
        "precision": round(score.precision(), 4),
        "recall": round(score.recall(), 4),
        "f1": round(score.f1(), 4),
        "silence_accuracy": round(score.silence_accuracy(), 4),
        "flag_accuracy": round(score.flag_accuracy(), 4),
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
