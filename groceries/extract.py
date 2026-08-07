"""Stage 2: turn candidate documents into structured store claims.

Runs against Claude on Amazon Bedrock via concurrent live calls. Bedrock has no
Anthropic Message Batches API; it does have S3-based batch inference at 50% off,
but that path cannot use prompt caching, which here covers ~86% of input tokens.
On-demand with caching wins.

Structured outputs guarantee the response parses, so there is no regex salvage
path and no retry-on-bad-JSON loop.

Trust boundary: the document text is untrusted third-party content and goes
into the user turn, so a Reddit comment can attempt prompt injection. The JSON
schema is the containment: `store`, `category`, `sentiment`, `price_signal`,
and `confidence` are closed enums, so the worst an injection achieves is
attacker-chosen prose in the free-text `claim`/`location`/`item` fields. Treat
those three as untrusted downstream — escape them before rendering as HTML.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .select import STORE_PATTERNS
from .types import (
    AnthropicClient,
    Candidate,
    Claim,
    ModelResponse,
    Pricing,
    RunStats,
    SourcedClaim,
    Usage,
)

DEFAULT_MODEL: Final = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION: Final = "us-east-1"
# A hard cap on thinking *plus* response text, not just the answer. Measured
# output is ~78 tokens, but adaptive thinking shares this budget, and a
# response truncated here surfaces as an unparseable-JSON failure.
MAX_TOKENS: Final = 8000
# Retry has exactly one owner: the SDK client is built with max_retries=0
# (see groceries/client.py), so this loop is the whole policy rather than an
# outer layer multiplying with the SDK's own.
MAX_ATTEMPTS: Final = 6

# Only models that accept this request shape belong here: `build_params`
# always sends `output_config.effort` and adaptive thinking, both of which
# Haiku 4.5 rejects, so listing it would advertise a route that 400s.
PRICING: Final[dict[str, Pricing]] = {
    "us.anthropic.claude-sonnet-4-6": Pricing(3.00, 15.00, 0.30, 3.75),
    "us.anthropic.claude-opus-5": Pricing(5.00, 25.00, 0.50, 6.25),
}
# Derived, never hand-listed: the schema enum must stay in lockstep with the
# stage-1 matchers or the model is structurally unable to name a store that
# stage 1 happily selects for.
STORES: Final[list[str]] = [*STORE_PATTERNS, "other"]

CATEGORIES: Final[list[str]] = [
    "produce", "meat", "seafood", "dairy", "bakery", "prepared_food", "pantry",
    "frozen", "alcohol", "specific_item", "price_overall", "quality_overall",
    "selection", "service_checkout", "cleanliness", "parking_access",
]

SYSTEM: Final = """You extract structured claims about grocery stores from Reddit posts and comments.

The reader is one shopper in Cambridge, Massachusetts deciding where to buy groceries. \
A claim is only useful to them if it says something evaluative about a store: what it is \
good or bad at, what something costs there, whether a specific item is worth buying.

Extract one entry per distinct (store, topic) judgment the text actually makes.

Rules:
- Only extract claims the text genuinely supports. Do not infer, extrapolate, or \
supply your own knowledge about these stores. An empty list is the correct answer for \
text that merely names a store as a landmark, asks a question without answering it, or \
discusses something unrelated to shopping there.
- `store` must be one of the listed names, or "other" if the text evaluates a grocery \
store not on the list. Never map a claim onto a listed store because it seems similar.
- `claim` is a single self-contained sentence in your own words, understandable without \
the original text. Include the specific detail (the item, the price, the comparison) \
rather than a generic summary.
- `location` is the specific branch only if the text names one (e.g. "Somerville", \
"Alewife", "Central Square"). Use "" when the text is about the chain generally.
- `item` is the specific product only when the claim is about one. Use "" otherwise.
- `sentiment` is whether this claim gives the reader a reason to shop at this \
store or a reason not to. "positive" = a reason to go; "negative" = a reason to \
avoid; "mixed" = the text asserts both a reason for and a reason against; \
"neutral" = a factual statement that gives the reader no reason either way. \
A purely descriptive observation ("the store is large", "they sell Bob Evans \
mashed potatoes for $3.99") is "neutral", not "positive" — do not treat an \
affirmative sentence as a positive one.
- `price_signal` describes what the text says about cost at this store: "cheap", \
"expensive", "fair", or "none" if cost is not discussed.
- `confidence` is "high" when the text states the claim directly from apparent firsthand \
experience, "medium" when it is hedged or secondhand, "low" when the text supports the \
claim only weakly. It measures how well the text backs the claim, never how sure you \
are of the underlying fact.
- Sarcasm and jokes are common. Judge the intended meaning, not the literal words.
- Text may be years old. Extract what it says; the pipeline records the date separately.

Categories:
- produce, meat, seafood, dairy, bakery, frozen, pantry, alcohol: the claim is about
  that department's quality, price, or selection at this store.
- prepared_food: hot bar, deli counter, sandwiches, ready meals.
- specific_item: the claim is about one named product rather than a department.
- price_overall: what the store costs generally, or a cost comparison against another store.
- quality_overall: the store's overall standard, with no single department named.
- selection: what the store does or does not carry — breadth, gaps, specialty items.
- service_checkout: staff, lines, self-checkout, bagging, returns.
- cleanliness: store condition.
- parking_access: parking, transit access, how hard the store is to get to and out of.

Pick the most specific category the text supports. Prefer a department category over
quality_overall when the text names a department, and specific_item over a department
when it names one product.

Return an empty list for text like these:
- "I got off at Haymarket and walked to the North End." — a place, not a store visit.
- "Does anyone know if the Star Market on Beacon is open late?" — a question with no answer.
- "Wegmans in Westwood is a large store." — descriptive, no reason to go or avoid.
- "Market Basket treats its employees well." — about the company, not the groceries."""

SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "enum": STORES},
                    "location": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "item": {"type": "string"},
                    "claim": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "mixed", "neutral"],
                    },
                    "price_signal": {
                        "type": "string",
                        "enum": ["cheap", "expensive", "fair", "none"],
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": [
                    "store", "location", "category", "item", "claim",
                    "sentiment", "price_signal", "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class ExtractionError(RuntimeError):
    """Raised when a document could not be extracted after all retries."""


def doc_key(doc: Candidate) -> str:
    """Stable per-document id used for resume bookkeeping."""
    return f"{doc['kind'][0]}_{doc['subreddit']}_{doc['id']}"


def user_message(doc: Candidate) -> str:
    return (
        f"Source: r/{doc['subreddit']}\n"
        f"Stores matched by the pre-filter: {', '.join(doc['stores'])}\n"
        f"(The pre-filter is keyword-based and may be wrong — trust the text.)\n\n"
        f"---\n{doc['text']}\n---\n\n"
        f"Extract the grocery-store claims this text supports."
    )


def build_params(
    doc: Candidate, model: str, thinking: dict[str, str]
) -> dict[str, Any]:
    """Assemble the messages.create kwargs for one document."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
        ],
        "thinking": thinking,
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        "messages": [{"role": "user", "content": user_message(doc)}],
    }


def thinking_config(enabled: bool) -> dict[str, str]:
    return {"type": "adaptive"} if enabled else {"type": "disabled"}


def parse_response(raw: Any) -> ModelResponse:
    """Narrow an SDK response object into the fields this pipeline uses.

    This is the single boundary where the SDK's untyped response surface is
    converted into a typed value; everything downstream works on ModelResponse.
    """
    usage_obj = getattr(raw, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    )
    text: str | None = None
    for block in getattr(raw, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            break
    return ModelResponse(
        stop_reason=getattr(raw, "stop_reason", None), text=text, usage=usage
    )


def parse_claims(text: str | None) -> list[Claim]:
    """Structured outputs guarantee valid JSON; empty text means no claims."""
    if not text:
        return []
    payload: dict[str, Any] = json.loads(text)
    claims: list[Claim] = list(payload.get("claims", []))
    return claims


UNSAFE_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(value: str) -> str:
    """Strip control and ANSI-escape characters from model free text.

    `claim`, `location` and `item` are attacker-influenceable through the
    document text and are printed to a terminal and written to the verdicts
    file. Closed enums constrain every other field; nothing constrains these.
    """
    return UNSAFE_CHARS.sub("", value)


def attach_provenance(claims: Iterable[Claim], doc: Candidate) -> list[SourcedClaim]:
    out: list[SourcedClaim] = []
    for claim in claims:
        sourced: SourcedClaim = dict(claim)  # type: ignore[assignment]
        sourced["claim"] = sanitize(claim["claim"])
        sourced["location"] = sanitize(claim["location"])
        sourced["item"] = sanitize(claim["item"])
        sourced["source_id"] = doc["id"]
        sourced["source_key"] = doc_key(doc)
        sourced["subreddit"] = doc["subreddit"]
        sourced["kind"] = doc["kind"]
        sourced["created_utc"] = doc["created_utc"]
        sourced["permalink"] = doc["permalink"]
        sourced["score"] = doc["score"]
        out.append(sourced)
    return out


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, 5xx, and connection failures are worth another attempt."""
    name = type(exc).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def backoff_delay(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff, capped. Jitter is added by the caller."""
    return min(base * (2.0**attempt), cap)


@dataclass
class Extractor:
    """Extracts claims for one document at a time against a Claude client."""

    client: AnthropicClient
    model: str = DEFAULT_MODEL
    thinking: bool = True
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = lambda: random.uniform(0, 1)

    def call(self, doc: Candidate) -> ModelResponse:
        """One document, with retry on transient failures."""
        params = build_params(doc, self.model, thinking_config(self.thinking))
        last: BaseException | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return parse_response(self.client.messages.create(**params))
            except Exception as exc:  # noqa: BLE001 - re-raised below if fatal
                last = exc
                if not is_retryable(exc):
                    raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
                if attempt == MAX_ATTEMPTS - 1:
                    break  # no retry remains; sleeping before giving up is waste
                self.sleep(backoff_delay(attempt) + self.jitter())
        raise ExtractionError(f"retries exhausted: {last}")

    def extract(self, doc: Candidate) -> tuple[list[SourcedClaim], ModelResponse]:
        """Claims for one document, plus the raw response for accounting."""
        response = self.call(doc)
        if response.stop_reason == "refusal":
            return [], response
        if response.text is None:
            raise ExtractionError(
                f"response carried no text block (stop_reason={response.stop_reason})"
            )
        try:
            claims = parse_claims(response.text)
        except json.JSONDecodeError as exc:
            # Structured outputs should make this impossible, but a response
            # truncated at max_tokens would land here. Fail the document, not
            # the run, and record stop_reason so it is diagnosable.
            raise ExtractionError(
                f"unparseable response (stop_reason={response.stop_reason}): {exc}"
            ) from exc
        return attach_provenance(claims, doc), response


def record(stats: RunStats, claims: Sequence[SourcedClaim], response: ModelResponse) -> None:
    """Fold one document's outcome into the running totals."""
    stats.docs += 1
    stats.usage = stats.usage + response.usage
    if response.stop_reason == "refusal":
        stats.refusals += 1
        return
    if not claims:
        stats.empty += 1
    stats.claims += len(claims)


def pricing_for(model: str) -> Pricing | None:
    """None when we have no rate card — the report says so rather than guessing."""
    return PRICING.get(model)


def estimate_total(stats: RunStats, model: str, total_docs: int) -> float | None:
    """Extrapolate spend for a full run from a sample; None without a rate card."""
    pricing = pricing_for(model)
    if pricing is None or stats.docs == 0:
        return None
    return float(pricing.cost(stats.usage) / stats.docs * total_docs)


def format_report(stats: RunStats, model: str, elapsed: float, total_docs: int | None = None) -> str:
    pricing = pricing_for(model)
    n = max(stats.docs, 1)
    rate = stats.docs / elapsed if elapsed > 0 else 0.0
    lines = [
        f"docs={stats.docs:,}  claims={stats.claims:,}  no-claim={stats.empty:,}  "
        f"refusals={stats.refusals}  failed={stats.failed}",
        f"tokens: in={stats.usage.input_tokens:,} "
        f"cache_read={stats.usage.cache_read_tokens:,} "
        f"cache_write={stats.usage.cache_write_tokens:,} "
        f"out={stats.usage.output_tokens:,}",
        f"cache hit rate: {100.0 * stats.cache_hit_rate():.1f}%",
    ]
    if stats.stopped:
        lines.insert(0, f"RUN STOPPED EARLY: {stats.stopped}")
    if pricing is None:
        lines.append(f"est. cost: unknown — no rate card for {model}")
    else:
        cost = pricing.cost(stats.usage)
        lines.append(
            f"est. cost ${cost:.2f}  (${cost / n * 1000:.2f} per 1,000 docs) "
            f"— indicative; Bedrock bills at AWS rates"
        )
    lines.append(f"elapsed {elapsed / 60:.1f}m  ({rate:.1f} docs/s)")
    if total_docs is not None and total_docs > stats.docs:
        projected = estimate_total(stats, model, total_docs)
        hours = (total_docs / rate / 3600) if rate > 0 else float("inf")
        money = "unknown cost" if projected is None else f"${projected:,.0f}"
        lines.append(
            f"\nextrapolated to the full {total_docs:,}-doc run: "
            f"{money}, ~{hours:.1f}h at this rate"
        )
    return "\n".join(lines)
