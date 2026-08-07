"""Shared types for the grocery-claims pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict

Sentiment = Literal["positive", "negative", "mixed", "neutral"]
PriceSignal = Literal["cheap", "expensive", "fair", "none"]
Confidence = Literal["high", "medium", "low"]


class RawDoc(TypedDict, total=False):
    """A record as it appears in the fetched Reddit dumps."""

    id: str
    created_utc: int
    author: str
    parent_id: str
    score: int
    body: str
    title: str
    selftext: str
    permalink: str
    link_id: str


class Candidate(TypedDict):
    """A document that survived stage-1 selection."""

    id: str
    subreddit: str
    kind: str
    created_utc: int
    score: int | None
    author: str
    parent_body: str
    permalink: str
    stores: list[str]
    text: str
    truncated: bool


class Claim(TypedDict):
    """One extracted claim, exactly as the model returns it."""

    store: str
    location: str
    category: str
    item: str
    claim: str
    sentiment: Sentiment
    price_signal: PriceSignal
    confidence: Confidence
    # "cheaper than Shaw's" is not evidence about Shaw's in isolation; without
    # the comparator the pair reads as two independent observations.
    comparator_store: str
    # A closing-down sale or a pandemic stock-out was true once and is not a
    # durable property of the store.
    transient: bool


class SourcedClaim(Claim):
    """A claim with the provenance the pipeline attaches after extraction.

    Provenance is mandatory, not optional: `attach_provenance` always writes
    every field. Declaring it optional pushed `.get(..., default)` calls into
    the scoring math, where a missing `created_utc` silently became *today* —
    promoting corrupt data to the freshest, highest-weight evidence in its
    cell. Malformed rows are now rejected at the read boundary instead.
    """

    source_id: str
    source_key: str
    subreddit: str
    kind: str
    created_utc: int
    permalink: str
    score: int | None
    # Carried purely so stage 3 can cap how many claims one regular poster
    # contributes to a single cell. Deleted accounts arrive as "" or
    # "[deleted]"; both are treated as anonymous rather than as one prolific
    # author, so they are never capped together.
    author: str


@dataclass(frozen=True)
class Usage:
    """Token usage for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ModelResponse:
    """The parts of a model response this pipeline cares about."""

    stop_reason: str | None
    text: str | None
    usage: Usage


@dataclass(frozen=True)
class Pricing:
    """Per-MTok rates. Indicative only; Bedrock bills at AWS rates."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens / 1e6 * self.input
            + usage.output_tokens / 1e6 * self.output
            + usage.cache_read_tokens / 1e6 * self.cache_read
            + usage.cache_write_tokens / 1e6 * self.cache_write
        )


@dataclass
class RunStats:
    """Mutable counters accumulated across an extraction run."""

    docs: int = 0
    claims: int = 0
    empty: int = 0
    refusals: int = 0
    failed: int = 0
    consecutive_failures: int = 0
    stopped: str | None = None
    usage: Usage = field(default_factory=Usage)

    def cache_hit_rate(self) -> float:
        billed = self.usage.cache_read_tokens + self.usage.input_tokens
        if billed == 0:
            return 0.0
        return self.usage.cache_read_tokens / billed


class MessagesAPI(Protocol):
    """The one SDK method this pipeline calls.

    Kept deliberately loose: the Anthropic SDK's response objects are a large
    union, and `parse_response` is the single place where that untyped surface
    is narrowed into `ModelResponse`.
    """

    def create(self, **kwargs: object) -> object: ...


class AnthropicClient(Protocol):
    @property
    def messages(self) -> MessagesAPI: ...
