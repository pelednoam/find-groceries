"""Shared fixtures and fakes for the pipeline tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from groceries.types import Candidate, Claim, SourcedClaim


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 900
    cache_creation_input_tokens: int = 0


@dataclass
class FakeBlock:
    type: str
    text: str = ""


@dataclass
class FakeMessage:
    content: list[FakeBlock]
    stop_reason: str | None = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    """Records calls and replays a scripted sequence of results."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0) if self.results else FakeMessage([])
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.messages = FakeMessages(results or [])


class RateLimitError(Exception):
    """Mirrors the SDK class name that `is_retryable` keys on."""


class ServerError(Exception):
    def __init__(self, status_code: int = 503) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class BadRequestError(Exception):
    def __init__(self) -> None:
        super().__init__("bad request")
        self.status_code = 400


def message_with_claims(claims: Sequence[Mapping[str, Any]], **kw: Any) -> FakeMessage:
    payload = json.dumps({"claims": claims})
    return FakeMessage(content=[FakeBlock("text", payload)], **kw)


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        id="abc123",
        subreddit="boston",
        kind="comments",
        created_utc=1_700_000_000,
        score=7,
        author="alice",
        parent_body="",
        permalink="/r/boston/comments/xyz/_/abc123/",
        stores=["Market Basket"],
        text="Market Basket produce is way cheaper than Star Market.",
        truncated=False,
    )


@pytest.fixture
def raw_claim() -> Claim:
    return Claim(
        store="Market Basket",
        location="Somerville",
        category="produce",
        item="",
        claim="Produce is cheaper than at Star Market.",
        sentiment="positive",
        price_signal="cheap",
        confidence="high",
        comparator_store="",
        transient=False,
    )


@pytest.fixture
def sourced_claim(raw_claim: Claim) -> SourcedClaim:
    claim: SourcedClaim = dict(raw_claim)  # type: ignore[assignment]
    claim["source_id"] = "abc123"
    claim["source_key"] = "c_boston_abc123"
    claim["subreddit"] = "boston"
    claim["kind"] = "comments"
    claim["created_utc"] = 1_700_000_000
    claim["permalink"] = "/r/boston/comments/xyz/_/abc123/"
    claim["score"] = 7
    claim["author"] = "alice"
    return claim
