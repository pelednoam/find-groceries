"""Tests for Bedrock client construction."""

from __future__ import annotations

import sys
import types
from typing import Any

from groceries import client


def test_build_client_requests_no_sdk_retries(monkeypatch: Any) -> None:
    """Extractor owns retry; SDK retries would nest inside one attempt."""
    seen: dict[str, Any] = {}

    class FakeBedrock:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        @property
        def messages(self) -> Any:  # pragma: no cover - satisfies the protocol
            return None

    fake = types.ModuleType("anthropic")
    fake.AnthropicBedrock = FakeBedrock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    built = client.build_client("us-east-1")
    assert isinstance(built, FakeBedrock)
    assert seen == {"aws_region": "us-east-1", "max_retries": 0}
