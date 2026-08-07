"""Construction of the Bedrock-backed Claude client.

Lives in the package rather than the CLI so that the `cast` narrowing the SDK's
concrete type onto our Protocol sits next to the Protocol it satisfies — and so
it falls inside the coverage gate rather than in the one layer excluded from it.

`max_retries=0` is deliberate: `Extractor` owns retry. The SDK's own retry loop
would otherwise nest inside what `Extractor` counts as a single attempt, making
the real per-document attempt ceiling the product of the two and hiding the
SDK's sleeps inside the package's backoff budget.
"""

from __future__ import annotations

from typing import cast

from .types import AnthropicClient


def build_client(region: str) -> AnthropicClient:
    from anthropic import AnthropicBedrock

    return cast(AnthropicClient, AnthropicBedrock(aws_region=region, max_retries=0))
