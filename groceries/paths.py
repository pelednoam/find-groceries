"""Canonical on-disk locations, so the three CLIs cannot disagree.

Stage 1's default output was `working_set.jsonl` while stage 2's default input
was `working_set_local.jsonl`. Regenerating the working set therefore appeared
to succeed and changed nothing, because stage 2 kept reading a stale file that
stage 1 had stopped writing months earlier. One constant each, defined once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA: Final[Path] = ROOT / "data"
CORPUS: Final[Path] = DATA / "reddit"
EXTRACT_DIR: Final[Path] = DATA / "extraction"

WORKING_SET: Final[Path] = EXTRACT_DIR / "working_set.jsonl"
CLAIMS: Final[Path] = EXTRACT_DIR / "claims.jsonl"
DONE: Final[Path] = EXTRACT_DIR / "done.txt"
FAILED: Final[Path] = EXTRACT_DIR / "failed.jsonl"
VERDICTS: Final[Path] = EXTRACT_DIR / "store_verdicts.json"
LOCK: Final[Path] = EXTRACT_DIR / ".extract.lock"
