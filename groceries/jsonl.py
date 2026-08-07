"""One JSONL reader, used by every stage.

Stage 1 writes candidates, stage 2 writes claims, stage 3 reads them back —
three call sites that had the same three-line body with different return
annotations.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> tuple[list[Any], int]:
    """Parse a JSONL file. Returns (rows, n_unparseable).

    A SIGKILL mid-flush leaves a partial final line, and the resumed run then
    appends after it. Raising there would hold every good row in a paid-for
    artifact hostage to one truncated line, so bad lines are skipped and
    counted for the caller to report.
    """
    rows: list[Any] = []
    unparseable = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                unparseable += 1
    return rows, unparseable


def write_atomic(path: Path, chunks: Iterable[str]) -> int:
    """Write via a temp file and rename, so a crash cannot destroy the old one.

    Both stage 1 and stage 3 previously truncated their output on open, which
    meant a failure partway through replaced a good artifact with a partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process: a fixed name lets two concurrent writers
    # interleave into the same temp file and then publish the result.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk)
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return n
