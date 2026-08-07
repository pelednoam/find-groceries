#!/usr/bin/env bash
# Every gate this project has, in the order that fails fastest.
#
#   scripts/check.sh
#
# Backend:  mypy --strict, then pytest with a 99% branch-coverage floor.
# Frontend: tsc --strict (mypy's opposite number), then a headless run of the
#           real page against the real payload, including a poisoned one.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== mypy (strict) =="
.venv/bin/mypy

echo
echo "== pytest =="
.venv/bin/python -m pytest tests -q

echo
echo "== tsc (strict) =="
npx tsc --noEmit -p tsconfig.json

echo
echo "== build docs/app.js =="
npx tsc -p tsconfig.json
if ! git diff --quiet -- docs/app.js; then
  echo "docs/app.js was stale — rebuilt. Commit the result."
  exit 1
fi

echo
echo "== site smoke =="
node tests/site_smoke.js

echo
echo "all gates pass"
