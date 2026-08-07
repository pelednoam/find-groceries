# The published site

Served by GitHub Pages at
**[pelednoam.github.io/find-groceries](https://pelednoam.github.io/find-groceries)**.

Four files, no build step, no dependencies, no CDN:

| file | what it is |
|---|---|
| `index.html` | structure |
| `styles.css` | ~250 lines, light and dark |
| `app.js` | vanilla ES2020, no framework |
| `verdicts.json` | 1.0MB payload built from stage 3 |

## Rebuilding after a new extraction

```
.venv/bin/python scripts/aggregate_claims.py   # stage 3 -> data/extraction/store_verdicts.json
.venv/bin/python scripts/build_site.py         # -> docs/verdicts.json
node tests/site_smoke.js                       # needs `npm install jsdom`
git add docs && git commit && git push
```

Pages serves `main:/docs`, so a push is a deploy.

## Why the payload is not in `docs/data/`

The repo gitignores `data/`, which matches a directory of that name at **any**
depth. `docs/data/verdicts.json` would have been silently omitted from every
commit and the live site would have 404'd on its only data file, with nothing
in `git status` to explain it. The payload lives at `docs/verdicts.json`.

## The one rule in `app.js`

Every string in the payload except the object keys is text a stranger typed on
Reddit. It reaches the DOM through `textContent`, never `innerHTML`, and the
only attribute built from it is a permalink, which is pattern-checked against
`/r/<sub>/` before it becomes an `href`.

`tests/site_smoke.js` re-runs the entire app against a payload seeded with
`<img onerror>`, `<script>`, and a `javascript:` permalink, and asserts none of
them become elements or execute. If you change how claims are rendered, that
test is the thing to keep passing.

## What the shopping list actually does

For each term it tries, in order:

1. **an item the corpus discusses** — "rotisserie chicken" has its own evidence
2. **the department the word belongs to** — "milk" → `dairy`, via a keyword map
   in `groceries/site.py`
3. **nothing** — reported as "no evidence either way" rather than guessed at

Stores are then ranked by the *average* value across the terms they have
evidence for, not the sum, so a store with two matches cannot beat one with
eight just by having fewer chances to look bad. The slider weights sentiment
against price level; at either extreme it is one or the other.
