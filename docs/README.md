# The published site

Served by GitHub Pages at
**[pelednoam.github.io/find-groceries](https://pelednoam.github.io/find-groceries)**.

TypeScript, compiled to a single script. No framework, no bundler, no CDN.

| path | what it is |
|---|---|
| `src/app.ts` | the application — **edit this** |
| `src/types.ts` | the shape of `verdicts.json` |
| `app.js` | tsc output, committed because Pages serves static files |
| `index.html`, `styles.css` | structure and ~300 lines of CSS, light and dark |
| `verdicts.json` | 1.0MB payload built from stage 3 |
| `vendor/leaflet/` | Leaflet 1.9.4, vendored rather than pulled from a CDN |

## Type checking

`tsconfig.json` is the front end's `mypy --strict`: `strict`, plus
`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `noUnusedLocals`,
`noImplicitReturns` and friends. `noUncheckedIndexedAccess` is the one that
earns its keep — it makes `DATA.stores[name]` a `| undefined` you have to
handle, which is exactly the shape of the payload's optional fields.

`docs/src/types.ts` and `groceries/site.py` are two hand-written descriptions
of one wire format, and nothing but a test connects them:
`tests/test_site.py::TestPayloadContract` asserts the emitted payload has
exactly the keys the interface declares. Without it, a field renamed in Python
would surface as a blank panel in the browser and as an error nowhere — tsc
cannot see the Python, mypy cannot see the TypeScript.

## Rebuilding after a new extraction

```
.venv/bin/python scripts/aggregate_claims.py        # stage 3 -> store_verdicts.json
.venv/bin/python scripts/fetch_store_locations.py   # OSM -> data/locations.json (rarely)
.venv/bin/python scripts/build_site.py              # -> docs/verdicts.json
./scripts/check.sh                                  # mypy, pytest, tsc, smoke
git add docs && git commit && git push
```

Pages serves `main:/docs`, so a push is a deploy. `scripts/check.sh` fails if
`docs/app.js` is out of date with `docs/src/`, so the deployed script can never
silently lag the source it was compiled from.

## Why the payload is not in `docs/data/`

The repo gitignores `data/`, which matches a directory of that name at **any**
depth. `docs/data/verdicts.json` would have been silently omitted from every
commit and the live site would have 404'd on its only data file, with nothing
in `git status` to explain it. The payload lives at `docs/verdicts.json`.

## The map

Pins are real store locations from OpenStreetMap (© OpenStreetMap
contributors, ODbL), fetched at build time by
`scripts/fetch_store_locations.py` and matched to a chain using **stage 1's
own regexes** — not a second copy of the store list. An OSM entry that matches
nothing known is dropped rather than guessed at.

A pin is linked to a branch only when every word of the branch name appears in
the pin's city or street. 114 of 142 locations link that way; the rest fall
back to the chain verdict. Branch names that are not places — "the Acre",
"inside 128" — never link, because attaching them to the nearest pin would
invent a fact.

Leaflet is vendored under `vendor/` rather than loaded from a CDN, so the map
tiles are the only third-party request the page makes.

## The one rule in `app.ts`

Every string in the payload except the object keys is text a stranger typed on
Reddit. It reaches the DOM through `textContent`, never `innerHTML`, and the
only attribute built from it is a permalink, pattern-checked against
`/r/<sub>/` before it becomes an `href`. Leaflet popups take an HTML string, so
the map builds a detached node and hands Leaflet that instead.

`tests/site_smoke.js` re-runs the entire app against a payload seeded with
`<img onerror>`, `<script>`, and a `javascript:` permalink, and asserts none of
them become elements or execute. If you change how claims are rendered, that
test is the thing to keep passing.

## What the shopping list actually does

For each term it tries, in order:

1. **an item the corpus discusses** — "rotisserie chicken" has its own evidence
2. **the department the word belongs to** — "milk" -> `dairy`, via a keyword map
   in `groceries/site.py`
3. **nothing** — reported as "no evidence either way" rather than guessed at

Stores are then ranked by the *average* value across the terms they have
evidence for, not the sum, so a store with two matches cannot beat one with
eight just by having fewer chances to look bad. The slider weights sentiment
against price level; at either extreme it is one or the other.
