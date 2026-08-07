# Extraction gold set

24 hand-labelled documents used to measure stage 2. Run it with:

    .venv/bin/python scripts/evaluate_extraction.py

Every case exists because it is a **documented failure mode**, not because it
was convenient. Nine of the twenty-four are labelled with no claims at all,
which is deliberate: most of the corpus supports no claim, and the risk that
matters is a model that finds one anyway. `silence_accuracy` in the report is
that number on its own.

Text is drawn from `r/boston`, `r/CambridgeMA`, `r/Somerville` and
`r/traderjoes` and trimmed to isolate one judgment, or written to exercise a
rule the prompt states explicitly (sarcasm, comparator reciprocity, transient
sales, parking-vs-geography, unlisted stores, inherited referents, injection).

## What is scored

Claims match on the `(store, category, sentiment)` triple. The free-text
`claim` field is **not** scored — many phrasings are correct, and grading
prose against prose needs a judge, which is a second thing to be wrong about.
`transient` and `comparator_store` are scored separately over the claims that
did match, because stage 3 uses both: a missed `transient` files a
closing-down sale as a durable property of the store, and a missed
`comparator_store` counts one comparison as two independent opinions.

## Baseline

Sonnet 4.6, 2026-08-07, single sample:

    exact match 96%  precision 94%  recall 94%  correct silence 100%  flags 100%

One case disagrees: `star-closing-sale` files a closing-down 50%-off sale under
`price_overall` where the label says `deals_loyalty`. Both are defensible, and
the part that matters is right either way — the model sets `transient`, so the
liquidation does not enter the aggregate as "this store is cheap".

The first run of this set scored 83%. Three of the four disagreements were
**defects in the labels, not the model**: an over-reaching implicature label, a
case with two defensible answers on a scored field, and a claim the model found
that the label had simply omitted. The fourth was a real prompt gap — the model
was emitting `other` for stores the text never names — which is now a stated
rule with a worked example.

That ratio is the thing to watch. A gold set is only useful while it is harder
on itself than on the model; if a disagreement is resolved by editing the label
more often than by fixing the prompt, the set has started measuring the
labeller. Fix the prompt when the model is wrong, fix the label when it is
right, and stop adjusting once you are choosing between two defensible answers.

`--min-exact-match` defaults to 0 (off). Set it to catch regressions, not to
demand perfection: 24 cases sampled once from a non-deterministic model means
one case flipping moves the score four points. 0.80 is a reasonable gate; 0.96
would be flaky.

## Adding a case

Append a line with `id`, `why` (the failure mode — required, it is what makes
the case reviewable), `text`, `parent_body`, `stores` (what stage 1's
pre-filter matched) and `expected`. A line whose `id` starts with `#` is
ignored, so cases can be parked without deleting them.

Label conservatively. A case with two defensible answers measures the labeller,
not the model — trim the text until only one answer is defensible, or drop it.
