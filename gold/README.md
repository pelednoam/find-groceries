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

## Adding a case

Append a line with `id`, `why` (the failure mode — required, it is what makes
the case reviewable), `text`, `parent_body`, `stores` (what stage 1's
pre-filter matched) and `expected`. A line whose `id` starts with `#` is
ignored, so cases can be parked without deleting them.

Label conservatively. A case with two defensible answers measures the labeller,
not the model — trim the text until only one answer is defensible, or drop it.
