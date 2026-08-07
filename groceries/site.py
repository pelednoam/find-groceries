"""Build the payload the published site reads.

The verdict file is 3.9MB, most of it evidence quotes the reader will never
open. This trims it to what the UI actually renders and adds the one thing
the verdicts do not carry: a mapping from everyday shopping words to the
categories and items the pipeline indexed, so a shopping list can be answered.

Everything here is public. The claims are Reddit text — untrusted, and the
site renders them as text nodes, never as HTML — and no author names are
carried through, because the UI has no use for them and not publishing them
is free.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .jsonl import write_atomic
from .locations import attach_branches
from .merge import Calibration, combine, fit_calibration

# Evidence quotes per cell in the published payload. Three is what the UI
# shows before "more"; the rest is weight the reader never sees.
SITE_EXAMPLES: Final = 3
# Items below this much evidence are one passing remark and add noise to
# search without adding an answer.
MIN_ITEM_WEIGHT: Final = 1.0

# Everyday shopping words -> the category that answers them. Only needed for
# words a shopper would write on a list; the UI falls back to the item index
# first and to overall quality last.
LIST_CATEGORIES: Final[Mapping[str, tuple[str, ...]]] = {
    "produce": (
        "produce", "fruit", "fruits", "veg", "vegetable", "vegetables", "veggies",
        "salad", "lettuce", "greens", "tomato", "tomatoes", "onion", "onions",
        "potato", "potatoes", "apple", "apples", "banana", "bananas", "berries",
        "avocado", "avocados", "herbs", "cilantro", "spinach", "broccoli",
        "carrot", "carrots", "garlic", "lemon", "lime", "limes", "mushroom",
        "mushrooms", "pepper", "peppers", "cucumber",
    ),
    "meat": (
        "meat", "beef", "steak", "steaks", "chicken", "pork", "lamb", "bacon",
        "sausage", "ground beef", "mince", "ribs", "turkey", "brisket", "butcher",
    ),
    "seafood": (
        "seafood", "fish", "salmon", "tuna", "shrimp", "prawns", "lobster",
        "scallops", "cod", "haddock", "oysters", "clams", "mussels", "crab",
    ),
    "dairy": (
        "dairy", "milk", "cheese", "butter", "yogurt", "yoghurt", "cream",
        "eggs", "egg", "sour cream", "cottage cheese", "half and half",
    ),
    "bakery": (
        "bakery", "bread", "baguette", "bagel", "bagels", "rolls", "cake",
        "pastry", "pastries", "croissant", "donuts", "doughnuts", "muffins",
        "sourdough", "tortillas", "pita",
    ),
    "prepared_food": (
        "prepared", "deli", "hot bar", "salad bar", "sandwich", "sandwiches",
        "rotisserie", "pizza", "sushi", "ready meal", "ready meals", "takeout",
        "lunch",
    ),
    "pantry": (
        "pantry", "pasta", "rice", "beans", "canned", "cans", "flour", "sugar",
        "oil", "olive oil", "spices", "sauce", "cereal", "coffee", "tea",
        "snacks", "chips", "peanut butter", "condiments", "vinegar", "noodles",
        "soy sauce", "stock", "broth",
    ),
    "frozen": (
        "frozen", "ice cream", "frozen pizza", "frozen veg", "freezer",
        "frozen vegetables", "popsicles",
    ),
    "alcohol": (
        "alcohol", "beer", "wine", "liquor", "spirits", "booze", "cider",
        "whiskey", "vodka", "seltzer",
    ),
}

# Inverted once, at build time, rather than in the browser on every keystroke.
def keyword_index() -> dict[str, str]:
    return {
        word: category
        for category, words in LIST_CATEGORIES.items()
        for word in words
    }


def slim_cell(cell: Mapping[str, Any], examples: int = SITE_EXAMPLES) -> dict[str, Any]:
    """Keep what the UI renders; drop what it does not."""
    out: dict[str, Any] = {
        "n": cell["n_claims"],
        "w": round(float(cell["weighted_evidence"]), 1),
        "s": round(float(cell["sentiment"]), 3),
    }
    if cell.get("price_signal"):
        out["p"] = cell["price_signal"]
        out["pl"] = cell["price_level"]
    if cell.get("price_distribution"):
        out["pd"] = cell["price_distribution"]
    out["e"] = [
        {
            "t": e["claim"],
            "d": e["date"],
            "u": e["permalink"],
            "c": e["confidence"],
            **({"l": e["location"]} if e.get("location") else {}),
        }
        for e in list(cell["evidence"])[:examples]
    ]
    return out


def slim_group(
    group: Mapping[str, Mapping[str, Any]], examples: int = SITE_EXAMPLES
) -> dict[str, dict[str, Any]]:
    return {k: slim_cell(v, examples) for k, v in group.items()}


REGION_HINTS: Final = re.compile(
    r"^(nh|new hampshire|southern nh|maine|suburbs|inside \d+|north shore|"
    r"south shore|cape cod|the \w+)$"
)


def is_branch(name: str) -> bool:
    """Whether a branch key names a store rather than a region.

    The model emits "NH" and "the suburbs" alongside "Somerville". Both are
    real answers to "where", but only one is a branch, and listing them
    together implies a precision the region entries do not have.
    """
    return not REGION_HINTS.match(name.strip().casefold())


def _merge_block(
    verdicts: Mapping[str, Any],
    crosscheck: Mapping[str, Any] | None,
    places: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Combine the two sources, chain-level and branch-level.

    Google has no category breakdown — one number per location — so the merge
    can only sharpen the *overall* verdict for a store or a branch. Nothing
    here touches the per-category cells, and it must not: there is no Google
    opinion about a store's produce to combine with.
    """
    if not crosscheck:
        return None
    totals = verdicts["store_totals"]
    ratings: Mapping[str, Any] = crosscheck["stores"]

    # Fit on the decayed values, because those are what `combine` will feed
    # back in. Fitting on all-time means and applying to recent ones would
    # put a systematic tilt through every merged number.
    pairs = [
        (_google_norm(r), float(totals[store]["sentiment"]))
        for store, r in ratings.items()
        if store in totals and _usable(r) and not r.get("thin", False)
    ]
    cal = fit_calibration(pairs)
    if cal is None:
        return None

    stores: dict[str, Any] = {}
    for store, t in totals.items():
        rating = ratings.get(store)
        m = combine(
            float(t["sentiment"]), float(t["weighted_evidence"]),
            rating if rating is not None and _usable(rating) else None, cal,
        )
        if m is not None:
            stores[store] = m.as_dict()

    branches = _merge_branches(verdicts, crosscheck, places, cal)
    return {
        "calibration": cal.as_dict(),
        "stores": stores,
        "branches": branches,
        "note": (
            "Google contributes one number per location, so the merge applies "
            "to the overall verdict only, never to a category."
        ),
    }


def _merge_branches(
    verdicts: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    places: Sequence[Mapping[str, Any]],
    cal: Calibration,
) -> dict[str, dict[str, Any]]:
    """Merge per (store, branch), pooling every pin linked to that branch.

    This is where the combination earns its keep: only 26% of Reddit claims
    name a branch, while Google has ratings for almost every location.
    """
    by_osm: Mapping[str, Any] = crosscheck["locations"]
    # A branch can have more than one pin; pool their ratings by count.
    pooled: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for place in places:
        branch = place.get("branch")
        rating = by_osm.get(place["osm"])
        if branch and rating and _usable(rating):
            pooled.setdefault((place["store"], branch), []).append(rating)

    out: dict[str, dict[str, Any]] = {}
    branch_totals: Mapping[str, Mapping[str, Any]] = verdicts.get("branch_totals", {})
    seen: set[tuple[str, str]] = set()
    for store, by_branch in branch_totals.items():
        for branch, t in by_branch.items():
            seen.add((store, branch))
            m = combine(
                float(t["sentiment"]), float(t["weighted_evidence"]),
                _pool(pooled.get((store, branch), [])), cal,
            )
            if m is not None:
                out.setdefault(store, {})[branch] = m.as_dict()
    # Branches with a rating but no Reddit evidence at all are still worth
    # showing: "nobody on Reddit discussed this one, Google says 4.3".
    for (store, branch), ratings in pooled.items():
        if (store, branch) in seen:
            continue
        m = combine(None, 0.0, _pool(ratings), cal)
        if m is not None:
            out.setdefault(store, {})[branch] = m.as_dict()
    return out


def _google_norm(rating: Mapping[str, Any]) -> float:
    """The recency-decayed value, falling back to the all-time one."""
    return float(rating.get("norm_recent", rating["norm"]))


def _usable(rating: Mapping[str, Any]) -> bool:
    """Whether a rating carries what the merge needs.

    This is a boundary onto a separately-generated file, so a block written
    by an older version of the cross-check must degrade to "no Google here"
    rather than to a KeyError halfway through building the payload.
    """
    return "norm" in rating and "n" in rating


def _pool(ratings: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Count-weighted mean of several locations' ratings."""
    usable = [r for r in ratings if _usable(r) and int(r["n"]) > 0]
    if not usable:
        return None
    total = sum(int(r["n"]) for r in usable)
    # Pool on effective weight, so an old location does not out-vote a
    # recent one purely on raw count.
    eff = sum(float(r.get("n_eff", r["n"])) for r in usable) or 1.0
    norm = sum(_google_norm(r) * float(r.get("n_eff", r["n"])) for r in usable) / eff
    return {"n": total, "n_eff": eff, "norm": norm, "norm_recent": norm}


def build_payload(
    verdicts: Mapping[str, Any],
    locations: Mapping[str, Any] | None = None,
    crosscheck: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reshape the verdict document into the site payload."""
    stores = {s: slim_group(cats) for s, cats in verdicts["stores"].items()}

    branches: dict[str, dict[str, Any]] = {}
    regions: dict[str, dict[str, Any]] = {}
    for store, by_location in verdicts["branches"].items():
        for location, cats in by_location.items():
            target = branches if is_branch(location) else regions
            target.setdefault(store, {})[location] = slim_group(cats, 2)

    items: dict[str, dict[str, Any]] = {}
    for store, by_item in verdicts["items"].items():
        for item, cell in by_item.items():
            if float(cell["weighted_evidence"]) < MIN_ITEM_WEIGHT:
                continue
            items.setdefault(store, {})[item] = slim_cell(cell, 2)

    # Locations are optional: the site is useful without a map, and a failed
    # Overpass call should degrade to no map rather than to no site.
    places: list[dict[str, Any]] = []
    attribution = ""
    if locations:
        raw = list(locations.get("places", []))
        attribution = str(locations.get("attribution", ""))
        linked = attach_branches(
            raw, {store: list(b) for store, b in branches.items()}
        )
        for place in raw:
            # A pin for a store nobody discusses is a pin with nothing behind
            # it; the map exists to show where the evidence applies.
            if place["store"] not in stores:
                continue
            entry = dict(place)
            branch = linked.get(place["osm"])
            if branch:
                entry["branch"] = branch
            places.append(entry)

    merged = _merge_block(verdicts, crosscheck, places)

    return {
        "generated_at": verdicts["generated_at"],
        "method": verdicts["method"],
        "corpus": verdicts["corpus"],
        "places": places,
        "places_attribution": attribution,
        # Carried whole and never merged into `stores`. See groceries/crosscheck.py.
        "crosscheck": dict(crosscheck) if crosscheck else None,
        "merged": merged,
        "totals": {
            s: {
                "n": t["n_claims"],
                "w": round(float(t["weighted_evidence"]), 1),
                "s": round(float(t["sentiment"]), 3),
                "thin": t["insufficient_evidence"],
            }
            for s, t in verdicts["store_totals"].items()
        },
        "stores": stores,
        "branches": branches,
        "regions": regions,
        "items": items,
        "keywords": keyword_index(),
        "categories": sorted({c for cats in stores.values() for c in cats}),
    }


def write_payload(payload: Mapping[str, Any], path: Path) -> int:
    """Write the payload as compact JSON. Returns bytes written."""
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    write_atomic(path, [blob])
    return len(blob.encode("utf-8"))
