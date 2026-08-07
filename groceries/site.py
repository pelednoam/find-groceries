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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .jsonl import write_atomic

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


def build_payload(verdicts: Mapping[str, Any]) -> dict[str, Any]:
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

    return {
        "generated_at": verdicts["generated_at"],
        "method": verdicts["method"],
        "corpus": verdicts["corpus"],
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
