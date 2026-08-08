"""Vocabulary for reading a grocery receipt.

A till slip is not English. It is 24 characters of upper-case abbreviation
with the vowels removed, a brand prefix nobody outside the chain recognises,
and a size suffix — "GV WHL MLK GAL", "ORG BBY SPNCH", "GRT VAL SHRD CHDR".
None of that matches the shopping vocabulary the rest of this project uses,
so it has to be expanded before anything can be looked up.

This module holds the *vocabulary and rules*; the parsing runs in the browser
so that a photograph of a receipt never leaves the reader's device. Keeping
the table here means it ships in the payload, and means it can be tested
against real receipt lines rather than eyeballed.

Nothing here is clever. It is a list of what shops actually print, assembled
from the abbreviations that recur across US grocery receipts, and it is meant
to be extended when a line fails to match.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

# Token -> what it means. Applied per word after the line is split, so
# "WHL MLK" becomes "whole milk" and can then meet the keyword map.
ABBREVIATIONS: Final[Mapping[str, str]] = {
    # dairy
    "mlk": "milk", "whl": "whole", "skm": "skim", "hlf": "half",
    "chz": "cheese", "chdr": "cheddar", "moz": "mozzarella", "yog": "yogurt",
    "ygt": "yogurt", "crm": "cream", "buttr": "butter", "btr": "butter",
    "egg": "eggs", "eggs": "eggs", "sr": "sour",
    # produce
    "ban": "bananas", "bnna": "bananas", "bnns": "bananas",
    "apl": "apple", "appl": "apple", "tom": "tomato", "tmto": "tomato",
    "let": "lettuce", "lttce": "lettuce", "spnch": "spinach", "spnc": "spinach",
    "brocc": "broccoli", "brccl": "broccoli", "carr": "carrots",
    "ptato": "potato", "pot": "potato", "onin": "onion", "onn": "onion",
    "avcdo": "avocado", "avo": "avocado", "cuke": "cucumber", "cucmbr": "cucumber",
    "grp": "grapes", "strwb": "strawberries", "blueb": "blueberries",
    "lmn": "lemon", "lme": "lime", "pepp": "pepper", "mush": "mushrooms",
    "grlc": "garlic", "clntro": "cilantro", "prod": "produce",
    # meat and fish
    "chkn": "chicken", "chk": "chicken", "chick": "chicken",
    "brst": "breast", "thgh": "thigh", "grnd": "ground", "gr": "ground",
    "bf": "beef", "bef": "beef", "prk": "pork", "bcn": "bacon",
    "saus": "sausage", "ssg": "sausage", "trky": "turkey", "tky": "turkey",
    "slmn": "salmon", "sal": "salmon", "shrmp": "shrimp", "shrp": "shrimp",
    "tlpa": "tilapia", "ck": "cooked",
    # bakery and pantry
    "brd": "bread", "bred": "bread", "bgl": "bagel", "bgls": "bagels",
    "tort": "tortillas", "crckr": "crackers", "cerl": "cereal", "crl": "cereal",
    "pnut": "peanut", "pb": "peanut butter", "jly": "jelly",
    "past": "pasta", "spag": "spaghetti", "rce": "rice", "flr": "flour",
    "sgr": "sugar", "oliv": "olive", "ol": "oil", "vin": "vinegar",
    "sce": "sauce", "sup": "soup", "bns": "beans", "cof": "coffee",
    "cofe": "coffee", "coff": "coffee", "tea": "tea",
    # frozen, drinks, household
    "frz": "frozen", "frzn": "frozen", "icecrm": "ice cream", "ic": "ice cream",
    "pza": "pizza", "juc": "juice", "jc": "juice", "sltzr": "seltzer",
    "wtr": "water", "sda": "soda", "beer": "beer", "wne": "wine",
    "tp": "toilet paper", "ppr": "paper", "twl": "towel", "detrg": "detergent",
    # qualifiers that carry meaning worth keeping
    "org": "organic", "orgnc": "organic", "gf": "gluten free",
    "lg": "large", "sm": "small", "med": "medium",
    "shrd": "shredded", "slcd": "sliced", "bnls": "boneless", "sknls": "skinless",
    "bby": "baby", "fresh": "fresh", "frsh": "fresh", "nat": "natural",
    "lowfat": "low fat", "lf": "low fat", "unswt": "unsweetened",
}

# Units and pack words. They survive the size regex when a receipt prints
# them without a number ("MLK GAL", "@ 0.59/lb") and they say nothing about
# what was bought.
UNITS: Final[frozenset[str]] = frozenset({
    "oz", "lb", "lbs", "kg", "ml", "lt", "ltr", "gal", "qt", "pt", "ct",
    "pk", "pack", "each", "ea", "count", "bag", "box", "btl", "bottle", "can",
})

# Brand and own-label prefixes. They say who made it, never what it is, and
# they crowd out the words that do.
BRAND_PREFIXES: Final[frozenset[str]] = frozenset({
    "gv", "grt", "val", "greatvalue", "mm", "marketside", "eq", "equate",
    "sb", "signature", "select", "kroger", "kr", "safeway", "wegmans",
    "365", "wf", "tj", "traderjoes", "annies", "kirkland", "ks",
    "storebrand", "essential", "everyday", "smartway", "hannaford",
    "stopshop", "shaws", "star", "mb", "demoulas",
})

# Lines that are not items: totals, payment, loyalty, the shop's own address.
NOISE: Final[tuple[str, ...]] = (
    r"^\s*(sub)?total\b", r"\btax\b", r"\bchange\b", r"\bcash\b", r"\bdebit\b",
    r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bamex\b", r"\bapprov",
    r"\bauth\b", r"\bref(und|erence)?\s*#", r"\baid\b", r"\bterminal\b",
    r"\bthank you\b", r"\bcustomer copy\b", r"\bmerchant\b", r"\bsurvey\b",
    r"\bsave[ds]?\b.*\b(today|total)\b", r"\bcoupon\b", r"\bloyalty\b",
    r"\bmember\b", r"\bpoints?\b", r"\bbalance\b", r"\bitems? sold\b",
    r"\breceipt\b", r"\bstore\s*#", r"\breg\b", r"\bcashier\b", r"\btran\b",
    # The register header every US chain prints: "ST# 02175 OP# 000123 TE# 09".
    r"\b(st|op|te|tr|trn|inv)\s*#",
    r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", r"^\s*[\d\s()+-]{7,}$",
    r"^\s*[*=-]{3,}\s*$",
)

# A trailing money amount, with or without the sign a discount line carries.
PRICE: Final = re.compile(r"(-?\$?\s*\d{1,4}[.,]\d{2})\s*[a-z]?\s*$", re.I)
# Leading quantity: "2 @ 1.99", "3 X", "2 EA".
QUANTITY: Final = re.compile(r"^\s*(\d{1,2})\s*(?:@|x|ea\b)", re.I)
# Weighed goods print their own line: "1.24 lb @ 2.99/lb".
WEIGHED: Final = re.compile(r"\b\d+(?:\.\d+)?\s*(lb|kg|oz|g)\b\s*@", re.I)
# Barcodes, PLU codes and department numbers.
CODES: Final = re.compile(r"\b\d{4,}\b")
# Size suffixes: "12OZ", "1GAL", "2LT", "6PK".
SIZES: Final = re.compile(r"\b\d+(\.\d+)?\s*(oz|lb|g|kg|ml|l|lt|ct|pk|gal|qt|pt)\b", re.I)


def vocabulary() -> dict[str, object]:
    """Everything the browser-side parser needs, as plain data."""
    return {
        "abbreviations": dict(ABBREVIATIONS),
        "brands": sorted(BRAND_PREFIXES),
        "units": sorted(UNITS),
        "noise": list(NOISE),
        "price": PRICE.pattern,
        "quantity": QUANTITY.pattern,
        "weighed": WEIGHED.pattern,
        "codes": CODES.pattern,
        "sizes": SIZES.pattern,
    }


def is_noise(line: str) -> bool:
    """Whether a receipt line is something other than a purchased item."""
    lowered = line.lower()
    return any(re.search(p, lowered) for p in NOISE)


def expand(text: str) -> str:
    """Turn a receipt line into words the rest of the project understands.

    Drops the price, the codes, the sizes and the brand, then expands what is
    left. "GV WHL MLK GAL 3.28" becomes "whole milk".
    """
    line = PRICE.sub("", text)
    line = WEIGHED.sub(" ", line)
    line = SIZES.sub(" ", line)
    line = CODES.sub(" ", line)
    words: list[str] = []
    for raw in re.split(r"[^A-Za-z]+", line):
        word = raw.lower()
        if not word or word in BRAND_PREFIXES:
            continue
        if word in UNITS:
            continue
        expanded = ABBREVIATIONS.get(word, word)
        # A single letter that is not a real word is till noise.
        if len(expanded) < 2:
            continue
        words.extend(expanded.split())
    # "whole whole milk" happens when a line repeats a qualifier.
    out: list[str] = []
    for word in words:
        if not out or out[-1] != word:
            out.append(word)
    return " ".join(out)
