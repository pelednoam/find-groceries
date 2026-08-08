"""The receipt vocabulary, checked against lines shops actually print.

The parsing itself runs in the browser, so this file is where the rules get
to meet real till text. Every expansion below is taken from the shape of a US
grocery receipt: brand prefix, abbreviated noun, size suffix, price.
"""

from __future__ import annotations

import re

import pytest

from groceries.receipts import (
    ABBREVIATIONS,
    BRAND_PREFIXES,
    CODES,
    NOISE,
    PRICE,
    QUANTITY,
    SIZES,
    UNITS,
    WEIGHED,
    expand,
    is_noise,
    vocabulary,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("GV WHL MLK GAL 3.28", "whole milk"),
        ("ORG BBY SPNCH 5OZ 4.98", "organic baby spinach"),
        ("CHKN BRST BNLS SKNLS 9.41", "chicken breast boneless skinless"),
        ("GRT VAL SHRD CHDR 8OZ 2.12", "shredded cheddar"),
        ("BANANAS 1.24 lb @ 0.59/lb 0.73", "bananas"),
        ("365 ORG GRND BF 1LB 7.99", "organic ground beef"),
        ("SLTZR 12PK 3.98", "seltzer"),
        ("TJ FRZN PZA 4.49", "frozen pizza"),
        ("KS EGGS LG 24CT 6.79", "eggs large"),
        ("PNUT BTR CRMY 16OZ 2.48", "peanut butter crmy"),
    ],
)
def test_expand_real_lines(line: str, expected: str) -> None:
    assert expand(line) == expected


def test_expand_drops_the_barcode_but_keeps_the_word() -> None:
    # PLU and department numbers are four digits or more; a 6-pack is not.
    assert expand("078742370000 WTR 24PK 3.98") == "water"


def test_expand_collapses_a_repeated_qualifier() -> None:
    assert expand("ORG ORGNC APL 3LB 5.99") == "organic apple"


def test_expand_of_a_line_with_nothing_in_it() -> None:
    assert expand("GV 12OZ 1.99") == ""


def test_expand_drops_a_stray_letter() -> None:
    # Receipts pad with single-letter tax and department flags.
    assert expand("BRD WHT F N 2.79") == "bread wht"


@pytest.mark.parametrize(
    "line",
    [
        "SUBTOTAL 42.17",
        "TOTAL 44.83",
        "TAX 1 6.25% 2.66",
        "DEBIT TEND 44.83",
        "CHANGE DUE 0.00",
        "VISA ************1234",
        "THANK YOU FOR SHOPPING",
        "ST# 02175 OP# 000123 TE# 09 TR# 04412",
        "# ITEMS SOLD 14",
        "COUPON 1.00-",
        "08/06/2026 14:22",
        "(617) 555-0142",
        "-----------------",
    ],
)
def test_noise_lines_are_recognised(line: str) -> None:
    assert is_noise(line)


@pytest.mark.parametrize(
    "line",
    ["GV WHL MLK GAL 3.28", "BANANAS 0.73", "ORG BBY SPNCH 5OZ 4.98"],
)
def test_item_lines_are_not_noise(line: str) -> None:
    assert not is_noise(line)


@pytest.mark.parametrize(
    ("line", "amount"),
    [
        ("GV WHL MLK GAL 3.28", "3.28"),
        ("COUPON -1.00", "-1.00"),
        ("BREAD $2.49", "$2.49"),
        ("CHEESE 4.19 F", "4.19"),
    ],
)
def test_price_is_read_off_the_end(line: str, amount: str) -> None:
    match = PRICE.search(line)
    assert match is not None
    assert match.group(1).strip() == amount


def test_price_needs_two_decimals() -> None:
    # "12OZ" and "2LT" must not read as money.
    assert PRICE.search("SODA 2LT") is None


@pytest.mark.parametrize(
    ("line", "qty"), [("2 @ 1.99 SLTZR", "2"), ("3 X YOGURT", "3"), ("2 EA APPLES", "2")]
)
def test_quantity_prefix(line: str, qty: str) -> None:
    match = QUANTITY.match(line)
    assert match is not None
    assert match.group(1) == qty


def test_quantity_does_not_fire_on_a_leading_code() -> None:
    assert QUANTITY.match("0071 BANANAS") is None


def test_weighed_line_is_stripped() -> None:
    assert WEIGHED.search("1.24 lb @ 2.99/lb") is not None
    assert WEIGHED.search("MILK 1 GAL") is None


def test_sizes_and_codes() -> None:
    assert SIZES.sub("", "MLK 1GAL").strip() == "MLK"
    assert CODES.sub("", "078742 MLK").strip() == "MLK"
    # A three-digit number could be a real part of a name; leave it alone.
    assert CODES.sub("", "PIE 365") == "PIE 365"


def test_vocabulary_is_plain_json_data() -> None:
    v = vocabulary()
    assert set(v) == {
        "abbreviations", "brands", "units", "noise",
        "price", "quantity", "weighed", "codes", "sizes",
    }
    assert v["abbreviations"] == dict(ABBREVIATIONS)
    assert v["brands"] == sorted(BRAND_PREFIXES)
    assert v["units"] == sorted(UNITS)
    assert v["noise"] == list(NOISE)


def test_every_pattern_in_the_vocabulary_compiles() -> None:
    # The browser builds RegExp from these strings; a Python-only construct
    # would fail silently there, so keep them to the common subset.
    v = vocabulary()
    patterns = [str(v[k]) for k in ("price", "quantity", "weighed", "codes", "sizes")]
    patterns += [str(p) for p in NOISE]
    for pattern in patterns:
        re.compile(pattern)
        assert "(?P<" not in pattern
        assert "\\A" not in pattern and "\\Z" not in pattern


def test_expansions_are_lower_case_words() -> None:
    for short, long in ABBREVIATIONS.items():
        assert short == short.lower()
        assert long == long.lower()
        # A one-character expansion would be dropped by `expand` anyway.
        assert len(long) >= 2


def test_brands_and_units_do_not_overlap_the_abbreviations() -> None:
    # A word that is both a brand and an abbreviation would resolve
    # differently in the two languages; keep the sets disjoint.
    assert not BRAND_PREFIXES & set(ABBREVIATIONS)
    assert not UNITS & set(ABBREVIATIONS)


# Real output from Tesseract on a photographed receipt: rotated a degree,
# blurred, lit unevenly from one side. Every mangling below is one the OCR
# actually made — "5OZ" read as "50Z", "ST#" as "ST# ©", a price as "757K)".
# The parser has to survive this, because this is what the feature receives.
OCR_OUTPUT = """MARKET BASKET #21
1 MEMORIAL DR, CAMBRIDGE MA
ST# ©2175 OP# 000123 TE# 09 TR#¥ 04412
GV WHL MLK GAL 3.28
CHKN BRST BNLS SKNLS. 9.41
ORG BBY SPNCH 50Z 4.98
BRD WHT 2002 757K)
COF GRND 120Z 8.49
2 @ 1.99 SLTZR 12PK 3.98
BANANAS

1,24 lb @ 0.59/1lb 0.73
GRT VAL SHRD CHDR 80Z 2.12
FRZN PZA 4CT 4.49
SUBTOTAL 40.27
TAX 1 6.25% 0.00
TOTAL 40.27
DEBIT TEND 40.27
CHANGE DUE 0.00
08/01/2026 14:22
# ITEMS SOLD 10
THANK YOU FOR SHOPPING"""


# Enough of the corpus vocabulary to stand in for the payload's keyword map,
# which is what the browser matches against.
TERMS = frozenset({
    "milk", "chicken breast", "spinach", "bread", "coffee", "seltzer",
    "bananas", "frozen pizza", "cheese",
})


def _items(text: str) -> list[str]:
    """What the browser's parser would keep, expanded.

    Mirrors `parseLines` in docs/src/receipts.ts: drop the short and the
    noisy, then drop anything left that carries neither a price nor a word
    the corpus knows — that is how the shop's own address and the header
    fall out.
    """
    kept: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if len(line) < 3 or is_noise(line):
            continue
        expanded = expand(line)
        if not expanded:
            continue
        if PRICE.search(line) is None and not any(t in expanded for t in TERMS):
            continue
        kept.append(expanded)
    return kept


def test_a_photographed_receipt_survives_its_ocr() -> None:
    items = _items(OCR_OUTPUT)
    assert items == [
        "whole milk",
        "chicken breast boneless skinless",
        "organic baby spinach",
        "bread wht",
        "coffee ground",
        "seltzer",
        "bananas",
        "shredded cheddar",
        "frozen pizza",
    ]


def test_no_total_or_payment_line_survives_the_ocr() -> None:
    joined = " ".join(_items(OCR_OUTPUT))
    for word in ("total", "tax", "debit", "change", "thank", "sold", "memorial"):
        assert word not in joined


@pytest.mark.parametrize(
    ("misread", "expected"),
    [
        ("ORG BBY SPNCH 50Z 4.98", "organic baby spinach"),  # 5OZ -> 50Z
        ("COF GRND 120Z 8.49", "coffee ground"),  # 12OZ -> 120Z
        ("BRD WHT 2002 757K)", "bread wht"),  # a price read as letters
        ("CHKN BRST BNLS SKNLS. 9.41", "chicken breast boneless skinless"),
    ],
)
def test_common_ocr_manglings(misread: str, expected: str) -> None:
    assert expand(misread) == expected


@pytest.mark.parametrize("header", ["ST# ©2175 OP# 000123", "TR#¥ 04412", "# ITEMS SOLD 10"])
def test_register_headers_are_noise_even_when_misread(header: str) -> None:
    assert is_noise(header)
