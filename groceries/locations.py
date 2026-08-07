"""Store locations, from OpenStreetMap.

The corpus has opinions but no addresses. Reddit says "the Somerville Market
Basket"; it does not say 400 Somerville Ave. Rather than invent coordinates
for real businesses — which would put a wrong pin on a public map — this pulls
them from OpenStreetMap and matches on the *same* regexes stage 1 uses, so the
two vocabularies cannot drift.

Matching is name-only and deliberately conservative. An OSM entry that does
not match a known store is dropped rather than guessed at, and a branch is
associated with a pin only when the branch name appears in the address.

Data © OpenStreetMap contributors, ODbL. Attribution is carried through to
the payload and rendered on the map.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final, TypedDict

from .select import STORES

OVERPASS_URL: Final = "https://overpass-api.de/api/interpreter"
# Cambridge and everything a Cambridge resident might plausibly drive to.
BBOX: Final = (42.20, -71.35, 42.56, -70.92)
SHOP_TYPES: Final = "supermarket|greengrocer|wholesale|convenience|deli|health_food"

QUERY: Final = f"""[out:json][timeout:90];
(
  nwr["shop"~"^({SHOP_TYPES})$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out center tags;
"""

ATTRIBUTION: Final = "© OpenStreetMap contributors (ODbL)"


class Place(TypedDict):
    """One physical store."""

    store: str          # canonical name, from STORES
    name: str           # as OSM has it
    lat: float
    lon: float
    address: str
    city: str
    osm: str            # e.g. "node/473641811", so a pin is checkable


def match_store(name: str) -> str | None:
    """Canonical store for an OSM name, or None.

    Uses stage 1's patterns rather than a second list. `Target` and
    `Haymarket` are matched here without the grocery-context gate those
    names need in prose: an OSM feature tagged shop=supermarket named
    "Target" is a shop, not a sales figure or a bus stop.
    """
    lowered = name.lower()
    for store, pattern in STORES.items():
        if pattern.search(lowered):
            return store
    return None


def _address(tags: Mapping[str, str]) -> str:
    number = tags.get("addr:housenumber", "").strip()
    street = tags.get("addr:street", "").strip()
    return " ".join(p for p in (number, street) if p)


def extract_places(overpass: Mapping[str, Any]) -> list[Place]:
    """Turn an Overpass response into deduplicated Places."""
    places: dict[str, Place] = {}
    for element in overpass.get("elements", []):
        tags = element.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name:
            continue
        store = match_store(name)
        if store is None:
            continue
        # Nodes carry lat/lon directly; ways and relations carry a centre.
        centre = element.get("center") or element
        lat, lon = centre.get("lat"), centre.get("lon")
        if not isinstance(lat, float | int) or not isinstance(lon, float | int):
            continue
        osm = f"{element['type']}/{element['id']}"
        places[osm] = Place(
            store=store,
            name=name,
            lat=round(float(lat), 6),
            lon=round(float(lon), 6),
            address=_address(tags),
            city=(tags.get("addr:city") or "").strip(),
            osm=osm,
        )
    return sorted(places.values(), key=lambda p: (p["store"], p["city"], p["address"]))


def _tokens(value: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", value.lower()) if len(t) > 2}


def attach_branches(
    places: Iterable[Place], branch_keys: Mapping[str, Iterable[str]]
) -> dict[str, str]:
    """Map each place to a branch name that has its own evidence, if any.

    Deliberately strict: the branch name has to appear in the pin's city or
    address. "Somerville" matches the Somerville store; "the Acre" matches
    nothing and is left unlinked rather than attached to whatever is nearest.
    Returns {osm_id: branch_name}; a pin with no confident branch is absent.
    """
    linked: dict[str, str] = {}
    for place in places:
        haystack = _tokens(f"{place['city']} {place['address']}")
        if not haystack:
            continue
        best = ""
        for branch in branch_keys.get(place["store"], []):
            wanted = _tokens(branch)
            # Every word of the branch name must be present, so "Union Square"
            # does not match a store merely on a street called Union.
            if wanted and wanted <= haystack and len(branch) > len(best):
                best = branch
        if best:
            linked[place["osm"]] = best
    return linked
