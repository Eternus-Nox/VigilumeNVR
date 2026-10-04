"""How a notification says WHO is WHERE.

A recognized subject's alert reads like a sentence — "Adam is at the front
door", "Sam is in the driveway" — rather than "Adam at Front Door". The name
comes from the matched profile; the place from the camera's friendly name.

The place is a camera NAME, typed by the operator, so this only rewrites it
when it is plainly an ordinary place ("Front Door", "Driveway", "Back Yard"):
those get an article, lower case and the right preposition. Anything else
("Cam 2", "Adam's Office", "IPC-4K") is used exactly as typed after "at" (or
the preposition its last word calls for), because "at the cam 2" is worse than
the plain name.
"""
from __future__ import annotations

import re
from typing import Iterable

#: Places you are IN, by how the name ENDS (spaces ignored, so "Drive Way" and
#: "Back Yard" count).
_IN = (
    "driveway", "drive", "yard", "garden", "garage", "carport", "street",
    "road", "lane", "alley", "lot", "parking", "lawn", "field", "barn", "shed",
    "workshop", "shop", "office", "room", "kitchen", "hallway", "hall",
    "basement", "attic", "nursery", "lobby", "stairwell", "den", "study",
)
#: Places you are ON.
_ON = (
    "porch", "patio", "deck", "balcony", "sidewalk", "stairs", "steps",
    "terrace", "walkway", "path", "roof", "dock", "stoop",
)
#: Places you are AT. Also the default preposition for any name.
_AT = (
    "door", "doorbell", "gate", "entrance", "entry", "front", "back", "side",
    "window", "mailbox", "fence", "pool",
)

_WORD = re.compile(r"^[A-Za-z][A-Za-z-]*$")


def place_phrase(friendly: str) -> str:
    """"at the front door", "in the driveway", "on the porch", "at Cam 2"."""
    name = " ".join((friendly or "").replace("_", " ").split())
    if not name:
        return "here"
    compact = name.lower().replace(" ", "").replace("-", "")
    preposition, place_word = "at", False
    for prep, words in (("in", _IN), ("on", _ON), ("at", _AT)):
        if any(compact.endswith(w) for w in words):
            preposition, place_word = prep, True
            break
    words = name.split()
    if place_word and all(_WORD.match(w) for w in words):
        text = name.lower()
        if not text.startswith("the "):
            text = f"the {text}"
        return f"{preposition} {text}"
    return f"{preposition} {name}"


def distinct_names(names: Iterable[str]) -> list[str]:
    """Non-empty names, duplicates (any case) dropped, first spelling kept."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names or ():
        n = (n or "").strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def join_names(names: Iterable[str]) -> str:
    """"Adam", "Adam and Sarah", "Adam, Sarah and Ben"."""
    out = distinct_names(names)
    if len(out) <= 1:
        return out[0] if out else ""
    return f"{', '.join(out[:-1])} and {out[-1]}"


def named_title(names: Iterable[str], friendly: str, *, still: bool = False) -> str:
    """"Adam is at the front door", "Adam and Sarah are still in the yard".
    Empty when there is no name to lead with."""
    names = distinct_names(names)
    if not names:
        return ""
    who = join_names(names)
    verb = "are" if len(names) > 1 else "is"
    title = f"{who} {verb}{' still' if still else ''} {place_phrase(friendly)}"
    return title[0].upper() + title[1:]
