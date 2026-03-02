"""Shared text sanitization utilities for safe UTF-8 storage.

Functions here are used by the parsing layer, the annotation engine,
and the storage layer to ensure all text reaching PostgreSQL is valid
NFC-normalised UTF-8.
"""

from __future__ import annotations

import unicodedata

# Windows-1252 "C1 control" range (0x80–0x9F).  In ISO-8859-1 these are
# undefined control codes, but Windows-1252 maps them to printable characters.
# Publisher EPUBs and LLM outputs sometimes contain these bytes.
CP1252_TO_UNICODE: dict[int, str] = {
    0x80: "\u20AC",  # €
    0x82: "\u201A",  # ‚
    0x83: "\u0192",  # ƒ
    0x84: "\u201E",  # „
    0x85: "\u2026",  # …
    0x86: "\u2020",  # †
    0x87: "\u2021",  # ‡
    0x88: "\u02C6",  # ˆ
    0x89: "\u2030",  # ‰
    0x8A: "\u0160",  # Š
    0x8B: "\u2039",  # ‹
    0x8C: "\u0152",  # Œ
    0x8E: "\u017D",  # Ž
    0x91: "\u2018",  # '
    0x92: "\u2019",  # '
    0x93: "\u201C",  # "
    0x94: "\u201D",  # "
    0x95: "\u2022",  # •
    0x96: "\u2013",  # –
    0x97: "\u2014",  # —
    0x98: "\u02DC",  # ˜
    0x99: "\u2122",  # ™
    0x9A: "\u0161",  # š
    0x9B: "\u203A",  # ›
    0x9C: "\u0153",  # œ
    0x9E: "\u017E",  # ž
    0x9F: "\u0178",  # Ÿ
}


def sanitize_text(text: str) -> str:
    """Normalise and clean text for safe UTF-8 storage in PostgreSQL.

    Applies:
    - Unicode NFC normalisation (compose decomposed sequences)
    - Windows-1252 C1 control-code fixup (0x80–0x9F → proper Unicode)
    - Strip null bytes and remaining C0/C1 control characters (keep \\n, \\r, \\t)
    - UTF-8 round-trip verification

    This function is idempotent — calling it multiple times produces the
    same result.  It is safe (and encouraged) to call at multiple layers
    as a defence-in-depth measure.
    """
    if not text:
        return text

    # 1. NFC normalisation — ensures composed forms (é not e+combining accent)
    text = unicodedata.normalize("NFC", text)

    # 2. Fix C1 control codes that are actually Windows-1252 characters.
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in CP1252_TO_UNICODE:
            chars.append(CP1252_TO_UNICODE[cp])
        elif cp < 0x20 and ch not in ("\n", "\r", "\t"):
            continue  # strip C0 control chars except whitespace
        elif cp == 0x7F:
            continue  # strip DEL
        elif 0x80 <= cp <= 0x9F:
            continue  # strip unmapped C1 controls
        else:
            chars.append(ch)
    text = "".join(chars)

    # 3. Verify clean UTF-8 round-trip (belt-and-suspenders)
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    return text
