"""Text preprocessing utilities for cyberbullying detection.

Preserves adversarial signals (leetspeak, symbols, casing) while normalizing
noisy social media tokens (URLs, mentions, emojis).
"""

from __future__ import annotations

import re

try:
    import demoji
    USE_DEMOJI = True
except ImportError:
    USE_DEMOJI = False

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
ALLOWED_PH = re.compile(r"\[(?:name|username|user)\]|\(person'?s name\)", re.I)
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Preprocess text for transformer-based cyberbullying classification.
    
    - Normalizes emojis into descriptive text (via demoji if installed).
    - Maps URLs to `[url]`.
    - Maps user mentions and placeholder tags to `[user]`.
    - Unwraps hashtags (e.g. `#loser` -> `loser`).
    - Collapses excess whitespace.
    - Preserves casing, symbols, and digits to retain leetspeak/typo context.
    """
    if text is None:
        return ""
    t = str(text)
    if USE_DEMOJI:
        try:
            t = demoji.replace_with_desc(t, sep=" ")
        except Exception:
            pass
    t = URL_RE.sub("[url]", t)
    t = ALLOWED_PH.sub("[user]", t)
    t = MENTION_RE.sub("[user]", t)
    t = HASHTAG_RE.sub(r"\1", t)
    return WS_RE.sub(" ", t).strip()
