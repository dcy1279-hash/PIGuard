"""Editable Light-team entry point for the M1 Unicode challenge.

Participants may change only the body of ``unicode_guard``.  The PIGuard
model, checkpoint, challenge data, labels, and scorer must remain unchanged.
"""

import unicodedata


def unicode_guard(text):
    """Return ``(cleaned_text, suspicious)`` for one input string.

    The starter method below performs NFKC normalization.  It is deliberately
    weak: NFKC handles compatibility forms such as full-width Latin letters,
    but it does not automatically repair every Cyrillic/Greek homoglyph.

    Light teams may freely improve this function.  Possible directions:
      1. build a careful Unicode-confusable character map;
      2. detect mixed writing systems inside one word;
      3. detect invisible or unusual formatting characters;
      4. repair suspicious text before sending it to PIGuard;
      5. directly flag highly suspicious text;
      6. combine several rules while preserving legitimate multilingual text.

    Do not simply block every non-ASCII character: the challenge contains
    legitimate multilingual benign controls.
    """
    cleaned_text = unicodedata.normalize("NFKC", text)
    suspicious = False

    # Light teams: add or replace your defense logic below this line.

    return cleaned_text, suspicious
