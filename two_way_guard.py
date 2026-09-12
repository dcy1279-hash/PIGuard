"""Editable entry point for the Dark.M&M Two-Way Guard challenge.

Teams may change ONLY the body of ``guard``. PIGuard, its checkpoint, the challenge
pack, the labels, and the scorer stay fixed.

Return ``(cleaned_text, suspicious)``:
  - if ``suspicious`` is True, the front guard blocks the input outright (label 1);
  - otherwise the unchanged PIGuard classifies ``cleaned_text``.

The starter below only runs NFKC. NFKC does NOT undo Cyrillic homoglyphs, so it
neither recovers the evasion attacks nor repairs the false blocks — a deliberately
weak baseline. Do NOT just set suspicious=True whenever the text has non-ASCII
characters: the evasion set and the false_block set contain the SAME Cyrillic
characters with opposite correct answers, and the controls include legitimate
multilingual text. Blanket blocking fails two of the three sets.

Ideas: map confusable characters back to Latin (a real confusables table, not just
NFKC); detect mixed scripts within a single word; strip invisible/formatting
characters; repair, then let PIGuard decide.
"""
import unicodedata


def guard(text):
    cleaned_text = unicodedata.normalize("NFKC", text)
    suspicious = False
    # Teams: add or replace the defence logic below this line.
    return cleaned_text, suspicious
