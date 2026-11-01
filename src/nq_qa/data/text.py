"""Text utilities for the Natural Questions (NQ) CSV export.

The CSV stores passages as space-separated *NQ tokens*: punctuation is split off
(``firm 's``, ``Brady , the``), quotes are written ``` `` ``` / ``''`` and structural HTML
elements appear as stand-alone tokens (``<P>``, ``</Td>``).  This module

* strips the HTML tokens,
* turns the token stream back into natural text **while remembering the character
  offsets of every token**, so that an answer found at token level can be mapped to a
  character span of the final passage without any fuzzy matching, and
* locates the short answer (possibly several comma-joined spans) inside the passage.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# ``<P>``, ``</P>``, ``<Table>``, ``<Td_colspan="2">`` ... (attributes are glued with "_").
HTML_TOKEN_RE = re.compile(r"^</?[A-Za-z][A-Za-z0-9]*(?:_[^\s<>]*)?>$")
_FIRST_TAG_RE = re.compile(r"\s*<([A-Za-z][A-Za-z0-9]*)")
# NQ joins several short-answer spans with ", " (no space before the comma), whereas commas
# *inside* a span are tokenised as " , ".  The look-behind therefore only matches joins.
_SPAN_SEPARATOR_RE = re.compile(r"(?<=\S), (?=\S)")

_ATTACH_LEFT = frozenset({",", ".", ";", "!", "?", "%", ")", "]", "}", "...", "\u2026"})
_CONTRACTIONS = frozenset({"'s", "'re", "'ve", "'ll", "'d", "'m", "n't"})
_ATTACH_RIGHT = frozenset({"(", "[", "{", "$", "\u00a3", "\u20ac"})
_TIGHT = frozenset({"-", "/"})


def is_html_token(token: str) -> bool:
    """True for structural HTML tokens such as ``<P>`` or ``</Table>``."""
    return bool(HTML_TOKEN_RE.match(token))


def first_html_tag(raw: str) -> str:
    """Name of the first HTML tag of a raw passage (``"P"``, ``"Table"``...) or ``"plain"``."""
    match = _FIRST_TAG_RE.match(raw)
    return match.group(1) if match else "plain"


def strip_html_tokens(raw: str) -> list[str]:
    """Split a raw passage into NQ tokens and drop the HTML ones."""
    return [tok for tok in raw.split() if not is_html_token(tok)]


def _is_number(token: str | None) -> bool:
    return token is not None and token.isdigit()


def detokenize(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild readable text from NQ tokens.

    Returns ``(text, spans)`` where ``spans[i] = (start, end)`` is the character range
    of ``tokens[i]`` inside ``text`` (so ``text[start:end]`` is the detokenised token).
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    previous_glues_right = True  # no leading space before the very first token
    quote_open = False

    for i, token in enumerate(tokens):
        text = token
        glue_left = False  # attach to the previous token (no space before)
        glue_right = False  # attach to the next token (no space after)
        lowered = token.lower()

        if token == "``":
            text, glue_right, quote_open = '"', True, True
        elif token == "''":
            text, glue_left, quote_open = '"', True, False
        elif token == '"':
            if quote_open:
                glue_left, quote_open = True, False
            else:
                glue_right, quote_open = True, True
        elif token in _ATTACH_LEFT or lowered in _CONTRACTIONS:
            glue_left = True
        elif token in _ATTACH_RIGHT:
            glue_right = True
        elif token in _TIGHT:
            glue_left = glue_right = True
        elif token == ":":
            before = tokens[i - 1] if i > 0 else None
            after = tokens[i + 1] if i + 1 < len(tokens) else None
            if _is_number(before) and _is_number(after):  # 7 : 00 -> 7:00
                glue_left = glue_right = True
            else:
                glue_left = True

        if i > 0 and not (previous_glues_right or glue_left):
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor))
        previous_glues_right = glue_right

    return "".join(parts), spans


def find_token_span(
    haystack: Sequence[str], needle: Sequence[str], *, ignore_case: bool = False
) -> int:
    """Index of the first occurrence of ``needle`` in ``haystack`` (token level) or ``-1``."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    if ignore_case:
        hay = [tok.lower() for tok in haystack]
        ndl = [tok.lower() for tok in needle]
    else:
        hay, ndl = list(haystack), list(needle)
    first = ndl[0]
    for i in range(n - m + 1):
        if hay[i] == first and hay[i : i + m] == ndl:
            return i
    return -1


def split_multi_span(short_answer: str) -> list[str]:
    """Split a short answer made of several comma-joined NQ spans into its parts."""
    parts = [part.strip() for part in _SPAN_SEPARATOR_RE.split(short_answer)]
    return [part for part in parts if part]
