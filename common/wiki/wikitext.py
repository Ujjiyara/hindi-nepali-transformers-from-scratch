"""Wikitext -> plain text.

Wikitext is not a regular language -- templates and tables nest arbitrarily --
so a single regex pass cannot strip it. The approach here is to remove the
nestable constructs by applying an *innermost-first* pattern repeatedly until
the text stops changing, then handle the flat constructs in one pass.

This is deliberately a cleaner, not a parser. It aims to leave readable prose
and throw away everything else; a template that survives as a stray word is
much less harmful to a language model than an infobox rendered as a wall of
key=value pairs.
"""

from __future__ import annotations

import html
import re

# Link namespaces whose contents are not prose. Includes the localised Hindi
# and Nepali forms -- an English-only list silently leaves image captions and
# category tails in the corpus.
_MEDIA_PREFIXES = "|".join(
    [
        "File", "Image", "Media", "Category",
        "चित्र", "श्रेणी", "फ़ाइल", "फाइल", "तस्वीर", "सञ्चारमाध्यम", "मिडिया",
    ]
)

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_SELFCLOSE_RE = re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE)
_REF_PAIR_RE = re.compile(r"<ref[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_DROP_BLOCK_RE = re.compile(
    r"<(math|code|pre|syntaxhighlight|source|gallery|timeline|score|imagemap)\b[^>]*>"
    r".*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
# Wikisource proofread pages wrap running headers, footers and page numbers in
# <noinclude>. That content is per-scan furniture, not the work's text, so it
# is dropped rather than unwrapped.
_NOINCLUDE_RE = re.compile(r"<noinclude\b[^>]*>.*?</noinclude\s*>", re.DOTALL | re.IGNORECASE)
_NOINCLUDE_SELFCLOSE_RE = re.compile(r"<noinclude\b[^>]*/\s*>", re.IGNORECASE)
_PAGEQUALITY_RE = re.compile(r"<pagequality\b[^>]*/?\s*>", re.IGNORECASE)
# Innermost table: {| ... |} containing no other table opener.
_TABLE_RE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.DOTALL)
# Innermost template/parser function: {{ ... }} containing no braces.
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
# Innermost media/category link, i.e. one containing no further [[ ]].
_MEDIA_LINK_RE = re.compile(
    rf"\[\[\s*(?:{_MEDIA_PREFIXES})\s*:(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]",
    re.DOTALL | re.IGNORECASE,
)
_PIPED_LINK_RE = re.compile(r"\[\[[^\[\]|]*\|([^\[\]|]*)\]\]")
_PLAIN_LINK_RE = re.compile(r"\[\[([^\[\]|]*)\]\]")
_EXT_LINK_TEXT_RE = re.compile(r"\[(?:https?:|//)\S+\s+([^\]]*)\]")
_EXT_LINK_BARE_RE = re.compile(r"\[(?:https?:|//)\S+\]")
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_HEADING_RE = re.compile(r"^\s*={1,6}\s*(.*?)\s*={1,6}\s*$", re.MULTILINE)
_LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
_TABLE_LEFTOVER_RE = re.compile(r"^[!|].*$", re.MULTILINE)
_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# A stripped line shorter than this is almost always a leftover fragment
# (a stray category tail, a table cell, a one-word caption) rather than prose.
MIN_LINE_CHARS = 25

_MAX_NESTING_PASSES = 40


def _strip_nested(text: str, pattern: re.Pattern[str], replacement: str = " ") -> str:
    """Apply an innermost-match pattern until the text stops changing.

    Each pass removes the deepest constructs, exposing the next level out. The
    pass cap prevents a pathological page from looping forever.
    """
    for _ in range(_MAX_NESTING_PASSES):
        new_text = pattern.sub(replacement, text)
        if new_text == text:
            break
        text = new_text
    return text


def strip_wikitext(text: str) -> str:
    """Convert raw wikitext to plain prose.

    Order matters: comments and <ref> blocks are removed first because they can
    contain unbalanced braces that would otherwise confuse template removal.
    """
    text = _COMMENT_RE.sub(" ", text)
    text = _NOINCLUDE_RE.sub(" ", text)
    text = _NOINCLUDE_SELFCLOSE_RE.sub(" ", text)
    text = _PAGEQUALITY_RE.sub(" ", text)
    text = _REF_PAIR_RE.sub(" ", text)
    text = _REF_SELFCLOSE_RE.sub(" ", text)
    text = _DROP_BLOCK_RE.sub(" ", text)

    # Nestable constructs, innermost first.
    text = _strip_nested(text, _TABLE_RE)
    text = _strip_nested(text, _TEMPLATE_RE)
    text = _strip_nested(text, _MEDIA_LINK_RE)

    # Links: keep the human-readable half.
    text = _PIPED_LINK_RE.sub(r"\1", text)
    text = _PLAIN_LINK_RE.sub(r"\1", text)
    text = _EXT_LINK_TEXT_RE.sub(r"\1", text)
    text = _EXT_LINK_BARE_RE.sub(" ", text)

    text = _HEADING_RE.sub(r"\1", text)
    text = _TABLE_LEFTOVER_RE.sub("", text)
    text = _LIST_PREFIX_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _MAGIC_WORD_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)

    # Keep only lines that look like prose.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= MIN_LINE_CHARS]
    text = "\n".join(lines)

    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
