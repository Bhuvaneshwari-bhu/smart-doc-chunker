import re
from collections import Counter


def clean_text(text: str) -> str:
    text = _basic_normalize(text)
    text = _remove_noise(text)
    text = _remove_artifacts(text)
    text = _fix_spacing(text)
    text = _drop_garbage_lines(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 1 — basic normalization
# ---------------------------------------------------------------------------

def _basic_normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Stage 2 — noise removal
# ---------------------------------------------------------------------------

# Legal/typographic symbols that add no semantic value in plain-text pipelines
_NOISE_CHARS = re.compile(r"[§©®™]")

# Repeated punctuation: keep exactly one of the repeated character
_REPEATED_PUNCT = re.compile(r"([!?])\1+")


def _remove_noise(text: str) -> str:
    text = _NOISE_CHARS.sub("", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Stage 3 — document artifacts
# ---------------------------------------------------------------------------

# "Page 1", "Page-12", "Page 1 of 20", "12 |", "| 12"
_PAGE_NUMBER = re.compile(
    r"(?i)"
    r"(\bpage[\s\-]+\d+(\s+of\s+\d+)?\b"   # Page 1 / Page 1 of 20 / Page-12
    r"|\b\d+\s*\|"                           # 12 |
    r"|\|\s*\d+\b)",                         # | 12
)


def _remove_artifacts(text: str) -> str:
    text = _PAGE_NUMBER.sub("", text)

    # Remove short repetitive lines (header/footer heuristic):
    # a line is a candidate if it has < 5 words; remove it when the exact
    # same normalised line appears more than once in the document.
    lines = text.split("\n")
    line_counts: Counter = Counter(
        ln.strip() for ln in lines if 0 < len(ln.strip().split()) < 5
    )
    cleaned: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        word_count = len(stripped.split()) if stripped else 0
        if word_count > 0 and word_count < 5 and line_counts[stripped] > 1:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Stage 4 — spacing before punctuation
# ---------------------------------------------------------------------------

# One or more spaces immediately before . , ! ? : ; ) ]
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,:;!?)>\]])")

# Missing space after sentence-ending punctuation followed by a capital letter
_MISSING_SPACE_AFTER = re.compile(r"([.!?])([A-Z])")


def _fix_spacing(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MISSING_SPACE_AFTER.sub(r"\1 \2", text)
    return text


# ---------------------------------------------------------------------------
# Stage 5 — garbage line removal
# ---------------------------------------------------------------------------

# Lines that consist entirely of non-alphanumeric characters (dashes, stars…)
_SYMBOL_ONLY = re.compile(r"^[^a-zA-Z0-9]+$")


def _drop_garbage_lines(text: str) -> str:
    result: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()

        # keep empty lines so paragraph breaks are preserved —
        # but only up to one consecutive blank line (already handled by stage 1)
        if not stripped:
            result.append("")
            continue

        # fewer than 3 characters and not purely numeric → discard
        if len(stripped) < 3 and not stripped.isdigit():
            continue

        # only symbols → discard
        if _SYMBOL_ONLY.match(stripped):
            continue

        result.append(line)

    # Collapse any newly created runs of blank lines back to max two newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return cleaned
