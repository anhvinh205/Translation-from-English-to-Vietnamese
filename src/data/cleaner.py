import hashlib
import re
import unicodedata
from typing import Iterable


def extract_pair(sample: dict) -> tuple[str, str]:
    """Extract (en, vi) sentence pairs across different WikiMatrix HF schemas."""
    if "english" in sample and "non_english" in sample:
        return str(sample["english"]).strip(), str(sample["non_english"]).strip()
    if "translation" in sample and isinstance(sample["translation"], dict):
        trans = sample["translation"]
        if "en" in trans and "vi" in trans:
            return str(trans["en"]).strip(), str(trans["vi"]).strip()
    if "en" in sample and "vi" in sample:
        return str(sample["en"]).strip(), str(sample["vi"]).strip()
    raise KeyError("Unsupported dataset schema: missing ('english', 'non_english') or ('en', 'vi').")


def is_valid_pair(
    en_text: str,
    vi_text: str,
    min_words: int = 3,
    max_words: int = 100,
    max_ratio: float = 2.8,
) -> bool:
    """Filter out noisy web-crawled sentences, URLs, and length imbalances in WikiMatrix."""
    if not en_text or not vi_text:
        return False
    if en_text.lower() == vi_text.lower():
        return False
    if re.search(r"https?://|www\.", en_text) or re.search(r"https?://|www\.", vi_text):
        return False

    en_words = len(en_text.split())
    vi_words = len(vi_text.split())
    if en_words < min_words or vi_words < min_words:
        return False
    if en_words > max_words or vi_words > max_words:
        return False

    ratio = max(en_words, vi_words) / max(1, min(en_words, vi_words))
    return ratio <= max_ratio


def _normalize_text(text: str) -> str:
    """Normalize Unicode (NFKC) and whitespace for consistent deduplication."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())


def deduplicate_pairs(pairs: Iterable[dict]) -> list[dict]:
    """Deduplicate bilingual pairs using normalized SHA-256 fingerprint."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in pairs:
        en_key = item.get("en", "")
        vi_key = item.get("vi", "")
        if not en_key or not vi_key:
            continue
        en_norm = _normalize_text(en_key)
        vi_norm = _normalize_text(vi_key)
        digest = hashlib.sha256(f"{en_norm}||{vi_norm}".encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)
    return unique


def unique_indices(texts: Iterable[str]) -> list[int]:
    """Indices of the first occurrence of every distinct (normalized) text.

    Empty texts are skipped. Used to keep a source sentence from appearing in more
    than one of the train / translation-memory / eval splits (evaluation leakage).
    """
    seen: set[str] = set()
    keep: list[int] = []
    for idx, text in enumerate(texts):
        key = _normalize_text(text or "")
        if not key or key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return keep
