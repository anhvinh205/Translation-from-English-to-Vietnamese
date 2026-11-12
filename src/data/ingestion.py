"""Load, clean, de-duplicate and split WikiMatrix into Train / Translation-Memory / Eval."""

from datasets import load_dataset

from src.core.logger import get_logger
from src.core.settings import AppSettings
from src.data.cleaner import extract_pair, is_valid_pair, unique_indices
from src.data.prompt_builder import build_sft_or_rag_prompt

logger = get_logger(__name__)

# Oversampling factor to ensure enough valid pairs after filtering
_OVERSAMPLE_FACTOR = 1.3
# Default split ratios: 45% train, 45% TM, 10% eval
_TRAIN_RATIO = 0.45
_TM_RATIO = 0.45


def load_wikimatrix_splits(settings: AppSettings):
    """Load, clean, and partition WikiMatrix into Train (SFT/DPO), TM (RAG), and Eval splits."""
    cfg = settings.dataset
    raw_ds = load_dataset(cfg.name, cfg.subset, split="train")
    raw_ds = raw_ds.shuffle(seed=42)

    total_needed = cfg.train_sample_size + cfg.tm_sample_size + cfg.eval_sample_size
    candidate_limit = min(len(raw_ds), int(total_needed * _OVERSAMPLE_FACTOR))
    subset_ds = raw_ds.select(range(candidate_limit))

    def _process(example):
        try:
            en, vi = extract_pair(example)
        except KeyError:
            return {"en": "", "vi": "", "valid": False, "text": ""}
        valid = is_valid_pair(
            en,
            vi,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
            max_ratio=cfg.max_length_ratio,
        )
        # Only build SFT prompt for valid pairs (saves compute)
        text = build_sft_or_rag_prompt(en, vi) if valid else ""
        return {"en": en, "vi": vi, "valid": valid, "text": text}

    mapped = subset_ds.map(_process, remove_columns=subset_ds.column_names)
    clean_ds = mapped.filter(lambda x: x["valid"])

    # Keep every source sentence once: a sentence present in both the translation memory
    # and the eval split would let RAG "retrieve the answer" and inflate the scores.
    keep = unique_indices(clean_ds["en"])
    if len(keep) < len(clean_ds):
        logger.info("Dropped %d duplicate source sentences", len(clean_ds) - len(keep))
        clean_ds = clean_ds.select(keep)

    n_total = len(clean_ds)
    n_train = min(cfg.train_sample_size, int(n_total * _TRAIN_RATIO))
    n_tm = min(cfg.tm_sample_size, int(n_total * _TM_RATIO))
    n_eval = max(0, min(cfg.eval_sample_size, n_total - n_train - n_tm))

    if n_eval == 0:
        logger.warning(
            "Eval split is empty! Total clean samples: %d, train: %d, tm: %d. "
            "Consider reducing train/tm sample sizes or increasing the dataset.",
            n_total,
            n_train,
            n_tm,
        )

    train_ds = clean_ds.select(range(n_train))
    tm_ds = clean_ds.select(range(n_train, n_train + n_tm))
    eval_ds = clean_ds.select(range(n_train + n_tm, n_train + n_tm + n_eval))

    logger.info("Data splits: train=%d, tm=%d, eval=%d (total clean=%d)", n_train, n_tm, n_eval, n_total)
    return train_ds, tm_ds, eval_ds
