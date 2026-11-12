"""Helpers shared by the training scripts."""

from typing import Any, Callable


def drop_overlong(
    dataset: Any,
    tokenizer: Any,
    max_length: int,
    text_fn: Callable[[dict], list[str]],
) -> tuple[Any, int]:
    """Remove the samples whose tokenized text is longer than ``max_length``.

    Truncating a training sample would cut the answer (the only part the loss is computed
    on) and teach the model to never finish a sentence, so overlong samples are dropped
    instead. ``text_fn`` receives a batch (dict of columns) and returns the texts to
    measure. Returns ``(filtered dataset, number of dropped samples)``.
    """
    if len(dataset) == 0:
        return dataset, 0

    def _measure(batch: dict) -> dict:
        encoded = tokenizer(text_fn(batch))["input_ids"]
        return {"_n_tokens": [len(ids) for ids in encoded]}

    measured = dataset.map(_measure, batched=True)
    kept = measured.filter(lambda example: example["_n_tokens"] <= max_length)
    return kept.remove_columns("_n_tokens"), len(dataset) - len(kept)
