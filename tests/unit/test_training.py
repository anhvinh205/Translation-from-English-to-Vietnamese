import random

import pytest
from datasets import Dataset

from src.data.prompt_builder import build_sft_or_rag_prompt
from src.training import dpo_align
from src.training.dpo_align import _generate_degraded_translation, _strip_eos, build_synthetic_dpo_dataset
from src.training.sft_qlora import IGNORE_INDEX, PromptMaskingDataCollator, mask_prompt_labels
from src.training.utils import drop_overlong

MARKER = [10, 11, 12]

SENTENCES = [
    "Chính phủ đã công bố một chương trình mới nhằm hỗ trợ nông dân ở các vùng nông thôn",
    "Ngôi làng nằm ở phía bắc của tỉnh",
    "Ông học toán tại Đại học Paris trước khi chuyển đến London",
    "Tôi đi học",
    "Dịch máy rất vui",
]
WORD_POOL = ["gạo", "máy", "học", "mạng", "dữ", "liệu", "thành", "phố"]


# ------------------------------------------------------------------ SFT label masking
def test_prompt_and_padding_are_masked_and_the_answer_is_kept():
    ids = [1, 2, 10, 11, 12, 7, 8, 9, 0, 0]  # prompt(2) marker(3) answer(3) pad(2)
    attention = [1] * 8 + [0, 0]
    assert mask_prompt_labels(ids, attention, MARKER) == [IGNORE_INDEX] * 5 + [7, 8, 9] + [IGNORE_INDEX] * 2


def test_a_sequence_whose_marker_was_cut_off_is_ignored_entirely():
    # Truncation can remove the assistant marker; training on the prompt would be the original bug.
    assert mask_prompt_labels([1, 2, 3, 4], [1] * 4, MARKER) == [IGNORE_INDEX] * 4


def test_a_marker_at_the_very_end_leaves_nothing_to_train_on():
    assert mask_prompt_labels([1, 10, 11, 12], [1] * 4, MARKER) == [IGNORE_INDEX] * 4


def test_collator_masks_the_prompt_and_pads_dynamically(toy_tokenizer):
    pytest.importorskip("torch")
    texts = [
        build_sft_or_rag_prompt("good morning", "chào buổi sáng"),
        build_sft_or_rag_prompt("thank you very much indeed", "cảm ơn bạn rất nhiều"),
    ]
    tokenizer = toy_tokenizer(texts)
    batch = PromptMaskingDataCollator(tokenizer, max_length=256)([{"text": t} for t in texts])

    input_ids, labels, attention = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert input_ids.shape == labels.shape == attention.shape
    assert input_ids.shape[1] == max(len(tokenizer(t)["input_ids"]) for t in texts)  # no padding to a fixed size

    marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for row in range(len(texts)):
        ids = input_ids[row].tolist()
        start = next(i for i in range(len(ids)) if ids[i : i + len(marker)] == marker)
        answer_start = start + len(marker)
        n_real = int(attention[row].sum())
        assert (labels[row][:answer_start] == IGNORE_INDEX).all()  # system + user prompt
        assert labels[row][answer_start:n_real].tolist() == ids[answer_start:n_real]  # the translation
        assert labels[row][n_real - 1].item() == end_of_turn  # the model learns to stop
        assert (labels[row][n_real:] == IGNORE_INDEX).all()  # padding


def test_collator_ignores_samples_truncated_before_the_answer(toy_tokenizer):
    pytest.importorskip("torch")
    text = build_sft_or_rag_prompt("a very long sentence " * 20, "bản dịch")
    tokenizer = toy_tokenizer([text])
    batch = PromptMaskingDataCollator(tokenizer, max_length=20)([{"text": text}])
    assert (batch["labels"] == IGNORE_INDEX).all()


def test_drop_overlong_removes_samples_instead_of_truncating_them(toy_tokenizer):
    tokenizer = toy_tokenizer(["a b c"])
    dataset = Dataset.from_list([{"text": "a b c"}, {"text": "a " * 50}, {"text": "a b"}])
    kept, dropped = drop_overlong(dataset, tokenizer, 10, lambda batch: batch["text"])
    assert dropped == 1
    assert kept["text"] == ["a b c", "a b"]
    assert kept.column_names == ["text"]  # the helper column is gone


# ------------------------------------------------------------------ DPO data
def test_the_rejected_translation_always_differs_from_the_chosen_one():
    rng = random.Random(0)
    for sentence in SENTENCES:
        for _ in range(100):
            assert _generate_degraded_translation(sentence, rng, WORD_POOL).strip() != sentence.strip()
            assert _generate_degraded_translation(sentence, rng).strip() != sentence.strip()


def test_every_degradation_strategy_is_used():
    rng = random.Random(1)
    sentence = SENTENCES[0]
    original = sentence.split()
    kinds = set()
    for _ in range(300):
        words = _generate_degraded_translation(sentence, rng, WORD_POOL).split()
        if len(words) < len(original):
            kinds.add("shorter")  # truncation or a dropped particle
        elif len(words) > len(original):
            kinds.add("repeated")
        elif sorted(words) == sorted(original):
            kinds.add("reordered")
        else:
            kinds.add("substituted")
    assert kinds == {"shorter", "repeated", "reordered", "substituted"}


def _fake_splits(monkeypatch, rows):
    dataset = Dataset.from_list(rows)
    monkeypatch.setattr(dpo_align, "load_wikimatrix_splits", lambda settings: (dataset, None, None))


def _rows(n=30):
    return [{"en": f"English sentence number {i}", "vi": f"Câu tiếng Việt số {i} rất đẹp và dài"} for i in range(n)]


def test_the_synthetic_dpo_dataset_is_well_formed_and_reproducible(monkeypatch, settings):
    _fake_splits(monkeypatch, _rows())
    first = build_synthetic_dpo_dataset(settings, num_samples=20)
    second = build_synthetic_dpo_dataset(settings, num_samples=20)

    assert first.column_names == ["prompt", "chosen", "rejected"]
    assert len(first) == 20
    assert first["rejected"] == second["rejected"]  # same seed -> same dataset
    for row in first:
        assert row["chosen"] != row["rejected"]
        assert row["chosen"].endswith("<|im_end|>") and row["rejected"].endswith("<|im_end|>")
        assert row["prompt"].count("<|im_start|>") == 3


def test_a_different_seed_gives_different_negatives(monkeypatch, settings):
    _fake_splits(monkeypatch, _rows())
    a = build_synthetic_dpo_dataset(settings, num_samples=20, seed=1)
    b = build_synthetic_dpo_dataset(settings, num_samples=20, seed=2)
    assert a["rejected"] != b["rejected"]


def test_the_global_random_state_is_left_untouched(monkeypatch, settings):
    _fake_splits(monkeypatch, _rows())
    random.seed(123)
    expected = random.random()
    random.seed(123)
    build_synthetic_dpo_dataset(settings, num_samples=10)
    assert random.random() == expected


def test_strip_eos_removes_only_a_trailing_marker():
    assert _strip_eos("Tôi đi học.<|im_end|>", "<|im_end|>") == "Tôi đi học."
    assert _strip_eos("Tôi đi học.", "<|im_end|>") == "Tôi đi học."
    assert _strip_eos("abc", None) == "abc"


def test_training_arguments_keep_the_columns_the_collator_needs(settings):
    pytest.importorskip("transformers")
    from src.training.sft_qlora import build_training_arguments

    # Without remove_unused_columns=False the Trainer drops the raw "text" column before the
    # collator runs, and SFT fails immediately (ValueError in transformers 4.46+).
    with_eval = build_training_arguments(settings, evaluate=True, use_cuda=False, use_bf16=False)
    assert with_eval.remove_unused_columns is False
    assert with_eval.eval_strategy == "steps"
    assert with_eval.load_best_model_at_end is True

    without_eval = build_training_arguments(settings, evaluate=False, use_cuda=False, use_bf16=False)
    assert without_eval.remove_unused_columns is False
    assert without_eval.eval_strategy == "no"
    assert without_eval.load_best_model_at_end is False
