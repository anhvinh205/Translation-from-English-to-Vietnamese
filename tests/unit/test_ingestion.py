import logging

import pytest
from datasets import Dataset

from src.data import ingestion
from src.data.cleaner import _normalize_text


def _rows(valid=0, duplicates=0, invalid=0):
    rows = []
    for i in range(valid):
        rows.append(
            {
                "english": f"Sentence number {i} talks about topic {i} in great detail.",
                "non_english": f"Câu số {i} nói về chủ đề {i} một cách rất chi tiết.",
            }
        )
    for i in range(duplicates):  # the same English source again, with another translation
        rows.append(
            {
                "english": f"sentence NUMBER {i} talks about topic {i} in great detail.",
                "non_english": f"Bản dịch khác số {i} cho cùng một câu nguồn tiếng Anh.",
            }
        )
    for i in range(invalid):
        rows.append(
            {
                "english": f"Visit https://example.com/{i} now please",
                "non_english": f"Truy cập https://example.com/{i} ngay",
            }
        )
    return rows


@pytest.fixture
def serve_rows(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(ingestion, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))

    return _install


def test_splits_are_clean_deduplicated_and_disjoint(make_settings, serve_rows):
    settings = make_settings(dataset={"train_sample_size": 25, "tm_sample_size": 25, "eval_sample_size": 10})
    serve_rows(_rows(valid=60, duplicates=10, invalid=5))

    train, tm, eval_ = ingestion.load_wikimatrix_splits(settings)

    assert (len(train), len(tm), len(eval_)) == (25, 25, 10)
    sources = [_normalize_text(en) for split in (train, tm, eval_) for en in split["en"]]
    assert len(sources) == len(set(sources))  # no source sentence appears twice, in or across splits
    assert not any("https://" in en for split in (train, tm, eval_) for en in split["en"])
    assert all(text.endswith("<|im_end|>") for text in train["text"])


@pytest.mark.parametrize("n_valid", [1, 2, 3, 5, 10, 17, 30])
def test_split_sizes_are_never_negative_and_never_exceed_the_data(make_settings, serve_rows, n_valid):
    settings = make_settings(dataset={"train_sample_size": 25, "tm_sample_size": 25, "eval_sample_size": 10})
    serve_rows(_rows(valid=n_valid))

    train, tm, eval_ = ingestion.load_wikimatrix_splits(settings)

    assert min(len(train), len(tm), len(eval_)) >= 0
    assert len(train) + len(tm) + len(eval_) <= n_valid


def test_empty_eval_split_is_reported(make_settings, serve_rows, caplog):
    settings = make_settings(dataset={"train_sample_size": 5, "tm_sample_size": 5, "eval_sample_size": 0})
    serve_rows(_rows(valid=12))

    logger = logging.getLogger("src.data.ingestion")
    logger.addHandler(caplog.handler)  # the project logger does not propagate to the root logger
    try:
        with caplog.at_level(logging.WARNING, logger="src.data.ingestion"):
            _, _, eval_ = ingestion.load_wikimatrix_splits(settings)
    finally:
        logger.removeHandler(caplog.handler)

    assert len(eval_) == 0
    assert any("Eval split is empty" in record.getMessage() for record in caplog.records)


def test_a_dataset_without_any_valid_pair_yields_empty_splits(make_settings, serve_rows):
    serve_rows(_rows(invalid=5))
    train, tm, eval_ = ingestion.load_wikimatrix_splits(make_settings())
    assert (len(train), len(tm), len(eval_)) == (0, 0, 0)
