"""End-to-end SFT -> DPO -> serving with a tiny random Qwen2 model and a toy tokenizer.

Everything runs on CPU, offline, in well under a minute: it proves that the pieces fit together
(Trainer + masking collator, TRL DPO with a frozen SFT reference, adapter loading, generation).
"""

import json
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from datasets import Dataset  # noqa: E402

from src.core.errors import EngineBusyError, ModelOutOfMemoryError  # noqa: E402
from src.core.settings import AppSettings  # noqa: E402
from src.data.prompt_builder import build_sft_or_rag_prompt  # noqa: E402
from src.rag.engine import HybridRAGTranslationEngine  # noqa: E402
from src.rag.reranker import CrossEncoderReranker  # noqa: E402
from tests.conftest import FakeVectorStore, build_toy_tokenizer  # noqa: E402

pytestmark = pytest.mark.slow

ROWS = [
    {"en": f"the cat number {i} sits on the warm mat", "vi": f"con mèo số {i} ngồi trên tấm thảm ấm"} for i in range(24)
]


def _split(rows):
    return Dataset.from_list(
        [{**row, "valid": True, "text": build_sft_or_rag_prompt(row["en"], row["vi"])} for row in rows]
    )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module")
def tiny_model_dir(workdir) -> str:
    from transformers import AutoModelForCausalLM, Qwen2Config

    corpus = [build_sft_or_rag_prompt(r["en"], r["vi"]) for r in ROWS]
    corpus += [
        build_sft_or_rag_prompt("hello world", few_shot_examples=[{"en": "a b", "vi": "c d", "rerank_score": 0.5}])
    ]
    tokenizer = build_toy_tokenizer(corpus)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    path = workdir / "tiny-qwen"
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def e2e_settings(workdir, tiny_model_dir) -> AppSettings:
    return AppSettings(
        model={
            "base_model_id": tiny_model_dir,
            "lora_output_dir": str(workdir / "sft"),
            "dpo_output_dir": str(workdir / "dpo"),
        },
        dataset={"max_seq_length": 160, "max_new_tokens": 6},
        qlora={
            "r": 4,
            "lora_alpha": 8,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "learning_rate": 1e-3,
            "logging_steps": 1,
            "eval_steps": 2,
            "save_steps": 2,
            "max_eval_samples": 8,
        },
        dpo={"batch_size": 2, "num_train_epochs": 1, "learning_rate": 1e-3},
        inference={"max_prompt_tokens": 200, "num_beams": 1, "queue_timeout_seconds": 5},
        cache={"sqlite_path": str(workdir / "cache.db"), "allowed_root": str(workdir)},
        vector_store={"persist_dir": str(workdir / "chroma")},
        device="cpu",
    )


@pytest.fixture(scope="module")
def trained(e2e_settings):
    """Run SFT, then DPO from the SFT adapter, once for the whole module."""
    from src.training import dpo_align, sft_qlora

    train, eval_ = _split(ROWS[:16]), _split(ROWS[16:])
    with pytest.MonkeyPatch.context() as patch:
        splits = lambda settings: (train, None, eval_)  # noqa: E731
        patch.setattr(sft_qlora, "load_wikimatrix_splits", splits)
        patch.setattr(dpo_align, "load_wikimatrix_splits", splits)
        sft_qlora.run_sft_qlora(e2e_settings)
        dpo_align.run_dpo_training(num_samples=8, settings=e2e_settings)
    return e2e_settings


def _adapter_files(directory: str) -> set[str]:
    return {p.name for p in Path(directory).iterdir()}


def test_sft_trains_evaluates_and_saves_an_adapter(trained):
    files = _adapter_files(trained.model.lora_output_dir)
    assert {"adapter_config.json", "adapter_model.safetensors"} <= files

    # the Trainer evaluated and logged finite losses (the masking collator produced real labels)
    states = sorted(Path(trained.model.lora_output_dir).glob("checkpoint-*/trainer_state.json"))
    assert states, "no checkpoint was written"
    history = json.loads(states[-1].read_text())["log_history"]
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    assert train_losses and eval_losses
    assert all(0 < value < 100 for value in train_losses + eval_losses)


def test_dpo_continues_from_the_sft_adapter_and_exports_only_the_policy(trained):
    files = _adapter_files(trained.model.dpo_output_dir)
    assert {"adapter_config.json", "adapter_model.safetensors"} <= files
    # the frozen reference copy of the SFT adapter must not be exported next to the policy
    assert "reference" not in files


def _engine(settings, store=None) -> HybridRAGTranslationEngine:
    return HybridRAGTranslationEngine(
        settings=settings,
        load_llm=True,
        vector_store=store or FakeVectorStore(),
        reranker=CrossEncoderReranker(settings, use_neural_cross_encoder=False),
    )


def test_the_engine_serves_the_trained_adapter(trained):
    engine = _engine(trained)
    try:
        assert engine.active_adapter == "dpo-qlora"  # DPO output takes precedence over SFT
        assert engine.model is not None and engine.tokenizer.truncation_side == "left"

        result = engine.translate("the cat number 3 sits on the warm mat", use_rag=False)
        assert isinstance(result["translation"], str)
        assert result["cache_hit"] is False
        assert result["latency_ms"] > 0

        again = engine.translate("the cat number 3 sits on the warm mat", use_rag=False)
        assert again["cache_hit"] is True and again["translation"] == result["translation"]
    finally:
        engine.close()


def test_rag_prompts_respect_the_token_budget_with_a_real_tokenizer(trained):
    engine = _engine(trained)
    try:
        engine.index_memory(
            [{"en": f"the cat number {i} sits on the warm mat", "vi": f"con mèo số {i}"} for i in range(8)]
        )
        query = "the cat number 3 sits on the warm mat"
        retrieved = engine.retrieve_context(query, top_k=8, score_threshold=0.0)
        assert len(retrieved) > 3

        prompt, used = engine._fit_prompt(query, retrieved)

        assert engine._count_tokens(prompt) <= trained.inference.max_prompt_tokens
        assert 0 < len(used) < len(retrieved)  # the lowest-ranked examples were dropped...
        assert prompt.endswith("<|im_start|>assistant\n")  # ...never the query / assistant header
        assert f"EN: {query}" in prompt

        result = engine.translate(query, use_rag=True, top_k=8, score_threshold=0.0)
        assert 0 < len(result["retrieved_memory"]) < len(retrieved)
    finally:
        engine.close()


def test_out_of_memory_is_mapped_and_the_generation_slot_is_released(trained, monkeypatch):
    engine = _engine(trained)
    try:

        def oom(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        monkeypatch.setattr(engine.model, "generate", oom)
        with pytest.raises(ModelOutOfMemoryError):
            engine.translate("the cat number 1 sits on the warm mat", use_rag=False, use_cache=False)

        monkeypatch.undo()  # generation works again, so the slot must have been released
        assert engine.translate("the cat number 1 sits on the warm mat", use_rag=False, use_cache=False)
    finally:
        engine.close()


def test_a_failing_generation_does_not_leak_the_generation_slot(trained, monkeypatch):
    engine = _engine(trained)
    try:

        def broken(*args, **kwargs):
            raise RuntimeError("kernel crashed")

        monkeypatch.setattr(engine.model, "generate", broken)
        for _ in range(3):  # more attempts than there are slots
            with pytest.raises(RuntimeError, match="kernel crashed"):
                engine.translate("the cat number 1 sits on the warm mat", use_rag=False, use_cache=False)
        assert engine._generation_slots.acquire(blocking=False)
    finally:
        engine.close()


def test_only_one_generation_runs_at_a_time(trained, monkeypatch):
    engine = _engine(trained)
    try:
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        original = engine.model.generate

        def tracked(*args, **kwargs):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.05)
                return original(*args, **kwargs)
            finally:
                with lock:
                    state["active"] -= 1

        monkeypatch.setattr(engine.model, "generate", tracked)
        errors = []

        def work(i):
            try:
                for repeat in range(3):
                    engine.translate(f"the cat number {i} sits on the warm mat", use_rag=False, use_cache=False)
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        # Also guards the thread-safety of the shared HF tokenizer ("RuntimeError: Already borrowed").
        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert state["peak"] == 1  # max_concurrent_generations defaults to 1
    finally:
        engine.close()


def test_requests_wait_for_a_slot_then_fail_fast_when_it_never_frees_up(trained):
    settings = trained.model_copy(
        update={"inference": trained.inference.model_copy(update={"queue_timeout_seconds": 0.1})}
    )
    engine = _engine(settings)
    try:
        assert engine._generation_slots.acquire(blocking=False)  # someone else holds the only slot
        with pytest.raises(EngineBusyError):
            engine.translate("the cat number 1 sits on the warm mat", use_rag=False, use_cache=False)
    finally:
        engine.close()


def test_dpo_can_also_start_from_the_base_model_without_an_sft_adapter(trained, workdir, monkeypatch):
    from src.training import dpo_align

    settings = trained.model_copy(
        update={
            "model": trained.model.model_copy(
                update={
                    "lora_output_dir": str(workdir / "missing-sft"),
                    "dpo_output_dir": str(workdir / "dpo-from-base"),
                }
            )
        }
    )
    monkeypatch.setattr(dpo_align, "load_wikimatrix_splits", lambda s: (_split(ROWS[:8]), None, None))
    dpo_align.run_dpo_training(num_samples=8, settings=settings)
    assert "adapter_model.safetensors" in _adapter_files(settings.model.dpo_output_dir)
