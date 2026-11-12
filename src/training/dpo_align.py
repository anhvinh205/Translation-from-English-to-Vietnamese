"""Stage 2: Direct Preference Optimization (DPO) for natural Vietnamese fluency."""

import os
import random
from typing import Optional

from datasets import Dataset

from src.core.logger import get_logger
from src.core.settings import AppSettings, load_settings
from src.data.ingestion import load_wikimatrix_splits
from src.data.prompt_builder import build_dpo_preference_sample
from src.training.utils import drop_overlong

logger = get_logger(__name__)

# Name of the frozen copy of the SFT adapter that DPO uses as its reference policy.
REFERENCE_ADAPTER = "reference"


def _generate_degraded_translation(
    chosen_vi: str,
    rng: Optional[random.Random] = None,
    word_pool: Optional[list[str]] = None,
) -> str:
    """Return a worse translation that is guaranteed to differ from ``chosen_vi``."""
    degraded = _apply_degradation(chosen_vi, rng, word_pool)
    if degraded.strip() == chosen_vi.strip():
        words = chosen_vi.split()
        degraded = " ".join(words[:-1]) if len(words) > 1 else chosen_vi + " ."
    return degraded


def _apply_degradation(
    chosen_vi: str,
    rng: Optional[random.Random] = None,
    word_pool: Optional[list[str]] = None,
) -> str:
    """Generate a plausible-but-worse translation to use as the rejected sample.

    Strategies (diverse negatives are what gives DPO a useful signal):
    1. truncate       - drop the last ~15-30% of the words (incomplete translation)
    2. shuffle        - swap adjacent word pairs (broken word order)
    3. repeat         - duplicate a word (typical decoding artefact)
    4. drop_particle  - remove a short function word (missing "của", "là", "và", ...)
    5. substitute     - replace 1-2 words by words from other sentences (wrong terminology);
                        only used when ``word_pool`` is provided.
    """
    rng = rng or random.Random()
    words = chosen_vi.split()

    def substitute(count: int) -> str:
        result = words.copy()
        for _ in range(count):
            result[rng.randrange(len(result))] = rng.choice(word_pool)
        return " ".join(result)

    if len(words) <= 3:
        return substitute(1) if word_pool else chosen_vi + " ."

    strategies = ["truncate", "shuffle", "repeat", "drop_particle"]
    if word_pool:
        strategies.append("substitute")
    strategy = rng.choice(strategies)

    if strategy == "truncate":
        cut = max(1, int(len(words) * rng.uniform(0.15, 0.30)))
        return " ".join(words[:-cut])

    if strategy == "shuffle":
        result = words.copy()
        for _ in range(min(3, len(result) // 2)):
            i = rng.randint(0, len(result) - 2)
            result[i], result[i + 1] = result[i + 1], result[i]
        return " ".join(result)

    if strategy == "repeat":
        idx = rng.randint(0, len(words) - 1)
        return " ".join(words[:idx] + [words[idx], words[idx]] + words[idx + 1 :])

    if strategy == "substitute":
        return substitute(rng.randint(1, 2))

    # drop_particle
    particles = [w for w in words if len(w) <= 3]
    if particles:
        to_remove = rng.choice(particles)
        result = [w for w in words if w != to_remove]
        if result:
            return " ".join(result)
    return " ".join(words[:-1])  # fallback: drop the last word


def build_synthetic_dpo_dataset(
    settings: Optional[AppSettings] = None,
    num_samples: int = 2000,
    seed: int = 42,
) -> Dataset:
    """Create preference pairs: ``chosen`` is the clean WikiMatrix target, ``rejected`` a degraded one."""
    rng = random.Random(seed)  # local RNG: reproducible and no global side effects
    if settings is None:
        settings = load_settings()
    train_ds, _, _ = load_wikimatrix_splits(settings)
    subset = train_ds.select(range(min(num_samples, len(train_ds))))

    # Vocabulary of the sample, used to fabricate "wrong terminology" negatives (sorted -> deterministic).
    word_pool = sorted({w for row in subset for w in row["vi"].split() if w.isalpha() and len(w) >= 2})

    records = []
    for row in subset:
        en_text = row["en"]
        chosen_vi = row["vi"]
        rejected_vi = _generate_degraded_translation(chosen_vi, rng, word_pool)
        records.append(build_dpo_preference_sample(en_text, chosen_vi, rejected_vi))

    logger.info("Built %d DPO preference triplets.", len(records))
    return Dataset.from_list(records)


def _strip_eos(text: str, eos_token: Optional[str]) -> str:
    """TRL appends the EOS token to completions itself; avoid a doubled end-of-turn marker."""
    if eos_token and text.endswith(eos_token):
        return text[: -len(eos_token)]
    return text


def run_dpo_training(num_samples: int = 2000, settings: Optional[AppSettings] = None) -> None:
    """Run DPO alignment training using TRL's DPOTrainer."""
    try:
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer
    except ImportError as exc:
        raise ImportError(
            "DPO training requires TRL (processing_class needs trl>=0.12). "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    settings = settings or load_settings()
    max_len = settings.dataset.max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(settings.model.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the DPO dataset; drop (instead of truncating) samples that do not fit.
    dpo_dataset = build_synthetic_dpo_dataset(settings=settings, num_samples=num_samples)
    dpo_dataset, dropped = drop_overlong(
        dpo_dataset,
        tokenizer,
        max_len,
        lambda batch: [p + c for p, c in zip(batch["prompt"], batch["chosen"])],
    )
    logger.info("Dropped %d preference samples longer than %d tokens (%d kept)", dropped, max_len, len(dpo_dataset))
    eos = tokenizer.eos_token
    dpo_dataset = dpo_dataset.map(
        lambda row: {"chosen": _strip_eos(row["chosen"], eos), "rejected": _strip_eos(row["rejected"], eos)}
    )

    bnb_config = None
    use_bf16 = False
    if settings.device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.model.base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if settings.device == "cuda" else None,
    )
    if settings.device == "cuda":
        base_model = prepare_model_for_kbit_training(base_model)

    sft_adapter_path = settings.model.lora_output_dir
    ref_adapter_name = None
    if os.path.exists(os.path.join(sft_adapter_path, "adapter_config.json")):
        logger.info("Continuing from the SFT adapter at %s.", sft_adapter_path)
        # "default" = trainable policy, initialised from the SFT adapter.
        model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
        # Frozen second copy of the SFT adapter = reference policy. Without it TRL would
        # compare against the *base* model (adapter disabled) and the KL anchor would be wrong.
        model.load_adapter(sft_adapter_path, adapter_name=REFERENCE_ADAPTER)
        model.set_adapter("default")
        ref_adapter_name = REFERENCE_ADAPTER
    else:
        logger.warning("No SFT adapter found at %s. Running DPO from the base model.", sft_adapter_path)
        lora_config = LoraConfig(
            r=settings.qlora.r,
            lora_alpha=settings.qlora.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=settings.qlora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    dpo_config = TRLDPOConfig(
        output_dir=settings.model.dpo_output_dir,
        beta=settings.dpo.beta,
        learning_rate=settings.dpo.learning_rate,
        num_train_epochs=settings.dpo.num_train_epochs,
        per_device_train_batch_size=settings.dpo.batch_size,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        fp16=(settings.device == "cuda" and not use_bf16),
        bf16=(settings.device == "cuda" and use_bf16),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        # Samples were pre-filtered to fit max_len; the rejected text can be a few tokens longer.
        max_length=max_len + 16,
        max_prompt_length=max_len,
        ref_adapter_name=ref_adapter_name,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training with %d samples...", len(dpo_dataset))
    trainer.train()
    # Only the trained policy ("default") is exported, never the frozen reference copy.
    model.save_pretrained(settings.model.dpo_output_dir, selected_adapters=["default"])
    tokenizer.save_pretrained(settings.model.dpo_output_dir)
    logger.info("DPO adapter saved to: %s", settings.model.dpo_output_dir)


if __name__ == "__main__":
    run_dpo_training()
