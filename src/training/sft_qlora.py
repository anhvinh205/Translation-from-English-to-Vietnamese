"""Stage 1: supervised fine-tuning with QLoRA; the loss is computed on the answer tokens only."""

from typing import Optional

from src.core.logger import get_logger
from src.core.settings import AppSettings, load_settings
from src.data.ingestion import load_wikimatrix_splits
from src.data.prompt_builder import ASSISTANT_HEADER
from src.training.utils import drop_overlong

logger = get_logger(__name__)

IGNORE_INDEX = -100


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Index of the first occurrence of ``needle`` in ``haystack`` (-1 if absent)."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def mask_prompt_labels(input_ids: list[int], attention_mask: list[int], marker_ids: list[int]) -> list[int]:
    """Labels of one sequence: only the tokens after the assistant marker are trained on.

    Prompt and padding positions get ``-100`` (ignored by the loss). If the marker is not
    present (it was cut by truncation) the whole sequence is ignored rather than trained on.
    """
    start = _find_subsequence(input_ids, marker_ids)
    answer_start = start + len(marker_ids) if start >= 0 else len(input_ids)
    return [
        IGNORE_INDEX if (position < answer_start or attention_mask[position] == 0) else token
        for position, token in enumerate(input_ids)
    ]


class PromptMaskingDataCollator:
    """Pads dynamically and masks the prompt so the loss covers only the Vietnamese translation."""

    ASSISTANT_MARKER = ASSISTANT_HEADER

    def __init__(self, tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = tokenizer.encode(self.ASSISTANT_MARKER, add_special_tokens=False)
        if not self.marker_ids:
            raise ValueError("The assistant marker is tokenized to nothing; check the tokenizer.")

    def __call__(self, features: list[dict]) -> dict:
        import torch

        encodings = self.tokenizer(
            [f["text"] for f in features],
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = torch.tensor(
            [
                mask_prompt_labels(ids, mask, self.marker_ids)
                for ids, mask in zip(input_ids.tolist(), attention_mask.tolist())
            ],
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_training_arguments(settings: AppSettings, evaluate: bool, use_cuda: bool, use_bf16: bool):
    """TrainingArguments of the SFT stage (kept in a function so the critical flags are testable)."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=settings.model.lora_output_dir,
        per_device_train_batch_size=settings.qlora.train_batch_size,
        per_device_eval_batch_size=settings.qlora.eval_batch_size,
        gradient_accumulation_steps=settings.qlora.gradient_accumulation_steps,
        learning_rate=settings.qlora.learning_rate,
        num_train_epochs=settings.qlora.num_train_epochs,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=(use_cuda and not use_bf16),
        bf16=(use_cuda and use_bf16),
        logging_steps=settings.qlora.logging_steps,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=settings.qlora.eval_steps if evaluate else None,
        save_strategy="steps",
        save_steps=settings.qlora.save_steps,
        save_total_limit=2,
        load_best_model_at_end=evaluate,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        # REQUIRED: the collator reads the raw "text" column. With the default (True) the
        # Trainer drops every column the model's forward() does not accept - including
        # "text" - before the collator runs, and training fails immediately.
        remove_unused_columns=False,
    )


def run_sft_qlora(settings: Optional[AppSettings] = None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
    )

    settings = settings or load_settings()
    train_ds, _, eval_ds = load_wikimatrix_splits(settings)

    tokenizer = AutoTokenizer.from_pretrained(settings.model.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Never truncate a training sample (that would cut the answer): drop the overlong ones.
    max_len = settings.dataset.max_seq_length
    train_ds, dropped = drop_overlong(train_ds, tokenizer, max_len, lambda batch: batch["text"])
    logger.info("Dropped %d training samples longer than %d tokens (%d kept)", dropped, max_len, len(train_ds))

    # Use a small evaluation subset to keep periodic evaluation cheap.
    eval_subset = None
    if len(eval_ds) > 0:
        eval_subset = eval_ds.select(range(min(settings.qlora.max_eval_samples, len(eval_ds))))
        eval_subset, _ = drop_overlong(eval_subset, tokenizer, max_len, lambda batch: batch["text"])
        if len(eval_subset) == 0:
            eval_subset = None

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

    model = AutoModelForCausalLM.from_pretrained(
        settings.model.base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if settings.device == "cuda" else None,
    )
    if settings.device == "cuda":
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=settings.qlora.r,
        lora_alpha=settings.qlora.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=settings.qlora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = PromptMaskingDataCollator(tokenizer=tokenizer, max_length=max_len)

    args = build_training_arguments(
        settings,
        evaluate=eval_subset is not None,
        use_cuda=settings.device == "cuda",
        use_bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_subset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(settings.model.lora_output_dir)
    tokenizer.save_pretrained(settings.model.lora_output_dir)
    logger.info("SFT LoRA adapter saved to: %s", settings.model.lora_output_dir)


if __name__ == "__main__":
    run_sft_qlora()
