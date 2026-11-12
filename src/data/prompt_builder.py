"""ChatML prompt construction for zero-shot, RAG few-shot, SFT and DPO."""

import re
from typing import Optional

# Any "<|name|>" marker: <|im_start|>, <|im_end|>, <|endoftext|>, <|vision_start|>, ...
_SPECIAL_TOKEN = re.compile(r"<\|[A-Za-z0-9_]{1,40}\|>")

SYSTEM_PROMPT = (
    "Bạn là hệ thống dịch máy chuyên nghiệp từ tiếng Anh sang tiếng Việt. "
    "Hãy ưu tiên tính chính xác thuật ngữ và văn phong tiếng Việt tự nhiên."
)
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def sanitize_chatml(text: str) -> str:
    """Remove every ``<|...|>`` special-token marker so user text cannot forge chat turns.

    The substitution is repeated until the text is stable: a single pass is not
    enough because removing an inner marker can glue the surrounding pieces into
    a new one (``"<|im_<|im_start|>start|>"`` -> ``"<|im_start|>"``).
    """
    previous = None
    while previous != text:
        previous = text
        text = _SPECIAL_TOKEN.sub("", text)
    return text


def _format_context(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["Các cặp câu dịch tham khảo từ Bộ nhớ dịch thuật (Translation Memory):"]
    for idx, ex in enumerate(examples, 1):
        ex_en = sanitize_chatml(ex["en"])
        ex_vi = sanitize_chatml(ex["vi"])
        score_info = f" (Rerank Score: {ex['rerank_score']:.2f})" if "rerank_score" in ex else ""
        lines.append(f"{idx}.{score_info} EN: {ex_en}\n   VI: {ex_vi}")
    return "\n".join(lines) + "\n\n"


def _assemble(en_safe: str, examples: list[dict]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{_format_context(examples)}Hãy dịch câu sau sang tiếng Việt:\n"
        f"EN: {en_safe}<|im_end|>\n"
        f"{ASSISTANT_HEADER}"
    )


def build_sft_or_rag_prompt(
    en_text: str,
    vi_text: Optional[str] = None,
    few_shot_examples: Optional[list[dict]] = None,
    max_prompt_chars: Optional[int] = None,
) -> str:
    """Build the ChatML prompt for zero-shot, few-shot RAG or SFT training.

    ``max_prompt_chars`` bounds the prompt by dropping the lowest-ranked retrieved
    examples; the prompt is *never* cut from the end, because that would remove the
    sentence to translate and the assistant header.
    """
    en_safe = sanitize_chatml(en_text.strip())
    examples = list(few_shot_examples or [])
    prompt = _assemble(en_safe, examples)

    if max_prompt_chars is not None:
        while examples and len(prompt) > max_prompt_chars:
            examples.pop()
            prompt = _assemble(en_safe, examples)

    if vi_text is not None:
        prompt += f"{sanitize_chatml(vi_text.strip())}<|im_end|>"
    return prompt


def build_dpo_preference_sample(en_text: str, chosen_vi: str, rejected_vi: str) -> dict:
    """Format a Direct Preference Optimization (DPO) triplet."""
    return {
        "prompt": build_sft_or_rag_prompt(en_text),
        "chosen": f"{sanitize_chatml(chosen_vi.strip())}<|im_end|>",
        "rejected": f"{sanitize_chatml(rejected_vi.strip())}<|im_end|>",
    }
