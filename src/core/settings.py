"""Typed application settings: YAML config files + environment overrides, validated by Pydantic."""

import functools
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from src.core.cache import validate_cache_path

MIN_API_KEY_LENGTH = 12


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "sentence-transformers/parallel-sentences-wikimatrix"
    subset: str = "en-vi"
    src_lang: str = "en"
    tgt_lang: str = "vi"
    # Max tokens of a *training* sequence (prompt + answer). Longer samples are dropped,
    # never silently truncated (a truncated answer teaches the model to never finish).
    max_seq_length: int = Field(384, ge=64)
    max_new_tokens: int = Field(256, ge=1)
    train_sample_size: int = 50000
    tm_sample_size: int = 50000
    eval_sample_size: int = 1500
    min_words: int = 3
    max_words: int = 100
    max_length_ratio: float = 2.8
    min_semantic_similarity: float = 0.65


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_output_dir: str = "./models/qwen-wikimatrix-sft-lora"
    dpo_output_dir: str = "./models/qwen-wikimatrix-dpo-lora"


class QLoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    logging_steps: int = Field(100, ge=1)
    eval_steps: int = Field(500, ge=1)
    save_steps: int = Field(500, ge=1)
    max_eval_samples: int = Field(500, ge=1)

    @model_validator(mode="after")
    def _save_steps_multiple_of_eval_steps(self) -> "QLoRAConfig":
        # Required by Trainer(load_best_model_at_end=True).
        if self.save_steps % self.eval_steps != 0:
            raise ValueError("qlora.save_steps must be a multiple of qlora.eval_steps")
        return self


class DPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    batch_size: int = 8


class VectorStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    persist_dir: str = "./data/chroma_tm_db"
    collection_name: str = "wikimatrix_en_vi_tm"


class HybridSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k_dense: int = 5
    top_k_bm25: int = 5
    dense_weight: float = 0.6
    bm25_weight: float = 0.4


class RerankerConfig(BaseModel):
    # "model_id" would otherwise clash with pydantic's protected "model_" namespace
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_final: int = 3
    score_threshold: float = 0.25


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_path: str = "./data/translation_cache.db"
    # The cache file must live inside this directory (guards against path traversal).
    allowed_root: str = "./data"
    enabled: bool = True
    ttl_hours: int = Field(168, ge=1)  # 7 days
    max_entries: int = Field(100000, ge=1)

    @model_validator(mode="after")
    def _validate_sqlite_path(self) -> "CacheConfig":
        validate_cache_path(self.sqlite_path, allowed_root=self.allowed_root)
        return self


class CORSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Token budget of the *whole* prompt (system + retrieved examples + query).
    # When it is exceeded, retrieved examples are dropped - the tail is never cut.
    max_prompt_tokens: int = Field(2048, ge=128)
    num_beams: int = Field(3, ge=1, le=8)
    # How many generations may run at once (protects the GPU); extra requests wait
    # up to ``queue_timeout_seconds`` and are then rejected with HTTP 503.
    max_concurrent_generations: int = Field(1, ge=1)
    queue_timeout_seconds: float = Field(30.0, gt=0)


class SecurityConfig(BaseModel):
    """Authentication and rate limiting. Secrets belong in the environment, not in YAML."""

    model_config = ConfigDict(extra="forbid")

    # If both lists are empty authentication is disabled (local development only).
    api_keys: list[SecretStr] = Field(default_factory=list)
    admin_api_keys: list[SecretStr] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_default: str = "120/minute"
    rate_limit_translate: str = "30/minute"
    rate_limit_memory_index: str = "10/minute"
    rate_limit_admin: str = "5/minute"

    @field_validator("api_keys", "admin_api_keys")
    @classmethod
    def _check_key_strength(cls, keys: list[SecretStr]) -> list[SecretStr]:
        for key in keys:
            if len(key.get_secret_value()) < MIN_API_KEY_LENGTH:
                raise ValueError(f"API keys must be at least {MIN_API_KEY_LENGTH} characters long")
        return keys

    @field_validator(
        "rate_limit_default",
        "rate_limit_translate",
        "rate_limit_memory_index",
        "rate_limit_admin",
    )
    @classmethod
    def _check_rate_limit(cls, value: str) -> str:
        try:
            from limits import parse_many
        except ImportError:  # slowapi (and therefore limits) not installed
            return value
        try:
            parse_many(value)
        except Exception as exc:
            raise ValueError(f"invalid rate limit {value!r}: {exc}") from exc
        return value

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys or self.admin_api_keys)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    device: str = "cpu"

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, v: str) -> str:
        # Resolved when the settings object is built (not at import time).
        if v == "auto":
            return "cuda" if _cuda_available() else "cpu"
        return v


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts; override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# environment variable -> (location inside the settings dict, converter)
_ENV_OVERRIDES: dict[str, tuple[tuple[str, str], Callable[[str], Any]]] = {
    "BASE_MODEL_ID": (("model", "base_model_id"), str),
    "CHROMA_PERSIST_DIR": (("vector_store", "persist_dir"), str),
    "CORS_ORIGINS": (("cors", "allowed_origins"), _split_csv),
    "CACHE_ENABLED": (("cache", "enabled"), _as_bool),
    "MAX_CONCURRENT_GENERATIONS": (("inference", "max_concurrent_generations"), int),
    "API_KEYS": (("security", "api_keys"), _split_csv),
    "ADMIN_API_KEYS": (("security", "admin_api_keys"), _split_csv),
    "RATE_LIMIT_ENABLED": (("security", "rate_limit_enabled"), _as_bool),
    "RATE_LIMIT_DEFAULT": (("security", "rate_limit_default"), str),
    "RATE_LIMIT_TRANSLATE": (("security", "rate_limit_translate"), str),
    "RATE_LIMIT_MEMORY_INDEX": (("security", "rate_limit_memory_index"), str),
    "RATE_LIMIT_ADMIN": (("security", "rate_limit_admin"), str),
}


def build_settings(
    train_cfg_path: str = "configs/train_config.yaml",
    rag_cfg_path: str = "configs/rag_config.yaml",
    env: Optional[Mapping[str, str]] = None,
) -> AppSettings:
    """Build (uncached) settings from the YAML files plus environment overrides."""
    env = os.environ if env is None else env
    merged = _deep_merge(_load_yaml(Path(train_cfg_path)), _load_yaml(Path(rag_cfg_path)))

    # Auto-detect device if not explicitly set
    merged.setdefault("device", "auto")

    # Empty values (e.g. ``API_KEYS=`` coming from docker-compose) are ignored.
    overrides: dict[str, Any] = {}
    for name, ((section, field), convert) in _ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is not None and raw.strip():
            overrides.setdefault(section, {})[field] = convert(raw.strip())

    return AppSettings(**_deep_merge(merged, overrides))


@functools.lru_cache(maxsize=1)
def load_settings(
    train_cfg_path: str = "configs/train_config.yaml",
    rag_cfg_path: str = "configs/rag_config.yaml",
) -> AppSettings:
    """Process-wide settings (cached). Use ``build_settings`` when you need a fresh object."""
    return build_settings(train_cfg_path, rag_cfg_path)
