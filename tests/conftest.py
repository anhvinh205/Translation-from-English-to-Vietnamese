"""Shared fixtures. Everything is hermetic: no network, no model download, temp paths only."""

import os

# Must be set before huggingface_hub / chromadb are imported by any test module.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("MOCK_LLM", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402

from src.core.settings import AppSettings  # noqa: E402
from src.rag.engine import HybridRAGTranslationEngine  # noqa: E402
from src.rag.reranker import CrossEncoderReranker  # noqa: E402


class FakeVectorStore:
    """In-memory stand-in for ChromaTranslationStore (Jaccard overlap instead of embeddings)."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.fail = False  # simulate an outage of the vector database

    @staticmethod
    def _key(text: str) -> str:
        return " ".join(text.split()).lower()

    def add_pairs(self, pairs, batch_size=500):
        unique = {}
        for p in pairs:
            en, vi = (p.get("en") or "").strip(), (p.get("vi") or "").strip()
            if en and vi:
                unique[self._key(en)] = {"en": en, "vi": vi}
        self.items.update(unique)
        return len(unique)

    def dense_search(self, query_en, k):
        if self.fail:
            raise RuntimeError("vector store is down")
        query = set(re.findall(r"\w+", query_en.lower()))
        hits = []
        for item in self.items.values():
            tokens = set(re.findall(r"\w+", item["en"].lower()))
            if query and tokens and query & tokens:
                hits.append({**item, "dense_score": round(len(query & tokens) / len(query | tokens), 4)})
        hits.sort(key=lambda h: h["dense_score"], reverse=True)
        return hits[:k]

    def count(self):
        return len(self.items)


class FakeEmbeddings:
    """Deterministic hashed bag-of-words embeddings (LangChain ``Embeddings`` interface)."""

    DIM = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def make_settings(tmp_path):
    """Factory for AppSettings that only touch ``tmp_path``; keyword args override sections."""

    def _make(**overrides) -> AppSettings:
        cache = {"sqlite_path": str(tmp_path / "cache.db"), "allowed_root": str(tmp_path)}
        cache.update(overrides.pop("cache", {}))
        vector_store = {"persist_dir": str(tmp_path / "chroma")}
        vector_store.update(overrides.pop("vector_store", {}))
        return AppSettings(cache=cache, vector_store=vector_store, device="cpu", **overrides)

    return _make


@pytest.fixture
def settings(make_settings) -> AppSettings:
    return make_settings()


@pytest.fixture
def make_engine():
    """Factory for engines wired with fakes (no model, no Chroma, no cross-encoder download)."""
    created = []

    def _make(settings: AppSettings, **kwargs) -> HybridRAGTranslationEngine:
        kwargs.setdefault("vector_store", FakeVectorStore())
        kwargs.setdefault("reranker", CrossEncoderReranker(settings, use_neural_cross_encoder=False))
        engine = HybridRAGTranslationEngine(settings=settings, load_llm=False, **kwargs)
        created.append(engine)
        return engine

    yield _make
    for engine in created:
        engine.close()


@pytest.fixture
def engine(settings, make_engine) -> HybridRAGTranslationEngine:
    return make_engine(settings)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


def build_toy_tokenizer(corpus: list[str]):
    """A tiny offline word-level tokenizer that treats the ChatML markers as special tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    splitter = pre_tokenizers.Whitespace()
    words = sorted({piece for text in corpus for piece, _ in splitter.pre_tokenize_str(text)})
    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {token: idx for idx, token in enumerate(specials + [w for w in words if w not in specials])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = splitter
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
        # like the real Qwen tokenizer: a decoder-only model does not accept token_type_ids
        model_input_names=["input_ids", "attention_mask"],
    )


@pytest.fixture
def toy_tokenizer():
    pytest.importorskip("transformers")
    return build_toy_tokenizer


def make_response(status=200, body=None, headers=None, text=""):
    """A ``requests.Response`` with a JSON body (or raw text), for mocking the API."""
    response = requests.Response()
    response.status_code = status
    response._content = (json.dumps(body) if body is not None else text).encode("utf-8")
    response.headers.update(headers or {})
    return response
