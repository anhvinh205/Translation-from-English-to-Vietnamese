import pytest
from pydantic import ValidationError

from src.core import settings as settings_module
from src.core.settings import AppSettings, build_settings


def test_repository_configs_load():
    s = build_settings(env={})
    assert s.dataset.max_seq_length == 384
    assert s.cors.allowed_origins == ["http://localhost:8501"]
    assert s.cache.allowed_root == "./data"
    assert not s.security.auth_enabled
    assert s.security.rate_limit_translate == "30/minute"


def test_missing_config_files_fall_back_to_defaults(tmp_path):
    s = build_settings(str(tmp_path / "nope1.yaml"), str(tmp_path / "nope2.yaml"), env={})
    assert s.model.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert s.inference.max_prompt_tokens == 2048


def test_environment_overrides():
    env = {
        "BASE_MODEL_ID": "org/some-model",
        "CHROMA_PERSIST_DIR": "/x/y",
        "CORS_ORIGINS": "http://a.com, http://b.com",
        "CACHE_ENABLED": "false",
        "MAX_CONCURRENT_GENERATIONS": "2",
        "API_KEYS": "a" * 16 + "," + "b" * 16,
        "ADMIN_API_KEYS": "c" * 20,
        "RATE_LIMIT_ENABLED": "0",
        "RATE_LIMIT_TRANSLATE": "5/minute",
    }
    s = build_settings(env=env)
    assert s.model.base_model_id == "org/some-model"
    assert s.vector_store.persist_dir == "/x/y"
    assert s.cors.allowed_origins == ["http://a.com", "http://b.com"]
    assert s.cache.enabled is False
    assert s.inference.max_concurrent_generations == 2
    assert [k.get_secret_value() for k in s.security.api_keys] == ["a" * 16, "b" * 16]
    assert [k.get_secret_value() for k in s.security.admin_api_keys] == ["c" * 20]
    assert s.security.auth_enabled
    assert s.security.rate_limit_enabled is False
    assert s.security.rate_limit_translate == "5/minute"


def test_empty_environment_values_are_ignored():
    # docker-compose passes `API_KEYS=${API_KEYS:-}`, i.e. an empty string, when unset
    s = build_settings(env={"API_KEYS": "", "ADMIN_API_KEYS": "  ", "CORS_ORIGINS": ""})
    assert not s.security.auth_enabled
    assert s.cors.allowed_origins == ["http://localhost:8501"]


def test_secrets_never_appear_in_repr_or_dumps():
    s = build_settings(env={"API_KEYS": "super-secret-key-123"})
    assert "super-secret-key-123" not in repr(s)
    assert "super-secret-key-123" not in s.model_dump_json()


def test_weak_api_keys_are_rejected():
    with pytest.raises(ValidationError):
        build_settings(env={"API_KEYS": "short"})


@pytest.mark.parametrize("bad", ["lots", "10/fortnight", "", "abc/minute"])
def test_invalid_rate_limits_are_rejected(bad):
    with pytest.raises(ValidationError):
        AppSettings(security={"rate_limit_translate": bad})


def test_unknown_settings_are_rejected():
    with pytest.raises(ValidationError):
        AppSettings(unknown_option=1)


@pytest.mark.parametrize("path", ["../escape.db", "data/../../etc/x.db", "./data/cache.txt", "/etc/cache.db"])
def test_cache_path_is_validated(path):
    with pytest.raises(ValidationError):
        AppSettings(cache={"sqlite_path": path})


def test_cache_path_inside_allowed_root_is_accepted(tmp_path):
    s = AppSettings(cache={"sqlite_path": str(tmp_path / "c.db"), "allowed_root": str(tmp_path)})
    assert s.cache.sqlite_path.endswith("c.db")


def test_device_is_resolved_when_settings_are_built_not_at_import(monkeypatch):
    monkeypatch.setattr(settings_module, "_cuda_available", lambda: True)
    assert AppSettings(device="auto").device == "cuda"
    monkeypatch.setattr(settings_module, "_cuda_available", lambda: False)
    assert AppSettings(device="auto").device == "cpu"
    assert AppSettings(device="cuda").device == "cuda"


def test_save_steps_must_be_a_multiple_of_eval_steps():
    with pytest.raises(ValidationError):
        AppSettings(qlora={"eval_steps": 300, "save_steps": 500})
