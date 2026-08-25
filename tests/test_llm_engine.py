"""
tests/test_llm_engine.py
========================
Unit tests for :mod:`backend.llm_engine`.

Tests are split into two categories:
- **Offline tests** (always run): exercise error handling, config reads,
  and the ``check_health`` fallback without needing Ollama.
- **Online tests** (marked ``@pytest.mark.online``): actually call Ollama.
  Skip automatically if the daemon is not running.

Run all:
    python -m pytest tests/test_llm_engine.py -v

Run including online (Ollama required):
    python -m pytest tests/test_llm_engine.py -v -m online
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend import llm_engine
from backend.llm_engine import LLMError


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def ollama_available():
    """Skip the test if Ollama is not running."""
    if not llm_engine.check_health():
        pytest.skip("Ollama daemon not running.")


# ────────────────────────────────────────────────────────────────────
# Offline tests (no Ollama needed)
# ────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_check_health_returns_bool(self):
        result = llm_engine.check_health()
        assert isinstance(result, bool)

    def test_check_health_unreachable(self):
        """When Ollama is not at the configured host, returns False."""
        with patch.object(config, "OLLAMA_HOST", "http://127.0.0.1:1"):
            assert llm_engine.check_health() is False


class TestListModels:
    def test_list_local_models_returns_list(self):
        result = llm_engine.list_local_models()
        assert isinstance(result, list)


class TestEnsureModel:
    def test_ensure_model_no_server(self):
        """Should raise LLMError when Ollama is not reachable."""
        with patch.object(config, "OLLAMA_HOST", "http://127.0.0.1:1"):
            with pytest.raises(LLMError, match="not running"):
                llm_engine.ensure_model("llama3.2:3b")


class TestEmbedErrors:
    def test_embed_empty_string_raises(self):
        with pytest.raises(LLMError, match="empty string"):
            llm_engine.embed("")

    def test_embed_whitespace_only_raises(self):
        with pytest.raises(LLMError, match="empty string"):
            llm_engine.embed("   \n  ")


class TestGetClientMissing:
    def test_no_ollama_sdk(self):
        """If the ollama package is not installed, _get_client should raise."""
        with patch.object(llm_engine, "_ollama_sdk", None):
            with pytest.raises(LLMError, match="not installed"):
                llm_engine._get_client()


# ────────────────────────────────────────────────────────────────────
# Online tests (require Ollama running)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.online
class TestGenerate:
    def test_generate_returns_string(self, ollama_available):
        result = llm_engine.generate("Say 'hello' and nothing else.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_system_prompt(self, ollama_available):
        result = llm_engine.generate(
            "What is 2+2?",
            system="You are a calculator. Reply with only the number.",
        )
        assert "4" in result


@pytest.mark.online
class TestGenerateStream:
    def test_stream_yields_tokens(self, ollama_available):
        tokens = list(llm_engine.generate_stream("Say 'hello'."))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 0


@pytest.mark.online
class TestEmbed:
    def test_embed_returns_vector(self, ollama_available):
        vec = llm_engine.embed("Test sentence for embedding.")
        assert isinstance(vec, list)
        assert len(vec) == config.EMBEDDING_DIM

    def test_embed_batch_returns_list(self, ollama_available):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vecs = llm_engine.embed_batch(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == config.EMBEDDING_DIM

    def test_embed_batch_handles_empty_strings(self, ollama_available):
        texts = ["Hello", "", "World"]
        vecs = llm_engine.embed_batch(texts)
        assert len(vecs) == 3
        # The empty string should get a zero vector.
        assert all(x == 0.0 for x in vecs[1])
        # Non-empty should have actual values.
        assert any(x != 0.0 for x in vecs[0])
