"""
tests/test_vector_store.py
==========================
Unit tests for :mod:`backend.vector_store`.

Tests cover:
- NumPy fallback index in isolation (no ChromaDB needed).
- The ``VectorStore`` wrapper's domain-specific indexers (mocked
  embeddings when Ollama is not available).
- Round-trip: index → search → verify results.

Run:
    python -m pytest tests/test_vector_store.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.vector_store import VectorStore, _NumpyFallbackIndex


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory for vector data."""
    return tmp_path / "vector_test"


@pytest.fixture
def dummy_embeddings():
    """Return a function that creates deterministic fake embeddings."""
    def _make(texts: list[str]) -> list[list[float]]:
        """Each text gets a simple hash-based vector of dimension 768."""
        results = []
        for t in texts:
            np.random.seed(hash(t) % (2**31))
            vec = np.random.randn(config.EMBEDDING_DIM).tolist()
            results.append(vec)
        return results
    return _make


@pytest.fixture
def mock_embed(dummy_embeddings):
    """Patch llm_engine.embed and embed_batch to avoid needing Ollama."""
    def _embed(text, **kwargs):
        return dummy_embeddings([text])[0]

    def _embed_batch(texts, **kwargs):
        return dummy_embeddings(texts)

    with patch("backend.llm_engine.embed", side_effect=_embed), \
         patch("backend.llm_engine.embed_batch", side_effect=_embed_batch):
        yield


# ────────────────────────────────────────────────────────────────────
# _NumpyFallbackIndex tests
# ────────────────────────────────────────────────────────────────────

class TestNumpyFallbackIndex:
    def test_add_and_search(self, tmp_dir, dummy_embeddings):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = _NumpyFallbackIndex(tmp_dir, "test_collection")

        docs = ["apple pie recipe", "banana smoothie recipe", "car repair guide"]
        embs = dummy_embeddings(docs)
        ids = ["doc_1", "doc_2", "doc_3"]

        idx.add(ids, docs, embs)
        assert idx.count() == 3

        # Search for something food-related — should return food docs.
        query_emb = dummy_embeddings(["fruit recipe"])[0]
        results = idx.search(query_emb, top_k=2)
        assert len(results) == 2
        assert all("id" in r and "document" in r and "distance" in r for r in results)

    def test_upsert_replaces(self, tmp_dir, dummy_embeddings):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = _NumpyFallbackIndex(tmp_dir, "test_upsert")

        embs = dummy_embeddings(["original"])
        idx.add(["id_1"], ["original"], embs)
        assert idx.count() == 1

        # Upsert with same ID.
        embs2 = dummy_embeddings(["updated"])
        idx.add(["id_1"], ["updated"], embs2)
        assert idx.count() == 1
        results = idx.search(embs2[0], top_k=1)
        assert results[0]["document"] == "updated"

    def test_reset_clears(self, tmp_dir, dummy_embeddings):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = _NumpyFallbackIndex(tmp_dir, "test_reset")
        idx.add(["id_1"], ["doc"], dummy_embeddings(["doc"]))
        assert idx.count() == 1
        idx.reset()
        assert idx.count() == 0

    def test_persistence(self, tmp_dir, dummy_embeddings):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = _NumpyFallbackIndex(tmp_dir, "test_persist")
        idx.add(["id_1"], ["persisted doc"], dummy_embeddings(["persisted doc"]))

        # Create a new instance from the same directory.
        idx2 = _NumpyFallbackIndex(tmp_dir, "test_persist")
        assert idx2.count() == 1
        results = idx2.search(dummy_embeddings(["persisted doc"])[0], top_k=1)
        assert results[0]["document"] == "persisted doc"

    def test_empty_search(self, tmp_dir):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = _NumpyFallbackIndex(tmp_dir, "test_empty")
        results = idx.search([0.0] * config.EMBEDDING_DIM, top_k=5)
        assert results == []


# ────────────────────────────────────────────────────────────────────
# VectorStore wrapper tests (with mocked embeddings)
# ────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self, tmp_dir, mock_embed):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        vs = VectorStore(persist_dir=tmp_dir)
        count = vs.add_documents(
            "test_col",
            ids=["a", "b", "c"],
            documents=["I love apples", "Bananas are great", "Cars need fuel"],
        )
        assert count == 3
        assert vs.count("test_col") == 3

        results = vs.search("test_col", "fruit", top_k=2)
        assert len(results) == 2

    def test_reset_collection(self, tmp_dir, mock_embed):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        vs = VectorStore(persist_dir=tmp_dir)
        vs.add_documents("to_delete", ids=["x"], documents=["delete me"])
        assert vs.count("to_delete") >= 1
        vs.reset("to_delete")
        assert vs.count("to_delete") == 0

    def test_empty_documents_noop(self, tmp_dir, mock_embed):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        vs = VectorStore(persist_dir=tmp_dir)
        count = vs.add_documents("empty_col", ids=[], documents=[])
        assert count == 0
