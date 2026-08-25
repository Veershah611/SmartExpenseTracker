"""
tests/test_rag_engine.py
========================
Unit tests for :mod:`backend.rag_engine`.

Most tests mock the LLM and embedding calls so they run offline.
Online integration tests (marked ``@pytest.mark.online``) hit a real
Ollama instance and a seeded database.

Run:
    python -m pytest tests/test_rag_engine.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.rag_engine import (
    MAX_HISTORY_TURNS,
    SYSTEM_PROMPT,
    RAGEngine,
    _load_student_context,
)
from backend.vector_store import VectorStore


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_db(tmp_path):
    """Create a minimal seeded SQLite database for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")

    # Create tables.
    schema_path = PROJECT_ROOT / "backend" / "schema.sql"
    conn.executescript(schema_path.read_text(encoding="utf-8"))

    # Insert a student.
    conn.execute(
        """INSERT INTO students (id, name, roll_no, course, semester,
                                 hostel_resident, monthly_budget, persona_note)
           VALUES (1, 'Test Student', '22TEST001', 'B.Tech CS', 5,
                   1, 15000.0, 'A test student.')"""
    )

    # Insert categories.
    for name, icon, share in config.EXPENSE_CATEGORIES:
        conn.execute(
            "INSERT OR IGNORE INTO categories (name, icon, typical_share) VALUES (?, ?, ?)",
            (name, icon, share),
        )

    # Insert a few expenses.
    conn.executemany(
        """INSERT INTO expenses (student_id, category_id, txn_date, amount,
                                 merchant, description, payment_mode, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (1, 1, "2025-08-20", 150.0, "Campus Canteen", "Lunch", "UPI", "manual"),
            (1, 3, "2025-08-21", 120.0, "Ola Auto", "Auto to campus", "Cash", "manual"),
            (1, 6, "2025-08-22", 499.0, "Netflix", "Monthly subscription", "Credit Card", "recurring_auto"),
        ],
    )

    # Insert knowledge.
    conn.execute(
        "INSERT INTO knowledge_base (topic, content, tags) VALUES (?, ?, ?)",
        ("50/30/20 rule", "Split your budget: 50% needs, 30% wants, 20% savings.", "budgeting"),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_embed():
    """Patch embedding functions to avoid needing Ollama."""
    def _embed(text, **kwargs):
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(config.EMBEDDING_DIM).tolist()

    def _embed_batch(texts, **kwargs):
        return [_embed(t) for t in texts]

    with patch("backend.llm_engine.embed", side_effect=_embed), \
         patch("backend.llm_engine.embed_batch", side_effect=_embed_batch):
        yield


@pytest.fixture
def mock_generate():
    """Patch generate functions to return canned responses."""
    with patch(
        "backend.llm_engine.generate_with_messages",
        return_value="You spent ₹150 at the canteen on Aug 20th.",
    ) as gen_mock:
        yield gen_mock


@pytest.fixture
def mock_generate_stream():
    """Patch streaming generation to yield canned tokens."""
    def _stream(*args, **kwargs):
        for word in ["You ", "spent ", "₹150."]:
            yield word

    with patch(
        "backend.llm_engine.generate_stream_with_messages",
        side_effect=_stream,
    ) as stream_mock:
        yield stream_mock


# ────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────

class TestStudentContext:
    def test_load_student_context(self, seeded_db):
        ctx = _load_student_context(seeded_db, student_id=1)
        assert "Test Student" in ctx
        assert "B.Tech CS" in ctx
        assert "15,000" in ctx or "15000" in ctx

    def test_load_missing_student(self, seeded_db):
        ctx = _load_student_context(seeded_db, student_id=999)
        assert ctx == ""


class TestRAGEnginePromptAssembly:
    def test_build_messages_structure(self, seeded_db, mock_embed):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs")
        vs.index_knowledge_base(seeded_db)
        vs.index_expenses(seeded_db, student_id=1)

        engine = RAGEngine(vector_store=vs, db_path=seeded_db)
        messages = engine._build_messages(
            "How much did I spend on food?",
            student_id=1,
            history=[],
        )

        # Should have system + user messages at minimum.
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        # System prompt should be present.
        assert "financial assistant" in messages[0]["content"].lower()
        # Context should be injected in the user message.
        assert "Context" in messages[-1]["content"]

    def test_history_truncation(self, seeded_db, mock_embed):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs2")
        engine = RAGEngine(vector_store=vs, db_path=seeded_db)

        # Create a long history.
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(30)
        ]

        messages = engine._build_messages(
            "test",
            student_id=1,
            history=long_history,
        )

        # History in the prompt should be truncated.
        history_messages = [m for m in messages if m["role"] != "system"][:-1]  # exclude final user
        assert len(history_messages) <= MAX_HISTORY_TURNS


class TestRAGEngineChat:
    def test_chat_returns_string(self, seeded_db, mock_embed, mock_generate):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs3")
        vs.index_knowledge_base(seeded_db)
        engine = RAGEngine(vector_store=vs, db_path=seeded_db)

        reply = engine.chat("How much did I spend?", student_id=1)
        assert isinstance(reply, str)
        assert "150" in reply

    def test_chat_persists_history(self, seeded_db, mock_embed, mock_generate):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs4")
        engine = RAGEngine(vector_store=vs, db_path=seeded_db)

        engine.chat("Hello", student_id=1)

        history = engine.load_history(student_id=1)
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_chat_stream_yields_tokens(self, seeded_db, mock_embed, mock_generate_stream):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs5")
        engine = RAGEngine(vector_store=vs, db_path=seeded_db)

        tokens = list(engine.chat_stream("Test", student_id=1))
        assert len(tokens) == 3
        assert "".join(tokens) == "You spent ₹150."

    def test_clear_history(self, seeded_db, mock_embed, mock_generate):
        vs = VectorStore(persist_dir=seeded_db.parent / "vs6")
        engine = RAGEngine(vector_store=vs, db_path=seeded_db)

        engine.chat("Hello", student_id=1)
        assert len(engine.load_history(1)) == 2

        engine.clear_history(1)
        assert len(engine.load_history(1)) == 0
