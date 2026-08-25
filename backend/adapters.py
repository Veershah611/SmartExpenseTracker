"""
backend/adapters.py
===================
Contract-shaped wrappers around teammate modules whose real signatures differ.
**Owned by the Core Integrator.**

Why this file exists
--------------------
Two modules arrived built to a different convention than ``INTEGRATION.md``
specifies. Rather than edit code another role owns, or force a rewrite the night
before a demo, the differences are bridged here.

The divergence is not a mistake on their part -- it is a genuine design choice.
Their functions take ``(student_id, sqlite3.Connection)`` and query the database
themselves; the shell passes a DataFrame it has already loaded and cached. Both
work. Theirs re-queries SQLite per chart; the shell reads once per rerun and
shares it. This module absorbs the difference so neither side has to move.

Registered as the *fallback* for the affected contracts, which means it is used
only while the teammate's module lacks the contracted function. The moment they
rename or add it, ``integration.py`` binds their version directly and these
wrappers stop being called.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """
    Short-lived connection for modules that expect to be handed one.

    Opened and closed per call rather than cached: Streamlit reruns constantly,
    and a module-level connection shared across reruns is the classic source of
    "database is locked" during a live demo.
    """
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Predictive Broke Alert  (Analytics & Forecasting Developer)
# --------------------------------------------------------------------------- #
# Their module exposes generate_broke_alert(student_id, conn) and returns
# {burn_data, alert_text, source, prompt}. The shell's contract is
# predict_broke_alert(expenses, student) -> {message, severity}.

# Their four-level severity mapped onto the three the UI renders. 'safe' and
# 'warning' both stay non-red: a projection Rs 12 over a Rs 15,000 budget is a
# rounding error, and colouring it like an emergency trains students to ignore
# the alert that matters.
_SEVERITY_MAP = {
    "safe": "info",
    "warning": "warning",
    "danger": "danger",
    "critical": "danger",
}


def predict_broke_alert(expenses: pd.DataFrame, student: dict) -> dict[str, Any]:
    """
    Contract-shaped Broke Alert, delegating to ``forecasting.generate_broke_alert``.

    ``expenses`` is accepted to satisfy the contract but not forwarded -- their
    implementation reads the database itself. Keeping the parameter means the
    call site does not change when they adopt the contracted signature.
    """
    from backend import forecasting

    student_id = int(student["id"])
    with _connection() as conn:
        result = forecasting.generate_broke_alert(student_id=student_id, conn=conn)

    burn = result.get("burn_data") or {}
    return {
        "message": result.get("alert_text", ""),
        "severity": _SEVERITY_MAP.get(burn.get("severity", ""), "info"),
        # Passed through so the UI can label an LLM-written alert differently
        # from the rule-based fallback.
        "source": result.get("source", "unknown"),
        "burn_data": burn,
    }


# --------------------------------------------------------------------------- #
# Subscription Ghost Hunter  (Data & Database Engineer -- not yet delivered)
# --------------------------------------------------------------------------- #
def find_recurring_charges(expenses: pd.DataFrame) -> pd.DataFrame:
    """Delegate to the reference detector until Role 2 ships theirs."""
    from backend import analytics

    return analytics.detect_recurring(expenses)


# --------------------------------------------------------------------------- #
# Chat client bridge  (Vision & OCR Specialist)
# --------------------------------------------------------------------------- #
class LLMEngineChatClient:
    """
    Presents ``llm_engine`` through the ``ChatClient`` protocol ocr_engine expects.

    Their module falls back to constructing ``ollama.Client`` directly when no
    client is injected. Injecting this instead means receipt parsing goes
    through the shared connection, which brings three things their direct call
    does not have: LM Studio support, the shared timeout policy, and the JSON
    salvaging in ``chat_json`` for models that wrap output in prose or fences.
    """

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        """Accept Ollama-style kwargs, return an Ollama-style response dict."""
        from backend import llm_engine

        messages = kwargs.get("messages") or []
        options = kwargs.get("options") or {}

        content = llm_engine.chat(
            messages,
            model=kwargs.get("model"),
            temperature=options.get("temperature"),
            json_mode=kwargs.get("format") == "json",
        )
        # ocr_engine reads response["message"]["content"].
        return {"message": {"content": content}}


# --------------------------------------------------------------------------- #
# Conversational assistant + Quick Log  (RAG & NLP Developer)
# --------------------------------------------------------------------------- #
# They shipped a stateful RAGEngine class and nlp_quick_log.parse_expense();
# the contract asks for module-level answer_question() and parse_quick_log().

_RAG_ENGINE: Any = None


def _rag_engine() -> Any:
    """Build the RAGEngine once and reuse it -- it holds the vector store open."""
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        from backend.rag_engine import RAGEngine
        from backend.vector_store import VectorStore

        _RAG_ENGINE = RAGEngine(vector_store=VectorStore())
    return _RAG_ENGINE


def answer_question(question: str, student_id: int) -> str:
    """Contract-shaped wrapper over ``RAGEngine.chat``."""
    return _rag_engine().chat(question, student_id=student_id)


def parse_quick_log(sentence: str) -> dict[str, Any]:
    """
    Contract-shaped wrapper over ``nlp_quick_log.parse_expense``.

    Theirs returns an INSERT-ready row keyed by ``category_id``; the
    confirmation form needs a category *name*, so the id is resolved back here.
    """
    from backend import nlp_quick_log

    parsed = nlp_quick_log.parse_expense(sentence)

    category_name = ""
    category_id = parsed.get("category_id")
    if category_id is not None:
        with _connection() as conn:
            row = conn.execute(
                "SELECT name FROM categories WHERE id = ?", (category_id,)
            ).fetchone()
        if row:
            category_name = row["name"]

    return {
        "amount": parsed.get("amount", 0.0),
        "category": category_name,
        "merchant": parsed.get("merchant", ""),
        "description": parsed.get("description", ""),
        "txn_date": parsed.get("txn_date"),
    }


def describe_backend() -> dict[str, str]:
    """
    Vector-store diagnostics for the sidebar.

    Their VectorStore does not expose the reporting helper the shell's original
    one did, so the state is inspected here rather than added to their file.
    """
    try:
        import chromadb  # noqa: F401

        index = "ChromaDB"
    except Exception:  # noqa: BLE001
        index = "NumPy (fallback)"

    from backend import llm_engine

    status = llm_engine.get_status()
    if not status.available:
        embeddings = "unavailable (no LLM running)"
    else:
        bare = {name.split(":")[0] for name in status.models}
        if config.OLLAMA_EMBED_MODEL.split(":")[0] in bare:
            embeddings = f"Ollama: {config.OLLAMA_EMBED_MODEL}"
        else:
            embeddings = "lexical hashing (no embed model pulled)"

    return {"index": index, "embeddings": embeddings, "mode": "adapter"}
