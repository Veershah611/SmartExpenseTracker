"""
frontend/state.py
=================
Global session state and cached data access. **Owned by the Core Integrator.**

Streamlit re-executes the entire script on every widget interaction. Two
consequences drive this module:

1. Anything that must survive a rerun (the selected student, the chat
   transcript) has to live in ``st.session_state``.
2. Anything expensive (a SQLite read, an analytics pass) has to be cached, or
   the app re-queries the database on every keystroke.

Centralising both here means no tab has to think about it, and there is exactly
one place to clear when data changes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from backend import integration  # noqa: E402

# Session keys, named once so a typo cannot silently create a second key.
KEY_STUDENT_ID = "student_id"
KEY_CHAT = "chat_messages"
KEY_DATE_RANGE = "date_range"
KEY_LAST_ERROR = "last_error"


# --------------------------------------------------------------------------- #
# Session bootstrap
# --------------------------------------------------------------------------- #
def init_session() -> None:
    """
    Seed session state on first run. Idempotent -- safe to call every rerun.

    Called once at the top of ``app.py`` before any tab renders, so no tab has
    to defend against a missing key.
    """
    st.session_state.setdefault(KEY_STUDENT_ID, config.DEFAULT_STUDENT_ID)
    st.session_state.setdefault(KEY_CHAT, [])
    st.session_state.setdefault(KEY_DATE_RANGE, None)
    st.session_state.setdefault(KEY_LAST_ERROR, None)


def get_student_id() -> int:
    """The 'logged-in' mock student. Single source of truth for every tab."""
    return int(st.session_state.get(KEY_STUDENT_ID, config.DEFAULT_STUDENT_ID))


def set_student_id(student_id: int) -> None:
    """
    Switch student and clear everything scoped to the previous one.

    Without the reset, switching students would leave the old chat transcript
    and date filter in place -- which looks like a bug during a demo.
    """
    if student_id != get_student_id():
        st.session_state[KEY_STUDENT_ID] = int(student_id)
        st.session_state[KEY_CHAT] = []
        st.session_state[KEY_DATE_RANGE] = None
        clear_data_cache()


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
# Every loader below goes through the integration seam rather than importing
# `backend.database` directly, so a teammate replacing that module changes
# nothing here.

@st.cache_data(ttl=300, show_spinner=False)
def load_students() -> pd.DataFrame:
    """All personas for the sidebar selector."""
    return integration.feature("data").call("get_students")


@st.cache_data(ttl=300, show_spinner=False)
def load_student(student_id: int) -> dict[str, Any] | None:
    """One student record."""
    return integration.feature("data").call("get_student", student_id)


@st.cache_data(ttl=300, show_spinner=False)
def load_expenses(
    student_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Expenses for one student, optionally date-filtered.

    Arguments are plain strings rather than ``date`` objects so Streamlit's
    cache key stays stable -- two equal ``date`` objects hash identically, but
    strings make the cache key obvious when debugging a stale-data problem.
    """
    return integration.feature("data").call(
        "get_expenses", student_id, start_date, end_date
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_budgets(student_id: int) -> pd.DataFrame:
    """
    Per-category monthly budget limits.

    Budgets are not part of the ``data`` contract's required set, so this probes
    for the function rather than assuming it -- a teammate shipping a minimal
    module still leaves the app working, just without budget rings.
    """
    module = integration.feature("data").module
    if module is not None and hasattr(module, "get_budgets"):
        return module.get_budgets(student_id)
    return pd.DataFrame(columns=["month", "category", "limit_amount"])


@st.cache_data(ttl=300, show_spinner=False)
def load_goals(student_id: int) -> pd.DataFrame:
    """Savings goals, if the data module provides them."""
    module = integration.feature("data").module
    if module is not None and hasattr(module, "get_savings_goals"):
        return module.get_savings_goals(student_id)
    return pd.DataFrame(columns=["title", "target_amount", "saved_amount", "progress_pct"])


def clear_data_cache() -> None:
    """
    Drop every cached read.

    Called after any write (a new expense, a receipt scan, a quick log) so the
    dashboard reflects it immediately instead of showing stale figures for the
    next five minutes.
    """
    load_students.clear()
    load_student.clear()
    load_expenses.clear()
    load_budgets.clear()
    load_goals.clear()


# --------------------------------------------------------------------------- #
# Chat transcript
# --------------------------------------------------------------------------- #
def get_chat() -> list[dict[str, str]]:
    """Current conversation, as ``[{'role': ..., 'content': ...}]``."""
    return st.session_state.setdefault(KEY_CHAT, [])


def append_chat(role: str, content: str) -> None:
    """Add one turn to the in-memory transcript."""
    get_chat().append({"role": role, "content": content})


def clear_chat() -> None:
    """Reset the conversation, in session and in the database if available."""
    st.session_state[KEY_CHAT] = []
    module = integration.feature("data").module
    if module is not None and hasattr(module, "clear_chat_history"):
        try:
            module.clear_chat_history(get_student_id())
        except Exception:  # noqa: BLE001 -- clearing history must never block the UI
            pass


# --------------------------------------------------------------------------- #
# Vector store bootstrap
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Indexing your transactions for search...")
def get_indexed_vector_store():
    """
    Build the vector store and index it once per session.

    Without this the store is empty, retrieval returns nothing, and the
    assistant answers from general knowledge instead of the student's data --
    which is indistinguishable from the model making things up. Indexing has to
    happen somewhere, and startup is the only place that guarantees it.

    ``cache_resource`` (not ``cache_data``) because the store holds an open
    ChromaDB handle and must be shared, not copied, across reruns.

    Returns ``None`` if anything fails: semantic search is an enhancement, and
    losing it must not take the chat tab down.
    """
    try:
        from backend.vector_store import VectorStore

        store = VectorStore()
        if store.count(config.RAG_COLLECTION_KNOWLEDGE) == 0:
            store.index_knowledge_base()
        if store.count(config.RAG_COLLECTION_EXPENSES) == 0:
            store.index_expenses()
        return store
    except Exception:  # noqa: BLE001
        return None


def reindex_expenses() -> None:
    """Re-index after a write so new transactions become searchable."""
    store = get_indexed_vector_store()
    if store is None:
        return
    try:
        store.index_expenses()
    except Exception:  # noqa: BLE001
        pass
