"""
config.py
=========
Single source of truth for every path, model name and tunable constant used by
the Smart Expense Tracker.

Design rationale
----------------
Every other module imports from here instead of hard-coding strings. That means
a judge/demo machine can change the Ollama model or the DB location by editing
one ``.env`` file -- no code changes, no hunting through modules.

Values resolve in this order:  environment variable  ->  .env file  ->  default.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# .env loading (optional dependency -- the app must still boot without it)
# --------------------------------------------------------------------------- #
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is a soft dependency
    # Not fatal: we simply fall back to real environment variables + defaults.
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# BASE_DIR is resolved from this file's location so the app works no matter
# which directory `streamlit run` was launched from.
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
RECEIPTS_DIR: Path = DATA_DIR / "receipts"

DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "data/expenses.db")
VECTOR_STORE_PATH: Path = BASE_DIR / os.getenv("VECTOR_STORE_PATH", "data/vector_store")

# Create the directory tree at import time so no module ever has to guard for it.
for _directory in (DATA_DIR, RECEIPTS_DIR, VECTOR_STORE_PATH):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ollama / LLM settings
# --------------------------------------------------------------------------- #
# The integrator owns this connection for the whole team, so both local
# runtimes are supported. "auto" probes Ollama first, then LM Studio, so a
# teammate running either one needs no config change.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")   # auto | ollama | lmstudio

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LMSTUDIO_HOST: str = os.getenv("LMSTUDIO_HOST", "http://localhost:1234")

# llama3.2 (3B) is the sweet spot for a laptop demo: ~2 GB, fast first token.
# Swap to "llama3.1:8b", "gemma3:4b" or "mistral" on a stronger machine.
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")

# nomic-embed-text: 768-dim, ~270 MB, far lighter than pulling in PyTorch via
# sentence-transformers just to embed a few hundred expense rows.
#   ollama pull nomic-embed-text
# If it is not installed, vector_store.py falls back to embedding with the chat
# model (lower quality, no extra download), and then to a lexical index if
# Ollama is unreachable entirely. The demo never hard-fails on a missing model.
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_DIM: int = 768

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # low = factual
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


# --------------------------------------------------------------------------- #
# RAG settings
# --------------------------------------------------------------------------- #
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))        # chunks fed to the LLM
RAG_COLLECTION_EXPENSES: str = "expense_documents"
RAG_COLLECTION_KNOWLEDGE: str = "finance_knowledge"


# --------------------------------------------------------------------------- #
# Demo / domain defaults
# --------------------------------------------------------------------------- #
DEFAULT_STUDENT_ID: int = int(os.getenv("DEFAULT_STUDENT_ID", "1"))
CURRENCY_SYMBOL: str = os.getenv("CURRENCY_SYMBOL", "\u20b9")  # Indian Rupee

# Canonical spending categories. The tuple is (name, emoji, typical monthly
# share of a student's budget) -- the share seeds the mock data generator and
# powers the "you are over/under the typical student" insight.
EXPENSE_CATEGORIES: list[tuple[str, str, float]] = [
    ("Food & Canteen",    "\U0001F35B", 0.28),
    ("Hostel & Rent",     "\U0001F3E0", 0.22),
    ("Transport",         "\U0001F68C", 0.10),
    ("Books & Stationery","\U0001F4DA", 0.08),
    ("Academics & Fees",  "\U0001F393", 0.09),
    ("Entertainment",     "\U0001F3AC", 0.07),
    ("Shopping",          "\U0001F6CD", 0.06),
    ("Health & Medical",  "\U0001F48A", 0.04),
    ("Mobile & Internet", "\U0001F4F1", 0.04),
    ("Miscellaneous",     "\U0001F4E6", 0.02),
]

PAYMENT_MODES: list[str] = ["UPI", "Cash", "Debit Card", "Credit Card", "Net Banking"]

# Chart palette -- colour-blind safe, consistent across every tab.
CHART_COLORS: list[str] = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


def as_currency(amount: float) -> str:
    """Format a number as Indian-style currency, e.g. ``1234.5 -> '₹1,234.50'``."""
    try:
        return f"{CURRENCY_SYMBOL}{float(amount):,.2f}"
    except (TypeError, ValueError):
        return f"{CURRENCY_SYMBOL}0.00"
