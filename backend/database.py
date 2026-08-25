"""
backend/database.py
===================
The only module in the project that talks to SQLite directly.

Everything above this layer (analytics, insights, RAG, the Streamlit UI) works
with pandas DataFrames or plain dicts and never sees a cursor. That boundary is
what makes the rest of the codebase testable without a database fixture, and it
means swapping SQLite for Postgres later would touch this file alone.

Conventions
-----------
* Connections are short-lived and always closed -- see :func:`get_connection`.
* ``row_factory`` is ``sqlite3.Row`` so rows behave like dicts.
* Foreign keys are enforced on every connection (SQLite disables them by default).
* Reads return DataFrames; writes return the new row id.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# Allow `import config` when this module is imported from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


class DatabaseError(RuntimeError):
    """Raised when the database is missing or a query fails.

    Wrapping ``sqlite3.Error`` in a project-specific exception keeps SQLite
    details from leaking into the UI layer, which only needs to know that the
    data could not be loaded.
    """


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #
@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """
    Yield a configured SQLite connection and guarantee it is closed.

    Usage::

        with get_connection() as conn:
            conn.execute(...)

    Commits on clean exit, rolls back if the block raises. Streamlit reruns the
    whole script on every interaction, so leaking connections here would pile up
    file handles fast -- hence the context manager rather than a module global.
    """
    path = Path(db_path) if db_path else config.DB_PATH

    if not path.exists():
        raise DatabaseError(
            f"Database not found at {path}.\n"
            "Run:  python backend/scripts/generate_mock_data.py --force"
        )

    conn = None
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def database_exists(db_path: Path | str | None = None) -> bool:
    """Cheap check the UI uses to show a 'generate demo data' prompt instead of a stack trace."""
    return Path(db_path or config.DB_PATH).exists()


def _read_sql(query: str, params: tuple | dict = ()) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Central place for read error handling."""
    with get_connection() as conn:
        try:
            return pd.read_sql_query(query, conn, params=params)
        except (pd.errors.DatabaseError, sqlite3.Error) as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Students
# --------------------------------------------------------------------------- #
def get_students() -> pd.DataFrame:
    """All student personas, for the sidebar selector."""
    return _read_sql(
        """SELECT id, name, roll_no, course, semester, hostel_resident,
                  monthly_budget, persona_note
           FROM students ORDER BY id"""
    )


def get_student(student_id: int) -> dict[str, Any] | None:
    """One student as a plain dict, or ``None`` if the id does not exist."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, name, roll_no, course, semester, hostel_resident,
                      monthly_budget, persona_note
               FROM students WHERE id = ?""",
            (student_id,),
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #
def get_categories() -> pd.DataFrame:
    """Canonical category list with icons and benchmark shares."""
    return _read_sql(
        "SELECT id, name, icon, typical_share FROM categories ORDER BY name"
    )


def get_category_map() -> dict[str, int]:
    """``{category_name: id}`` -- used when inserting an expense by name."""
    return {row.name: int(row.id) for row in get_categories().itertuples()}


# --------------------------------------------------------------------------- #
# Expenses
# --------------------------------------------------------------------------- #
# Shared SELECT body. Kept in one constant so the column set returned to the
# analytics layer is identical no matter which helper was called.
_EXPENSE_SELECT = """
    SELECT e.id,
           e.student_id,
           e.txn_date,
           e.amount,
           e.merchant,
           e.description,
           e.payment_mode,
           e.source,
           e.is_recurring,
           e.receipt_path,
           c.name AS category,
           c.icon AS category_icon
    FROM expenses e
    JOIN categories c ON c.id = e.category_id
"""


def get_expenses(
    student_id: int,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch expenses for one student, optionally filtered by date range and category.

    Returns a DataFrame with ``txn_date`` already parsed to datetime64 -- every
    consumer wants dates, not strings, and parsing once here avoids repeating it
    in each analytics function.

    Filters are composed as parameterised SQL rather than filtered in pandas so
    the ``(student_id, txn_date)`` index does the work.
    """
    query = _EXPENSE_SELECT + " WHERE e.student_id = ?"
    params: list[Any] = [student_id]

    if start_date:
        query += " AND e.txn_date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND e.txn_date <= ?"
        params.append(str(end_date))
    if categories:
        # Build the right number of placeholders; never interpolate values.
        placeholders = ",".join("?" for _ in categories)
        query += f" AND c.name IN ({placeholders})"
        params.extend(categories)

    query += " ORDER BY e.txn_date DESC, e.id DESC"

    frame = _read_sql(query, tuple(params))
    if not frame.empty:
        frame["txn_date"] = pd.to_datetime(frame["txn_date"])
    return frame


def add_expense(
    student_id: int,
    category: str,
    amount: float,
    merchant: str,
    txn_date: str | date | None = None,
    description: str = "",
    payment_mode: str = "UPI",
    source: str = "manual",
    receipt_path: str | None = None,
) -> int:
    """
    Insert one expense and return its new id.

    Validates before touching the database so the caller gets a clear message
    rather than an opaque CHECK-constraint failure.
    """
    if amount is None or float(amount) <= 0:
        raise ValueError("Amount must be greater than zero.")
    if not merchant or not str(merchant).strip():
        raise ValueError("Merchant is required.")

    category_map = get_category_map()
    if category not in category_map:
        raise ValueError(
            f"Unknown category {category!r}. Expected one of: "
            + ", ".join(sorted(category_map))
        )

    txn_date = str(txn_date or date.today())

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO expenses
                   (student_id, category_id, txn_date, amount, merchant,
                    description, payment_mode, source, receipt_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (student_id, category_map[category], txn_date, float(amount),
             str(merchant).strip(), description, payment_mode, source, receipt_path),
        )
        return int(cursor.lastrowid)


def delete_expense(expense_id: int, student_id: int) -> bool:
    """
    Delete one expense. ``student_id`` is part of the WHERE clause so a stale
    id from another student's session can never delete the wrong row.

    Returns True if a row was actually removed.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM expenses WHERE id = ? AND student_id = ?",
            (expense_id, student_id),
        )
        return cursor.rowcount > 0


def get_date_range(student_id: int) -> tuple[date, date]:
    """
    Earliest and latest transaction dates for a student.

    The dashboard uses this to seed its date pickers, so filters open on real
    data instead of an arbitrary window that might be empty.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MIN(txn_date) AS first, MAX(txn_date) AS last "
            "FROM expenses WHERE student_id = ?",
            (student_id,),
        ).fetchone()

    today = date.today()
    if not row or not row["first"]:
        return today - timedelta(days=30), today
    return date.fromisoformat(row["first"]), date.fromisoformat(row["last"])


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #
def get_budgets(student_id: int, month: str | None = None) -> pd.DataFrame:
    """
    Per-category budget limits. ``month`` is a ``'YYYY-MM'`` key; omit it to get
    every month.
    """
    query = """
        SELECT b.month, c.name AS category, b.limit_amount
        FROM budgets b
        JOIN categories c ON c.id = b.category_id
        WHERE b.student_id = ?
    """
    params: list[Any] = [student_id]
    if month:
        query += " AND b.month = ?"
        params.append(month)
    return _read_sql(query + " ORDER BY b.month, c.name", tuple(params))


# --------------------------------------------------------------------------- #
# Savings goals
# --------------------------------------------------------------------------- #
def get_savings_goals(student_id: int) -> pd.DataFrame:
    """Goals with a computed ``progress_pct`` column for the dashboard widget."""
    frame = _read_sql(
        """SELECT id, title, target_amount, saved_amount, deadline
           FROM savings_goals WHERE student_id = ? ORDER BY deadline""",
        (student_id,),
    )
    if not frame.empty:
        # Guard against a zero target sneaking in and raising ZeroDivisionError.
        frame["progress_pct"] = (
            frame["saved_amount"] / frame["target_amount"].replace(0, pd.NA) * 100
        ).fillna(0).clip(0, 100).round(1)
    return frame


# --------------------------------------------------------------------------- #
# Knowledge base (RAG grounding)
# --------------------------------------------------------------------------- #
def get_knowledge_base() -> pd.DataFrame:
    """Curated finance advice, embedded into the vector store at startup."""
    return _read_sql("SELECT id, topic, content, tags FROM knowledge_base ORDER BY id")


# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
def get_chat_history(student_id: int, limit: int = 50) -> list[dict[str, str]]:
    """
    Recent conversation turns, oldest first.

    SQLite gives us the *newest* N rows efficiently, so we select DESC with a
    LIMIT and reverse in Python -- the alternative would scan the whole table.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT role, content FROM chat_history
               WHERE student_id = ? ORDER BY id DESC LIMIT ?""",
            (student_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def add_chat_message(student_id: int, role: str, content: str) -> int:
    """Persist one chat turn so a Streamlit rerun does not wipe the conversation."""
    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_history (student_id, role, content) VALUES (?, ?, ?)",
            (student_id, role, content),
        )
        return int(cursor.lastrowid)


def clear_chat_history(student_id: int) -> int:
    """Wipe one student's conversation. Returns the number of rows deleted."""
    with get_connection() as conn:
        return conn.execute(
            "DELETE FROM chat_history WHERE student_id = ?", (student_id,)
        ).rowcount
