"""
backend/nlp_quick_log.py
========================
**Natural Language Quick Log** — the creative feature for Role #5.

Converts messy, casual text like:

    "Spent 50 on chai at canteen"
    "Auto to college 120 yesterday"
    "Bought books from Amazon for 850 using debit card"

into a validated, INSERT-ready dictionary that maps directly to the
``expenses`` table schema.

Pipeline
--------
1. Build a few-shot prompt with 6 diverse examples.
2. Send to Ollama (``llama3.2:3b``).
3. Extract the JSON from the LLM response (handles markdown fences,
   trailing text, partial output).
4. Validate every field: amount > 0, date is real, category matches one
   of the 10 canonical names.
5. Return a dict ready for ``INSERT INTO expenses``.

If the first attempt produces bad JSON, a single retry with a stricter
prompt is made before raising ``ParseError``.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import config
from backend import llm_engine

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Custom exception
# ────────────────────────────────────────────────────────────────────

class ParseError(ValueError):
    """Raised when the natural-language input cannot be parsed into a valid expense."""


# ────────────────────────────────────────────────────────────────────
# Category matching
# ────────────────────────────────────────────────────────────────────

# Build lookup structures once at import time.
_CANONICAL_CATEGORIES: list[str] = [name for name, _icon, _share in config.EXPENSE_CATEGORIES]

# Keyword → canonical category mapping for fuzzy matching.
_CATEGORY_KEYWORDS: dict[str, str] = {
    # Food & Canteen
    "food": "Food & Canteen", "canteen": "Food & Canteen", "mess": "Food & Canteen",
    "lunch": "Food & Canteen", "dinner": "Food & Canteen", "breakfast": "Food & Canteen",
    "snack": "Food & Canteen", "chai": "Food & Canteen", "tea": "Food & Canteen",
    "coffee": "Food & Canteen", "swiggy": "Food & Canteen", "zomato": "Food & Canteen",
    "restaurant": "Food & Canteen", "pizza": "Food & Canteen", "biryani": "Food & Canteen",
    "juice": "Food & Canteen", "eat": "Food & Canteen", "meal": "Food & Canteen",
    "delivery": "Food & Canteen", "order": "Food & Canteen",
    # Hostel & Rent
    "hostel": "Hostel & Rent", "rent": "Hostel & Rent", "room": "Hostel & Rent",
    "accommodation": "Hostel & Rent", "pg": "Hostel & Rent",
    # Transport
    "transport": "Transport", "auto": "Transport", "cab": "Transport",
    "uber": "Transport", "ola": "Transport", "rapido": "Transport",
    "bus": "Transport", "train": "Transport", "fuel": "Transport",
    "petrol": "Transport", "metro": "Transport", "travel": "Transport",
    "commute": "Transport", "ride": "Transport", "brts": "Transport",
    # Books & Stationery
    "book": "Books & Stationery", "stationery": "Books & Stationery",
    "notebook": "Books & Stationery", "pen": "Books & Stationery",
    "xerox": "Books & Stationery", "photocopy": "Books & Stationery",
    "print": "Books & Stationery", "textbook": "Books & Stationery",
    # Academics & Fees
    "fee": "Academics & Fees", "tuition": "Academics & Fees",
    "exam": "Academics & Fees", "course": "Academics & Fees",
    "workshop": "Academics & Fees", "certification": "Academics & Fees",
    "coursera": "Academics & Fees", "udemy": "Academics & Fees",
    # Entertainment
    "movie": "Entertainment", "netflix": "Entertainment",
    "spotify": "Entertainment", "game": "Entertainment", "gaming": "Entertainment",
    "cricket": "Entertainment", "outing": "Entertainment",
    "pvr": "Entertainment", "cinema": "Entertainment", "party": "Entertainment",
    # Shopping
    "shopping": "Shopping", "amazon": "Shopping", "flipkart": "Shopping",
    "myntra": "Shopping", "clothes": "Shopping", "shoes": "Shopping",
    "online order": "Shopping", "decathlon": "Shopping",
    # Health & Medical
    "health": "Health & Medical", "medical": "Health & Medical",
    "medicine": "Health & Medical", "pharmacy": "Health & Medical",
    "doctor": "Health & Medical", "gym": "Health & Medical",
    "hospital": "Health & Medical",
    # Mobile & Internet
    "mobile": "Mobile & Internet", "recharge": "Mobile & Internet",
    "jio": "Mobile & Internet", "airtel": "Mobile & Internet",
    "internet": "Mobile & Internet", "wifi": "Mobile & Internet",
    "data pack": "Mobile & Internet",
    # Miscellaneous
    "laundry": "Miscellaneous", "salon": "Miscellaneous",
    "haircut": "Miscellaneous", "gift": "Miscellaneous",
    "charity": "Miscellaneous",
}


def _fuzzy_match_category(raw: str) -> str:
    """
    Map a raw category string (from LLM output or user text) to one of
    the 10 canonical category names.

    Strategy:
    1. Exact match (case-insensitive).
    2. Substring match against canonical names.
    3. Keyword lookup.
    4. Fallback to "Miscellaneous".
    """
    if not raw:
        return "Miscellaneous"

    cleaned = raw.strip()

    # 1. Exact match (case-insensitive).
    for canonical in _CANONICAL_CATEGORIES:
        if cleaned.lower() == canonical.lower():
            return canonical

    # 2. Substring match (e.g. "food" matches "Food & Canteen").
    lower = cleaned.lower()
    for canonical in _CANONICAL_CATEGORIES:
        if lower in canonical.lower() or canonical.lower() in lower:
            return canonical

    # 3. Keyword lookup.
    for keyword, canonical in _CATEGORY_KEYWORDS.items():
        if keyword in lower:
            return canonical

    # 4. Fallback.
    logger.warning("Could not match category '%s' — defaulting to Miscellaneous.", raw)
    return "Miscellaneous"


def _resolve_category_id(category_name: str, db_path: Path | str) -> int:
    """Look up the category_id for a canonical category name."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (category_name,)
        ).fetchone()
    finally:
        conn.close()
    if row:
        return row[0]
    # Should never happen with seeded data, but be safe.
    raise ParseError(f"Category '{category_name}' not found in the database.")


# ────────────────────────────────────────────────────────────────────
# Date resolution
# ────────────────────────────────────────────────────────────────────

def _resolve_date(raw: str | None) -> str:
    """
    Convert a date string to ISO format (``YYYY-MM-DD``).

    Handles:
    - ``None`` / empty → today
    - "today" → today
    - "yesterday" → yesterday
    - "day before yesterday" → 2 days ago
    - ISO format pass-through
    - Common formats: DD/MM/YYYY, DD-MM-YYYY
    """
    if not raw or not raw.strip():
        return date.today().isoformat()

    cleaned = raw.strip().lower()

    if cleaned == "today":
        return date.today().isoformat()
    if cleaned == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    if cleaned in ("day before yesterday", "day before"):
        return (date.today() - timedelta(days=2)).isoformat()

    # Try ISO format first (YYYY-MM-DD).
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    # DD/MM/YYYY or DD-MM-YYYY.
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue

    # If nothing works, default to today.
    logger.warning("Could not parse date '%s' — defaulting to today.", raw)
    return date.today().isoformat()


# ────────────────────────────────────────────────────────────────────
# JSON extraction from LLM output
# ────────────────────────────────────────────────────────────────────

def _extract_json(raw_response: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from the LLM's raw text output.

    Handles common LLM quirks:
    - JSON wrapped in ```json ... ``` markdown fences
    - Trailing explanation text after the JSON
    - Extra whitespace / newlines
    """
    text = raw_response.strip()

    # Strip markdown code fences.
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the first { ... } block.
    brace_start = text.find("{")
    if brace_start == -1:
        raise ParseError(f"No JSON object found in LLM response:\n{raw_response[:300]}")

    # Walk forward to find the matching closing brace.
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[brace_start : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as exc:
                    raise ParseError(
                        f"Invalid JSON in LLM response: {exc}\n"
                        f"Extracted: {json_str[:300]}"
                    ) from exc

    raise ParseError(f"Unmatched braces in LLM response:\n{raw_response[:300]}")


# ────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────

def _build_parse_prompt(user_text: str) -> str:
    """
    Build a few-shot prompt that reliably makes the LLM output a JSON
    object from natural-language expense descriptions.
    """
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()

    return f"""\
You are a precise expense parser.  Convert the user's natural-language \
description into a JSON object with EXACTLY these keys:
  "amount"       — number (positive, no currency symbol)
  "category"     — string (one of: Food & Canteen, Hostel & Rent, Transport, \
Books & Stationery, Academics & Fees, Entertainment, Shopping, \
Health & Medical, Mobile & Internet, Miscellaneous)
  "merchant"     — string (best guess for the shop/vendor/app name)
  "description"  — string (short summary of what was bought)
  "payment_mode" — string (one of: UPI, Cash, Debit Card, Credit Card, Net Banking; \
default to UPI if not mentioned)
  "date"         — string in YYYY-MM-DD format (use "{today_str}" if not mentioned; \
"yesterday" means "{yesterday_str}")

Respond with ONLY the JSON object — no explanation, no markdown fences.

Example 1:
Input: "Spent 50 on chai at canteen"
Output: {{"amount": 50, "category": "Food & Canteen", "merchant": "Campus Canteen", "description": "Chai", "payment_mode": "UPI", "date": "{today_str}"}}

Example 2:
Input: "Auto to college 120 yesterday"
Output: {{"amount": 120, "category": "Transport", "merchant": "Ola Auto", "description": "Auto to campus", "payment_mode": "Cash", "date": "{yesterday_str}"}}

Example 3:
Input: "Bought books from Amazon for 850 using debit card"
Output: {{"amount": 850, "category": "Books & Stationery", "merchant": "Amazon Books", "description": "Books purchase", "payment_mode": "Debit Card", "date": "{today_str}"}}

Example 4:
Input: "Netflix subscription 499"
Output: {{"amount": 499, "category": "Entertainment", "merchant": "Netflix", "description": "Netflix monthly plan", "payment_mode": "Credit Card", "date": "{today_str}"}}

Example 5:
Input: "Paid 4200 for hostel mess fees on 15/08/2025"
Output: {{"amount": 4200, "category": "Hostel & Rent", "merchant": "Mess Fee Counter", "description": "Monthly mess and hostel charges", "payment_mode": "Net Banking", "date": "2025-08-15"}}

Example 6:
Input: "Jio recharge 299 cash"
Output: {{"amount": 299, "category": "Mobile & Internet", "merchant": "Jio Recharge", "description": "Monthly mobile recharge", "payment_mode": "Cash", "date": "{today_str}"}}

Now parse this:
Input: "{user_text}"
Output:"""


def _build_retry_prompt(user_text: str, failed_output: str) -> str:
    """Stricter retry prompt when the first attempt returns bad JSON."""
    today_str = date.today().isoformat()
    return f"""\
The previous attempt to parse this expense description failed.

Input: "{user_text}"
Previous (bad) output: {failed_output[:200]}

Try again.  Return ONLY a valid JSON object with these keys:
  amount (number), category (string), merchant (string),
  description (string), payment_mode (string), date (string YYYY-MM-DD).

Use "{today_str}" for today's date if not mentioned.
Output:"""


# ────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────

def _validate_and_normalise(
    parsed: dict[str, Any],
    student_id: int,
    db_path: Path | str,
) -> dict[str, Any]:
    """
    Validate and normalise every field of the parsed expense dict.

    Returns a dict ready for ``INSERT INTO expenses``.

    Raises :class:`ParseError` if critical fields are missing or invalid.
    """
    # ── amount ───────────────────────────────────────────────────
    raw_amount = parsed.get("amount")
    if raw_amount is None:
        raise ParseError("Missing 'amount' in parsed output.")
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"Invalid amount: {raw_amount}") from exc
    if amount <= 0:
        raise ParseError(f"Amount must be positive, got {amount}.")

    # ── category ─────────────────────────────────────────────────
    raw_category = str(parsed.get("category", ""))
    category = _fuzzy_match_category(raw_category)
    category_id = _resolve_category_id(category, db_path)

    # ── merchant ─────────────────────────────────────────────────
    merchant = str(parsed.get("merchant", "Unknown")).strip()
    if not merchant:
        merchant = "Unknown"

    # ── description ──────────────────────────────────────────────
    description = str(parsed.get("description", "")).strip()

    # ── payment_mode ─────────────────────────────────────────────
    raw_mode = str(parsed.get("payment_mode", "UPI")).strip()
    # Normalise to one of the canonical modes.
    mode_lower = raw_mode.lower()
    payment_mode = "UPI"  # default
    for canonical_mode in config.PAYMENT_MODES:
        if mode_lower == canonical_mode.lower():
            payment_mode = canonical_mode
            break
    else:
        # Partial match fallback.
        for canonical_mode in config.PAYMENT_MODES:
            if mode_lower in canonical_mode.lower() or canonical_mode.lower() in mode_lower:
                payment_mode = canonical_mode
                break

    # ── date ──────────────────────────────────────────────────────
    raw_date = parsed.get("date")
    txn_date = _resolve_date(str(raw_date) if raw_date else None)

    return {
        "student_id": student_id,
        "category_id": category_id,
        "txn_date": txn_date,
        "amount": round(amount, 2),
        "merchant": merchant,
        "description": description,
        "payment_mode": payment_mode,
        "source": "nlp_parse",
        "is_recurring": 0,
        # Also include the human-readable category name for display.
        "_category_name": category,
    }


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def parse_expense(
    text: str,
    student_id: int | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Parse a natural-language expense description into a validated,
    INSERT-ready dictionary.

    Parameters
    ----------
    text : str
        The user's natural-language expense description, e.g.
        ``"Spent 50 on chai at canteen"``.
    student_id : int | None
        The student recording this expense.  Defaults to
        ``config.DEFAULT_STUDENT_ID``.
    db_path : Path | str | None
        Path to the SQLite database.  Defaults to ``config.DB_PATH``.

    Returns
    -------
    dict
        Keys: ``student_id``, ``category_id``, ``txn_date``, ``amount``,
        ``merchant``, ``description``, ``payment_mode``, ``source``,
        ``is_recurring``, ``_category_name`` (display-only).

    Raises
    ------
    ParseError
        If the input cannot be parsed after a retry.
    LLMError
        If the Ollama daemon is unreachable.

    Example
    -------
    >>> result = parse_expense("Spent 50 on chai at canteen")
    >>> result["amount"]
    50.0
    >>> result["_category_name"]
    'Food & Canteen'
    """
    student_id = student_id or config.DEFAULT_STUDENT_ID
    db_path = Path(db_path or config.DB_PATH)
    text = text.strip()
    if not text:
        raise ParseError("Cannot parse an empty string.")

    prompt = _build_parse_prompt(text)

    # Attempt 1.
    try:
        raw_response = llm_engine.generate(
            prompt,
            system="You are a precise expense parser.  Output ONLY valid JSON.",
            temperature=0.1,  # near-deterministic for structured output
        )
        parsed = _extract_json(raw_response)
        return _validate_and_normalise(parsed, student_id, db_path)
    except ParseError as first_err:
        logger.warning("First parse attempt failed: %s — retrying.", first_err)
    except llm_engine.LLMError:
        raise  # Don't retry connectivity errors.

    # Attempt 2 — stricter prompt.
    retry_prompt = _build_retry_prompt(text, raw_response)  # type: ignore[possibly-undefined]
    try:
        raw_response = llm_engine.generate(
            retry_prompt,
            system="You are a precise expense parser.  Output ONLY valid JSON.",
            temperature=0.0,
        )
        parsed = _extract_json(raw_response)
        return _validate_and_normalise(parsed, student_id, db_path)
    except ParseError as retry_err:
        raise ParseError(
            f"Failed to parse expense after retry.\n"
            f"Input: \"{text}\"\n"
            f"Last error: {retry_err}"
        ) from retry_err


def insert_parsed_expense(
    parsed: dict[str, Any],
    db_path: Path | str | None = None,
) -> int:
    """
    Insert a parsed expense dict (from :func:`parse_expense`) into the
    ``expenses`` table.

    Returns the new row ID.
    """
    db_path = Path(db_path or config.DB_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        cursor = conn.execute(
            """INSERT INTO expenses
                   (student_id, category_id, txn_date, amount, merchant,
                    description, payment_mode, source, is_recurring)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                parsed["student_id"],
                parsed["category_id"],
                parsed["txn_date"],
                parsed["amount"],
                parsed["merchant"],
                parsed["description"],
                parsed["payment_mode"],
                parsed["source"],
                parsed["is_recurring"],
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info(
            "Inserted expense #%d: %s %s at %s.",
            row_id,
            config.as_currency(parsed["amount"]),
            parsed.get("_category_name", ""),
            parsed["merchant"],
        )
        return row_id  # type: ignore[return-value]
    finally:
        conn.close()
