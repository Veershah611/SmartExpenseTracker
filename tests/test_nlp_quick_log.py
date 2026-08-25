"""
tests/test_nlp_quick_log.py
============================
Unit tests for :mod:`backend.nlp_quick_log`.

Tests cover:
- JSON extraction from various LLM output formats.
- Category fuzzy matching (keyword, substring, exact).
- Date resolution (today, yesterday, ISO, DD/MM/YYYY).
- Full parse pipeline (mocked LLM).
- Edge cases: empty input, missing amount, bad JSON.

Run:
    python -m pytest tests/test_nlp_quick_log.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.nlp_quick_log import (
    ParseError,
    _extract_json,
    _fuzzy_match_category,
    _resolve_date,
    parse_expense,
    insert_parsed_expense,
)


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_db(tmp_path):
    """Minimal database with categories for validation."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = PROJECT_ROOT / "backend" / "schema.sql"
    conn.executescript(schema_path.read_text(encoding="utf-8"))

    conn.execute(
        """INSERT INTO students (id, name, roll_no, course, semester,
                                 hostel_resident, monthly_budget)
           VALUES (1, 'Test', '22T001', 'B.Tech', 5, 1, 15000.0)"""
    )
    for name, icon, share in config.EXPENSE_CATEGORIES:
        conn.execute(
            "INSERT OR IGNORE INTO categories (name, icon, typical_share) VALUES (?, ?, ?)",
            (name, icon, share),
        )
    conn.commit()
    conn.close()
    return db_path


# ────────────────────────────────────────────────────────────────────
# _extract_json tests
# ────────────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"amount": 50, "category": "Food"}'
        result = _extract_json(raw)
        assert result["amount"] == 50

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"amount": 100, "merchant": "Swiggy"}\n```'
        result = _extract_json(raw)
        assert result["amount"] == 100

    def test_json_with_trailing_text(self):
        raw = '{"amount": 200}\nHere is the parsed expense.'
        result = _extract_json(raw)
        assert result["amount"] == 200

    def test_json_with_leading_text(self):
        raw = 'Sure, here is the result:\n{"amount": 300, "category": "Transport"}'
        result = _extract_json(raw)
        assert result["amount"] == 300

    def test_no_json_raises(self):
        with pytest.raises(ParseError, match="No JSON"):
            _extract_json("No JSON here at all.")

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError, match="Invalid JSON"):
            _extract_json('{"amount": }')


# ────────────────────────────────────────────────────────────────────
# _fuzzy_match_category tests
# ────────────────────────────────────────────────────────────────────

class TestFuzzyMatchCategory:
    @pytest.mark.parametrize("input_str, expected", [
        ("Food & Canteen", "Food & Canteen"),
        ("food & canteen", "Food & Canteen"),
        ("Transport", "Transport"),
        ("transport", "Transport"),
    ])
    def test_exact_match(self, input_str, expected):
        assert _fuzzy_match_category(input_str) == expected

    @pytest.mark.parametrize("input_str, expected", [
        ("food", "Food & Canteen"),
        ("canteen", "Food & Canteen"),
        ("chai", "Food & Canteen"),
        ("auto", "Transport"),
        ("cab", "Transport"),
        ("uber", "Transport"),
        ("books", "Books & Stationery"),
        ("netflix", "Entertainment"),
        ("gym", "Health & Medical"),
        ("jio", "Mobile & Internet"),
        ("recharge", "Mobile & Internet"),
        ("laundry", "Miscellaneous"),
        ("amazon", "Shopping"),
    ])
    def test_keyword_match(self, input_str, expected):
        assert _fuzzy_match_category(input_str) == expected

    def test_unknown_falls_to_miscellaneous(self):
        assert _fuzzy_match_category("xyzzy_unknown") == "Miscellaneous"

    def test_empty_falls_to_miscellaneous(self):
        assert _fuzzy_match_category("") == "Miscellaneous"


# ────────────────────────────────────────────────────────────────────
# _resolve_date tests
# ────────────────────────────────────────────────────────────────────

class TestResolveDate:
    def test_none_returns_today(self):
        assert _resolve_date(None) == date.today().isoformat()

    def test_empty_returns_today(self):
        assert _resolve_date("") == date.today().isoformat()

    def test_today(self):
        assert _resolve_date("today") == date.today().isoformat()

    def test_yesterday(self):
        expected = (date.today() - timedelta(days=1)).isoformat()
        assert _resolve_date("yesterday") == expected

    def test_iso_format_passthrough(self):
        assert _resolve_date("2025-08-15") == "2025-08-15"

    def test_dd_mm_yyyy_slash(self):
        assert _resolve_date("15/08/2025") == "2025-08-15"

    def test_dd_mm_yyyy_dash(self):
        assert _resolve_date("15-08-2025") == "2025-08-15"

    def test_garbage_returns_today(self):
        assert _resolve_date("not a date") == date.today().isoformat()


# ────────────────────────────────────────────────────────────────────
# Full parse_expense pipeline (mocked LLM)
# ────────────────────────────────────────────────────────────────────

class TestParseExpense:
    def _mock_generate(self, response_dict):
        """Return a patch that makes llm_engine.generate return fixed JSON."""
        return patch(
            "backend.nlp_quick_log.llm_engine.generate",
            return_value=json.dumps(response_dict),
        )

    def test_simple_parse(self, seeded_db):
        response = {
            "amount": 50,
            "category": "Food & Canteen",
            "merchant": "Campus Canteen",
            "description": "Chai",
            "payment_mode": "UPI",
            "date": date.today().isoformat(),
        }
        with self._mock_generate(response):
            result = parse_expense("Spent 50 on chai", student_id=1, db_path=seeded_db)
        assert result["amount"] == 50.0
        assert result["_category_name"] == "Food & Canteen"
        assert result["payment_mode"] == "UPI"
        assert result["source"] == "nlp_parse"

    def test_parse_with_fuzzy_category(self, seeded_db):
        response = {
            "amount": 120,
            "category": "cab",  # Not canonical — should be fuzzy-matched.
            "merchant": "Ola",
            "description": "Auto to campus",
            "payment_mode": "Cash",
            "date": "yesterday",
        }
        with self._mock_generate(response):
            result = parse_expense("Auto 120 yesterday", student_id=1, db_path=seeded_db)
        assert result["_category_name"] == "Transport"
        assert result["amount"] == 120.0

    def test_parse_with_markdown_fenced_json(self, seeded_db):
        raw = '```json\n{"amount": 299, "category": "Mobile & Internet", "merchant": "Jio", "description": "Recharge", "payment_mode": "UPI", "date": "today"}\n```'
        with patch("backend.nlp_quick_log.llm_engine.generate", return_value=raw):
            result = parse_expense("Jio recharge 299", student_id=1, db_path=seeded_db)
        assert result["amount"] == 299.0
        assert result["_category_name"] == "Mobile & Internet"

    def test_empty_input_raises(self, seeded_db):
        with pytest.raises(ParseError, match="empty"):
            parse_expense("", db_path=seeded_db)

    def test_missing_amount_raises(self, seeded_db):
        response = {
            "category": "Food",
            "merchant": "Canteen",
            "description": "Lunch",
            "payment_mode": "UPI",
            "date": "today",
        }
        bad = json.dumps(response)
        with patch("backend.nlp_quick_log.llm_engine.generate", return_value=bad):
            with pytest.raises(ParseError, match="amount"):
                parse_expense("Some lunch", student_id=1, db_path=seeded_db)

    def test_negative_amount_raises(self, seeded_db):
        response = {
            "amount": -50,
            "category": "Food",
            "merchant": "Canteen",
            "description": "Lunch",
            "payment_mode": "UPI",
            "date": "today",
        }
        bad = json.dumps(response)
        with patch("backend.nlp_quick_log.llm_engine.generate", return_value=bad):
            with pytest.raises(ParseError, match="positive"):
                parse_expense("Bad amount", student_id=1, db_path=seeded_db)


class TestInsertParsedExpense:
    def test_insert_and_retrieve(self, seeded_db):
        parsed = {
            "student_id": 1,
            "category_id": 1,
            "txn_date": "2025-08-25",
            "amount": 75.0,
            "merchant": "Campus Canteen",
            "description": "Tea",
            "payment_mode": "UPI",
            "source": "nlp_parse",
            "is_recurring": 0,
            "_category_name": "Food & Canteen",
        }
        row_id = insert_parsed_expense(parsed, db_path=seeded_db)
        assert isinstance(row_id, int)
        assert row_id > 0

        # Verify it's in the database.
        conn = sqlite3.connect(str(seeded_db))
        row = conn.execute(
            "SELECT amount, merchant, source FROM expenses WHERE id = ?",
            (row_id,),
        ).fetchone()
        conn.close()
        assert row == (75.0, "Campus Canteen", "nlp_parse")
