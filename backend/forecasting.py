"""
forecasting.py
==============
Budget forecasting and **Predictive Broke Alert** engine for the Smart
Expense Tracker.

Responsibilities
----------------
1. Calculate the student's *daily burn rate* for the current month.
2. Project whether the budget will be exhausted before the month ends.
3. Build a structured LLM prompt and call Ollama to generate a personalised,
   slightly urgent "Broke Alert" warning.
4. Fall back gracefully to a tiered rule-based alert when Ollama is offline.

Contract
--------
Every public function accepts a ``student_id`` and an open
``sqlite3.Connection``.  **No Streamlit imports.**  All constants come from
``config.py``.

Usage (from the frontend)::

    import sqlite3, config
    from backend.forecasting import calculate_burn_rate, generate_broke_alert

    conn = sqlite3.connect(config.DB_PATH)
    data  = calculate_burn_rate(student_id=1, conn=conn)
    alert = generate_broke_alert(student_id=1, conn=conn)
"""

from __future__ import annotations

import json
import sqlite3
from calendar import monthrange
from datetime import date

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project-level configuration — the single source of truth.
# ---------------------------------------------------------------------------
import config

_CURRENCY: str = config.CURRENCY_SYMBOL


# =========================================================================== #
#  Core calculation — burn rate + projection                                  #
# =========================================================================== #
def calculate_burn_rate(
    student_id: int,
    conn: sqlite3.Connection,
) -> dict:
    """
    Compute the student's spending velocity for the current month and
    project whether the monthly budget will hold.

    Returns
    -------
    dict
        Keys::

            total_spent        – amount spent so far this month (float)
            days_passed        – calendar days elapsed including today (int)
            days_remaining     – calendar days left in the month (int)
            days_in_month      – total days in the current month (int)
            daily_burn_rate    – total_spent / days_passed (float)
            monthly_budget     – from the students table (float)
            remaining_budget   – monthly_budget - total_spent (float)
            projected_spend    – daily_burn_rate * days_in_month (float)
            projected_deficit  – projected_spend - monthly_budget (float, may be negative = surplus)
            on_track           – True if projected_spend <= monthly_budget (bool)
            top_category       – name of the highest-spend category (str)
            top_category_amount – amount in that category (float)
            severity           – 'safe' | 'warning' | 'danger' | 'critical' (str)

    Raises
    ------
    ValueError
        If ``student_id`` does not exist in the database.
    """
    today = date.today()
    year, month = today.year, today.month
    days_in_month = monthrange(year, month)[1]
    days_passed = max(today.day, 1)  # avoid division by zero on the 1st
    days_remaining = days_in_month - days_passed

    # --- Monthly budget from the students table ---
    row = conn.execute(
        "SELECT monthly_budget FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Student {student_id} not found in the database.")
    monthly_budget: float = row[0]

    # --- Spending this month, broken down by category ---
    start_date = f"{year:04d}-{month:02d}-01"
    end_date = f"{year:04d}-{month:02d}-{days_in_month:02d}"

    query = """
        SELECT c.name AS category, SUM(e.amount) AS total
        FROM   expenses  e
        JOIN   categories c ON c.id = e.category_id
        WHERE  e.student_id = ?
          AND  e.txn_date BETWEEN ? AND ?
        GROUP  BY c.name
        ORDER  BY total DESC
    """
    df = pd.read_sql_query(query, conn, params=(student_id, start_date, end_date))

    total_spent = float(df["total"].sum()) if not df.empty else 0.0
    daily_burn_rate = total_spent / days_passed
    projected_spend = daily_burn_rate * days_in_month
    remaining_budget = monthly_budget - total_spent
    projected_deficit = projected_spend - monthly_budget

    # Top spending category
    if not df.empty:
        top_category = df.iloc[0]["category"]
        top_category_amount = float(df.iloc[0]["total"])
    else:
        top_category = "N/A"
        top_category_amount = 0.0

    # Severity classification
    spend_ratio = projected_spend / monthly_budget if monthly_budget > 0 else 0.0
    if spend_ratio <= 1.0:
        severity = "safe"
    elif spend_ratio <= 1.15:
        severity = "warning"
    elif spend_ratio <= 1.35:
        severity = "danger"
    else:
        severity = "critical"

    return {
        "total_spent": round(total_spent, 2),
        "days_passed": days_passed,
        "days_remaining": days_remaining,
        "days_in_month": days_in_month,
        "daily_burn_rate": round(daily_burn_rate, 2),
        "monthly_budget": round(monthly_budget, 2),
        "remaining_budget": round(remaining_budget, 2),
        "projected_spend": round(projected_spend, 2),
        "projected_deficit": round(projected_deficit, 2),
        "on_track": projected_spend <= monthly_budget,
        "top_category": top_category,
        "top_category_amount": round(top_category_amount, 2),
        "severity": severity,
    }


# =========================================================================== #
#  LLM prompt builder                                                         #
# =========================================================================== #
def build_broke_alert_prompt(burn_data: dict) -> str:
    """
    Assemble the system prompt for the Ollama LLM.

    The prompt injects the student's real numbers and instructs the model to
    generate a short, personalised "Broke Alert".

    Parameters
    ----------
    burn_data : dict
        The output of :func:`calculate_burn_rate`.

    Returns
    -------
    str
        The full system prompt string.
    """
    severity_desc = {
        "safe": "within budget — encourage them to keep it up",
        "warning": "slightly over budget — give a gentle heads-up",
        "danger": "significantly over budget — be direct and specific",
        "critical": "severely over budget — be urgent, clear, and action-oriented",
    }

    return f"""You are a friendly student finance advisor at Nirma University, Ahmedabad.
A student has asked you to review their spending this month. Here is their data:

- Daily burn rate: {_CURRENCY}{burn_data['daily_burn_rate']:,.0f}/day
- Days passed this month: {burn_data['days_passed']} of {burn_data['days_in_month']}
- Days remaining: {burn_data['days_remaining']}
- Total spent so far: {_CURRENCY}{burn_data['total_spent']:,.0f}
- Monthly budget: {_CURRENCY}{burn_data['monthly_budget']:,.0f}
- Remaining budget: {_CURRENCY}{burn_data['remaining_budget']:,.0f}
- Projected total by month end: {_CURRENCY}{burn_data['projected_spend']:,.0f}
- Projected {'DEFICIT' if burn_data['projected_deficit'] > 0 else 'surplus'}: {_CURRENCY}{abs(burn_data['projected_deficit']):,.0f}
- Top spending category: {burn_data['top_category']} ({_CURRENCY}{burn_data['top_category_amount']:,.0f})
- Status: {burn_data['severity'].upper()} — {severity_desc.get(burn_data['severity'], '')}

Your task:
1. Generate a "Broke Alert" message for this student.
2. Tone: friendly but urgent, like a concerned senior or mentor — NOT a banker or a robot.
3. Mention the EXACT numbers (burn rate, remaining budget, top category) — don't be vague.
4. If they are over budget, suggest ONE concrete, actionable tip to cut spending in their top category.
5. If they are within budget, congratulate them briefly and warn about any single category that dominates their spending.
6. Keep it under 100 words. No bullet points. No greetings. No sign-off. Just the alert message.
"""


# =========================================================================== #
#  Tiered rule-based fallback (when Ollama is offline)                        #
# =========================================================================== #
def _fallback_alert(burn_data: dict) -> str:
    """
    Generate a deterministic alert string using the burn-rate data.

    Three tiers so the dashboard never shows a flat, boring message even
    without the LLM.
    """
    remaining = burn_data["remaining_budget"]
    deficit = burn_data["projected_deficit"]
    top_cat = burn_data["top_category"]
    top_amt = burn_data["top_category_amount"]
    daily = burn_data["daily_burn_rate"]
    days_left = burn_data["days_remaining"]
    budget = burn_data["monthly_budget"]
    severity = burn_data["severity"]

    if severity == "safe":
        return (
            f"You're on track! At {_CURRENCY}{daily:,.0f}/day, you'll finish the month "
            f"with about {_CURRENCY}{abs(deficit):,.0f} to spare. "
            f"Your biggest spend is {top_cat} at {_CURRENCY}{top_amt:,.0f} — "
            f"keep an eye on it so it doesn't creep up in the last {days_left} days."
        )
    elif severity == "warning":
        return (
            f"Heads up — at your current pace of {_CURRENCY}{daily:,.0f}/day, "
            f"you're projected to overshoot your {_CURRENCY}{budget:,.0f} budget "
            f"by about {_CURRENCY}{deficit:,.0f}. "
            f"Your {top_cat} spending ({_CURRENCY}{top_amt:,.0f}) is the main driver. "
            f"Cutting back a little there over the next {days_left} days should get you back on track."
        )
    elif severity == "danger":
        return (
            f"🚨 Budget alert: you've spent {_CURRENCY}{burn_data['total_spent']:,.0f} "
            f"with {days_left} days to go. At {_CURRENCY}{daily:,.0f}/day, you're heading for a "
            f"{_CURRENCY}{deficit:,.0f} overshoot. {top_cat} alone accounts for "
            f"{_CURRENCY}{top_amt:,.0f}. Try limiting yourself to "
            f"{_CURRENCY}{remaining / max(days_left, 1):,.0f}/day for the rest of the month."
        )
    else:  # critical
        return (
            f"🔴 CRITICAL: You've already spent {_CURRENCY}{burn_data['total_spent']:,.0f} "
            f"against a {_CURRENCY}{budget:,.0f} budget — that's "
            f"{_CURRENCY}{abs(remaining):,.0f} {'over' if remaining < 0 else 'away from your limit'} "
            f"with {days_left} days remaining. "
            f"{top_cat} ({_CURRENCY}{top_amt:,.0f}) is your biggest drain. "
            f"Freeze all non-essential spending immediately. "
            f"Stick to the mess and avoid delivery apps until month-end."
        )


# =========================================================================== #
#  End-to-end alert generator (LLM with fallback)                             #
# =========================================================================== #
def generate_broke_alert(
    student_id: int,
    conn: sqlite3.Connection,
    ollama_host: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> dict:
    """
    Compute the burn rate, build the LLM prompt, call Ollama, and return
    the alert.

    Parameters
    ----------
    student_id : int
        Primary key from the ``students`` table.
    conn : sqlite3.Connection
        An open connection to ``expenses.db``.
    ollama_host : str | None
        Override ``config.OLLAMA_HOST`` (useful for tests).
    model : str | None
        Override ``config.OLLAMA_CHAT_MODEL``.
    timeout : int | None
        Override ``config.LLM_TIMEOUT_SECONDS``.

    Returns
    -------
    dict
        Keys::

            burn_data   – the full dict from calculate_burn_rate()
            alert_text  – the LLM-generated (or fallback) alert string
            source      – 'llm' | 'fallback'
            prompt      – the system prompt sent to Ollama (for debugging)
    """
    ollama_host = ollama_host or config.OLLAMA_HOST
    model = model or config.OLLAMA_CHAT_MODEL
    timeout = timeout or config.LLM_TIMEOUT_SECONDS

    burn_data = calculate_burn_rate(student_id, conn)
    prompt = build_broke_alert_prompt(burn_data)

    # --- Attempt LLM call ---
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": 200,  # ~100 words cap
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        alert_text = result.get("response", "").strip()

        if alert_text:
            return {
                "burn_data": burn_data,
                "alert_text": alert_text,
                "source": "llm",
                "prompt": prompt,
            }
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        # Ollama is down or returned garbage — fall through to the fallback.
        pass

    # --- Fallback ---
    return {
        "burn_data": burn_data,
        "alert_text": _fallback_alert(burn_data),
        "source": "fallback",
        "prompt": prompt,
    }
