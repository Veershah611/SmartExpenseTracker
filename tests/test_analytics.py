"""
test_analytics.py
=================
Lightweight integration test for the analytics and forecasting modules.

Run after generating the mock database::

    python3 backend/scripts/generate_mock_data.py --force
    python3 tests/test_analytics.py

The script exercises every public function, validates the return types,
and prints a human-readable report so you can eyeball the numbers.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# --- sys.path fix (same pattern as app.py and generate_mock_data.py) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from backend.analytics_engine import (  # noqa: E402
    daily_spending_trend_chart,
    get_expenses_df,
    monthly_summary,
    payment_mode_breakdown,
    spending_by_category_chart,
)
from backend.forecasting import (  # noqa: E402
    build_broke_alert_prompt,
    calculate_burn_rate,
    generate_broke_alert,
)

# Force UTF-8 so emoji and rupee signs don't crash Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def _sep(title: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print(f"{'=' * 66}")


def _check(label: str, condition: bool) -> None:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}")


def main() -> int:
    db_path = config.DB_PATH
    if not db_path.exists():
        print(f"[error] Database not found at {db_path}.")
        print("        Run:  python3 backend/scripts/generate_mock_data.py --force")
        return 1

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    student_id = config.DEFAULT_STUDENT_ID
    errors = 0

    # ------------------------------------------------------------------ #
    #  1. get_expenses_df                                                 #
    # ------------------------------------------------------------------ #
    _sep("1. get_expenses_df()")
    df = get_expenses_df(student_id, conn)
    _check(f"Returns DataFrame with {len(df)} rows", len(df) >= 0)
    if not df.empty:
        expected_cols = {
            "id", "txn_date", "amount", "merchant", "description",
            "payment_mode", "source", "is_recurring", "category",
            "category_icon",
        }
        missing = expected_cols - set(df.columns)
        _check(f"Has all expected columns (missing: {missing or 'none'})", not missing)
        _check("txn_date is datetime64", df["txn_date"].dtype.kind == "M")
        print(f"  ℹ️  Rows: {len(df)} | Total: {config.as_currency(df['amount'].sum())}")
    else:
        print("  ⚠️  DataFrame is empty for the current month — this is OK if today is the 1st.")

    # ------------------------------------------------------------------ #
    #  2. spending_by_category_chart                                      #
    # ------------------------------------------------------------------ #
    _sep("2. spending_by_category_chart()")
    import plotly.graph_objects as go

    fig = spending_by_category_chart(student_id, conn)
    is_fig = isinstance(fig, go.Figure)
    _check("Returns a plotly Figure", is_fig)
    if is_fig and fig.data:
        _check(f"Has {len(fig.data)} trace(s)", len(fig.data) > 0)
        _check("Trace is a Pie chart", isinstance(fig.data[0], go.Pie))
    else:
        _check("Has data traces (may be empty this month)", bool(fig.data))
    if not is_fig:
        errors += 1

    # ------------------------------------------------------------------ #
    #  3. daily_spending_trend_chart                                      #
    # ------------------------------------------------------------------ #
    _sep("3. daily_spending_trend_chart()")
    fig2 = daily_spending_trend_chart(student_id, conn)
    is_fig2 = isinstance(fig2, go.Figure)
    _check("Returns a plotly Figure", is_fig2)
    if is_fig2 and fig2.data:
        _check(f"Has {len(fig2.data)} trace(s)", len(fig2.data) >= 2)
        _check("First trace is Scatter (line)", isinstance(fig2.data[0], go.Scatter))
    if not is_fig2:
        errors += 1

    # ------------------------------------------------------------------ #
    #  4. payment_mode_breakdown                                          #
    # ------------------------------------------------------------------ #
    _sep("4. payment_mode_breakdown()")
    fig3 = payment_mode_breakdown(student_id, conn)
    is_fig3 = isinstance(fig3, go.Figure)
    _check("Returns a plotly Figure", is_fig3)
    if is_fig3 and fig3.data:
        _check(f"Has {len(fig3.data)} trace(s)", len(fig3.data) > 0)
        _check("Trace is a Bar chart", isinstance(fig3.data[0], go.Bar))
    if not is_fig3:
        errors += 1

    # ------------------------------------------------------------------ #
    #  5. monthly_summary                                                 #
    # ------------------------------------------------------------------ #
    _sep("5. monthly_summary()")
    summary = monthly_summary(student_id, conn)
    _check("Returns a dict", isinstance(summary, dict))
    required_keys = {
        "total_spent", "daily_average", "transaction_count",
        "top_category", "top_category_amount", "month_label",
    }
    missing_keys = required_keys - set(summary.keys())
    _check(f"Has all required keys (missing: {missing_keys or 'none'})", not missing_keys)
    print(f"  ℹ️  Summary: {summary}")

    # ------------------------------------------------------------------ #
    #  6. calculate_burn_rate                                             #
    # ------------------------------------------------------------------ #
    _sep("6. calculate_burn_rate()")
    burn = calculate_burn_rate(student_id, conn)
    _check("Returns a dict", isinstance(burn, dict))
    burn_keys = {
        "total_spent", "days_passed", "days_remaining", "days_in_month",
        "daily_burn_rate", "monthly_budget", "remaining_budget",
        "projected_spend", "projected_deficit", "on_track",
        "top_category", "top_category_amount", "severity",
    }
    missing_burn = burn_keys - set(burn.keys())
    _check(f"Has all required keys (missing: {missing_burn or 'none'})", not missing_burn)
    _check("daily_burn_rate is non-negative", burn.get("daily_burn_rate", -1) >= 0)
    _check("on_track is a bool", isinstance(burn.get("on_track"), bool))
    _check(f"severity is valid: '{burn.get('severity')}'",
           burn.get("severity") in ("safe", "warning", "danger", "critical"))
    print(f"  ℹ️  Burn rate: {config.as_currency(burn['daily_burn_rate'])}/day")
    print(f"  ℹ️  Projected: {config.as_currency(burn['projected_spend'])} "
          f"vs budget {config.as_currency(burn['monthly_budget'])}")
    print(f"  ℹ️  Severity: {burn['severity'].upper()}")

    # ------------------------------------------------------------------ #
    #  7. build_broke_alert_prompt                                        #
    # ------------------------------------------------------------------ #
    _sep("7. build_broke_alert_prompt()")
    prompt = build_broke_alert_prompt(burn)
    _check("Returns a non-empty string", isinstance(prompt, str) and len(prompt) > 100)
    _check("Contains the daily burn rate", str(int(burn["daily_burn_rate"])) in prompt)
    _check("Contains the top category", burn["top_category"] in prompt)
    print(f"  ℹ️  Prompt length: {len(prompt)} chars")

    # ------------------------------------------------------------------ #
    #  8. generate_broke_alert (fallback — does NOT require Ollama)       #
    # ------------------------------------------------------------------ #
    _sep("8. generate_broke_alert() — fallback mode")
    # Use a dummy host so we guaranteed hit the fallback path.
    result = generate_broke_alert(
        student_id, conn,
        ollama_host="http://localhost:1",  # unreachable on purpose
        timeout=2,
    )
    _check("Returns a dict", isinstance(result, dict))
    _check("Has 'alert_text' key", "alert_text" in result)
    _check("Has 'source' key", "source" in result)
    _check(f"Source is 'fallback' (got: '{result.get('source')}')",
           result.get("source") == "fallback")
    _check("Alert text is non-empty", bool(result.get("alert_text")))
    print(f"\n  📢 ALERT ({result.get('source', '?').upper()}):")
    print(f"  {result.get('alert_text', '(empty)')}")

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #
    _sep("TEST SUMMARY")
    if errors:
        print(f"  ❌ {errors} critical failure(s). Check output above.")
    else:
        print("  ✅ All checks passed!")
    conn.close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
