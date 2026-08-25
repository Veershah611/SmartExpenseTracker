"""
frontend/components/compare.py
==============================
The Compare tab -- month-against-month spending analysis.
**Owned by the Core Integrator.**

The partial-month trap
----------------------
Today is the 25th of a 31-day month. Comparing this month's total against last
month's full total shows a ~20% "drop" that is entirely an artefact of six days
that have not happened yet. A student reading that would conclude they are doing
well when they may not be.

So the comparison defaults to **like-for-like**: both months truncated to the
same number of elapsed days. The toggle is there for anyone who genuinely wants
raw calendar totals, and the caption always states which is being shown.

One-off semester fees are excluded by default for the same reason as elsewhere
-- a Rs 49,000 tuition payment swamps every real behavioural change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402


def _analytics():
    """The analytics module, or None if it failed to load."""
    status = integration.feature("analytics")
    return status.module if status.ready else None


def _delta_tone(change: float) -> str:
    """Spending up is bad news, so the colour mapping is inverted."""
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "flat"


def _render_headline(result: dict) -> None:
    """Four cards summarising the comparison."""
    columns = st.columns(4)

    with columns[0]:
        ui.metric_card(
            f"{result['month_a']} spend",
            ui.money_compact(result["total_a"]),
            f"{result['txns_a']} transactions",
        )
    with columns[1]:
        ui.metric_card(
            f"{result['month_b']} spend",
            ui.money_compact(result["total_b"]),
            f"{result['txns_b']} transactions",
        )
    with columns[2]:
        change = result["change"]
        ui.metric_card(
            "Difference",
            ui.money_signed(change),
            f"{result['change_pct']:+.1f}%",
            _delta_tone(change),
        )
    with columns[3]:
        daily_change = result["daily_a"] - result["daily_b"]
        ui.metric_card(
            "Daily average",
            ui.money_compact(result["daily_a"]),
            f"{ui.money_signed(daily_change)} vs {result['month_b']}",
            _delta_tone(daily_change),
        )


def _render_verdict(result: dict) -> None:
    """One plain-English sentence on the overall direction."""
    change = result["change"]
    month_a, month_b = result["month_a"], result["month_b"]

    if abs(result["change_pct"]) < 2:
        st.info(
            f"Spending is essentially flat between {month_b} and {month_a} - "
            f"a difference of {ui.money_compact(abs(change))}.",
            icon=":material/balance:",
        )
    elif change < 0:
        st.success(
            f"You spent {ui.money_compact(abs(change))} less in {month_a} than in "
            f"{month_b}, down {abs(result['change_pct']):.0f}%.",
            icon=":material/trending_down:",
        )
    else:
        st.warning(
            f"You spent {ui.money_compact(change)} more in {month_a} than in "
            f"{month_b}, up {result['change_pct']:.0f}%.",
            icon=":material/trending_up:",
        )

    movers = []
    if result["biggest_increase"]:
        item = result["biggest_increase"]
        movers.append(
            f"**Biggest increase:** {item['category']} "
            f"({ui.money_signed(item['change'])})"
        )
    if result["biggest_decrease"]:
        item = result["biggest_decrease"]
        movers.append(
            f"**Biggest saving:** {item['category']} "
            f"({ui.money_signed(item['change'])})"
        )
    if movers:
        st.markdown(" &nbsp;•&nbsp; ".join(movers))


def _render_table(categories: pd.DataFrame, month_a: str, month_b: str) -> None:
    """Category-by-category movement, biggest increase first."""
    ui.section("Category by category", "Sorted by change, increases first.")

    display = categories.copy()
    # A category with no spend in the baseline month has no meaningful
    # percentage; label it rather than printing NaN or a fake infinity.
    display["change_pct"] = display["change_pct"].apply(
        lambda value: "new" if pd.isna(value) else f"{value:+.0f}%"
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "category": st.column_config.TextColumn("Category"),
            "amount_a": st.column_config.NumberColumn(month_a, format="%.0f"),
            "amount_b": st.column_config.NumberColumn(month_b, format="%.0f"),
            "change": st.column_config.NumberColumn("Change", format="%+.0f"),
            "change_pct": st.column_config.TextColumn("Change %"),
        },
    )


def _render_chart(categories: pd.DataFrame, month_a: str, month_b: str) -> None:
    """Grouped bars, one pair per category."""
    if categories.empty:
        return
    ui.section("Side by side")

    long = categories.melt(
        id_vars="category",
        value_vars=["amount_a", "amount_b"],
        var_name="month",
        value_name="amount",
    )
    long["month"] = long["month"].map({"amount_a": month_a, "amount_b": month_b})

    st.vega_lite_chart(
        long,
        {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "y": {"field": "category", "type": "nominal", "sort": "-x",
                      "axis": {"title": None, "labelLimit": 140}},
                "x": {"field": "amount", "type": "quantitative",
                      "axis": {"title": None, "format": "~s"},
                      "scale": {"zero": True}},
                "yOffset": {"field": "month", "type": "nominal"},
                "color": {"field": "month", "type": "nominal",
                          "scale": {"range": ["#4C78A8", "#B0B7C3"]},
                          "legend": {"title": None, "orient": "top"}},
            },
            "height": 320,
        },
        width="stretch",
    )


def _render_ai_summary(result: dict, student: dict) -> None:
    """Optional LLM commentary, grounded in the computed deltas."""
    if not llm_engine.is_available():
        return
    if not st.button("Explain the change", type="primary"):
        return

    movers = result["categories"].head(5)
    facts = [
        f"Comparing {result['month_a']} against {result['month_b']}"
        + (f", both limited to the first {result['days_compared']} days for a "
           "fair comparison" if result["truncated"] else ""),
        f"Total: Rs {result['total_a']:,.0f} vs Rs {result['total_b']:,.0f} "
        f"(a change of Rs {result['change']:,.0f}, {result['change_pct']:+.1f}%)",
        f"Daily average: Rs {result['daily_a']:,.0f} vs Rs {result['daily_b']:,.0f}",
    ]
    for row in movers.itertuples():
        direction = "more" if row.change > 0 else "less"
        facts.append(
            f"{row.category}: Rs {row.amount_a:,.0f} vs Rs {row.amount_b:,.0f} "
            f"- Rs {abs(row.change):,.0f} {direction}"
        )

    prompt = (
        "You are a financial assistant for an Indian university student with a "
        f"monthly budget of Rs {student['monthly_budget']:,.0f}.\n\n"
        "VERIFIED FIGURES (copy these numbers exactly, never recalculate):\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + "\n\nIn 3-4 short sentences, explain what changed between the two "
          "months and what it suggests about their habits. Name the categories "
          "that moved most. Amounts in rupees, written like Rs 1,250. "
          "No markdown headings."
    )
    try:
        with st.chat_message("assistant"):
            st.write_stream(llm_engine.chat_stream(
                [{"role": "user", "content": prompt}]
            ))
    except llm_engine.LLMUnavailableError as exc:
        st.warning(str(exc), icon=":material/smart_toy:")


def render() -> None:
    """Draw the Compare tab."""
    analytics = _analytics()
    if analytics is None or not hasattr(analytics, "compare_months"):
        ui.empty_state("Comparison needs the analytics module.")
        return

    student_id = state.get_student_id()
    student = state.load_student(student_id)
    expenses = state.load_expenses(student_id)

    if expenses.empty:
        ui.empty_state("No transactions to compare yet.")
        return

    months = analytics.available_months(expenses)
    if len(months) < 2:
        ui.empty_state(
            "Only one month of data so far.",
            "Comparison needs at least two months.",
        )
        return

    ui.section(
        "Compare months",
        "How this month's spending moved against another month.",
    )

    controls = st.columns([2, 2, 3])
    with controls[0]:
        month_a = st.selectbox("This month", months, index=0)
    with controls[1]:
        # Default the baseline to the month immediately before the selection.
        default_b = months.index(month_a) + 1 if months.index(month_a) + 1 < len(months) else 0
        month_b = st.selectbox("Compared with", months, index=default_b)
    with controls[2]:
        st.write("")
        same_period = st.toggle(
            "Fair comparison (same number of days)",
            value=True,
            help="The current month is incomplete. Without this, the missing "
                 "days look like a spending drop.",
        )
        exclude_fees = st.toggle("Exclude one-off semester fees", value=True)

    if month_a == month_b:
        st.info("Pick two different months.", icon=":material/info:")
        return

    result = analytics.compare_months(
        expenses, month_a, month_b,
        same_period_only=same_period,
        exclude_one_off=exclude_fees,
    )

    if result["categories"].empty:
        ui.empty_state("No spending in one of those months.")
        return

    # State plainly what is being compared -- the numbers mean different things
    # depending on these two switches.
    notes = []
    if result["truncated"]:
        notes.append(
            f"both months limited to their first **{result['days_compared']} "
            "days** so the comparison is like-for-like"
        )
    if exclude_fees and (result["one_off_a"] or result["one_off_b"]):
        excluded = []
        if result["one_off_b"]:
            excluded.append(f"{ui.money_compact(result['one_off_b'])} in {month_b}")
        if result["one_off_a"]:
            excluded.append(f"{ui.money_compact(result['one_off_a'])} in {month_a}")
        notes.append("excluding one-off semester fees of " + " and ".join(excluded))
    if notes:
        st.caption("Showing " + "; ".join(notes) + ".")

    _render_headline(result)
    st.divider()
    _render_verdict(result)
    st.divider()
    _render_chart(result["categories"], month_a, month_b)
    _render_table(result["categories"], month_a, month_b)
    st.divider()
    _render_ai_summary(result, student)
