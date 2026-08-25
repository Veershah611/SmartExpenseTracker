"""
frontend/components/dashboard.py
================================
The Overview tab: KPIs, spending trend, category split, budget progress.
**Layout owned by the Core Integrator; chart rendering belongs to Analytics.**

Chart strategy
--------------
Every chart is drawn through a small ``_chart_*`` helper that first asks the
integration seam whether the Analytics developer has delivered
``frontend/components/charts.py``. If they have, their Plotly/Altair renderer is
used. If not, the shell falls back to Streamlit's built-in charts.

That means the dashboard looks complete today and gets better when their work
lands -- without the shell taking a hard dependency on a charting library that
teammate may not have chosen yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import integration  # noqa: E402
from frontend import state, ui  # noqa: E402


# --------------------------------------------------------------------------- #
# Chart delegation
# --------------------------------------------------------------------------- #
def _charts():
    """Return the Analytics developer's chart module, or ``None``."""
    status = integration.feature("charts")
    return status.module if status.ready else None


def _chart_trend(trend: pd.DataFrame) -> None:
    """Monthly spending trend -- their renderer if present, else native."""
    charts = _charts()
    if charts is not None:
        try:
            charts.render_trend_chart(trend)
            return
        except Exception as exc:  # noqa: BLE001 -- never let a chart kill the tab
            ui.error_box(exc, "Analytics chart failed, showing fallback")

    # Core spending only by default. Including the semester fees pushes the
    # y-axis to 80,000 and squashes all ten normal months into identical stubs,
    # which is the exact problem the core/one-off split exists to solve.
    has_one_off = float(trend["one_off"].sum()) > 0
    include_fees = False
    if has_one_off:
        include_fees = st.toggle(
            "Include one-off semester fees",
            value=False,
            help="Fees are 5x a normal month and flatten the trend line.",
        )

    long = trend.melt(
        id_vars="month",
        value_vars=["core", "one_off"] if include_fees else ["core"],
        var_name="kind", value_name="amount",
    )
    long["kind"] = long["kind"].map({"core": "Core spending", "one_off": "One-off fees"})

    st.vega_lite_chart(
        long,
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 3, "tooltip": True},
            "encoding": {
                "x": {"field": "month", "type": "nominal",
                      "axis": {"title": None, "labelAngle": -45}},
                # zero=True: bars must be read against a zero baseline, or a
                # 15k-to-24k range looks like a tenfold increase.
                "y": {"field": "amount", "type": "quantitative",
                      "axis": {"title": None, "format": "~s"},
                      "scale": {"zero": True}},
                "color": {"field": "kind", "type": "nominal",
                          "scale": {"range": ["#4C78A8", "#F58518"]},
                          "legend": {"title": None, "orient": "top"}
                          if include_fees else None},
            },
            "height": 260,
        },
        width="stretch",
    )


def _chart_categories(categories: pd.DataFrame) -> None:
    """Category split -- their renderer if present, else native."""
    charts = _charts()
    if charts is not None:
        try:
            charts.render_category_chart(categories)
            return
        except Exception as exc:  # noqa: BLE001
            ui.error_box(exc, "Analytics chart failed, showing fallback")

    # st.bar_chart sorts a nominal axis alphabetically, which buries the biggest
    # category in the middle. Vega-Lite is built into Streamlit (no new
    # dependency) and does accept an explicit sort.
    st.vega_lite_chart(
        categories,
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 3, "tooltip": True},
            "encoding": {
                "y": {
                    "field": "category",
                    "type": "nominal",
                    "sort": "-x",              # descending by amount
                    "axis": {"title": None, "labelLimit": 140},
                },
                "x": {
                    "field": "amount",
                    "type": "quantitative",
                    "axis": {"title": None, "format": "~s"},
                },
                "color": {"value": "#4C78A8"},
            },
            "height": 260,
        },
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _render_kpis(kpis: dict, student: dict) -> None:
    """Five headline cards across the top of the dashboard."""
    columns = st.columns(5)

    with columns[0]:
        ui.metric_card("Spent this month", ui.money_compact(kpis["current_month_spend"]))

    with columns[1]:
        change = kpis["month_change_pct"]
        # Spending up is bad news, so the colour mapping is deliberately
        # inverted relative to a stock metric widget.
        direction = "up" if change > 1 else "down" if change < -1 else "flat"
        ui.metric_card(
            "vs last month",
            f"{change:+.1f}%",
            "current month still in progress",
            direction,
        )

    with columns[2]:
        used = kpis["budget_used_pct"]
        remaining = kpis["budget_remaining"]
        ui.metric_card(
            "Budget used",
            f"{used:.0f}%",
            f"{ui.money_compact(abs(remaining))} {'left' if remaining >= 0 else 'over'}",
            "up" if used > 100 else "down" if used < 80 else "flat",
        )

    with columns[3]:
        ui.metric_card("Daily average", ui.money_compact(kpis["daily_average"]),
                       f"over {kpis['active_days']} active days")

    with columns[4]:
        ui.metric_card("Top category", kpis["top_category"],
                       ui.money_compact(kpis["top_category_amount"]), wrap=True)

    # One-off fees are excluded from every figure above; saying so explicitly
    # prevents the obvious "these numbers look too low" question in a demo.
    if kpis["one_off_total"] > 0:
        st.caption(
            f"Figures exclude {ui.money_compact(kpis['one_off_total'])} of one-off "
            "semester fees, which are reported separately below."
        )


def _render_budget_table(budget: pd.DataFrame, month: str) -> None:
    """Budget vs actual, with a progress bar per category."""
    ui.section(
        f"Budget vs actual - {month}",
        "One-off semester fees are excluded; no student budgets tuition monthly.",
    )
    if budget.empty:
        ui.empty_state("No budget data for this month.")
        return

    st.dataframe(
        budget,
        hide_index=True,
        width="stretch",
        column_config={
            "category": st.column_config.TextColumn("Category"),
            "spent": st.column_config.NumberColumn("Spent", format="%.0f"),
            "limit_amount": st.column_config.NumberColumn("Limit", format="%.0f"),
            "remaining": st.column_config.NumberColumn("Remaining", format="%.0f"),
            "used_pct": st.column_config.ProgressColumn(
                "Used",
                format="%.0f%%",
                min_value=0,
                # Cap the bar at 200% so one big overspend does not squash
                # every other row into an invisible sliver.
                max_value=200,
            ),
            "status": st.column_config.TextColumn("Status"),
        },
    )


def _render_goals(goals: pd.DataFrame) -> None:
    """Savings goal progress bars."""
    if goals.empty:
        return
    ui.section("Savings goals")
    for row in goals.itertuples():
        left, right = st.columns([3, 1])
        with left:
            st.progress(
                min(float(row.progress_pct) / 100, 1.0),
                text=f"{row.title} - {ui.money(row.saved_amount)} of "
                     f"{ui.money(row.target_amount)}",
            )
        with right:
            st.markdown(
                ui.pill(f"{row.progress_pct:.0f}%",
                        "ok" if row.progress_pct >= 60 else "warn"),
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Draw the Overview tab."""
    analytics_status = integration.feature("analytics")
    if not ui.guard(analytics_status, "Dashboard analytics"):
        return

    student_id = state.get_student_id()
    student = state.load_student(student_id)
    if student is None:
        st.error("Selected student not found. Regenerate the demo database.")
        return

    expenses = state.load_expenses(student_id)
    if expenses.empty:
        ui.empty_state(
            "No transactions recorded yet.",
            "Add one in the **Transactions** tab, or generate demo data with "
            "`python backend/scripts/generate_mock_data.py --force`.",
        )
        return

    try:
        kpis = analytics_status.call("kpi_summary", expenses, student["monthly_budget"])
        trend = analytics_status.call("monthly_trend", expenses)
        categories = analytics_status.call("category_breakdown", expenses)
    except (integration.FeatureError, integration.FeatureUnavailable) as exc:
        ui.error_box(exc, "Analytics failed")
        return

    _render_kpis(kpis, student)
    st.divider()

    # The two chart sources cover different periods and scopes, so the captions
    # have to follow whichever is actually rendering. The Analytics module plots
    # the CURRENT MONTH day by day, across all categories; the native fallback
    # plots EVERY MONTH with one-off fees separated out. Describing one while
    # showing the other is how a demo loses its audience.
    using_their_charts = _charts() is not None

    left, right = st.columns([3, 2])
    with left:
        if using_their_charts:
            ui.section("Daily spending", "Day by day through the current month.")
        else:
            ui.section("Monthly spending", "Core spending separated from one-off fees.")
        if trend.empty:
            ui.empty_state("Not enough history to plot a trend.")
        else:
            _chart_trend(trend)
    with right:
        if using_their_charts:
            ui.section("Where it goes", "Share of this month's spending by category.")
        else:
            ui.section("Where it goes", "Share of core spending by category.")
        if categories.empty:
            ui.empty_state("No categorised spending yet.")
        else:
            _chart_categories(categories.head(8))

    st.divider()

    latest_month = expenses["txn_date"].max().strftime("%Y-%m")
    budgets = state.load_budgets(student_id)
    module = analytics_status.module
    if module is not None and hasattr(module, "budget_vs_actual"):
        try:
            _render_budget_table(
                module.budget_vs_actual(expenses, budgets, latest_month), latest_month
            )
        except Exception as exc:  # noqa: BLE001
            ui.error_box(exc, "Budget comparison failed")

    _render_goals(state.load_goals(student_id))
