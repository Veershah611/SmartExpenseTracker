"""
frontend/components/insights.py
===============================
The Insights tab -- automated recommendations.
**Layout and LLM orchestration owned by the Core Integrator.**

Hosts two teammates' headline features side by side:

* **Predictive Broke Alert** -- Analytics & Forecasting Developer
  (``backend/forecasting.py::predict_broke_alert``)
* **Subscription Ghost Hunter** -- Data & Database Engineer
  (``backend/ghost_hunter.py::find_recurring_charges``)

Plus the Integrator's own contribution: an LLM-written summary grounded in
rule-based signals. The signals are computed in pandas first and the model is
asked only to phrase them, so an offline model costs polish, not correctness --
with no LLM the raw signals still render as a plain list.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402


# --------------------------------------------------------------------------- #
# Rule-based signals (deterministic, no LLM)
# --------------------------------------------------------------------------- #
def _burn_rate_signal(expenses: pd.DataFrame, monthly_budget: float) -> dict | None:
    """
    The Integrator's stand-in for the Broke Alert until Analytics deliver theirs.

    Straight-line projection: current daily rate extended to month end. Crude on
    purpose -- a transparent rule a judge can follow beats an opaque forecast,
    and the real model is a teammate's deliverable, not mine to pre-empt.
    """
    if expenses.empty:
        return None

    analytics = integration.feature("analytics")
    if analytics.ready and analytics.module is not None:
        splitter = getattr(analytics.module, "split_core_and_one_off", None)
        if splitter is not None:
            expenses = splitter(expenses)[0]
    if expenses.empty:
        return None

    latest = expenses["txn_date"].max()
    this_month = expenses[expenses["txn_date"].dt.to_period("M") == latest.to_period("M")]
    if this_month.empty:
        return None

    spent = float(this_month["amount"].sum())
    day_of_month = int(latest.day)

    # Days in the current month, without importing calendar arithmetic twice.
    next_month = (latest.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
    days_in_month = int((next_month - pd.Timedelta(days=1)).day)

    daily_rate = spent / max(day_of_month, 1)
    projected = daily_rate * days_in_month
    remaining_budget = monthly_budget - spent

    days_until_broke = (
        int(remaining_budget / daily_rate) if daily_rate > 0 and remaining_budget > 0
        else 0
    )

    return {
        "spent": spent,
        "daily_rate": daily_rate,
        "projected": projected,
        "monthly_budget": monthly_budget,
        "overspend": projected - monthly_budget,
        "days_left": days_in_month - day_of_month,
        "days_until_broke": days_until_broke,
        "on_track": projected <= monthly_budget,
    }


def _render_broke_alert(expenses: pd.DataFrame, student: dict) -> None:
    """Predictive Broke Alert -- teammate's model if delivered, else the fallback."""
    ui.section(
        "Predictive Broke Alert",
        "Projects month-end spending from the current burn rate.",
    )

    forecasting = integration.feature("broke_alert")
    if forecasting.ready:
        try:
            result = forecasting.call("predict_broke_alert", expenses, student)
            # Accept a ready-made message or a structured dict.
            if isinstance(result, dict):
                message = result.get("message", "")
                severity = result.get("severity", "info")
                {"danger": st.error, "warning": st.warning}.get(
                    severity, st.info
                )(message)
            else:
                st.info(str(result))
            return
        except (integration.FeatureError, integration.FeatureUnavailable) as exc:
            ui.error_box(exc, "Forecasting module failed, showing fallback")

    signal = _burn_rate_signal(expenses, student["monthly_budget"])
    if signal is None:
        ui.empty_state("Not enough data this month to project a burn rate.")
        return

    columns = st.columns(3)
    with columns[0]:
        ui.metric_card("Daily burn rate", ui.money(signal["daily_rate"]))
    with columns[1]:
        ui.metric_card("Projected month end", ui.money(signal["projected"]),
                       f"budget {ui.money(signal['monthly_budget'])}",
                       "up" if not signal["on_track"] else "down")
    with columns[2]:
        ui.metric_card("Days left in month", str(signal["days_left"]))

    # Severity has to be proportionate. A projection landing Rs 12 over a
    # Rs 15,000 budget is a rounding error, not an emergency -- shouting about it
    # in red trains the student to ignore the alert that actually matters.
    overspend = signal["overspend"]
    overspend_share = overspend / signal["monthly_budget"] if signal["monthly_budget"] else 0

    if signal["on_track"]:
        st.success(
            f"On track. At {ui.money(signal['daily_rate'])} a day you should finish "
            f"the month around {ui.money(signal['projected'])}, inside your "
            f"{ui.money(signal['monthly_budget'])} budget.",
            icon=":material/check_circle:",
        )
    elif overspend_share < 0.05:
        st.warning(
            f"Right on the line. At {ui.money(signal['daily_rate'])} a day you are "
            f"projected to finish at {ui.money(signal['projected'])}, just "
            f"{ui.money(overspend)} over your {ui.money(signal['monthly_budget'])} "
            f"budget. Easing off slightly for the last {signal['days_left']} days "
            "keeps you inside it.",
            icon=":material/balance:",
        )
    else:
        st.error(
            f"At {ui.money(signal['daily_rate'])} a day you are heading for "
            f"{ui.money(signal['projected'])} - about "
            f"{ui.money(overspend)} over budget. "
            f"Your money runs out in roughly {signal['days_until_broke']} days, "
            f"with {signal['days_left']} still to go.",
            icon=":material/trending_up:",
        )

    if not forecasting.ready:
        st.caption(
            "Straight-line fallback projection. The Analytics developer's model "
            "replaces this via `backend/forecasting.py`."
        )


def _render_ghost_hunter(expenses: pd.DataFrame) -> None:
    """Subscription Ghost Hunter -- recurring charges the student may have forgotten."""
    ui.section(
        "Subscription Ghost Hunter",
        "Merchants charging a stable amount on a monthly cadence.",
    )

    hunter = integration.feature("ghost_hunter")
    if not ui.guard(hunter, "Subscription Ghost Hunter"):
        return

    try:
        found = hunter.call("find_recurring_charges", expenses)
    except (integration.FeatureError, integration.FeatureUnavailable) as exc:
        ui.error_box(exc, "Ghost Hunter failed")
        return

    if not isinstance(found, pd.DataFrame) or found.empty:
        st.success("No hidden recurring charges found.", icon=":material/check_circle:")
        return

    annual = float(found["annual_cost"].sum()) if "annual_cost" in found else 0.0
    st.warning(
        f"Found **{len(found)}** recurring charges costing about "
        f"**{ui.money(annual)} a year**.",
        icon=":material/receipt_long:",
    )
    st.dataframe(
        found, hide_index=True, width="stretch",
        column_config={
            "merchant": st.column_config.TextColumn("Merchant"),
            "category": st.column_config.TextColumn("Category"),
            "avg_amount": st.column_config.NumberColumn("Per charge", format="%.0f"),
            "occurrences": st.column_config.NumberColumn("Seen"),
            "median_gap_days": st.column_config.NumberColumn("Every (days)", format="%.0f"),
            "annual_cost": st.column_config.NumberColumn("Yearly cost", format="%.0f"),
        },
    )
    if hunter.using_fallback:
        st.caption(
            "Using the reference detector. The Data Engineer's implementation "
            "replaces it via `backend/ghost_hunter.py`."
        )


def _render_ai_summary(expenses: pd.DataFrame, student: dict) -> None:
    """LLM-written summary of the deterministic signals."""
    ui.section("AI recommendations", "Written from your verified figures.")

    analytics = integration.feature("analytics")
    if not analytics.ready or expenses.empty:
        ui.empty_state("Not enough data to generate recommendations.")
        return

    facts: list[str] = []
    try:
        kpis = analytics.call("kpi_summary", expenses, student["monthly_budget"])
        facts.append(
            f"Monthly budget Rs {student['monthly_budget']:,.0f}; "
            f"spent Rs {kpis['current_month_spend']:,.0f} this month "
            f"({kpis['budget_used_pct']:.0f}%)."
        )

        module = analytics.module
        if module is not None and hasattr(module, "compare_to_benchmark"):
            benchmark = module.compare_to_benchmark(expenses, student["monthly_budget"])
            over = benchmark[benchmark["difference"] > 0].head(3)
            for row in over.itertuples():
                facts.append(
                    f"{row.category}: Rs {row.actual_monthly:,.0f}/month vs typical "
                    f"Rs {row.benchmark_monthly:,.0f} (Rs {row.difference:,.0f} more)."
                )

        signal = _burn_rate_signal(expenses, student["monthly_budget"])
        if signal and not signal["on_track"]:
            facts.append(
                f"Projected to finish the month Rs {signal['overspend']:,.0f} "
                "over budget at the current rate."
            )
    except Exception as exc:  # noqa: BLE001
        ui.error_box(exc, "Could not assemble signals")
        return

    # Always show the deterministic signals -- these are the substance, and they
    # do not depend on a model being available.
    for fact in facts:
        st.markdown(f"- {fact}")

    if not llm_engine.is_available():
        st.caption("Start a local LLM for a written summary of these signals.")
        return

    if not st.button("Write recommendations", type="primary"):
        return

    prompt = (
        "You are advising an Indian university student on their spending.\n"
        "VERIFIED FIGURES (use these exact numbers, do not recalculate):\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + "\n\nWrite exactly three short, specific recommendations as a numbered "
          "list. Each must reference one of the figures above and suggest a "
          "concrete action. Amounts in rupees, written like Rs 1,250."
    )
    try:
        with st.chat_message("assistant"):
            st.write_stream(llm_engine.chat_stream(
                [{"role": "user", "content": prompt}]
            ))
    except llm_engine.LLMUnavailableError as exc:
        st.warning(str(exc), icon=":material/smart_toy:")


def _render_anomalies(expenses: pd.DataFrame) -> None:
    """Unusually large transactions, compared within their own category."""
    analytics = integration.feature("analytics")
    module = analytics.module if analytics.ready else None
    if module is None or not hasattr(module, "detect_anomalies"):
        return

    try:
        anomalies = module.detect_anomalies(expenses)
    except Exception:  # noqa: BLE001
        return
    if anomalies.empty:
        return

    ui.section("Unusual transactions", "Large relative to their own category.")
    display = anomalies.head(8).copy()
    display["txn_date"] = display["txn_date"].dt.strftime("%d %b %Y")
    st.dataframe(
        display, hide_index=True, width="stretch",
        column_config={
            "txn_date": st.column_config.TextColumn("Date"),
            "merchant": st.column_config.TextColumn("Merchant"),
            "category": st.column_config.TextColumn("Category"),
            "amount": st.column_config.NumberColumn("Amount", format="%.0f"),
            "category_avg": st.column_config.NumberColumn("Category avg", format="%.0f"),
            "times_avg": st.column_config.NumberColumn("x avg", format="%.1f"),
        },
    )


def render() -> None:
    """Draw the Insights tab."""
    student_id = state.get_student_id()
    student = state.load_student(student_id)
    if student is None:
        st.error("Select a student first.")
        return

    expenses = state.load_expenses(student_id)
    if expenses.empty:
        ui.empty_state("No transactions yet - nothing to analyse.")
        return

    _render_broke_alert(expenses, student)
    st.divider()
    _render_ghost_hunter(expenses)
    st.divider()
    _render_ai_summary(expenses, student)
    st.divider()
    _render_anomalies(expenses)
