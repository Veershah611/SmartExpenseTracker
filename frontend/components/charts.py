"""
frontend/components/charts.py
=============================
Renders the Analytics developer's Plotly figures into the dashboard.
**Owned by the Core Integrator; the figures themselves are theirs.**

``backend/analytics_engine.py`` builds fully-configured ``go.Figure`` objects
but, correctly, never imports Streamlit. This module is the missing half: it
supplies the connection their functions expect, and calls ``st.plotly_chart``.

It satisfies the ``charts`` contract, so the dashboard picks it up automatically
and stops using the native fallback charts.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from frontend import state, ui  # noqa: E402


def _figure(builder_name: str):
    """
    Build one figure by name, opening and closing the connection around it.

    Returns ``None`` on any failure so the dashboard can fall back to its native
    chart rather than showing an error where a picture should be.
    """
    from backend import analytics_engine

    builder = getattr(analytics_engine, builder_name, None)
    if builder is None:
        return None

    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        return builder(student_id=state.get_student_id(), conn=conn)
    except Exception:  # noqa: BLE001 -- a missing chart must not break the tab
        return None
    finally:
        if conn is not None:
            conn.close()


def render_category_chart(categories: pd.DataFrame) -> None:
    """
    Category split.

    The DataFrame argument is part of the contract; their builder queries the
    database itself, so it is used only for the empty-state check.
    """
    figure = _figure("spending_by_category_chart")
    if figure is None:
        if categories.empty:
            ui.empty_state("No categorised spending yet.")
            return
        # Their builder failed -- raise so the dashboard's except clause draws
        # its own native chart instead of leaving a blank panel.
        raise RuntimeError("analytics_engine.spending_by_category_chart unavailable")

    st.plotly_chart(figure, width="stretch", key="chart_categories")


def render_trend_chart(trend: pd.DataFrame) -> None:
    """Daily spending trend for the current month."""
    figure = _figure("daily_spending_trend_chart")
    if figure is None:
        if trend.empty:
            ui.empty_state("Not enough history to plot a trend.")
            return
        raise RuntimeError("analytics_engine.daily_spending_trend_chart unavailable")

    st.plotly_chart(figure, width="stretch", key="chart_trend")


def render_payment_modes() -> None:
    """
    Payment-mode split.

    Not part of the contract -- an extra their module provides, surfaced on the
    Insights tab rather than left unused.
    """
    figure = _figure("payment_mode_breakdown")
    if figure is None:
        return
    st.plotly_chart(figure, width="stretch", key="chart_payment_modes")
