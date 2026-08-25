"""
analytics_engine.py
===================
Pure-Python analytics module that powers the **Analytics** tab of the Smart
Expense Tracker dashboard.

Responsibilities
----------------
- Join ``expenses`` with ``categories`` into a tidy Pandas DataFrame.
- Compute monthly KPIs (total spend, daily average, top category).
- Build and return fully-configured **Plotly** figure objects ready for
  ``st.plotly_chart()`` in the frontend layer.

Contract
--------
Every public function accepts a ``student_id`` and an **open**
``sqlite3.Connection``.  No function in this module may import ``streamlit``.
All constants (colours, currency symbol, chart palette) come from ``config.py``.

Usage (from the frontend)::

    import sqlite3, config
    from backend.analytics_engine import (
        spending_by_category_chart,
        daily_spending_trend_chart,
        monthly_summary,
        payment_mode_breakdown,
    )

    conn = sqlite3.connect(config.DB_PATH)
    fig  = spending_by_category_chart(student_id=1, conn=conn)
    # hand `fig` to st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import sqlite3
from calendar import monthrange
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Project-level configuration — the single source of truth.
# ---------------------------------------------------------------------------
import config

# ---------------------------------------------------------------------------
# Shared colour constants (from config).
# ---------------------------------------------------------------------------
_COLORS: list[str] = config.CHART_COLORS
_CURRENCY: str = config.CURRENCY_SYMBOL

# ---------------------------------------------------------------------------
# Plotly layout template — dark, transparent, consistent across all charts.
# ---------------------------------------------------------------------------
_LAYOUT_DEFAULTS: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1e1e2f",
        font_size=13,
        font_family="Inter, Segoe UI, Roboto, sans-serif",
    ),
)


# =========================================================================== #
#  Base data loader                                                           #
# =========================================================================== #
def get_expenses_df(
    student_id: int,
    conn: sqlite3.Connection,
    month: int | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Load expenses joined with category names for a given student and
    calendar month.

    Parameters
    ----------
    student_id : int
        Primary key from the ``students`` table.
    conn : sqlite3.Connection
        An open connection to ``expenses.db``.
    month, year : int | None
        Calendar month/year.  Default to the current month if omitted.

    Returns
    -------
    pd.DataFrame
        Columns: ``id, txn_date, amount, merchant, description,
        payment_mode, source, is_recurring, category, category_icon``.
        ``txn_date`` is a proper ``datetime64`` column.
    """
    today = date.today()
    month = month or today.month
    year = year or today.year

    # Build the date-range boundaries as ISO strings (the way the DB stores
    # dates) so the composite index on (student_id, txn_date) is used.
    start_date = f"{year:04d}-{month:02d}-01"
    last_day = monthrange(year, month)[1]
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"

    query = """
        SELECT e.id,
               e.txn_date,
               e.amount,
               e.merchant,
               e.description,
               e.payment_mode,
               e.source,
               e.is_recurring,
               c.name      AS category,
               c.icon      AS category_icon
        FROM   expenses  e
        JOIN   categories c ON c.id = e.category_id
        WHERE  e.student_id = ?
          AND  e.txn_date BETWEEN ? AND ?
        ORDER  BY e.txn_date
    """

    df = pd.read_sql_query(query, conn, params=(student_id, start_date, end_date))
    if not df.empty:
        df["txn_date"] = pd.to_datetime(df["txn_date"])
    return df


# =========================================================================== #
#  Chart 1 — Spending by Category (Donut)                                     #
# =========================================================================== #
def spending_by_category_chart(
    student_id: int,
    conn: sqlite3.Connection,
    month: int | None = None,
    year: int | None = None,
) -> go.Figure:
    """
    Return a Plotly **donut chart** of total spending grouped by category.

    Each slice label shows the category emoji + name and the absolute amount
    in local currency.  The centre annotation displays the grand total.
    """
    df = get_expenses_df(student_id, conn, month, year)
    today = date.today()
    month = month or today.month
    year = year or today.year

    if df.empty:
        return _empty_figure("No expenses recorded this month")

    cat_totals = (
        df.groupby("category", sort=False)["amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    grand_total = cat_totals["amount"].sum()

    fig = go.Figure(
        go.Pie(
            labels=cat_totals["category"],
            values=cat_totals["amount"],
            hole=0.52,
            marker=dict(colors=_COLORS[: len(cat_totals)]),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"Amount: {_CURRENCY}%{{value:,.0f}}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            pull=[0.03] * len(cat_totals),  # slight explosion for depth
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"Spending by Category — {_month_label(month, year)}",
            x=0.5,
            xanchor="center",
        ),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{config.as_currency(grand_total)}</b>",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )
    return fig


# =========================================================================== #
#  Chart 2 — Daily Spending Trend (Line + Rolling Average)                    #
# =========================================================================== #
def daily_spending_trend_chart(
    student_id: int,
    conn: sqlite3.Connection,
    month: int | None = None,
    year: int | None = None,
) -> go.Figure:
    """
    Return a Plotly **line chart** of daily aggregate spending for the month,
    with a 7-day rolling average overlay and a marker on the current day.
    """
    df = get_expenses_df(student_id, conn, month, year)
    today = date.today()
    month = month or today.month
    year = year or today.year

    if df.empty:
        return _empty_figure("No expenses recorded this month")

    # --- Aggregate per day, filling missing days with 0 ---
    last_day = monthrange(year, month)[1]
    # Only go up to today if viewing the current month
    if year == today.year and month == today.month:
        end_day = today.day
    else:
        end_day = last_day

    full_range = pd.date_range(
        start=f"{year:04d}-{month:02d}-01",
        end=f"{year:04d}-{month:02d}-{end_day:02d}",
        freq="D",
    )

    daily = (
        df.groupby(df["txn_date"].dt.date)["amount"]
        .sum()
        .reindex(full_range.date, fill_value=0.0)
    )
    daily.index = pd.to_datetime(daily.index)

    # 7-day rolling average (min_periods=1 so the first days aren't NaN).
    rolling_avg = daily.rolling(window=7, min_periods=1).mean()

    fig = go.Figure()

    # Area fill under the daily line.
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily.values,
            mode="lines+markers",
            name="Daily Spend",
            line=dict(color=_COLORS[0], width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(76,120,168,0.15)",
            hovertemplate=(
                "<b>%{x|%d %b}</b><br>"
                f"Spent: {_CURRENCY}%{{y:,.0f}}<extra></extra>"
            ),
        )
    )

    # Rolling average overlay.
    fig.add_trace(
        go.Scatter(
            x=rolling_avg.index,
            y=rolling_avg.values,
            mode="lines",
            name="7-Day Avg",
            line=dict(color=_COLORS[1], width=2, dash="dash"),
            hovertemplate=(
                "<b>%{x|%d %b}</b><br>"
                f"7-day avg: {_CURRENCY}%{{y:,.0f}}<extra></extra>"
            ),
        )
    )

    # Highlight today if it falls within the chart range.
    if year == today.year and month == today.month:
        today_ts = pd.Timestamp(today)
        if today_ts in daily.index:
            fig.add_trace(
                go.Scatter(
                    x=[today_ts],
                    y=[daily[today_ts]],
                    mode="markers",
                    name="Today",
                    marker=dict(
                        color=_COLORS[3],
                        size=12,
                        symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>Today</b><br>Spent: {_CURRENCY}%{{y:,.0f}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"Daily Spending Trend — {_month_label(month, year)}",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Date",
            dtick="D1",
            tickformat="%d",
            showgrid=False,
        ),
        yaxis=dict(
            title=f"Amount ({_CURRENCY})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )
    return fig


# =========================================================================== #
#  Chart 3 — Payment Mode Breakdown (Horizontal Bar)                          #
# =========================================================================== #
def payment_mode_breakdown(
    student_id: int,
    conn: sqlite3.Connection,
    month: int | None = None,
    year: int | None = None,
) -> go.Figure:
    """
    Return a Plotly **horizontal bar chart** of spending split by payment
    mode (UPI, Cash, Debit Card, etc.).
    """
    df = get_expenses_df(student_id, conn, month, year)
    today = date.today()
    month = month or today.month
    year = year or today.year

    if df.empty:
        return _empty_figure("No expenses recorded this month")

    mode_totals = (
        df.groupby("payment_mode", sort=False)["amount"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=mode_totals["amount"],
            y=mode_totals["payment_mode"],
            orientation="h",
            marker=dict(
                color=_COLORS[: len(mode_totals)],
                line=dict(width=0),
            ),
            text=[config.as_currency(v) for v in mode_totals["amount"]],
            textposition="auto",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Total: {_CURRENCY}%{{x:,.0f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"Spending by Payment Mode — {_month_label(month, year)}",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=f"Amount ({_CURRENCY})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(title=""),
    )
    return fig


# =========================================================================== #
#  KPI helper — Monthly Summary dict                                          #
# =========================================================================== #
def monthly_summary(
    student_id: int,
    conn: sqlite3.Connection,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    """
    Return a dictionary of headline KPIs for the given month::

        {
            "total_spent":      float,
            "daily_average":    float,
            "transaction_count": int,
            "top_category":     str,
            "top_category_amount": float,
            "month_label":      str,   # e.g. "August 2026"
        }

    Returns zeroed-out values (not an error) when no expenses exist.
    """
    df = get_expenses_df(student_id, conn, month, year)
    today = date.today()
    month = month or today.month
    year = year or today.year

    if df.empty:
        return {
            "total_spent": 0.0,
            "daily_average": 0.0,
            "transaction_count": 0,
            "top_category": "N/A",
            "top_category_amount": 0.0,
            "month_label": _month_label(month, year),
        }

    total = df["amount"].sum()

    # Days elapsed: if current month, use today's date; otherwise full month.
    if year == today.year and month == today.month:
        days = max(today.day, 1)
    else:
        days = monthrange(year, month)[1]

    cat_totals = df.groupby("category")["amount"].sum()
    top_cat = cat_totals.idxmax()

    return {
        "total_spent": round(total, 2),
        "daily_average": round(total / days, 2),
        "transaction_count": len(df),
        "top_category": top_cat,
        "top_category_amount": round(cat_totals[top_cat], 2),
        "month_label": _month_label(month, year),
    }


# =========================================================================== #
#  Internal helpers                                                           #
# =========================================================================== #
def _month_label(month: int, year: int) -> str:
    """Human-readable month label, e.g. 'August 2026'."""
    return f"{datetime(year, month, 1):%B %Y}"


def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centred annotation for empty-state UX."""
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#888"),
            )
        ],
    )
    return fig
