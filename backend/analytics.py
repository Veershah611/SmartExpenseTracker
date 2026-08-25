"""
backend/analytics.py
====================
Every number and chart series in the app is computed here, with pandas.

Design rules
------------
* Functions take a **DataFrame** (as returned by :func:`database.get_expenses`)
  rather than a ``student_id``. That makes them pure, trivially testable, and
  means the UI can filter once and reuse the result across a dozen widgets
  instead of re-querying SQLite for each chart.
* Every function tolerates an empty DataFrame and returns an empty result of
  the right shape. A student with no data in a date range must not crash a tab.
* Nothing here imports Streamlit.

The one-off fee problem
-----------------------
Semester tuition (Rs 48,000+) lands in two months of the year and is 5x a normal
month's total spend. Charting it raw flattens every other month into a
featureless line. Rather than delete the data, :func:`split_core_and_one_off`
separates it so the UI can show "core spending" trends while still reporting
the fees honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from backend import database  # noqa: E402

# A transaction is treated as a one-off capital expense when it is both in a
# fee-like category and unusually large. The size test matters: a Rs 900
# Coursera charge is ordinary spending and belongs in the core trend, while a
# Rs 48,000 tuition payment plainly does not.
ONE_OFF_CATEGORIES: set[str] = {"Academics & Fees"}
ONE_OFF_MIN_AMOUNT: float = 10_000.0


# --------------------------------------------------------------------------- #
# Core / one-off separation
# --------------------------------------------------------------------------- #
def split_core_and_one_off(
    expenses: pd.DataFrame,
    min_amount: float = ONE_OFF_MIN_AMOUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into ``(core, one_off)``.

    ``core`` is day-to-day spending a student can actually influence.
    ``one_off`` is large, scheduled, unavoidable payments such as tuition.

    Reported separately because averaging them together produces a "monthly
    average" that describes no real month.
    """
    if expenses.empty:
        return expenses.copy(), expenses.copy()

    is_one_off = (
        expenses["category"].isin(ONE_OFF_CATEGORIES)
        & (expenses["amount"] >= min_amount)
    )
    return expenses.loc[~is_one_off].copy(), expenses.loc[is_one_off].copy()


# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
def kpi_summary(
    expenses: pd.DataFrame,
    monthly_budget: float,
    exclude_one_off: bool = True,
) -> dict[str, float | int]:
    """
    Headline metrics for the dashboard cards.

    ``month_change_pct`` compares the most recent month present in the data with
    the one before it. The latest month is usually still in progress, so this is
    labelled in the UI as a partial-month comparison rather than a clean MoM.
    """
    empty = {
        "total_spend": 0.0, "transaction_count": 0, "daily_average": 0.0,
        "avg_transaction": 0.0, "current_month_spend": 0.0,
        "month_change_pct": 0.0, "budget_used_pct": 0.0,
        "budget_remaining": float(monthly_budget), "one_off_total": 0.0,
        "top_category": "-", "top_category_amount": 0.0, "active_days": 0,
    }
    if expenses.empty:
        return empty

    core, one_off = split_core_and_one_off(expenses)
    frame = core if exclude_one_off else expenses
    if frame.empty:
        empty["one_off_total"] = float(one_off["amount"].sum())
        return empty

    months = frame["txn_date"].dt.to_period("M")
    monthly_totals = frame.groupby(months)["amount"].sum().sort_index()

    current_month_spend = float(monthly_totals.iloc[-1])
    if len(monthly_totals) >= 2:
        previous = float(monthly_totals.iloc[-2])
        # Guard against a zero-spend previous month producing an infinite change.
        month_change = ((current_month_spend - previous) / previous * 100) if previous else 0.0
    else:
        month_change = 0.0

    # Divide by distinct days that actually have transactions rather than the
    # calendar span: a student with a two-week gap should not look thriftier.
    active_days = int(frame["txn_date"].dt.normalize().nunique())
    by_category = frame.groupby("category")["amount"].sum().sort_values(ascending=False)

    return {
        "total_spend": float(frame["amount"].sum()),
        "transaction_count": int(len(frame)),
        "daily_average": float(frame["amount"].sum() / max(active_days, 1)),
        "avg_transaction": float(frame["amount"].mean()),
        "current_month_spend": current_month_spend,
        "month_change_pct": float(month_change),
        "budget_used_pct": float(current_month_spend / monthly_budget * 100)
        if monthly_budget else 0.0,
        "budget_remaining": float(monthly_budget - current_month_spend),
        "one_off_total": float(one_off["amount"].sum()),
        "top_category": str(by_category.index[0]),
        "top_category_amount": float(by_category.iloc[0]),
        "active_days": active_days,
    }


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def monthly_trend(expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Spend per calendar month, split into core and one-off columns.

    Returns columns: ``month`` (str 'YYYY-MM'), ``core``, ``one_off``, ``total``.
    """
    columns = ["month", "core", "one_off", "total"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    core, one_off = split_core_and_one_off(expenses)

    core_by_month = core.groupby(core["txn_date"].dt.to_period("M"))["amount"].sum()
    one_off_by_month = (
        one_off.groupby(one_off["txn_date"].dt.to_period("M"))["amount"].sum()
        if not one_off.empty else pd.Series(dtype=float)
    )

    frame = pd.DataFrame({"core": core_by_month, "one_off": one_off_by_month}).fillna(0.0)
    frame = frame.sort_index()
    frame["total"] = frame["core"] + frame["one_off"]
    frame.index = frame.index.astype(str)
    return frame.reset_index(names="month")[columns].round(2)


def daily_series(expenses: pd.DataFrame, rolling_window: int = 7) -> pd.DataFrame:
    """
    Daily totals with a rolling average, reindexed so zero-spend days appear.

    Without the reindex, matplotlib/plotly would connect across gaps and hide
    exactly the quiet stretches that make the month-end squeeze visible.
    """
    columns = ["date", "amount", "rolling_avg"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame(columns=columns)

    daily = core.groupby(core["txn_date"].dt.normalize())["amount"].sum()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)

    frame = daily.reset_index()
    frame.columns = ["date", "amount"]
    frame["rolling_avg"] = (
        frame["amount"].rolling(window=rolling_window, min_periods=1).mean()
    )
    return frame.round(2)


def category_breakdown(expenses: pd.DataFrame, exclude_one_off: bool = True) -> pd.DataFrame:
    """Total, share, count and average per category, biggest first."""
    columns = ["category", "amount", "share_pct", "transactions", "avg_amount"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    frame = split_core_and_one_off(expenses)[0] if exclude_one_off else expenses
    if frame.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        frame.groupby("category")["amount"]
        .agg(amount="sum", transactions="count", avg_amount="mean")
        .reset_index()
        .sort_values("amount", ascending=False)
    )
    total = grouped["amount"].sum()
    grouped["share_pct"] = (grouped["amount"] / total * 100) if total else 0.0
    return grouped[columns].round(2)


def category_trend(expenses: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Month-by-month totals for the ``top_n`` categories -- a wide frame ready to
    hand straight to a stacked area chart.
    """
    if expenses.empty:
        return pd.DataFrame()

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame()

    top = core.groupby("category")["amount"].sum().nlargest(top_n).index
    subset = core[core["category"].isin(top)]

    pivot = subset.pivot_table(
        index=subset["txn_date"].dt.to_period("M"),
        columns="category",
        values="amount",
        aggfunc="sum",
    ).fillna(0.0)
    pivot.index = pivot.index.astype(str)
    return pivot.reset_index(names="month").round(2)


def weekday_pattern(expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Average spend per weekday, Monday first.

    Averages by *occurrence* of each weekday, not a plain mean of transactions,
    so a category with many small Saturday purchases does not distort the shape.
    """
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    if expenses.empty:
        return pd.DataFrame(columns=["weekday", "avg_amount", "total"])

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame(columns=["weekday", "avg_amount", "total"])

    daily = core.groupby(core["txn_date"].dt.normalize())["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["weekday"] = daily["date"].dt.day_name()

    grouped = daily.groupby("weekday")["amount"].agg(avg_amount="mean", total="sum")
    grouped = grouped.reindex(order).fillna(0.0).reset_index()
    grouped.columns = ["weekday", "avg_amount", "total"]
    return grouped.round(2)


def month_progression(expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Average spend by position within the month (day 1-31).

    This is what makes the "front-loaded spending / month-end crunch" insight
    provable rather than a guess.
    """
    if expenses.empty:
        return pd.DataFrame(columns=["day_of_month", "avg_amount"])

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame(columns=["day_of_month", "avg_amount"])

    daily = core.groupby(core["txn_date"].dt.normalize())["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["day_of_month"] = daily["date"].dt.day

    grouped = daily.groupby("day_of_month")["amount"].mean().reset_index()
    grouped.columns = ["day_of_month", "avg_amount"]
    return grouped.round(2)


# --------------------------------------------------------------------------- #
# Merchants, payment modes
# --------------------------------------------------------------------------- #
def top_merchants(expenses: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Where the money actually goes, ranked by total spend."""
    columns = ["merchant", "amount", "transactions", "category"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        expenses.groupby("merchant")
        .agg(
            amount=("amount", "sum"),
            transactions=("amount", "count"),
            # A merchant sits in one category in practice; take the commonest.
            category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else "-"),
        )
        .reset_index()
        .sort_values("amount", ascending=False)
        .head(limit)
    )
    return grouped[columns].round(2)


def payment_mode_split(expenses: pd.DataFrame) -> pd.DataFrame:
    """Spend by payment method -- sets up the 'UPI is invisible' insight."""
    columns = ["payment_mode", "amount", "transactions", "share_pct"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        expenses.groupby("payment_mode")["amount"]
        .agg(amount="sum", transactions="count")
        .reset_index()
        .sort_values("amount", ascending=False)
    )
    total = grouped["amount"].sum()
    grouped["share_pct"] = (grouped["amount"] / total * 100) if total else 0.0
    return grouped[columns].round(2)


# --------------------------------------------------------------------------- #
# Budget comparison
# --------------------------------------------------------------------------- #
def budget_vs_actual(
    expenses: pd.DataFrame,
    budgets: pd.DataFrame,
    month: str,
    exclude_one_off: bool = True,
) -> pd.DataFrame:
    """
    Compare actual spend against the limit for one ``'YYYY-MM'`` month.

    Outer-joined on purpose: a category that was budgeted but never spent on is
    as interesting as an overspend, and both must appear in the table.

    One-off fees are excluded by default. Nobody budgets tuition out of a
    monthly allowance, so including a Rs 49,000 semester payment against a
    Rs 1,350 monthly line reports "3,655% over budget" -- technically true, and
    useless. The UI reports those payments separately instead.
    """
    columns = ["category", "spent", "limit_amount", "remaining", "used_pct", "status"]

    month_budgets = (
        budgets[budgets["month"] == month][["category", "limit_amount"]]
        if not budgets.empty else pd.DataFrame(columns=["category", "limit_amount"])
    )

    if exclude_one_off and not expenses.empty:
        expenses = split_core_and_one_off(expenses)[0]

    if expenses.empty:
        actual = pd.DataFrame(columns=["category", "spent"])
    else:
        in_month = expenses[expenses["txn_date"].dt.strftime("%Y-%m") == month]
        actual = (
            in_month.groupby("category")["amount"].sum().reset_index()
            if not in_month.empty else pd.DataFrame(columns=["category", "amount"])
        )
        if not actual.empty:
            actual.columns = ["category", "spent"]

    if month_budgets.empty and actual.empty:
        return pd.DataFrame(columns=columns)

    merged = month_budgets.merge(actual, on="category", how="outer")
    # Fill only the numeric columns. A blanket .fillna(0.0) would also touch the
    # object-dtype 'category' column, which pandas downcasts with a FutureWarning.
    for column in ("limit_amount", "spent"):
        if column not in merged.columns:
            merged[column] = 0.0
    merged[["limit_amount", "spent"]] = (
        merged[["limit_amount", "spent"]].astype(float).fillna(0.0)
    )

    merged["remaining"] = merged["limit_amount"] - merged["spent"]
    merged["used_pct"] = np.where(
        merged["limit_amount"] > 0,
        merged["spent"] / merged["limit_amount"].replace(0, np.nan) * 100,
        0.0,
    )
    merged["used_pct"] = merged["used_pct"].fillna(0.0)

    merged["status"] = pd.cut(
        merged["used_pct"],
        bins=[-np.inf, 80, 100, np.inf],
        labels=["On track", "Near limit", "Over budget"],
    ).astype(str)

    return merged.sort_values("used_pct", ascending=False)[columns].round(2)


# --------------------------------------------------------------------------- #
# Pattern detection
# --------------------------------------------------------------------------- #
def detect_recurring(expenses: pd.DataFrame, min_occurrences: int = 3) -> pd.DataFrame:
    """
    Find likely subscriptions: the same merchant charging a stable amount at a
    roughly monthly cadence.

    Detection is behavioural, not a lookup of the ``is_recurring`` flag -- the
    point is to catch subscriptions the student never labelled. A charge counts
    as recurring when the amount barely varies (coefficient of variation < 15%)
    and the median gap between charges is 25-35 days.
    """
    columns = ["merchant", "category", "avg_amount", "occurrences",
               "median_gap_days", "annual_cost"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    found = []
    for merchant, group in expenses.groupby("merchant"):
        if len(group) < min_occurrences:
            continue

        amounts = group["amount"]
        mean_amount = amounts.mean()
        if mean_amount <= 0:
            continue
        # Coefficient of variation: stable price is the strongest subscription tell.
        if amounts.std(ddof=0) / mean_amount > 0.15:
            continue

        gaps = group["txn_date"].sort_values().diff().dt.days.dropna()
        if gaps.empty:
            continue
        median_gap = float(gaps.median())
        if not 25 <= median_gap <= 35:
            continue

        found.append({
            "merchant": merchant,
            "category": group["category"].mode().iloc[0],
            "avg_amount": round(float(mean_amount), 2),
            "occurrences": int(len(group)),
            "median_gap_days": round(median_gap, 1),
            "annual_cost": round(float(mean_amount) * 12, 2),
        })

    if not found:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(found).sort_values("annual_cost", ascending=False)[columns]


def detect_anomalies(expenses: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag transactions that are unusually large *for their own category*.

    Compared within-category deliberately: a Rs 2,000 tuition-adjacent charge is
    unremarkable, while a Rs 2,000 canteen charge is not. A global threshold
    would only ever surface the fee payments.
    """
    columns = ["txn_date", "merchant", "category", "amount", "category_avg", "times_avg"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame(columns=columns)

    flagged = []
    for category, group in core.groupby("category"):
        # Need enough history for a standard deviation to mean anything.
        if len(group) < 5:
            continue
        mean = group["amount"].mean()
        std = group["amount"].std(ddof=0)
        if std == 0 or pd.isna(std):
            continue

        outliers = group[(group["amount"] - mean) / std > z_threshold].copy()
        if outliers.empty:
            continue
        outliers["category_avg"] = round(float(mean), 2)
        outliers["times_avg"] = (outliers["amount"] / mean).round(1)
        flagged.append(outliers)

    if not flagged:
        return pd.DataFrame(columns=columns)

    result = pd.concat(flagged).sort_values("amount", ascending=False)
    return result[columns]


def compare_to_benchmark(expenses: pd.DataFrame, monthly_budget: float) -> pd.DataFrame:
    """
    Compare the student's category split against the ``typical_share`` benchmarks
    in config, expressed in rupees per month.

    Turns "you spend 34% on food" into "you spend Rs 1,100 a month more on food
    than a typical student on your budget" -- a number worth acting on.
    """
    columns = ["category", "actual_monthly", "benchmark_monthly", "difference", "verdict"]
    if expenses.empty:
        return pd.DataFrame(columns=columns)

    core, _ = split_core_and_one_off(expenses)
    if core.empty:
        return pd.DataFrame(columns=columns)

    months = max(core["txn_date"].dt.to_period("M").nunique(), 1)
    actual = (core.groupby("category")["amount"].sum() / months).round(2)

    rows = []
    for name, _icon, share in config.EXPENSE_CATEGORIES:
        benchmark = round(monthly_budget * share, 2)
        actual_amount = float(actual.get(name, 0.0))
        difference = round(actual_amount - benchmark, 2)
        rows.append({
            "category": name,
            "actual_monthly": actual_amount,
            "benchmark_monthly": benchmark,
            "difference": difference,
            "verdict": "Above typical" if difference > 0 else "Below typical",
        })

    return pd.DataFrame(rows).sort_values("difference", ascending=False)[columns]


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def build_full_report(student_id: int) -> dict:
    """
    Compute every analytic for one student in a single pass.

    Used by the insights engine and the RAG context builder, which both need the
    whole picture. Fetching once and sharing the DataFrame avoids ten separate
    SQLite round trips.
    """
    student = database.get_student(student_id)
    if student is None:
        raise ValueError(f"No student with id {student_id}")

    expenses = database.get_expenses(student_id)
    budgets = database.get_budgets(student_id)

    latest_month = (
        expenses["txn_date"].max().strftime("%Y-%m")
        if not expenses.empty else pd.Timestamp.today().strftime("%Y-%m")
    )

    return {
        "student": student,
        "latest_month": latest_month,
        "kpis": kpi_summary(expenses, student["monthly_budget"]),
        "monthly_trend": monthly_trend(expenses),
        "categories": category_breakdown(expenses),
        "budget_vs_actual": budget_vs_actual(expenses, budgets, latest_month),
        "recurring": detect_recurring(expenses),
        "anomalies": detect_anomalies(expenses),
        "benchmark": compare_to_benchmark(expenses, student["monthly_budget"]),
        "top_merchants": top_merchants(expenses),
        "payment_modes": payment_mode_split(expenses),
        "weekday": weekday_pattern(expenses),
    }
