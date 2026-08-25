"""
frontend/components/transactions.py
===================================
The Transactions tab -- browse, filter, and add expenses.
**Owned by the Core Integrator.**

Also hosts the **Natural Language Quick Log** (RAG & NLP Developer,
``backend/rag_engine.py::parse_quick_log``). The Integrator owns the write path
around it: whatever the parser returns is shown to the student for confirmation
before anything is inserted.

That confirmation step is deliberate. A 3B model will occasionally parse
"spent 50 on chai" as category "chai", so writing straight to the database on a
model's say-so would corrupt the data mid-demo. Parse, preview, confirm, insert.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402

CATEGORY_NAMES = [name for name, _icon, _share in config.EXPENSE_CATEGORIES]


# --------------------------------------------------------------------------- #
# Quick Log
# --------------------------------------------------------------------------- #
def _render_quick_log(student_id: int) -> None:
    """Natural-language expense entry, with a mandatory confirmation step."""
    ui.section(
        "Quick log",
        'Type it the way you would say it - "spent 50 on chai at the canteen".',
    )

    quick_log = integration.feature("quick_log")
    if not quick_log.ready:
        ui.feature_pending(quick_log, "Natural Language Quick Log")
        return
    if not llm_engine.is_available():
        ui.llm_required("Quick log needs a local LLM to parse your sentence.")
        return

    sentence = st.text_input(
        "Describe the expense",
        key="quick_log_input",
        placeholder="spent 250 on books at Crossword",
        label_visibility="collapsed",
    )

    if st.button("Parse", disabled=not sentence) and sentence:
        try:
            parsed = quick_log.call("parse_quick_log", sentence)
            st.session_state["quick_log_parsed"] = parsed
        except (integration.FeatureError, integration.FeatureUnavailable) as exc:
            ui.error_box(exc, "Quick log parsing failed")
        except llm_engine.LLMUnavailableError as exc:
            st.warning(str(exc), icon=":material/smart_toy:")

    parsed = st.session_state.get("quick_log_parsed")
    if not isinstance(parsed, dict):
        return

    # Preview in an editable form. The model proposes; the student decides.
    st.caption("Check these before saving - the model can misread a sentence.")
    with st.form("quick_log_confirm"):
        columns = st.columns(4)
        with columns[0]:
            amount = st.number_input(
                "Amount", min_value=0.0, step=10.0,
                value=float(parsed.get("amount", 0) or 0),
            )
        with columns[1]:
            suggested = str(parsed.get("category", ""))
            # The model may return a category outside our list; fall back to
            # Miscellaneous rather than raising on an invalid index.
            index = (
                CATEGORY_NAMES.index(suggested) if suggested in CATEGORY_NAMES
                else CATEGORY_NAMES.index("Miscellaneous")
            )
            category = st.selectbox("Category", CATEGORY_NAMES, index=index)
        with columns[2]:
            merchant = st.text_input("Merchant", value=str(parsed.get("merchant", "")))
        with columns[3]:
            txn_date = st.date_input("Date", value=date.today())

        if st.form_submit_button("Save expense", type="primary"):
            if suggested and suggested not in CATEGORY_NAMES:
                st.caption(f"Model suggested '{suggested}', mapped to {category}.")
            _insert_expense(
                student_id, category, amount, merchant, txn_date,
                description=f"Quick log: {sentence}", source="manual",
            )
            st.session_state.pop("quick_log_parsed", None)


# --------------------------------------------------------------------------- #
# Manual entry
# --------------------------------------------------------------------------- #
def _insert_expense(
    student_id: int, category: str, amount: float, merchant: str,
    txn_date: date, description: str = "", source: str = "manual",
) -> None:
    """Insert through the data contract, then refresh every cached read."""
    data = integration.feature("data")
    try:
        data.call(
            "add_expense", student_id, category, float(amount), merchant,
            txn_date.isoformat(), description, "UPI", source, None,
        )
    except (integration.FeatureError, integration.FeatureUnavailable) as exc:
        ui.error_box(exc, "Could not save the expense")
        return

    # Without this the dashboard would keep serving the pre-insert cache.
    state.clear_data_cache()
    st.success(f"Saved {ui.money(amount)} at {merchant}.", icon=":material/check:")
    st.rerun()


def _render_manual_form(student_id: int) -> None:
    """Plain form entry -- the path that works with no LLM at all."""
    with st.expander("Add an expense manually"):
        with st.form("manual_expense", clear_on_submit=True):
            columns = st.columns(5)
            with columns[0]:
                amount = st.number_input("Amount", min_value=0.0, step=10.0)
            with columns[1]:
                category = st.selectbox("Category", CATEGORY_NAMES)
            with columns[2]:
                merchant = st.text_input("Merchant")
            with columns[3]:
                payment_mode = st.selectbox("Paid by", config.PAYMENT_MODES)
            with columns[4]:
                txn_date = st.date_input("Date", value=date.today())
            description = st.text_input("Note (optional)")

            if st.form_submit_button("Add expense", type="primary"):
                if amount <= 0 or not merchant.strip():
                    st.error("Enter an amount above zero and a merchant name.")
                else:
                    _insert_expense(student_id, category, amount, merchant.strip(),
                                    txn_date, description)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def _render_table(expenses: pd.DataFrame) -> None:
    """Filterable transaction list."""
    ui.section(f"{len(expenses):,} transactions")

    columns = st.columns([2, 2, 1])
    with columns[0]:
        chosen = st.multiselect(
            "Categories", sorted(expenses["category"].unique()),
            placeholder="All categories",
        )
    with columns[1]:
        search = st.text_input("Search merchant", placeholder="e.g. Swiggy")
    with columns[2]:
        limit = st.selectbox("Show", [50, 100, 250, 1000], index=0)

    filtered = expenses
    if chosen:
        filtered = filtered[filtered["category"].isin(chosen)]
    if search:
        filtered = filtered[
            filtered["merchant"].str.contains(search, case=False, na=False)
        ]

    if filtered.empty:
        ui.empty_state("No transactions match those filters.")
        return

    # The total covers every matching row, not just the visible page -- say so,
    # or it reads as the total of the 50 rows on screen.
    st.caption(
        f"Showing {min(len(filtered), limit):,} of {len(filtered):,} matching "
        f"transactions. All {len(filtered):,} total "
        f"{ui.money(filtered['amount'].sum())}."
    )

    display = filtered.head(limit).copy()
    display["txn_date"] = display["txn_date"].dt.strftime("%d %b %Y")

    # Row selection drives deletion. The id column is carried through the frame
    # but not displayed -- the selection returns positional indices, which are
    # mapped back to real database ids below.
    selection = st.dataframe(
        display[["txn_date", "merchant", "category", "amount",
                 "payment_mode", "source"]],
        hide_index=True, width="stretch", height=420,
        on_select="rerun",
        selection_mode="multi-row",
        key="txn_table",
        column_config={
            "txn_date": st.column_config.TextColumn("Date", width="small"),
            "merchant": st.column_config.TextColumn("Merchant"),
            "category": st.column_config.TextColumn("Category"),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "payment_mode": st.column_config.TextColumn("Paid by", width="small"),
            "source": st.column_config.TextColumn("Logged via", width="small"),
        },
    )

    _render_delete(display, selection)


def _render_delete(display: pd.DataFrame, selection) -> None:
    """
    Delete the rows selected in the table.

    Deletion is irreversible and there is no undo, so it takes two clicks: the
    first arms it and shows exactly what will go, the second commits. The
    confirmation lists the rows rather than a count, because "delete 3
    transactions" is not enough information to catch a mis-click.
    """
    rows = getattr(getattr(selection, "selection", None), "rows", None) or []
    if not rows:
        st.caption("Select rows in the table to delete them.")
        return

    chosen = display.iloc[rows]
    total = float(chosen["amount"].sum())

    st.warning(
        f"**{len(chosen)} transaction(s) selected** - {ui.money(total)} total.",
        icon=":material/delete:",
    )
    for row in chosen.itertuples():
        st.caption(f"• {row.txn_date} — {row.merchant} — {ui.money(row.amount)}")

    confirm_key = "confirm_delete_txns"
    if not st.session_state.get(confirm_key):
        if st.button("Delete selected", type="secondary"):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.error("This cannot be undone.", icon=":material/warning:")
    left, right = st.columns(2)
    with left:
        if st.button("Yes, delete permanently", type="primary", width="stretch"):
            _delete_rows(chosen)
            st.session_state[confirm_key] = False
            st.rerun()
    with right:
        if st.button("Cancel", width="stretch"):
            st.session_state[confirm_key] = False
            st.rerun()


def _delete_rows(chosen: pd.DataFrame) -> None:
    """Delete each selected row through the data contract, then refresh caches."""
    module = integration.feature("data").module
    if module is None or not hasattr(module, "delete_expense"):
        st.error("The data module does not support deletion.")
        return

    student_id = state.get_student_id()
    deleted, failed = 0, 0
    for expense_id in chosen["id"]:
        try:
            if module.delete_expense(int(expense_id), student_id):
                deleted += 1
            else:
                failed += 1
        except Exception:  # noqa: BLE001
            failed += 1

    # Both the cached reads and the semantic index now hold rows that no longer
    # exist; leaving either stale would have the assistant quoting deleted spend.
    state.clear_data_cache()
    state.reindex_expenses()

    if deleted:
        st.success(f"Deleted {deleted} transaction(s).", icon=":material/check:")
    if failed:
        st.error(f"{failed} could not be deleted.")


def render() -> None:
    """Draw the Transactions tab."""
    student_id = state.get_student_id()
    if not ui.guard(integration.feature("data"), "Transaction data"):
        return

    _render_quick_log(student_id)
    st.divider()
    _render_manual_form(student_id)

    expenses = state.load_expenses(student_id)
    if expenses.empty:
        ui.empty_state(
            "No transactions yet.",
            "Add one above, or run "
            "`python backend/scripts/generate_mock_data.py --force`.",
        )
        return

    _render_table(expenses)
