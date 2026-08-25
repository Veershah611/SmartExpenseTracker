"""
frontend/components/assistant.py
================================
The Assistant tab -- conversational Q&A over the student's spending.
**Owned by the Core Integrator.**

Two problems this module exists to solve
----------------------------------------

**1. Numbers must be exact, not retrieved.**

Vector search finds *similar* rows; it cannot *sum* them. Asked "how much did I
spend on food this month?", pure retrieval returns a handful of individual food
transactions from scattered months and leaves the model to guess a total -- and
a 3B model guesses badly. A wrong rupee figure on stage is worse than no answer.

So every figure is computed in pandas from the full transaction set and injected
as a VERIFIED FIGURES block. The model is asked to *phrase* an answer, never to
calculate one. Retrieval is still used, but only for advice snippets -- never as
the source of a total.

**2. It is a money assistant, not a general chatbot.**

Left unguarded the model will happily answer "who won the world cup". Questions
are screened before any answer is generated: a cheap keyword pass handles the
obvious cases, and only genuinely ambiguous ones cost an LLM classification call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402

SUGGESTED_QUESTIONS = [
    "How much did I spend on food this month?",
    "Where is my money actually going?",
    "How can I save money?",
    "What subscriptions am I paying for?",
]

SYSTEM_PROMPT = (
    "You are a personal finance assistant for an Indian university student. "
    "You ONLY discuss the student's money: their spending, budget, savings, "
    "categories, transactions and financial habits. "
    "Address the student directly as \"you\"; never use their name in the third "
    "person.\n\n"
    "CRITICAL RULE ABOUT NUMBERS: you are given a VERIFIED FIGURES block "
    "computed directly from the student's transaction database. Every number in "
    "your answer MUST be copied from that block. Never estimate, never add up "
    "figures yourself, and never invent an amount. If a figure you need is not "
    "in the block, say you do not have it rather than guessing.\n\n"
    "Amounts are Indian rupees, written like Rs 1,250. Answer in 2-4 short "
    "sentences. Be specific and practical, and where useful suggest one concrete "
    "action. Do not use markdown headings."
)

# --------------------------------------------------------------------------- #
# Scope guard
# --------------------------------------------------------------------------- #
# Anything money-shaped is waved straight through without an LLM round trip.
_FINANCE_TERMS = {
    "spend", "spent", "spending", "money", "budget", "save", "saving", "savings",
    "expense", "expenses", "cost", "costs", "afford", "cheap", "expensive",
    "transaction", "transactions", "paid", "pay", "payment", "price", "bill",
    "rupee", "rupees", "rs", "inr", "cash", "upi", "card", "wallet", "balance",
    "food", "canteen", "mess", "hostel", "rent", "transport", "travel", "fuel",
    "books", "stationery", "fees", "tuition", "entertainment", "shopping",
    "subscription", "subscriptions", "netflix", "spotify", "recharge", "swiggy",
    "zomato", "medical", "health", "gym", "laundry", "goal", "goals",
    "category", "categories", "merchant", "month", "monthly", "week", "weekly",
    "daily", "average", "total", "broke", "overspend", "overspending", "debt",
    "loan", "income", "allowance", "pocket", "invest", "finance", "financial",
}

_REFUSAL = (
    "I can only help with your money -- spending, budgets, savings and "
    "transactions. Try asking something like *\"How much did I spend on food "
    "this month?\"* or *\"Where is my money going?\"*"
)


def _is_finance_question(question: str) -> bool:
    """
    Decide whether a question is in scope.

    Keyword pass first because it is free and catches almost everything. The
    LLM classifier is the fallback for genuinely ambiguous phrasings like
    "am I doing okay?" -- worth one short call, not worth one on every message.

    Fails open: if classification errors, the question is allowed through, since
    wrongly refusing a real question is the worse failure.
    """
    words = {word.strip("?.,!'\"").lower() for word in question.split()}
    if words & _FINANCE_TERMS:
        return True

    if not llm_engine.is_available():
        return True  # cannot classify; do not block

    try:
        verdict = llm_engine.chat(
            [{
                "role": "user",
                "content": (
                    "Is the following question about the user's personal "
                    "finances, money, spending, budgeting or savings? "
                    "Answer with exactly one word: YES or NO.\n\n"
                    f"Question: {question}"
                ),
            }],
            temperature=0.0,
        )
        return "yes" in verdict.strip().lower()[:5]
    except llm_engine.LLMUnavailableError:
        return True


# --------------------------------------------------------------------------- #
# Verified figures
# --------------------------------------------------------------------------- #
def _build_facts(student: dict, expenses: pd.DataFrame) -> str:
    """
    Every figure the assistant is allowed to quote, computed from real rows.

    Deliberately comprehensive: the model cannot look anything up, so whatever
    is missing here it simply cannot answer. Deliberately compact too -- a 3B
    model degrades as the prompt grows, so this is aggregates plus a short tail
    of recent transactions, not 624 raw rows.
    """
    if expenses.empty:
        return "No transaction data available for this student."

    currency = "Rs"
    lines: list[str] = [
        f"STUDENT: {student['name']}, {student['course']}, semester "
        f"{student['semester']}, "
        f"{'hostel resident' if student['hostel_resident'] else 'day scholar'}.",
        f"MONTHLY BUDGET: {currency} {student['monthly_budget']:,.0f}.",
        f"DATA RANGE: {expenses['txn_date'].min():%d %b %Y} to "
        f"{expenses['txn_date'].max():%d %b %Y} "
        f"({len(expenses):,} transactions).",
    ]

    latest_period = expenses["txn_date"].max().to_period("M")
    this_month = expenses[expenses["txn_date"].dt.to_period("M") == latest_period]
    previous = expenses[expenses["txn_date"].dt.to_period("M") == (latest_period - 1)]

    # Two facts that make month-to-month comparisons honest. Without them the
    # model reports "Rs 67,442 last month vs Rs 12,107 this month" as if
    # spending collapsed, when in truth last month carried a one-off semester
    # fee and this month is not finished.
    latest_day = expenses["txn_date"].max()
    days_in_month = (
        (latest_day.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
        - pd.Timedelta(days=1)
    ).day
    if latest_day.day < days_in_month:
        lines.append(
            f"NOTE: the current month is INCOMPLETE - {latest_day.day} of "
            f"{days_in_month} days elapsed. Comparisons with a full previous "
            "month are not like-for-like."
        )

    # A small model will not reliably act on a "mention the fee" instruction, so
    # the qualification is written into the figure's own label instead. Whatever
    # the model quotes then carries the caveat with it.
    def _month_total_line(label: str, frame: pd.DataFrame) -> str:
        total = frame["amount"].sum()
        fees = frame[
            frame["category"].isin({"Academics & Fees"}) & (frame["amount"] >= 10_000)
        ]["amount"].sum()
        if fees > 0:
            return (
                f"  {label}: {currency} {total:,.0f} total, but {currency} "
                f"{fees:,.0f} of that was a ONE-OFF SEMESTER FEE, so day-to-day "
                f"spending was only {currency} {total - fees:,.0f}."
            )
        return f"  {label}: {currency} {total:,.0f}"

    # ---- current month, by category (the most-asked question) -------------- #
    lines.append(f"\nCURRENT MONTH ({latest_period}):")
    lines.append(_month_total_line("Total spent", this_month)
                 + f" Across {len(this_month)} transactions.")
    for name, value in (
        this_month.groupby("category")["amount"].sum()
        .sort_values(ascending=False).items()
    ):
        count = int((this_month["category"] == name).sum())
        lines.append(f"  {name}: {currency} {value:,.0f} ({count} transactions)")

    # ---- previous month, for comparison questions -------------------------- #
    if not previous.empty:
        lines.append(f"\nPREVIOUS MONTH ({latest_period - 1}):")
        lines.append(_month_total_line("Total spent", previous))

        # Pre-computed so the model never has to subtract. Asked to do the
        # arithmetic itself, llama3.2 reported the gap between Rs 67,442 and
        # Rs 12,107 as "Rs 4,335" -- an invented number in an otherwise
        # correct sentence, which is the most dangerous kind of wrong.
        this_total = float(this_month["amount"].sum())
        previous_total = float(previous["amount"].sum())
        # Given a fact to rephrase, the model inverted the direction ("you spent
        # less LAST month"). Given a finished sentence to copy, it does not.
        direction = "less" if this_total < previous_total else "more"
        lines.append(
            f"  PRE-COMPUTED COMPARISON - if asked to compare the two months, "
            f"copy this sentence exactly: \"You spent {currency} "
            f"{abs(this_total - previous_total):,.0f} {direction} this month "
            f"({latest_period}) than last month ({latest_period - 1}).\""
        )
        for name, value in (
            previous.groupby("category")["amount"].sum()
            .sort_values(ascending=False).head(5).items()
        ):
            lines.append(f"  {name}: {currency} {value:,.0f}")

    # ---- all-time monthly averages ----------------------------------------- #
    month_count = max(expenses["txn_date"].dt.to_period("M").nunique(), 1)
    lines.append(f"\nAVERAGE PER MONTH (over {month_count} months):")
    for name, value in (
        (expenses.groupby("category")["amount"].sum() / month_count)
        .sort_values(ascending=False).head(6).items()
    ):
        lines.append(f"  {name}: {currency} {value:,.0f}")

    # ---- top merchants ----------------------------------------------------- #
    lines.append("\nTOP MERCHANTS (all time):")
    for name, value in (
        expenses.groupby("merchant")["amount"].sum()
        .sort_values(ascending=False).head(6).items()
    ):
        count = int((expenses["merchant"] == name).sum())
        lines.append(f"  {name}: {currency} {value:,.0f} over {count} payments")

    # ---- recurring charges -------------------------------------------------- #
    try:
        hunter = integration.feature("ghost_hunter")
        if hunter.ready:
            found = hunter.call("find_recurring_charges", expenses)
            if not found.empty:
                lines.append("\nRECURRING SUBSCRIPTIONS:")
                for row in found.itertuples():
                    lines.append(
                        f"  {row.merchant}: {currency} {row.avg_amount:,.0f} per "
                        f"month ({currency} {row.annual_cost:,.0f} per year)"
                    )
    except Exception:  # noqa: BLE001 -- a partial fact block beats none
        pass

    # ---- payment modes ------------------------------------------------------ #
    lines.append("\nPAYMENT METHODS (all time):")
    for name, value in (
        expenses.groupby("payment_mode")["amount"].sum()
        .sort_values(ascending=False).items()
    ):
        lines.append(f"  {name}: {currency} {value:,.0f}")

    # ---- recent transactions ------------------------------------------------ #
    lines.append("\nMOST RECENT TRANSACTIONS:")
    for row in expenses.sort_values("txn_date", ascending=False).head(12).itertuples():
        lines.append(
            f"  {row.txn_date:%d %b}: {currency} {row.amount:,.0f} at "
            f"{row.merchant} ({row.category})"
        )

    return "\n".join(lines)


def _retrieve(question: str) -> list[str]:
    """
    Advice snippets from the knowledge base, for "how do I save" style questions.

    Only the curated advice collection is searched. Retrieving individual
    transactions here would invite the model to treat a few sampled rows as if
    they were the whole picture -- the exact failure the verified figures block
    exists to prevent.
    """
    store = state.get_indexed_vector_store()
    if store is None:
        return []
    try:
        hits = store.search(config.RAG_COLLECTION_KNOWLEDGE, question, top_k=3)
        return [hit["document"] for hit in hits]
    except Exception:  # noqa: BLE001
        return []


def render() -> None:
    """Draw the Assistant tab."""
    student_id = state.get_student_id()
    student = state.load_student(student_id)
    if student is None:
        st.error("Select a student first.")
        return

    llm_status = llm_engine.get_status()
    expenses = state.load_expenses(student_id)
    st.caption(
        f"Answers computed from your {len(expenses):,} real transactions - "
        f"model: {llm_status.badge}"
    )

    if not llm_status.available:
        ui.llm_required("The assistant needs a local LLM to answer questions.")
        return

    pending: str | None = None
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        with column:
            if st.button(question, width="stretch", key=f"sq_{question}"):
                pending = question

    for message in state.get_chat():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask about your spending...")
    question = typed or pending
    if not question:
        return

    state.append_chat("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Scope check before anything expensive happens.
        with st.spinner("Checking..."):
            in_scope = _is_finance_question(question)

        if not in_scope:
            st.markdown(_REFUSAL)
            state.append_chat("assistant", _REFUSAL)
            return

        facts = _build_facts(student, expenses)
        advice = _retrieve(question)

        prompt = f"VERIFIED FIGURES (the only numbers you may use):\n{facts}"
        if advice:
            prompt += (
                "\n\nRELEVANT GUIDANCE (use for suggestions, not for numbers):\n"
                + "\n".join(f"- {snippet}" for snippet in advice)
            )
        prompt += f"\n\nQUESTION: {question}"

        try:
            answer = st.write_stream(llm_engine.chat_stream([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]))
            state.append_chat("assistant", answer)

            with st.expander("What the model was given"):
                st.caption(
                    "Every number above is computed in pandas from your actual "
                    "transactions. The model only phrases them."
                )
                st.code(facts, language="text")
                if advice:
                    st.caption("Retrieved guidance:")
                    for snippet in advice:
                        st.caption(f"- {snippet}")

        except llm_engine.LLMUnavailableError as exc:
            st.warning(str(exc), icon=":material/smart_toy:")
        except Exception as exc:  # noqa: BLE001
            ui.error_box(exc, "Could not answer")

    if state.get_chat():
        st.button("Clear conversation", on_click=state.clear_chat)
