"""
frontend/components/assistant.py
================================
The Assistant tab -- conversational Q&A over the student's spending.
**Chat UI and LLM orchestration owned by the Core Integrator;
retrieval owned by the RAG & NLP Developer.**

Two paths, one interface
------------------------
1. **RAG path** -- once ``backend/rag_engine.py`` lands, questions are routed to
   ``answer_question()`` and the retrieved context is shown in an expander.
2. **Direct path** -- until then, the Integrator's own orchestration answers
   using a compact, deterministic summary of the student's real figures.

The direct path is not a toy stub. It exists because the chat tab is the centre
of the demo, and it cannot be dark while one teammate finishes. It also proves
the LLM wiring works independently of the retrieval layer, which makes the two
failure modes distinguishable during integration.

A note on numbers
-----------------
The direct path computes every figure with pandas and injects them as facts.
The model is asked to *phrase* an answer, never to calculate one -- language
models are unreliable at arithmetic, and a wrong rupee figure on stage is worse
than no answer at all.
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
    "You are a friendly financial assistant for an Indian university student. "
    "Address the student directly as \"you\" - never refer to them by name in "
    "the third person. "
    "You are given VERIFIED FIGURES computed from their real transaction data. "
    "Use only those figures - never invent or recalculate numbers. "
    "Amounts are in Indian rupees; write them like Rs 1,250. "
    "Answer in 2-4 short sentences, be specific and practical, and suggest one "
    "concrete action when it is useful. Do not use markdown headings."
)


def _build_facts(student: dict, expenses: pd.DataFrame) -> str:
    """
    Compact, factual context block for the direct path.

    Deliberately terse: a 3B model on a laptop degrades quickly as the prompt
    grows, and the facts that matter fit in a few lines.
    """
    analytics = integration.feature("analytics")
    if not analytics.ready or expenses.empty:
        return "No transaction data available."

    lines: list[str] = [
        f"Student: {student['name']}, {student['course']}, semester "
        f"{student['semester']}.",
        f"Monthly budget: Rs {student['monthly_budget']:,.0f}.",
    ]

    try:
        kpis = analytics.call("kpi_summary", expenses, student["monthly_budget"])
        lines += [
            f"Spent this month so far: Rs {kpis['current_month_spend']:,.0f} "
            f"({kpis['budget_used_pct']:.0f}% of budget).",
            f"Change vs last month: {kpis['month_change_pct']:+.1f}%.",
            f"Daily average: Rs {kpis['daily_average']:,.0f}.",
            f"Biggest category: {kpis['top_category']} "
            f"(Rs {kpis['top_category_amount']:,.0f} all-time).",
        ]

        latest_month = expenses["txn_date"].max().strftime("%Y-%m")
        this_month = expenses[
            expenses["txn_date"].dt.strftime("%Y-%m") == latest_month
        ]
        if not this_month.empty:
            by_category = (
                this_month.groupby("category")["amount"].sum()
                .sort_values(ascending=False)
            )
            lines.append(f"Spending in {latest_month} by category:")
            lines += [f"  - {name}: Rs {value:,.0f}"
                      for name, value in by_category.items()]

        recurring = integration.feature("ghost_hunter")
        if recurring.ready:
            found = recurring.call("find_recurring_charges", expenses)
            if not found.empty:
                lines.append("Recurring charges detected:")
                lines += [
                    f"  - {row.merchant}: Rs {row.avg_amount:,.0f}/month "
                    f"(Rs {row.annual_cost:,.0f}/year)"
                    for row in found.itertuples()
                ]
    except Exception:  # noqa: BLE001 -- a partial fact block still beats none
        pass

    return "\n".join(lines)


def _answer_directly(question: str, facts: str) -> str:
    """Stream an answer from the LLM using the injected facts."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"VERIFIED FIGURES:\n{facts}\n\nQUESTION: {question}"},
    ]
    return st.write_stream(llm_engine.chat_stream(messages))


def render() -> None:
    """Draw the Assistant tab."""
    student_id = state.get_student_id()
    student = state.load_student(student_id)
    if student is None:
        st.error("Select a student first.")
        return

    rag = integration.feature("rag")
    llm_status = llm_engine.get_status()

    # Explain which path is live, so an integration problem is obvious at a
    # glance instead of being mistaken for a bad answer.
    if rag.ready:
        st.caption(
            f"Retrieval-augmented answers via `{rag.module_name}` - "
            f"model: {llm_status.badge}"
        )
    else:
        st.caption(
            f"Direct answers from verified figures - model: {llm_status.badge}. "
            "Semantic retrieval arrives with the RAG developer's module."
        )

    if not llm_status.available:
        ui.llm_required("The assistant needs a local LLM to answer questions.")
        return

    # -- suggested questions ------------------------------------------------ #
    pending: str | None = None
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        with column:
            if st.button(question, width="stretch", key=f"sq_{question}"):
                pending = question

    # -- transcript --------------------------------------------------------- #
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

    expenses = state.load_expenses(student_id)

    with st.chat_message("assistant"):
        try:
            if rag.ready:
                result = rag.call("answer_question", question, student_id)
                # Accept either a plain string or a richer dict, so the RAG
                # developer is not forced into one return shape.
                if isinstance(result, dict):
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Retrieved context ({len(sources)})"):
                            for source in sources:
                                st.caption(
                                    source if isinstance(source, str)
                                    else source.get("text", str(source))
                                )
                else:
                    answer = str(result)
                    st.markdown(answer)
            else:
                facts = _build_facts(student, expenses)
                answer = _answer_directly(question, facts)
                with st.expander("Figures given to the model"):
                    st.code(facts, language="text")

            state.append_chat("assistant", answer)

        except llm_engine.LLMUnavailableError as exc:
            st.warning(str(exc), icon=":material/smart_toy:")
        except (integration.FeatureError, integration.FeatureUnavailable) as exc:
            ui.error_box(exc, "The RAG module failed")
        except Exception as exc:  # noqa: BLE001
            ui.error_box(exc, "Could not answer")

    # Clearing lives at the bottom so it never sits above the transcript.
    if state.get_chat():
        st.button("Clear conversation", on_click=state.clear_chat)
