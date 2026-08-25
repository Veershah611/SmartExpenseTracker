"""
frontend/app.py
===============
Smart Expense Tracker -- application entry point.
**Owned by the Core Integrator & UI Lead.**

Run with::

    streamlit run frontend/app.py

This file deliberately contains no business logic. Its whole job is to:

* configure the page and apply the shared stylesheet
* establish global state (the "logged-in" mock student)
* render the sidebar: student switcher, LLM connection, team module status
* mount each teammate's tab

Every tab is rendered inside :func:`_safe_render`, so an exception in one
person's code degrades that tab alone. During a five-way integration, an
unhandled error in someone's module must never blank the whole app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts *this file's* directory on sys.path, not the project root, so
# the root has to be added before any local import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402
from frontend.components import (  # noqa: E402
    assistant,
    compare,
    dashboard,
    insights,
    scanner,
    transactions,
)

st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon=":material/savings:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_student_selector() -> None:
    """The mock 'logged-in' student. Global state for every tab."""
    st.sidebar.markdown("### Student")
    try:
        students = state.load_students()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error("Could not load students.")
        with st.sidebar.expander("Detail"):
            st.code(integration.format_traceback(exc), language="text")
        return

    if students.empty:
        st.sidebar.warning("No students in the database.")
        return

    options = students["id"].tolist()
    current = state.get_student_id()
    index = options.index(current) if current in options else 0

    chosen = st.sidebar.selectbox(
        "Signed in as",
        options=options,
        index=index,
        format_func=lambda sid: students.loc[students["id"] == sid, "name"].iloc[0],
        label_visibility="collapsed",
    )
    state.set_student_id(chosen)

    record = students[students["id"] == chosen].iloc[0]
    st.sidebar.caption(
        f"{record['course']} - semester {record['semester']}  \n"
        f"Budget {config.as_currency(record['monthly_budget'])} / month  \n"
        f"{'Hostel resident' if record['hostel_resident'] else 'Day scholar'}"
    )


def _render_llm_panel() -> None:
    """Connection status for the shared local LLM."""
    st.sidebar.markdown("### AI model")
    status = llm_engine.get_status()

    if status.available:
        st.sidebar.markdown(
            ui.pill(status.badge, "ok") + f" <span style='opacity:.7;font-size:.75rem'>"
            f"{status.detail}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(ui.pill("Offline", "bad"), unsafe_allow_html=True)
        st.sidebar.caption(
            "Start `ollama serve` or LM Studio. The dashboard and analytics "
            "work without it."
        )

    if st.sidebar.button("Refresh connection", width="stretch"):
        llm_engine.get_status(force_refresh=True)
        st.rerun()


def _render_team_panel() -> None:
    """
    Live integration status for all five roles.

    This panel is the Integrator's control surface: at a glance it shows which
    teammate modules have landed, which are running on a reference stub, and
    which have not arrived. It also doubles as a progress board during the build.
    """
    summary = integration.integration_summary()
    ready = summary["ready"] + summary["fallback"]

    st.sidebar.markdown("### Team modules")
    st.sidebar.caption(f"{ready} of {summary['total']} wired up")

    for status in integration.get_features().values():
        tone = ui.status_tone(status.state)
        label = status.label
        st.sidebar.markdown(
            f'<div class="team-row"><span>{status.key}</span>'
            f"{ui.pill(label, tone)}</div>",
            unsafe_allow_html=True,
        )

    if st.sidebar.button("Reload team modules", width="stretch",
                         help="Pick up teammate files added since startup"):
        integration.reload_features()
        state.clear_data_cache()
        st.rerun()

    with st.sidebar.expander("Integration detail"):
        for status in integration.get_features().values():
            st.markdown(
                f"**{status.key}** - {status.owner}  \n"
                f"`{status.module_name}`"
                + (f"  \nmissing: {', '.join(status.missing)}" if status.missing else "")
                + (f"  \n{status.error}" if status.error else "")
            )


def _render_environment_panel() -> None:
    """Diagnostics -- the first thing to check when something looks wrong."""
    with st.sidebar.expander("Environment"):
        info = integration.describe_environment()
        database_info = info.get("database", {})
        vectors = info.get("vectors", {})
        st.markdown(
            f"**Database**: {'found' if database_info.get('exists') else 'missing'}  \n"
            f"**Vector index**: {vectors.get('index', '-')}  \n"
            f"**Embeddings**: {vectors.get('embeddings', '-')}"
        )
        if not database_info.get("exists"):
            st.code(
                "python backend/scripts/generate_mock_data.py --force",
                language="bash",
            )


# --------------------------------------------------------------------------- #
# Tab mounting
# --------------------------------------------------------------------------- #
def _safe_render(render_function, tab_name: str) -> None:
    """
    Render one tab, containing any exception it raises.

    The blanket catch is intentional and is the core of the integrator's job:
    five people are editing five modules, and a crash in one of them must cost
    one tab, not the demo.
    """
    try:
        render_function()
    except Exception as exc:  # noqa: BLE001
        ui.error_box(exc, f"The {tab_name} tab hit an error")


def main() -> None:
    """Compose the application."""
    ui.inject_styles()
    state.init_session()

    # A missing database is the single most common first-run problem, so it is
    # handled up front with the exact command to fix it.
    if not integration.describe_environment().get("database", {}).get("exists"):
        st.title("Smart Expense Tracker")
        st.warning("No database found. Generate the demo data to get started.",
                   icon=":material/database:")
        st.code("python backend/scripts/generate_mock_data.py --force",
                language="bash")
        st.stop()

    _render_student_selector()
    st.sidebar.divider()
    _render_llm_panel()
    st.sidebar.divider()
    _render_team_panel()
    _render_environment_panel()

    student = state.load_student(state.get_student_id())
    st.title("Smart Expense Tracker")
    st.caption(
        f"AI-enabled financial assistant for university students"
        + (f" - viewing {student['name']}" if student else "")
    )

    tabs = st.tabs([
        "Overview",
        "Compare",
        "Assistant",
        "Insights",
        "Transactions",
        "Receipt Scanner",
    ])

    with tabs[0]:
        _safe_render(dashboard.render, "Overview")
    with tabs[1]:
        _safe_render(compare.render, "Compare")
    with tabs[2]:
        _safe_render(assistant.render, "Assistant")
    with tabs[3]:
        _safe_render(insights.render, "Insights")
    with tabs[4]:
        _safe_render(transactions.render, "Transactions")
    with tabs[5]:
        _safe_render(scanner.render, "Receipt Scanner")


if __name__ == "__main__":
    main()
