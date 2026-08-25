"""
frontend/ui.py
==============
Shared presentation helpers and the app's stylesheet.
**Owned by the Core Integrator.**

Every tab draws its cards, section headers and "module not ready" notices from
here. That is the point: with five people contributing screens, a shared
vocabulary of components is the only thing that keeps the app looking like one
product rather than five.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from backend import integration  # noqa: E402

# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
# Kept in one string rather than scattered `st.markdown` calls so restyling the
# app is a single edit. Colours are defined as CSS variables and adapt to
# Streamlit's light/dark themes.
STYLESHEET = """
<style>
  :root {
    --card-radius: 12px;
    --accent: #4C78A8;
    --ok: #2E7D32;
    --warn: #ED6C02;
    --bad: #C62828;
  }

  /* Tighten Streamlit's default vertical rhythm -- the dashboard fits on one
     screen only if the stock padding is reduced. */
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

  .metric-card {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: var(--card-radius);
    padding: 0.9rem 1.1rem;
    height: 100%;
  }
  .metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
    margin-bottom: 0.25rem;
  }
  /* Scale with viewport and never break a number across lines -- "Rs 12,106.6 / 1"
     was the result before clamp + nowrap. */
  .metric-card .value {
    font-size: clamp(1.05rem, 1.5vw, 1.5rem);
    font-weight: 650;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* Text values (a category name) may wrap at spaces; only numbers need the
     single-line treatment that was truncating "Hostel & Rent" to "Hostel & R". */
  .metric-card .value.wrap {
    white-space: normal;
    overflow-wrap: normal;
    word-break: keep-all;
    font-size: clamp(0.95rem, 1.25vw, 1.2rem);
  }
  .metric-card .delta { font-size: 0.82rem; margin-top: 0.2rem; }

  .delta-up   { color: var(--bad); }
  .delta-down { color: var(--ok); }
  .delta-flat { opacity: 0.65; }

  .pill {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid transparent;
  }
  .pill-ok       { background: rgba(46,125,50,0.15);  color: var(--ok);  border-color: rgba(46,125,50,0.35); }
  .pill-warn     { background: rgba(237,108,2,0.15);  color: var(--warn);border-color: rgba(237,108,2,0.35); }
  .pill-bad      { background: rgba(198,40,40,0.15);  color: var(--bad); border-color: rgba(198,40,40,0.35); }
  .pill-neutral  { background: rgba(128,128,128,0.15); opacity: 0.85; }

  .section-title { font-size: 1.05rem; font-weight: 650; margin: 0.4rem 0 0.1rem; }
  .section-note  { font-size: 0.83rem; opacity: 0.7; margin-bottom: 0.6rem; }

  /* Sidebar team-status rows */
  .team-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.8rem; padding: 0.18rem 0;
  }
  .team-row .owner { opacity: 0.75; }
</style>
"""


def inject_styles() -> None:
    """Apply the stylesheet. Called once from ``app.py``."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def money(amount: float) -> str:
    """Format currency consistently everywhere."""
    return config.as_currency(amount)


def money_compact(amount: float) -> str:
    """
    Whole-rupee currency for KPI cards.

    Paise carry no meaning in a headline figure and cost roughly three
    characters of width, which is what pushed values onto a second line.
    """
    try:
        return f"{config.CURRENCY_SYMBOL}{float(amount):,.0f}"
    except (TypeError, ValueError):
        return f"{config.CURRENCY_SYMBOL}0"


def section(title: str, note: str = "") -> None:
    """A titled section header with an optional one-line explanation."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_direction: str = "flat",
    wrap: bool = False,
) -> None:
    """
    A KPI card.

    Uses a custom card rather than ``st.metric`` because the stock widget cannot
    show a neutral delta, and colour-codes every delta green-good/red-bad --
    which is backwards for spending, where an increase is the bad outcome.
    """
    delta_html = ""
    if delta:
        delta_html = f'<div class="delta delta-{delta_direction}">{delta}</div>'
    value_class = "value wrap" if wrap else "value"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="{value_class}">{value}</div>'
        f"{delta_html}</div>",
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "neutral") -> str:
    """Return a small status pill. Tones: ok | warn | bad | neutral."""
    return f'<span class="pill pill-{tone}">{text}</span>'


def status_tone(state: str) -> str:
    """Map a budget/feature state onto a pill tone."""
    return {
        "On track": "ok", "Near limit": "warn", "Over budget": "bad",
        "ready": "ok", "fallback": "warn", "missing": "neutral", "error": "bad",
    }.get(state, "neutral")


# --------------------------------------------------------------------------- #
# Degradation notices
# --------------------------------------------------------------------------- #
def feature_pending(status: integration.FeatureStatus, what: str) -> None:
    """
    Explain that a teammate's module has not landed, without looking broken.

    Shows the exact module path and function names required, so whoever owns it
    can read their to-do straight off the screen during integration.
    """
    functions = ", ".join(f"`{name}()`" for name in status.missing) or "-"
    st.info(
        f"**{what}** is waiting on the *{status.owner}*.\n\n"
        f"Create `{status.module_name}` exposing {functions}, then press "
        f"**Reload team modules** in the sidebar.",
        icon=":material/pending:",
    )


def feature_broken(status: integration.FeatureStatus, what: str) -> None:
    """Report that a delivered module failed to import, with the reason."""
    st.error(
        f"**{what}** could not load from `{status.module_name}`.",
        icon=":material/error:",
    )
    if status.error:
        with st.expander("Error detail"):
            st.code(status.error, language="text")


def guard(status: integration.FeatureStatus, what: str) -> bool:
    """
    Standard gate at the top of a tab.

    Returns True when the feature can be used. Otherwise renders the right
    notice and returns False, so a tab body reads::

        if not ui.guard(status, "Receipt scanning"):
            return
    """
    if status.state == "error":
        feature_broken(status, what)
        return False
    if not status.ready:
        feature_pending(status, what)
        return False
    return True


def llm_required(message: str = "") -> None:
    """Shown where a feature genuinely cannot work without the model."""
    st.warning(
        (message or "This feature needs a local LLM.")
        + "\n\nStart **Ollama** (`ollama serve`) or LM Studio's local server, "
          "then press **Refresh** in the sidebar.",
        icon=":material/smart_toy:",
    )


def error_box(exc: BaseException, context: str) -> None:
    """
    Render an unexpected exception without killing the tab.

    The traceback is collapsed: a judge sees a tidy message, and the developer
    can still expand it on the spot.
    """
    st.error(f"{context}: {type(exc).__name__} - {exc}", icon=":material/warning:")
    with st.expander("Technical detail"):
        st.code(integration.format_traceback(exc), language="text")


def empty_state(message: str, hint: str = "") -> None:
    """Neutral placeholder for a chart or table with no rows to show."""
    st.caption(message + (f"  \n{hint}" if hint else ""))
