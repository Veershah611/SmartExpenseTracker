"""
frontend/components/scanner.py
==============================
The Receipt Scanner tab.
**Upload/preview/save flow owned by the Core Integrator;
OCR and item splitting owned by the Vision & OCR Specialist.**

Contract note
-------------
This tab originally called ``extract_text()`` then ``split_receipt()``. The
delivered module exposes ``process_receipt()``, which runs OCR, a deterministic
regex parse and LLM structuring in one call -- and cross-checks the LLM's total
against the regex total, rejecting the LLM result when they disagree. That is a
better design than the two-step contract, so the contract was changed to match
their module rather than the other way round.

Their module builds its own ``ollama.Client`` when no client is injected. The
shell injects ``adapters.LLMEngineChatClient`` instead, so receipt parsing uses
the shared connection and works against LM Studio too.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from backend import integration, llm_engine  # noqa: E402
from frontend import state, ui  # noqa: E402

CATEGORY_NAMES = [name for name, _icon, _share in config.EXPENSE_CATEGORIES]


def _save_upload(uploaded) -> Path:
    """Persist the upload so the OCR module receives a real file path."""
    config.RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix or ".png"
    target = config.RECEIPTS_DIR / f"receipt_{date.today():%Y%m%d}_{uploaded.name}"
    target = target.with_suffix(suffix)
    target.write_bytes(uploaded.getbuffer())
    return target


def _as_dict(result) -> dict:
    """
    Normalise the return value to a plain dict.

    Their module returns a ``ReceiptResult`` dataclass with ``as_dict()``, but
    accepting a bare dict too means a future rewrite on their side does not
    break this tab.
    """
    if isinstance(result, dict):
        return result
    if hasattr(result, "as_dict"):
        return result.as_dict()
    return {}


def _render_confirmation(student_id: int, parsed: dict, receipt_path: str) -> None:
    """Show extracted items and let the student save a single summary expense."""
    items = parsed.get("items") or []
    total = parsed.get("total")
    warnings = parsed.get("warnings") or []
    confidence = parsed.get("confidence")

    # Their module reports OCR confidence and self-diagnosed problems. Surfacing
    # both is the whole point of a confirmation step -- a low-confidence scan is
    # exactly when a student should check before saving.
    if confidence is not None:
        tone = "ok" if confidence >= 0.7 else "warn" if confidence >= 0.45 else "bad"
        st.markdown(
            f"OCR confidence: {ui.pill(f'{confidence:.0%}', tone)}",
            unsafe_allow_html=True,
        )
    for warning in warnings:
        st.warning(warning, icon=":material/warning:")

    if items:
        ui.section(f"{len(items)} items found")
        st.dataframe(
            [{"Item": item.get("name", ""), "Price": item.get("price", 0.0)}
             for item in items],
            hide_index=True, width="stretch",
        )

    if total is None and items:
        # Fall back to summing the items when no total could be read.
        try:
            total = sum(float(item.get("price", 0)) for item in items)
        except (TypeError, ValueError):
            total = 0.0

    with st.form("receipt_confirm"):
        columns = st.columns(4)
        with columns[0]:
            amount = st.number_input("Total", min_value=0.0, step=10.0,
                                     value=float(total or 0.0))
        with columns[1]:
            category = st.selectbox("Category", CATEGORY_NAMES)
        with columns[2]:
            merchant = st.text_input("Merchant",
                                     value=str(parsed.get("merchant") or ""))
        with columns[3]:
            txn_date = st.date_input("Date", value=date.today())

        if st.form_submit_button("Save expense", type="primary"):
            if amount <= 0 or not merchant.strip():
                st.error("Enter an amount above zero and a merchant name.")
                return
            data = integration.feature("data")
            try:
                data.call(
                    "add_expense", student_id, category, float(amount),
                    merchant.strip(), txn_date.isoformat(),
                    f"Receipt: {len(items)} items", "UPI", "receipt_ocr",
                    receipt_path,
                )
            except (integration.FeatureError, integration.FeatureUnavailable) as exc:
                ui.error_box(exc, "Could not save the expense")
                return

            state.clear_data_cache()
            st.session_state.pop("receipt_parsed", None)
            st.success(f"Saved {ui.money(amount)} from receipt.",
                       icon=":material/check:")
            st.rerun()


def render() -> None:
    """Draw the Receipt Scanner tab."""
    student_id = state.get_student_id()

    ui.section(
        "Smart Receipt Splitting",
        "Upload a receipt photo; items and total are extracted automatically.",
    )

    ocr = integration.feature("ocr")
    if not ui.guard(ocr, "Receipt scanning"):
        st.file_uploader(
            "Receipt image", type=["png", "jpg", "jpeg"], disabled=True,
            help="Enabled once the OCR module is delivered.",
        )
        return

    uploaded = st.file_uploader("Receipt image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    left, right = st.columns([1, 2])
    with left:
        st.image(uploaded, caption="Uploaded receipt", width="stretch")

    with right:
        try:
            receipt_path = _save_upload(uploaded)
        except OSError as exc:
            ui.error_box(exc, "Could not save the uploaded image")
            return

        # Route their LLM step through the shared engine; skip it entirely when
        # no model is running so the deterministic regex parse still works.
        from backend.adapters import LLMEngineChatClient

        use_llm = llm_engine.is_available()
        if not use_llm:
            st.caption("No local LLM running - using deterministic parsing only.")

        with st.spinner("Reading receipt..."):
            try:
                result = ocr.call(
                    "process_receipt", str(receipt_path),
                    use_llm=use_llm,
                    llm_client=LLMEngineChatClient() if use_llm else None,
                )
            except (integration.FeatureError, integration.FeatureUnavailable) as exc:
                # Tesseract is a separate binary from the pytesseract package,
                # and a missing PATH entry is by far the most common cause here.
                if "tesseract" in str(exc).lower():
                    st.error(
                        "Tesseract is not installed or not on PATH. Install it from "
                        "https://github.com/UB-Mannheim/tesseract/wiki, then restart "
                        "the app.",
                        icon=":material/error:",
                    )
                else:
                    ui.error_box(exc, "Receipt processing failed")
                return

        parsed = _as_dict(result)
        if not parsed:
            st.error(
                f"The OCR module returned `{type(result).__name__}`; expected a "
                "ReceiptResult or a dict."
            )
            return

        raw_text = parsed.get("raw_text", "")
        if not raw_text.strip():
            st.warning(
                "No text could be read from that image. Try a sharper, "
                "well-lit photo.",
                icon=":material/image_not_supported:",
            )
            return

        with st.expander("Raw OCR text"):
            st.code(raw_text, language="text")

        st.session_state["receipt_parsed"] = parsed

    parsed = st.session_state.get("receipt_parsed")
    if isinstance(parsed, dict):
        _render_confirmation(student_id, parsed, str(receipt_path))
