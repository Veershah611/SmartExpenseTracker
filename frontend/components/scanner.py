"""
frontend/components/scanner.py
==============================
The Receipt Scanner tab.
**Upload/preview/save flow owned by the Core Integrator;
OCR and item splitting owned by the Vision & OCR Specialist.**

The Integrator owns everything around their module: the uploader, the saved
image file, the confirmation step, and the database write. The specialist owns
exactly two functions:

    extract_text(image_path: str) -> str
    split_receipt(text: str) -> dict        # {"items": {name: price}, "total": float}

Splitting it this way means they can build and test their pipeline on a folder
of images with no Streamlit and no database involved -- which is the point of
their role being described as isolated.
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
    """
    Persist the upload so the OCR module receives a real file path.

    A path rather than bytes keeps the specialist's function usable from a plain
    script, which is how they will actually develop and debug it.
    """
    config.RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix or ".png"
    target = config.RECEIPTS_DIR / f"receipt_{date.today():%Y%m%d}_{uploaded.name}"
    target = target.with_suffix(suffix)
    target.write_bytes(uploaded.getbuffer())
    return target


def _render_confirmation(student_id: int, parsed: dict, receipt_path: str) -> None:
    """Show extracted items and let the student save a single summary expense."""
    items = parsed.get("items") or {}
    total = parsed.get("total")

    if items:
        ui.section(f"{len(items)} items found")
        st.dataframe(
            [{"Item": name, "Price": price} for name, price in items.items()],
            hide_index=True, width="stretch",
        )

    if total is None and items:
        # Fall back to summing the items when the module could not read a total.
        try:
            total = sum(float(value) for value in items.values())
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
                                     value=str(parsed.get("merchant", "")))
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
        # Still show the uploader so the flow is demonstrable and the specialist
        # has a live harness to test against the moment their module lands.
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

        with st.spinner("Reading receipt..."):
            try:
                text = ocr.call("extract_text", str(receipt_path))
            except (integration.FeatureError, integration.FeatureUnavailable) as exc:
                ui.error_box(exc, "Text extraction failed")
                return

        if not text or not str(text).strip():
            st.warning(
                "No text could be read from that image. Try a sharper, "
                "well-lit photo.",
                icon=":material/image_not_supported:",
            )
            return

        with st.expander("Raw OCR text"):
            st.code(text, language="text")

        with st.spinner("Splitting items..."):
            try:
                parsed = ocr.call("split_receipt", text)
            except (integration.FeatureError, integration.FeatureUnavailable) as exc:
                ui.error_box(exc, "Receipt splitting failed")
                return
            except llm_engine.LLMUnavailableError as exc:
                st.warning(str(exc), icon=":material/smart_toy:")
                return

        if not isinstance(parsed, dict):
            st.error(
                "The OCR module returned "
                f"`{type(parsed).__name__}`; expected a dict with an `items` key."
            )
            return

        st.session_state["receipt_parsed"] = parsed

    parsed = st.session_state.get("receipt_parsed")
    if isinstance(parsed, dict):
        _render_confirmation(student_id, parsed, str(receipt_path))
