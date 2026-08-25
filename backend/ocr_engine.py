"""Receipt image preprocessing, OCR, and structured item extraction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

from config import LLM_TIMEOUT_SECONDS, OLLAMA_CHAT_MODEL, OLLAMA_HOST


class ChatClient(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


@dataclass
class ReceiptItem:
    name: str
    price: float


@dataclass
class ReceiptResult:
    raw_text: str = ""
    merchant: str | None = None
    receipt_date: str | None = None
    items: list[ReceiptItem] = field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["items"] = [asdict(item) for item in self.items]
        return result


_PRICE_RE = re.compile(r"(?:₹|rs\.?|inr)?\s*([0-9]+(?:[,.][0-9]{1,2})?)\s*$", re.I)
_CURRENCY_RE = re.compile(r"(₹|\bINR\b|\bRs\.?\b)", re.I)


def _load_image(image_source: str | Path | bytes | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image_source, np.ndarray):
        image = image_source
    elif isinstance(image_source, Image.Image):
        image = cv2.cvtColor(np.asarray(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source))
    elif isinstance(image_source, bytes):
        image = cv2.imdecode(np.frombuffer(image_source, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        raise TypeError("image_source must be a path, bytes, PIL image, or NumPy array")

    if image is None or image.size == 0:
        raise ValueError("Could not decode receipt image")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _deskew(gray: np.ndarray) -> np.ndarray:
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    points = cv2.findNonZero(threshold)
    if points is None or len(points) < 20:
        return gray
    angle = cv2.minAreaRect(points)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image: np.ndarray) -> list[np.ndarray]:
    """Return OCR candidates: enhanced grayscale and adaptive threshold images."""
    gray = _deskew(image)
    height, width = gray.shape[:2]
    if min(height, width) < 1200:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    threshold = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [enhanced, threshold]


def _ocr(image: np.ndarray) -> tuple[str, float]:
    data = pytesseract.image_to_data(image, config="--oem 3 --psm 6", output_type=Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for index, (text, confidence) in enumerate(zip(data["text"], data["conf"])):
        text = text.strip()
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = -1
        if text and score >= 0:
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            lines.setdefault(key, []).append(text)
            confidences.append(score)
    text = "\n".join(" ".join(words) for words in lines.values())
    confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0
    return text, confidence


def extract_text(image_source: str | Path | bytes | Image.Image | np.ndarray) -> tuple[str, float]:
    """Extract the highest-confidence OCR text from a receipt image."""
    image = _load_image(image_source)
    candidates = [_ocr(candidate) for candidate in preprocess_image(image)]
    return max(candidates, key=lambda candidate: candidate[1], default=("", 0.0))


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def parse_items(text: str) -> tuple[list[ReceiptItem], dict[str, float | None]]:
    """Parse obvious item-price lines and receipt summary values."""
    items: list[ReceiptItem] = []
    summary: dict[str, float | None] = {"subtotal": None, "tax": None, "total": None}
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip(" .:-")
        match = _PRICE_RE.search(line)
        if not match:
            continue
        price = _number(match.group(1))
        label = line[: match.start()].strip(" .:-")
        lowered = label.lower()
        if "grand total" in lowered or re.search(r"\btotal\b", lowered):
            summary["total"] = price
        elif "sub" in lowered and "total" in lowered:
            summary["subtotal"] = price
        elif "tax" in lowered or "gst" in lowered:
            summary["tax"] = price
        elif "discount" in lowered:
            continue
        elif label:
            items.append(ReceiptItem(name=label, price=price))
    return items, summary


def _validated_llm_result(payload: str) -> ReceiptResult:
    data = json.loads(payload)
    items = [ReceiptItem(name=str(item["name"]).strip(), price=float(item["price"])) for item in data.get("items", [])]
    if any(not item.name or item.price < 0 for item in items):
        raise ValueError("LLM returned an invalid item")
    return ReceiptResult(
        merchant=data.get("merchant"),
        receipt_date=data.get("receipt_date"),
        items=items,
        subtotal=float(data["subtotal"]) if data.get("subtotal") is not None else None,
        tax=float(data["tax"]) if data.get("tax") is not None else None,
        total=float(data["total"]) if data.get("total") is not None else None,
        currency=data.get("currency"),
    )


def _llm_structure(text: str, client: ChatClient | None) -> ReceiptResult:
    if client is None:
        try:
            from ollama import Client

            client = Client(host=OLLAMA_HOST, timeout=LLM_TIMEOUT_SECONDS)
        except Exception as error:
            raise RuntimeError("Ollama is unavailable") from error
    prompt = (
        "Extract the receipt OCR below. Return JSON only, with exactly this schema:\n"
        '{"merchant": null, "receipt_date": null, "items": [{"name": "Tea", '
        '"price": 20.0}], "subtotal": null, "tax": null, "total": null, '
        '"currency": "INR"}\n'
        "Rules: keep values in their correct fields; prices and totals must be JSON "
        "numbers, never strings; use null when a value is absent; receipt_date must "
        "be a date or null; do not invent missing values; do not include totals, tax, "
        "discounts, or change as items.\nOCR text:\n" + text
    )
    response = client.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    content = response["message"]["content"] if isinstance(response, dict) else response.message.content
    return _validated_llm_result(content)


def process_receipt(
    image_source: str | Path | bytes | Image.Image | np.ndarray,
    *,
    use_llm: bool = True,
    llm_client: ChatClient | None = None,
) -> ReceiptResult:
    """Run OCR and optionally convert the result into structured receipt data."""
    raw_text, ocr_confidence = extract_text(image_source)
    items, summary = parse_items(raw_text)
    result = ReceiptResult(
        raw_text=raw_text,
        items=items,
        confidence=round(ocr_confidence, 3),
        currency="INR" if _CURRENCY_RE.search(raw_text) else None,
        **summary,
    )
    if use_llm and raw_text:
        try:
            structured = _llm_structure(raw_text, llm_client)
            deterministic_total = summary["total"]
            if (
                deterministic_total is not None
                and structured.total is not None
                and abs(structured.total - deterministic_total) > 0.01
            ):
                raise ValueError("LLM total disagrees with OCR total")
            structured.raw_text = raw_text
            structured.confidence = result.confidence
            if not structured.currency:
                structured.currency = result.currency
            result = structured
        except (ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
            result.warnings.append("LLM extraction failed; deterministic parsing was used")
    if result.total is None:
        result.warnings.append("Could not identify receipt total")
    if result.total is not None and result.items:
        item_sum = sum(item.price for item in result.items)
        if abs(item_sum - result.total) > max(1.0, result.total * 0.05):
            result.warnings.append("Item prices do not match receipt total")
    if result.confidence < 0.45:
        result.warnings.append("Low OCR confidence; please verify the extracted values")
    return result
