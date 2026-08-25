import unittest
from unittest.mock import patch

from backend.ocr_engine import ReceiptItem, _validated_llm_result, parse_items, process_receipt


class OcrEngineTests(unittest.TestCase):
    def test_parse_items_and_summary(self):
        items, summary = parse_items(
            "Campus Cafe\nTea 20\nSandwich Rs. 80.00\nGST 5\nTOTAL 105.00"
        )

        self.assertEqual(items, [ReceiptItem("Tea", 20.0), ReceiptItem("Sandwich", 80.0)])
        self.assertEqual(summary, {"subtotal": None, "tax": 5.0, "total": 105.0})

    def test_llm_result_requires_valid_item_values(self):
        with self.assertRaises(ValueError):
            _validated_llm_result('{"items": [{"name": "Tea", "price": -1}]}')

    @patch("backend.ocr_engine.extract_text", return_value=("Tea 20\nTOTAL 20", 0.9))
    def test_process_receipt_falls_back_when_llm_fails(self, _extract_text):
        with patch("backend.ocr_engine._llm_structure", side_effect=RuntimeError):
            result = process_receipt(b"unused", use_llm=True)

        self.assertEqual(result.items, [ReceiptItem("Tea", 20.0)])
        self.assertEqual(result.total, 20.0)
        self.assertIn("LLM extraction failed; deterministic parsing was used", result.warnings)


if __name__ == "__main__":
    unittest.main()