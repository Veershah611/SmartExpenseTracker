# Integration Guide

**Read this before writing code.** It is the contract between the five roles.

The app shell is already built and running. It works today with your module
missing — your tab shows a "waiting on you" notice instead of crashing. The
moment you push a file matching the contract below, your feature lights up.

---

## How to check your work is wired in

1. Start the app: `streamlit run frontend/app.py`
2. Look at **Team modules** in the sidebar. Your row shows one of:

   | Badge | Meaning |
   |---|---|
   | `live` | Your module loaded, all required functions found |
   | `stub` | Running on a reference implementation, not yours |
   | `pending` | Your module does not exist yet |
   | `error` | Your module exists but failed to import — expand **Integration detail** for the reason |

3. After pushing a file, hit **Reload team modules**. No restart needed.

The **Integration detail** expander names the exact functions still missing.

---

## Contracts

Signatures are enforced by `backend/integration.py`. If you need to change one,
change it there and tell the group — do not diverge silently.

### Role 2 — Data & Database Engineer

The database module already exists and is `live`. Your Ghost Hunter is separate:

```python
# backend/ghost_hunter.py
def find_recurring_charges(expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Detect subscriptions the student may have forgotten.

    Input: the DataFrame from database.get_expenses() — columns include
           txn_date (datetime64), amount, merchant, category.

    Return a DataFrame with columns:
        merchant, category, avg_amount, occurrences,
        median_gap_days, annual_cost

    Return an empty DataFrame (not None) when nothing is found.
    """
```

A reference detector runs until you ship: `analytics.detect_recurring()`
(stable amount, coefficient of variation < 15%, median gap 25–35 days).

> **Worth improving on:** the reference version reports the ₹4,290/month hostel
> mess fee as a "ghost subscription" — ₹51,482 of the ₹63,377 annual total. It
> is recurring, but the student knows about it and cannot cancel it. Excluding
> `Hostel & Rent`, or ranking by "cancellability", would make the feature much
> sharper.

### Role 3 — Analytics & Forecasting Developer

Two deliverables. The analytics module is already `live` with reference
implementations you can extend or replace.

```python
# backend/forecasting.py
def predict_broke_alert(expenses: pd.DataFrame, student: dict) -> dict:
    """
    Return either a plain string, or a dict:
        {"message": str, "severity": "info" | "warning" | "danger"}
    """
```

```python
# frontend/components/charts.py
def render_trend_chart(trend: pd.DataFrame) -> None:
    """Columns: month (str 'YYYY-MM'), core, one_off, total. Call st.* directly."""

def render_category_chart(categories: pd.DataFrame) -> None:
    """Columns: category, amount, share_pct, transactions, avg_amount."""
```

Charts currently fall back to Streamlit's native charts. **If you use Plotly or
Altair, add it to `requirements.txt`** — it is commented out there, because the
shell must not hard-depend on a library you might not choose.

> **Two traps already hit in the fallback charts, worth not repeating:**
> - Include the semester fees in the trend and the y-axis jumps to 80,000,
>   flattening all ten normal months. Plot `core` by default.
> - `st.bar_chart` sorts a category axis alphabetically, burying the largest
>   category mid-list. Sort explicitly by amount.
> - Keep bar charts zero-based. A 15k–24k range on a floating baseline reads as
>   a tenfold jump.

### Role 4 — Vision & OCR Specialist

```python
# backend/ocr_engine.py
def extract_text(image_path: str) -> str:
    """OpenCV preprocessing (greyscale, threshold, deskew) then OCR. Return raw text."""

def split_receipt(text: str) -> dict:
    """
    Return: {"items": {name: price}, "total": float, "merchant": str}
    'total' and 'merchant' are optional; the UI sums items when total is absent.
    """
```

You never touch the database or Streamlit. The shell saves the upload, hands you
a path, shows your output for confirmation, and writes the row.

Use `llm_engine.chat_json()` for the LLM formatting step — it already handles
markdown fences, chatty preambles, and one retry. Do not write your own parser.

Requires `opencv-python-headless` and `pytesseract` (plus the Tesseract binary).
`pytesseract` is installed; **OpenCV is not yet** — `pip install opencv-python-headless`.

### Role 5 — RAG & NLP Developer

```python
# backend/rag_engine.py
def answer_question(question: str, student_id: int) -> str | dict:
    """
    Return a string, or {"answer": str, "sources": list[str | dict]}.
    Sources render in an expander under the answer.
    """

def parse_quick_log(sentence: str) -> dict:
    """
    "spent 50 on chai" -> {"amount": 50.0, "category": "Food & Canteen",
                           "merchant": "...", "description": "..."}
    """
```

`backend/vector_store.py` is built and working — use it rather than wiring
ChromaDB yourself:

```python
from backend.vector_store import get_vector_store
store = get_vector_store("finance_knowledge")
store.add(ids, texts, metadatas)
results = store.query("how do I save money", top_k=5)
```

It falls back from ChromaDB to a NumPy index, and from `nomic-embed-text` to the
chat model to lexical hashing. **Currently running: NumPy index + chat-model
embeddings** (ChromaDB is not installed and has no reliable Python 3.13 wheel).
`ollama pull nomic-embed-text` improves retrieval quality if you want it.

Curated advice for grounding is already in the `knowledge_base` table
(12 entries) — `database.get_knowledge_base()`.

> **`parse_quick_log` must constrain the category.** Tested on this machine,
> llama3.2 parses *"Spent 50 on chai at campus canteen"* as
> `category: "chai"` — not one of the ten valid categories. Pass the category
> list in your prompt and validate the response against
> `config.EXPENSE_CATEGORIES` before returning. The UI shows a confirmation form
> and maps unknown categories to *Miscellaneous*, but do not rely on that.

> **On numeric questions:** vector similarity cannot sum. For *"how much did I
> spend on food this month?"*, retrieve context but compute the figure with
> pandas and inject it. The shell's direct path does this and answers exactly
> — verified against the database.

---

## Shared services (built, use them)

### LLM — `backend/llm_engine.py`

Do **not** open your own connection. Supports Ollama and LM Studio, auto-detected.

```python
from backend import llm_engine

llm_engine.ask("prompt", system="optional")       # one-shot string
llm_engine.chat(messages)                          # multi-turn
llm_engine.chat_stream(messages)                   # generator, for st.write_stream
llm_engine.chat_json(messages)                     # parsed JSON, with retry
llm_engine.is_available()                          # guard before calling
```

Raises `LLMUnavailableError` — **catch it and degrade**. The dashboard must stay
usable with the model offline.

### Data — `backend/database.py`

```python
get_students() / get_student(id)
get_expenses(student_id, start_date=None, end_date=None, categories=None)
add_expense(student_id, category, amount, merchant, ...)
get_budgets(student_id, month=None) / get_savings_goals(student_id)
get_knowledge_base()
```

Reads return DataFrames with `txn_date` already parsed to datetime64.

**After any write, call `state.clear_data_cache()`** or the dashboard serves
stale figures for five minutes.

---

## Ground rules

1. **Never import `streamlit` inside `backend/`.** It breaks testability and the
   one-way dependency (`frontend → backend → config`).
2. **Return empty, never `None`.** Empty DataFrame, empty dict. Every consumer
   handles empty; `None` causes an AttributeError two layers up.
3. **Read config from `config.py`.** No hard-coded paths, model names or
   category lists.
4. **Your exceptions are contained** — `integration.py` wraps every call, so a
   crash in your module degrades your tab only. Do not rely on this as error
   handling; it is a safety net, not a design.

## Environment

| Component | Status |
|---|---|
| Ollama | running, `llama3.2:latest` |
| Database | `data/expenses.db`, 1,874 transactions, 3 personas |
| Vector index | NumPy fallback (ChromaDB not installed) |
| Embeddings | chat model (`nomic-embed-text` not pulled) |
| Not installed | `opencv-python-headless`, `chromadb`, `plotly` |

```bash
python backend/scripts/generate_mock_data.py --force
```
