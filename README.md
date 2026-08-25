# Smart Expense Tracker

AI-enabled expense tracker for university students.
**Use Case 12 — TCS Technology Day, Nirma University.**

Runs fully offline: SQLite for transactions, a local vector store for semantic
search, and Ollama for the conversational assistant. No API keys, no cloud calls.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python backend/scripts/generate_mock_data.py --force
streamlit run frontend/app.py
```

You also need [Ollama](https://ollama.com) running locally:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## Folder structure

```
SmartExpenseTracker/
├── config.py                   # Shared contract: paths, models, categories
├── requirements.txt
├── .env.example                # Copy to .env to override any config value
│
├── backend/                    # Business logic — NO streamlit imports in here
│   ├── schema.sql              # SQLite DDL (tables, constraints, indexes)
│   ├── database.py             # Connection handling + typed CRUD helpers
│   ├── analytics.py            # Pandas aggregations powering every chart
│   ├── vector_store.py         # Chroma wrapper + NumPy fallback index
│   ├── llm_engine.py           # Ollama client, health checks, streaming
│   ├── rag_engine.py           # Retrieval + prompt assembly for the chat tab
│   ├── insights.py             # Rule-based signals + LLM-written recommendations
│   ├── ocr_engine.py           # OpenCV receipt pre-processing and parsing
│   └── scripts/
│       └── generate_mock_data.py   # Builds and seeds data/expenses.db
│
├── frontend/                   # Presentation only — the ONLY layer importing streamlit
│   ├── app.py                  # Entry point: tabs and layout, no business logic
│   └── components/             # One module per tab
│
├── data/                       # Runtime state, shared, git-ignored
│   ├── expenses.db             # Generated — not committed
│   ├── receipts/               # Uploaded receipt images
│   └── vector_store/           # Persisted embeddings
│
└── tests/
```

### Why this layout

`backend/` holds pure Python with **no Streamlit imports**. That keeps the
business logic testable from a plain script and means a future FastAPI or CLI
front end could reuse it untouched. `frontend/` is the only layer allowed to
`import streamlit`, and `app.py` does nothing but wire tabs together.

The dependency arrow points one way: **`frontend/` → `backend/` → `config.py`.**
Backend modules never import from `frontend/`.

`config.py` sits at the root rather than inside either layer because both
consume it — it is the shared contract, not the property of one side.

`data/` is also shared and deliberately outside both layers: it is runtime
state, not source code.

### A note on imports

Both entry points (`frontend/app.py` and `backend/scripts/generate_mock_data.py`)
add the project root to `sys.path` before importing anything local. This is what
lets you run them directly, from any working directory, without installing the
project as a package:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

---

## Data model

| Table | Purpose |
|---|---|
| `students` | Personas, budgets, hostel status |
| `categories` | 10 canonical categories with benchmark budget shares |
| `expenses` | Every transaction; indexed on `(student_id, txn_date)` |
| `budgets` | Per-category monthly limits |
| `savings_goals` | Goal tracking for the dashboard widget |
| `knowledge_base` | Curated finance advice — RAG grounding |
| `chat_history` | Conversation persistence across reruns |

---

## Regenerating demo data

```bash
python backend/scripts/generate_mock_data.py --force --months 12 --seed 7
```

The generator is seeded, so the same seed always produces the same database —
useful when rehearsing a demo.
