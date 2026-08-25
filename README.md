# Smart Expense Tracker

AI-enabled expense tracker for university students.
**Use Case 12 — TCS Technology Day, Nirma University.**

> **Building a module for this project? Read [INTEGRATION.md](INTEGRATION.md) first.**
> It defines the contract your file must match, and how to verify it is wired in.

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

You also need [Ollama](https://ollama.com) or LM Studio running locally:

```bash
ollama pull llama3.2
```

Optional, improves semantic search quality:

```bash
ollama pull nomic-embed-text
```

The app runs without either — the dashboard, analytics and Ghost Hunter work
with no LLM at all, and the vector store falls back to a lexical index.

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
│   ├── llm_engine.py           # Ollama / LM Studio client, streaming, JSON mode
│   ├── integration.py          # Teammate-module contracts and safe loading
│   ├── ghost_hunter.py         # (Role 2) Subscription Ghost Hunter
│   ├── forecasting.py          # (Role 3) Predictive Broke Alert
│   ├── ocr_engine.py           # (Role 4) OpenCV receipt pipeline
│   ├── rag_engine.py           # (Role 5) Retrieval + Quick Log parsing
│   └── scripts/
│       └── generate_mock_data.py   # Builds and seeds data/expenses.db
│
├── frontend/                   # Presentation only — the ONLY layer importing streamlit
│   ├── app.py                  # Entry point: tabs, sidebar, global state
│   ├── state.py                # Session state + cached data access
│   ├── ui.py                   # Stylesheet and shared components
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
