-- ===========================================================================
-- Smart Expense Tracker -- SQLite schema
-- ---------------------------------------------------------------------------
-- Design notes:
--   * Dates are stored as TEXT in ISO-8601 ('YYYY-MM-DD'). This is the SQLite
--     convention and sorts/compares correctly as a plain string.
--   * Money is stored as REAL. Fine for a prototype; a production system would
--     use INTEGER paise to avoid float drift.
--   * Foreign keys are declared AND enforced (PRAGMA foreign_keys = ON is set
--     on every connection in src/database.py).
-- ===========================================================================

CREATE TABLE IF NOT EXISTS students (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    roll_no         TEXT    NOT NULL UNIQUE,
    course          TEXT    NOT NULL,
    semester        INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 10),
    hostel_resident INTEGER NOT NULL DEFAULT 1 CHECK (hostel_resident IN (0, 1)),
    monthly_budget  REAL    NOT NULL CHECK (monthly_budget > 0),
    persona_note    TEXT,                -- short human description, used in LLM prompts
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS categories (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    icon           TEXT,
    typical_share  REAL    NOT NULL DEFAULT 0.0   -- benchmark share of a student budget
);

CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   INTEGER NOT NULL REFERENCES students(id)   ON DELETE CASCADE,
    category_id  INTEGER NOT NULL REFERENCES categories(id) ON DELETE RESTRICT,
    txn_date     TEXT    NOT NULL,                  -- ISO 'YYYY-MM-DD'
    amount       REAL    NOT NULL CHECK (amount > 0),
    merchant     TEXT    NOT NULL,
    description  TEXT,
    payment_mode TEXT    NOT NULL DEFAULT 'UPI',
    -- How the row entered the system. Drives the "automation" story:
    -- manual | receipt_ocr | sms_parse | recurring_auto
    source       TEXT    NOT NULL DEFAULT 'manual',
    receipt_path TEXT,
    is_recurring INTEGER NOT NULL DEFAULT 0 CHECK (is_recurring IN (0, 1)),
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Composite index: every analytics query filters by student then by date range.
CREATE INDEX IF NOT EXISTS idx_expenses_student_date ON expenses (student_id, txn_date);
CREATE INDEX IF NOT EXISTS idx_expenses_category     ON expenses (category_id);

CREATE TABLE IF NOT EXISTS budgets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   INTEGER NOT NULL REFERENCES students(id)   ON DELETE CASCADE,
    category_id  INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    month        TEXT    NOT NULL,                  -- 'YYYY-MM'
    limit_amount REAL    NOT NULL CHECK (limit_amount >= 0),
    UNIQUE (student_id, category_id, month)
);

CREATE TABLE IF NOT EXISTS savings_goals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    title         TEXT    NOT NULL,
    target_amount REAL    NOT NULL CHECK (target_amount > 0),
    saved_amount  REAL    NOT NULL DEFAULT 0,
    deadline      TEXT,                              -- ISO 'YYYY-MM-DD'
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Curated student-finance advice. Embedded into the vector store so the
-- assistant can ground "how do I save money?" answers in real guidance
-- instead of hallucinating generic tips.
CREATE TABLE IF NOT EXISTS knowledge_base (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    topic    TEXT NOT NULL,
    content  TEXT NOT NULL,
    tags     TEXT
);

-- Chat transcript persistence so a refresh does not wipe the demo conversation.
CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
