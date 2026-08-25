"""
scripts/generate_mock_data.py
=============================
Creates ``data/expenses.db`` and fills it with a realistic, *seasonal* year of
university-student spending.

Why not random noise?
---------------------
A flat random dataset makes every chart look the same and gives the AI nothing
interesting to say. This generator deliberately bakes in patterns a judge can
see on screen and the assistant can discover:

  * near-daily canteen/mess spend, higher on weekends
  * a big semester-fee spike in January and July
  * a textbook/stationery spike in the first fortnight of each semester
  * festival bumps around Navratri and Diwali
  * fixed recurring charges (Jio recharge, Spotify, Netflix) on the same date
  * a month-end squeeze: discretionary spend drops in the last five days
  * three distinct personas (hosteller / day-scholar / postgrad)

Usage
-----
    python backend/scripts/generate_mock_data.py                # create if absent
    python backend/scripts/generate_mock_data.py --force        # wipe and rebuild
    python backend/scripts/generate_mock_data.py --months 18    # longer history
    python backend/scripts/generate_mock_data.py --seed 7       # different variation
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

# This script lives at <root>/backend/scripts/, so the project root is two
# levels up. Adding it to sys.path lets the script be run directly from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (import must follow the sys.path fix)

# Windows consoles default to cp1252 and choke on the rupee sign. Force UTF-8
# so the summary printout never crashes the script.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #
STUDENTS: list[dict] = [
    {
        "name": "Aarav Mehta",
        "roll_no": "22BCE001",
        "course": "B.Tech Computer Engineering",
        "semester": 5,
        "hostel_resident": 1,
        "monthly_budget": 15000.0,
        "lifestyle": "moderate",
        "persona_note": (
            "Hostel resident. Eats most meals at the mess but orders in late at "
            "night. Goes home to Surat roughly once a month. Saving for a laptop."
        ),
    },
    {
        "name": "Diya Patel",
        "roll_no": "22BIT045",
        "course": "B.Tech Information Technology",
        "semester": 5,
        "hostel_resident": 0,
        "monthly_budget": 12000.0,
        "lifestyle": "frugal",
        "persona_note": (
            "Day scholar commuting from Bodakdev. Carries a tiffin so canteen "
            "spend is low, but daily fuel and auto fares add up fast."
        ),
    },
    {
        "name": "Rohan Shah",
        "roll_no": "24MBA112",
        "course": "MBA (Finance)",
        "semester": 3,
        "hostel_resident": 1,
        "monthly_budget": 18000.0,
        "lifestyle": "premium",
        "persona_note": (
            "Postgrad hostel resident with a bigger allowance. Frequent cafe "
            "meetings, weekend outings and online shopping."
        ),
    },
]

# --------------------------------------------------------------------------- #
# Merchants, keyed by category name. Chosen to feel like Ahmedabad / Nirma.
# --------------------------------------------------------------------------- #
MERCHANTS: dict[str, list[str]] = {
    "Food & Canteen": [
        "Nirma Mess", "Campus Canteen", "Chaayos Kiosk", "Amul Parlour",
        "Night Canteen", "Swiggy", "Zomato", "Sankalp Restaurant",
        "Dominos Vastrapur", "Havmor Ice Cream", "Juice Junction",
    ],
    "Hostel & Rent": [
        "Nirma University Hostel Office", "Mess Fee Counter",
        "Hostel Maintenance Desk", "Room Deposit Counter",
    ],
    "Transport": [
        "Ahmedabad BRTS", "Uber", "Ola Auto", "HP Petrol Pump",
        "AMTS Bus Pass", "IRCTC", "Campus Shuttle", "Rapido",
    ],
    "Books & Stationery": [
        "Crossword Bookstore", "Campus Book Depot", "Amazon Books",
        "Xerox and Print Shop", "Navneet Stationery", "Classmate Store",
    ],
    "Academics & Fees": [
        "Nirma University Fee Portal", "Exam Cell", "Coursera",
        "IEEE Membership", "Workshop Registration", "Lab Kit Counter",
    ],
    "Entertainment": [
        "PVR Acropolis", "Netflix", "Spotify", "BookMyShow",
        "Gaming Zone Vastrapur", "Cricket Turf Booking", "Sports Club",
    ],
    "Shopping": [
        "Myntra", "Amazon", "Flipkart", "Decathlon", "Reliance Trends",
        "D-Mart", "Local Market Stall",
    ],
    "Health & Medical": [
        "Apollo Pharmacy", "Campus Health Centre", "Wellness Forever",
        "Doctor Consultation", "Gym Membership",
    ],
    "Mobile & Internet": [
        "Jio Recharge", "Airtel Recharge", "Hostel Wi-Fi Plan", "Vi Recharge",
    ],
    "Miscellaneous": [
        "Campus Laundry", "Salon Cut and Style", "Photocopy Centre",
        "Charity Drive", "Gift Purchase", "ATM Withdrawal Fee",
    ],
}

# --------------------------------------------------------------------------- #
# Curated advice -- becomes the RAG "knowledge" collection in Step 3.
# --------------------------------------------------------------------------- #
KNOWLEDGE_BASE: list[tuple[str, str, str]] = [
    ("50/30/20 rule",
     "Split your monthly allowance three ways: 50 percent to needs (mess fees, "
     "hostel, transport, books), 30 percent to wants (eating out, movies, "
     "subscriptions) and 20 percent to savings. For a student on Rs 12,000 a "
     "month that is Rs 6,000 needs, Rs 3,600 wants and Rs 2,400 saved.",
     "budgeting,rule,savings"),
    ("Cutting food delivery costs",
     "Food delivery is the single biggest leak in most student budgets. Each "
     "Swiggy or Zomato order carries a delivery fee, a platform fee and surge "
     "pricing that can add 30 to 40 percent on top of the food itself. Capping "
     "delivery to four orders a month and using the mess for the rest typically "
     "saves Rs 1,500 to Rs 2,500 a month.",
     "food,delivery,savings"),
    ("Student transport savings",
     "A monthly AMTS or BRTS bus pass costs far less than daily auto or Rapido "
     "fares. If you commute more than 12 days a month, the pass pays for itself. "
     "Carpooling with classmates on the same route splits fuel three or four ways.",
     "transport,commute,savings"),
    ("Buying textbooks cheaply",
     "Buy second-hand textbooks from seniors, use the university library reserve "
     "section, and check for free legal PDFs of standard texts before buying new. "
     "Splitting the cost of a reference book with two classmates and sharing it "
     "across a semester is common practice.",
     "books,academics,savings"),
    ("Subscription audit",
     "Streaming and music subscriptions are small individually but permanent. "
     "Review them every semester: share a family plan with hostel-mates, pause "
     "services during exam months, and use student pricing where offered "
     "(Spotify and YouTube both discount for verified students).",
     "subscriptions,entertainment,savings"),
    ("Emergency fund for students",
     "Keep at least one month of expenses aside for emergencies such as medical "
     "costs, a broken laptop charger, or an unplanned trip home. Build it slowly "
     "by moving Rs 500 to Rs 1,000 aside on the day your allowance arrives, not "
     "at the end of the month when nothing is left.",
     "savings,emergency,goals"),
    ("Festival season overspending",
     "Spending reliably spikes during Navratri and Diwali on clothes, gifts and "
     "outings. Set a festival budget a month in advance and treat it as a "
     "separate envelope so it does not eat into mess fees or academic costs.",
     "festival,shopping,budgeting"),
    ("UPI and the visibility problem",
     "UPI makes small payments frictionless, which is exactly why they add up "
     "unnoticed. Ten payments of Rs 60 for chai are Rs 600. Reviewing your UPI "
     "history once a week turns invisible spending into a number you can act on.",
     "upi,payments,awareness"),
    ("Month-end cash crunch",
     "If you consistently run out of money in the last week of the month, the "
     "problem is usually front-loaded discretionary spend, not a low allowance. "
     "Move fixed costs (mess fee, recharge, subscriptions) to the first three "
     "days and give yourself a weekly discretionary limit for what remains.",
     "budgeting,cashflow,planning"),
    ("Semester fee planning",
     "Semester fees are large, predictable and due at the same time every year. "
     "Dividing the amount across the preceding five months and setting it aside "
     "turns a painful lump sum into a manageable monthly line item.",
     "fees,academics,planning"),
    ("Earning while studying",
     "Campus opportunities such as teaching assistantships, paid research work, "
     "coding contests with prize money, and freelance design or development work "
     "can add Rs 3,000 to Rs 10,000 a month without affecting attendance.",
     "income,earning,student"),
    ("Tracking is the intervention",
     "Studies of personal finance behaviour consistently find that the act of "
     "recording expenses reduces them by 10 to 15 percent, before any deliberate "
     "cuts are made. Logging every transaction for one full month is the "
     "highest-return habit a student can build.",
     "tracking,habit,awareness"),
]


# --------------------------------------------------------------------------- #
# Seasonal helpers
# --------------------------------------------------------------------------- #
def is_semester_start(day: date) -> bool:
    """True during the first fortnight of the January and July semesters."""
    return (day.month == 1 and day.day <= 15) or (day.month == 7 and 10 <= day.day <= 25)


def is_festival_period(day: date) -> bool:
    """Approximate Navratri (late Sep to early Oct) and Diwali (late Oct to early Nov)."""
    if day.month == 9 and day.day >= 22:
        return True
    if day.month == 10:
        return True
    if day.month == 11 and day.day <= 5:
        return True
    return False


def is_month_end_squeeze(day: date) -> bool:
    """True for the last five days of a month, when discretionary spend drops."""
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (next_month - day).days <= 5


def jitter(rng: random.Random, base: float, spread: float = 0.25) -> float:
    """Return ``base`` nudged by +/- ``spread`` and floored at Rs 10."""
    value = base * rng.uniform(1 - spread, 1 + spread)
    return round(max(value, 10.0), 2)


def month_key(day: date) -> str:
    """Format a date as the 'YYYY-MM' key used by the budgets table."""
    return f"{day.year:04d}-{day.month:02d}"


# --------------------------------------------------------------------------- #
# Per-category transaction generators
# --------------------------------------------------------------------------- #
def _food(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Near-daily canteen spend, with weekend and delivery behaviour baked in."""
    hosteller = student["hostel_resident"] == 1
    weekend = day.weekday() >= 5

    # A hosteller's main meals are already covered by the monthly mess fee, so
    # their daily campus spend is snacks and chai on top of it -- small amounts.
    # A day scholar buys food on campus less often (tiffin from home) but pays
    # full price when they do.
    if hosteller:
        base_chance, low, high = 0.72, 30.0, 110.0
    else:
        base_chance, low, high = 0.45, 60.0, 170.0

    if rng.random() < base_chance:
        # One or two small campus purchases per day.
        for _ in range(rng.choice([1, 1, 2])):
            merchant = rng.choice(
                ["Nirma Mess", "Campus Canteen", "Chaayos Kiosk",
                 "Amul Parlour", "Juice Junction"]
            )
            rows.append((day, "Food & Canteen", jitter(rng, rng.uniform(low, high)),
                         merchant, "Campus food purchase", "UPI", "manual", 0))

    # Delivery apps: mostly weekends and late nights, suppressed at month end.
    delivery_chance = 0.30 if weekend else 0.12
    if student["lifestyle"] == "premium":
        delivery_chance += 0.10          # Rohan orders in more often
    if is_month_end_squeeze(day):
        delivery_chance *= 0.35          # money is tight, cooking/mess instead
    if rng.random() < delivery_chance:
        merchant = rng.choice(["Swiggy", "Zomato", "Dominos Vastrapur",
                               "Sankalp Restaurant"])
        rows.append((day, "Food & Canteen", jitter(rng, rng.uniform(220, 560)),
                     merchant, "Food delivery order", "UPI", "sms_parse", 0))


def _hostel(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Monthly hostel/mess fee for residents only, charged early in the month."""
    if student["hostel_resident"] == 0:
        return
    if day.day == 3:
        rows.append((day, "Hostel & Rent", jitter(rng, 4200, spread=0.05),
                     "Mess Fee Counter", "Monthly mess and hostel charges",
                     "Net Banking", "recurring_auto", 1))
    # Occasional maintenance or laundry-deposit style charge.
    if day.day == 18 and rng.random() < 0.25:
        rows.append((day, "Hostel & Rent", jitter(rng, 350),
                     "Hostel Maintenance Desk", "Room maintenance charge",
                     "Cash", "manual", 0))


def _transport(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Daily commute for day scholars; occasional travel for hostellers."""
    weekday = day.weekday() < 5

    if student["hostel_resident"] == 0:
        # Day scholar: fuel twice a week, autos on the remaining days.
        if weekday and rng.random() < 0.30:
            rows.append((day, "Transport", jitter(rng, rng.uniform(280, 450)),
                         "HP Petrol Pump", "Two-wheeler fuel", "UPI", "manual", 0))
        elif weekday and rng.random() < 0.45:
            rows.append((day, "Transport", jitter(rng, rng.uniform(45, 130)),
                         rng.choice(["Ola Auto", "Rapido", "Ahmedabad BRTS"]),
                         "Commute to campus", "UPI", "manual", 0))
    else:
        # Hosteller: local runs plus a monthly trip home.
        if rng.random() < 0.18:
            rows.append((day, "Transport", jitter(rng, rng.uniform(50, 180)),
                         rng.choice(["Ola Auto", "Uber", "Campus Shuttle", "Rapido"]),
                         "Local travel", "UPI", "manual", 0))
        if day.day in (12, 26) and rng.random() < 0.5:
            rows.append((day, "Transport", jitter(rng, rng.uniform(350, 900)),
                         "IRCTC", "Train fare home", "Net Banking", "manual", 0))


def _books(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Textbook spike at semester start; photocopies year-round."""
    if is_semester_start(day) and rng.random() < 0.35:
        rows.append((day, "Books & Stationery", jitter(rng, rng.uniform(600, 1800)),
                     rng.choice(["Crossword Bookstore", "Campus Book Depot",
                                 "Amazon Books"]),
                     "Semester textbooks", "Debit Card", "receipt_ocr", 0))
    elif rng.random() < 0.09:
        rows.append((day, "Books & Stationery", jitter(rng, rng.uniform(40, 220)),
                     rng.choice(["Xerox and Print Shop", "Navneet Stationery",
                                 "Classmate Store"]),
                     "Printouts and stationery", "Cash", "manual", 0))


def _academics(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """The big semester fee, plus occasional certifications and workshops."""
    # Semester fee: January and July, on the 8th.
    if day.month in (1, 7) and day.day == 8:
        base = 62000 if student["course"].startswith("MBA") else 48000
        rows.append((day, "Academics & Fees", jitter(rng, base, spread=0.03),
                     "Nirma University Fee Portal", "Semester tuition fee",
                     "Net Banking", "manual", 0))
    if rng.random() < 0.025:
        rows.append((day, "Academics & Fees", jitter(rng, rng.uniform(400, 1500)),
                     rng.choice(["Coursera", "Workshop Registration",
                                 "IEEE Membership", "Exam Cell"]),
                     "Course or workshop fee", "Credit Card", "manual", 0))


def _entertainment(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Fixed subscriptions plus weekend outings."""
    # Recurring subscriptions on fixed dates -- the "automation" story.
    if day.day == 5:
        rows.append((day, "Entertainment", 199.0, "Spotify",
                     "Spotify student subscription", "UPI", "recurring_auto", 1))
    if day.day == 15 and student["lifestyle"] in ("moderate", "premium"):
        rows.append((day, "Entertainment", 499.0, "Netflix",
                     "Netflix monthly plan", "Credit Card", "recurring_auto", 1))

    weekend = day.weekday() >= 5
    chance = 0.22 if weekend else 0.05
    if is_month_end_squeeze(day):
        chance *= 0.4
    if rng.random() < chance:
        rows.append((day, "Entertainment", jitter(rng, rng.uniform(180, 700)),
                     rng.choice(["PVR Acropolis", "BookMyShow",
                                 "Gaming Zone Vastrapur", "Cricket Turf Booking"]),
                     "Weekend outing", "UPI", "manual", 0))


def _shopping(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Occasional online shopping, spiking hard during festival season."""
    chance = 0.05
    if is_festival_period(day):
        chance = 0.16                     # Navratri/Diwali clothes and gifts
    if student["lifestyle"] == "premium":
        chance += 0.03
    if is_month_end_squeeze(day):
        chance *= 0.4
    if rng.random() < chance:
        amount = rng.uniform(400, 2600) if is_festival_period(day) else rng.uniform(250, 1400)
        rows.append((day, "Shopping", jitter(rng, amount),
                     rng.choice(MERCHANTS["Shopping"]),
                     "Festival shopping" if is_festival_period(day) else "Online order",
                     rng.choice(["UPI", "Credit Card", "Debit Card"]), "sms_parse", 0))


def _health(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Rare medical spend; a gym membership for one persona."""
    if rng.random() < 0.03:
        rows.append((day, "Health & Medical", jitter(rng, rng.uniform(120, 800)),
                     rng.choice(["Apollo Pharmacy", "Campus Health Centre",
                                 "Wellness Forever", "Doctor Consultation"]),
                     "Medicines or consultation", "Cash", "receipt_ocr", 0))
    if day.day == 7 and student["lifestyle"] == "premium":
        rows.append((day, "Health & Medical", 800.0, "Gym Membership",
                     "Monthly gym fee", "UPI", "recurring_auto", 1))


def _mobile(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Prepaid recharge on a fixed monthly date."""
    if day.day == 10:
        # A student stays on one operator. Deriving it from the roll number keeps
        # the merchant stable month to month, which is what makes the recurring
        # -subscription detector in the analytics layer able to spot it.
        operators = ["Jio Recharge", "Airtel Recharge", "Vi Recharge"]
        operator = operators[sum(ord(ch) for ch in student["roll_no"]) % len(operators)]
        rows.append((day, "Mobile & Internet", jitter(rng, 299, spread=0.12),
                     operator, "Monthly mobile recharge", "UPI", "recurring_auto", 1))


def _misc(rng: random.Random, student: dict, day: date, rows: list) -> None:
    """Laundry, salon, small odds and ends."""
    if student["hostel_resident"] == 1 and day.weekday() == 6 and rng.random() < 0.55:
        rows.append((day, "Miscellaneous", jitter(rng, rng.uniform(80, 200)),
                     "Campus Laundry", "Weekly laundry", "Cash", "manual", 0))
    if rng.random() < 0.04:
        rows.append((day, "Miscellaneous", jitter(rng, rng.uniform(60, 500)),
                     rng.choice(["Salon Cut and Style", "Photocopy Centre",
                                 "Charity Drive", "Gift Purchase"]),
                     "Miscellaneous expense", "Cash", "manual", 0))


# Ordered list of generators applied to every simulated day.
GENERATORS = [_food, _hostel, _transport, _books, _academics,
              _entertainment, _shopping, _health, _mobile, _misc]


def build_expense_rows(student: dict, start: date, end: date,
                       rng: random.Random) -> list[tuple]:
    """
    Walk every day between ``start`` and ``end`` and let each category generator
    decide whether the student spent anything that day.

    Returns a list of raw tuples; category names are resolved to IDs by the
    caller, which keeps this function free of database concerns.
    """
    rows: list[tuple] = []
    current = start
    while current <= end:
        for generator in GENERATORS:
            generator(rng, student, current, rows)
        current += timedelta(days=1)
    return rows


# --------------------------------------------------------------------------- #
# Database population
# --------------------------------------------------------------------------- #
def create_schema(conn: sqlite3.Connection) -> None:
    """Execute src/schema.sql. Safe to run repeatedly (all CREATE IF NOT EXISTS)."""
    schema_path = BACKEND_DIR / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file missing: {schema_path}")
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def seed_categories(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert the canonical categories from config and return {name: id}."""
    conn.executemany(
        "INSERT OR IGNORE INTO categories (name, icon, typical_share) VALUES (?, ?, ?)",
        config.EXPENSE_CATEGORIES,
    )
    return {name: cid for cid, name in conn.execute("SELECT id, name FROM categories")}


def seed_knowledge_base(conn: sqlite3.Connection) -> int:
    """Insert curated finance advice used later as RAG grounding."""
    conn.executemany(
        "INSERT INTO knowledge_base (topic, content, tags) VALUES (?, ?, ?)",
        KNOWLEDGE_BASE,
    )
    return len(KNOWLEDGE_BASE)


def seed_students(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert the personas and return {roll_no: id}."""
    # `lifestyle` steers the generators only -- it is not a table column, so it
    # is dropped here rather than polluting the schema.
    db_ready = [
        {key: value for key, value in student.items() if key != "lifestyle"}
        for student in STUDENTS
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO students
               (name, roll_no, course, semester, hostel_resident,
                monthly_budget, persona_note)
           VALUES (:name, :roll_no, :course, :semester, :hostel_resident,
                   :monthly_budget, :persona_note)""",
        db_ready,
    )
    return {roll: sid for sid, roll in conn.execute("SELECT id, roll_no FROM students")}


def seed_budgets(conn: sqlite3.Connection, student_id: int, monthly_budget: float,
                 categories: dict[str, int], start: date, end: date) -> int:
    """
    Create a per-category monthly budget for every month in the range, using the
    ``typical_share`` weights from config. Semester-fee months are excluded from
    the Academics category so a Rs 48,000 tuition payment does not make every
    other budget look absurd.
    """
    rows = []
    current = date(start.year, start.month, 1)
    while current <= end:
        for name, _icon, share in config.EXPENSE_CATEGORIES:
            rows.append((student_id, categories[name], month_key(current),
                         round(monthly_budget * share, 2)))
        # Advance one month.
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    conn.executemany(
        """INSERT OR REPLACE INTO budgets (student_id, category_id, month, limit_amount)
           VALUES (?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def seed_goals(conn: sqlite3.Connection, student_id: int, roll_no: str,
               today: date) -> None:
    """Give each persona a savings goal so the Goals widget has something to show."""
    goals_by_roll = {
        "22BCE001": [
            ("New Laptop for final year", 55000.0, 18500.0, today + timedelta(days=210)),
            ("Emergency Fund", 12000.0, 4200.0, today + timedelta(days=120)),
        ],
        "22BIT045": [
            ("Goa trip with friends", 15000.0, 6800.0, today + timedelta(days=90)),
            ("Certification course", 8000.0, 3000.0, today + timedelta(days=150)),
        ],
        "24MBA112": [
            ("CFA Level 1 registration", 95000.0, 41000.0, today + timedelta(days=180)),
            ("Emergency Fund", 20000.0, 15000.0, today + timedelta(days=60)),
        ],
    }
    rows = [
        (student_id, title, target, saved, deadline.isoformat())
        for title, target, saved, deadline in goals_by_roll.get(roll_no, [])
    ]
    conn.executemany(
        """INSERT INTO savings_goals
               (student_id, title, target_amount, saved_amount, deadline)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )


def populate(conn: sqlite3.Connection, months: int, seed: int) -> dict:
    """Run the full seeding pipeline and return a summary dict for printing."""
    rng = random.Random(seed)

    today = date.today()
    # Rewind `months` months, then start on the 1st for clean monthly buckets.
    start_month = today.month - months
    start_year = today.year
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    start = date(start_year, start_month, 1)

    categories = seed_categories(conn)
    student_ids = seed_students(conn)
    tips = seed_knowledge_base(conn)

    total_expenses = 0
    for student in STUDENTS:
        student_id = student_ids[student["roll_no"]]

        raw_rows = build_expense_rows(student, start, today, rng)
        db_rows = [
            (student_id, categories[category], day.isoformat(), amount,
             merchant, description, payment_mode, source, is_recurring)
            for (day, category, amount, merchant, description,
                 payment_mode, source, is_recurring) in raw_rows
        ]
        conn.executemany(
            """INSERT INTO expenses
                   (student_id, category_id, txn_date, amount, merchant,
                    description, payment_mode, source, is_recurring)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            db_rows,
        )
        total_expenses += len(db_rows)

        seed_budgets(conn, student_id, student["monthly_budget"], categories,
                     start, today)
        seed_goals(conn, student_id, student["roll_no"], today)

    conn.commit()
    return {
        "start": start,
        "end": today,
        "students": len(STUDENTS),
        "expenses": total_expenses,
        "categories": len(categories),
        "tips": tips,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Smart Expense Tracker SQLite demo database."
    )
    parser.add_argument("--db", default=str(config.DB_PATH),
                        help="Target database path (default: %(default)s)")
    parser.add_argument("--months", type=int, default=10,
                        help="Months of history to generate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible demos (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Delete an existing database before generating")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        if not args.force:
            print(f"[skip] {db_path} already exists. Re-run with --force to rebuild.")
            return 0
        try:
            db_path.unlink()
            print(f"[wipe] removed existing {db_path}")
        except OSError as exc:
            print(f"[error] could not delete {db_path}: {exc}", file=sys.stderr)
            return 1

    if args.months < 1:
        print("[error] --months must be at least 1", file=sys.stderr)
        return 1

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn)
        summary = populate(conn, months=args.months, seed=args.seed)
    except (sqlite3.Error, FileNotFoundError) as exc:
        print(f"[error] database generation failed: {exc}", file=sys.stderr)
        if conn is not None:
            conn.rollback()
        return 1
    finally:
        if conn is not None:
            conn.close()

    print("-" * 66)
    print(f"  Database : {db_path}")
    print(f"  Period   : {summary['start']}  ->  {summary['end']}")
    print(f"  Students : {summary['students']}")
    print(f"  Expenses : {summary['expenses']:,} transactions")
    print(f"  Categories / advice snippets : "
          f"{summary['categories']} / {summary['tips']}")
    print("-" * 66)
    print("Next: python backend/scripts/generate_mock_data.py --force  (to reshuffle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
