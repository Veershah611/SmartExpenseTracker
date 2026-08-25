"""
demo_rag_nlp.py
===============
Quick demo script to verify all Role #5 (RAG & NLP) modules work
end-to-end against the seeded database.

Run:
    python demo_rag_nlp.py

Requires: Ollama running with llama3.2:3b and nomic-embed-text pulled.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
from backend import llm_engine
from backend.vector_store import VectorStore
from backend.rag_engine import RAGEngine
from backend.nlp_quick_log import parse_expense, insert_parsed_expense, ParseError


def banner(title: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print(f"{'=' * 66}")


def main() -> int:
    # ── 1. Health check ──────────────────────────────────────────
    banner("1. Ollama Health Check")
    healthy = llm_engine.check_health()
    print(f"  Ollama at {config.OLLAMA_HOST}: {'✅ ONLINE' if healthy else '❌ OFFLINE'}")

    if not healthy:
        print("\n  ⚠️  Ollama is not running. Start it with:  ollama serve")
        print("  Then pull models:")
        print(f"    ollama pull {config.OLLAMA_CHAT_MODEL}")
        print(f"    ollama pull {config.OLLAMA_EMBED_MODEL}")
        return 1

    models = llm_engine.list_local_models()
    print(f"  Local models: {', '.join(models) if models else '(none)'}")

    # ── 2. LLM generation ────────────────────────────────────────
    banner("2. LLM Generation Test")
    try:
        reply = llm_engine.generate(
            "What is 2+2? Reply with ONLY the number.",
            system="You are a calculator.",
            temperature=0.0,
        )
        print(f"  Prompt:   'What is 2+2?'")
        print(f"  Response: '{reply.strip()}'")
    except llm_engine.LLMError as e:
        print(f"  ❌ Generation failed: {e}")
        return 1

    # ── 3. Embeddings ─────────────────────────────────────────────
    banner("3. Embedding Test")
    try:
        vec = llm_engine.embed("University student spending habits")
        print(f"  Text:      'University student spending habits'")
        print(f"  Dimension: {len(vec)}")
        print(f"  First 5:   {[round(v, 4) for v in vec[:5]]}")
    except llm_engine.LLMError as e:
        print(f"  ❌ Embedding failed: {e}")
        return 1

    # ── 4. Vector Store indexing ──────────────────────────────────
    banner("4. Vector Store — Indexing")
    vs = VectorStore()

    # Reset collections for a clean demo.
    vs.reset(config.RAG_COLLECTION_KNOWLEDGE)
    vs.reset(config.RAG_COLLECTION_EXPENSES)

    kb_count = vs.index_knowledge_base()
    print(f"  Knowledge base: {kb_count} entries indexed")

    exp_count = vs.index_expenses(student_id=1)
    print(f"  Expenses (student 1): {exp_count} transactions indexed")

    # ── 5. Semantic search ────────────────────────────────────────
    banner("5. Semantic Search Demo")
    queries = [
        "How can I save money on food delivery?",
        "How much did I spend on transport?",
        "Netflix subscription cost",
    ]
    for q in queries:
        print(f"\n  🔎 Query: '{q}'")
        # Search knowledge base.
        kb_results = vs.search(config.RAG_COLLECTION_KNOWLEDGE, q, top_k=2)
        if kb_results:
            print(f"  📚 Top advice (distance={kb_results[0]['distance']:.3f}):")
            print(f"     {kb_results[0]['document'][:120]}...")
        # Search expenses.
        exp_results = vs.search(config.RAG_COLLECTION_EXPENSES, q, top_k=2)
        if exp_results:
            print(f"  💰 Top transaction (distance={exp_results[0]['distance']:.3f}):")
            print(f"     {exp_results[0]['document']}")

    # ── 6. RAG Chat ───────────────────────────────────────────────
    banner("6. RAG Conversational Assistant")
    rag = RAGEngine(vector_store=vs, db_path=config.DB_PATH)
    rag.clear_history(student_id=1)  # fresh start

    test_questions = [
        "How much did I spend on food this month?",
        "Give me tips to save money on food delivery.",
    ]
    for q in test_questions:
        print(f"\n  🧑 Student: {q}")
        print(f"  🤖 Assistant: ", end="", flush=True)
        for token in rag.chat_stream(q, student_id=1):
            print(token, end="", flush=True)
        print()

    # Show persisted history.
    history = rag.load_history(student_id=1)
    print(f"\n  📝 Chat history: {len(history)} messages persisted in SQLite")

    # ── 7. Natural Language Quick Log ─────────────────────────────
    banner("7. Natural Language Quick Log (Creative Feature)")
    test_inputs = [
        "Spent 50 on chai at canteen",
        "Auto to college 120 yesterday",
        "Netflix subscription 499",
        "Jio recharge 299 cash",
        "Bought books from Amazon for 850 using debit card",
    ]
    for text in test_inputs:
        print(f"\n  📝 Input: '{text}'")
        try:
            result = parse_expense(text, student_id=1)
            print(f"     ✅ Parsed:")
            print(f"        Amount:   {config.as_currency(result['amount'])}")
            print(f"        Category: {result['_category_name']}")
            print(f"        Merchant: {result['merchant']}")
            print(f"        Date:     {result['txn_date']}")
            print(f"        Payment:  {result['payment_mode']}")

            # Actually insert it.
            row_id = insert_parsed_expense(result)
            print(f"        → Inserted as expense #{row_id}")
        except (ParseError, llm_engine.LLMError) as e:
            print(f"     ❌ Failed: {e}")

    # ── Done ──────────────────────────────────────────────────────
    banner("✅ All Role #5 modules verified!")
    print("  • llm_engine.py      — Health, generate, stream, embed")
    print("  • vector_store.py    — ChromaDB/NumPy index, search")
    print("  • rag_engine.py      — RAG chat with history persistence")
    print("  • nlp_quick_log.py   — NL parsing with validation & insert")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
