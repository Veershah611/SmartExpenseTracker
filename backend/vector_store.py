"""
backend/vector_store.py
=======================
Semantic-search layer over two document collections:

1. **finance_knowledge** — curated student-finance advice from the
   ``knowledge_base`` table.  Gives the RAG assistant grounded, real
   tips instead of hallucinated platitudes.

2. **expense_documents** — every transaction for a student, formatted
   as a natural-language sentence so the LLM can reason over them
   (e.g. *"On 2025-10-15, spent ₹340 at Swiggy on Food delivery via UPI"*).

Storage backends
----------------
* **Primary**: ChromaDB ``PersistentClient`` stored in
  ``data/vector_store/``.
* **Fallback**: A pure-NumPy index (cosine similarity on an ``.npy``
  matrix) that kicks in automatically if ChromaDB is not installed or
  fails to initialise.  The fallback is slower but has zero external
  dependencies beyond NumPy.

All embeddings are generated via :mod:`backend.llm_engine` (Ollama
``nomic-embed-text``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

import config
from backend import llm_engine

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Try to import ChromaDB — fall back gracefully
# ────────────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings as _ChromaSettings

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.info("ChromaDB not available — using NumPy fallback index.")


# ────────────────────────────────────────────────────────────────────
# NumPy fallback index
# ────────────────────────────────────────────────────────────────────

class _NumpyFallbackIndex:
    """
    Minimal vector index backed by a NumPy matrix and a JSON doc list.

    Files written to ``store_dir``:
    - ``{collection}_vectors.npy``  — (N, D) float32 matrix
    - ``{collection}_docs.json``    — list of {id, document, metadata}
    """

    def __init__(self, store_dir: Path, collection_name: str) -> None:
        self._dir = store_dir
        self._name = collection_name
        self._vec_path = store_dir / f"{collection_name}_vectors.npy"
        self._doc_path = store_dir / f"{collection_name}_docs.json"
        self._vectors: np.ndarray | None = None  # (N, D)
        self._docs: list[dict[str, Any]] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if self._vec_path.exists() and self._doc_path.exists():
            self._vectors = np.load(str(self._vec_path))
            with open(self._doc_path, "r", encoding="utf-8") as fh:
                self._docs = json.load(fh)
        else:
            self._vectors = None
            self._docs = []

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None and len(self._docs) > 0:
            np.save(str(self._vec_path), self._vectors)
            with open(self._doc_path, "w", encoding="utf-8") as fh:
                json.dump(self._docs, fh, ensure_ascii=False)

    # ── public API ───────────────────────────────────────────────

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add or replace documents (upsert semantics)."""
        metadatas = metadatas or [{} for _ in ids]
        new_vecs = np.array(embeddings, dtype=np.float32)

        # Remove existing docs with the same IDs (upsert).
        existing_ids = {d["id"] for d in self._docs}
        keep_mask = [d["id"] not in set(ids) for d in self._docs]

        if self._vectors is not None and len(self._docs) > 0:
            kept_vecs = self._vectors[keep_mask] if any(keep_mask) else np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            kept_docs = [d for d, k in zip(self._docs, keep_mask) if k]
        else:
            kept_vecs = np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            kept_docs = []

        self._vectors = np.vstack([kept_vecs, new_vecs]) if kept_vecs.shape[0] > 0 else new_vecs
        self._docs = kept_docs + [
            {"id": i, "document": doc, "metadata": meta}
            for i, doc, meta in zip(ids, documents, metadatas)
        ]
        self._save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most similar documents by cosine similarity."""
        if self._vectors is None or len(self._docs) == 0:
            return []

        qvec = np.array(query_embedding, dtype=np.float32)
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(qvec)
        # Guard against zero-norm vectors.
        safe_norms = np.where(norms == 0, 1.0, norms)
        safe_q_norm = q_norm if q_norm > 0 else 1.0
        similarities = self._vectors @ qvec / (safe_norms * safe_q_norm)

        k = min(top_k, len(self._docs))
        if k >= len(self._docs):
            # Fewer docs than requested — just sort everything.
            top_indices = np.argsort(-similarities)[:k]
        else:
            top_indices = np.argpartition(-similarities, k)[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            doc = self._docs[idx]
            results.append({
                "id": doc["id"],
                "document": doc["document"],
                "metadata": doc.get("metadata", {}),
                "distance": float(1 - similarities[idx]),  # cosine distance
            })
        return results

    def reset(self) -> None:
        """Wipe the collection from disk and memory."""
        self._vectors = None
        self._docs = []
        for p in (self._vec_path, self._doc_path):
            if p.exists():
                p.unlink()

    def count(self) -> int:
        return len(self._docs)


# ────────────────────────────────────────────────────────────────────
# Main VectorStore class
# ────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Unified interface to the semantic-search layer.

    Automatically selects ChromaDB when available, otherwise falls back
    to :class:`_NumpyFallbackIndex`.

    Parameters
    ----------
    persist_dir : Path | None
        Where to store the index.  Defaults to ``config.VECTOR_STORE_PATH``.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._dir = persist_dir or config.VECTOR_STORE_PATH
        self._dir.mkdir(parents=True, exist_ok=True)
        self._use_chroma = _CHROMA_AVAILABLE
        self._chroma_client: Any | None = None
        self._fallback_indices: dict[str, _NumpyFallbackIndex] = {}

        if self._use_chroma:
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self._dir),
                    settings=_ChromaSettings(anonymized_telemetry=False),
                )
                logger.info("ChromaDB initialised at %s", self._dir)
            except Exception as exc:
                logger.warning(
                    "ChromaDB init failed (%s) — falling back to NumPy index.",
                    exc,
                )
                self._use_chroma = False

    # ── internal helpers ─────────────────────────────────────────

    def _get_chroma_collection(self, name: str) -> Any:
        """Get or create a Chroma collection."""
        return self._chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_fallback(self, name: str) -> _NumpyFallbackIndex:
        if name not in self._fallback_indices:
            self._fallback_indices[name] = _NumpyFallbackIndex(self._dir, name)
        return self._fallback_indices[name]

    # ── generic CRUD ─────────────────────────────────────────────

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Embed and upsert documents into a named collection.

        Returns the number of documents added.
        """
        if not documents:
            return 0

        # Embed all documents in one batch call.
        embeddings = llm_engine.embed_batch(documents)

        if self._use_chroma:
            col = self._get_chroma_collection(collection_name)
            # ChromaDB upsert handles add-or-replace cleanly.
            kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
            }
            if metadatas:
                kwargs["metadatas"] = metadatas
            col.upsert(**kwargs)
        else:
            fb = self._get_fallback(collection_name)
            fb.add(ids, documents, embeddings, metadatas)

        logger.info("Indexed %d docs into '%s'.", len(documents), collection_name)
        return len(documents)

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search against a named collection.

        Parameters
        ----------
        collection_name : str
            Which collection to search.
        query : str
            Natural-language query.
        top_k : int | None
            Number of results. Defaults to ``config.RAG_TOP_K``.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id``, ``document``, ``metadata``, ``distance``.
            Lower distance = more similar.
        """
        top_k = top_k or config.RAG_TOP_K
        query_embedding = llm_engine.embed(query)

        if self._use_chroma:
            col = self._get_chroma_collection(collection_name)
            raw = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, col.count() or 1),
            )
            results: list[dict[str, Any]] = []
            if raw and raw.get("ids") and raw["ids"][0]:
                for i, doc_id in enumerate(raw["ids"][0]):
                    results.append({
                        "id": doc_id,
                        "document": raw["documents"][0][i] if raw.get("documents") else "",
                        "metadata": raw["metadatas"][0][i] if raw.get("metadatas") else {},
                        "distance": raw["distances"][0][i] if raw.get("distances") else 0.0,
                    })
            return results
        else:
            fb = self._get_fallback(collection_name)
            return fb.search(query_embedding, top_k)

    def reset(self, collection_name: str) -> None:
        """Wipe a collection completely (data + embeddings)."""
        if self._use_chroma:
            try:
                self._chroma_client.delete_collection(name=collection_name)
                logger.info("Deleted Chroma collection '%s'.", collection_name)
            except Exception:
                pass  # collection didn't exist
        else:
            fb = self._get_fallback(collection_name)
            fb.reset()
            self._fallback_indices.pop(collection_name, None)

    def count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        if self._use_chroma:
            try:
                col = self._get_chroma_collection(collection_name)
                return col.count()
            except Exception:
                return 0
        else:
            fb = self._get_fallback(collection_name)
            return fb.count()

    # ── domain-specific indexing ─────────────────────────────────

    def index_knowledge_base(self, db_path: Path | str | None = None) -> int:
        """
        Read the ``knowledge_base`` table and index every entry into the
        ``finance_knowledge`` collection.

        Returns the number of entries indexed.
        """
        db_path = Path(db_path or config.DB_PATH)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            rows = conn.execute(
                "SELECT id, topic, content, tags FROM knowledge_base"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning("knowledge_base table is empty — nothing to index.")
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for row_id, topic, content, tags in rows:
            ids.append(f"kb_{row_id}")
            # Combine topic + content for a richer embedding.
            documents.append(f"{topic}: {content}")
            metadatas.append({
                "source": "knowledge_base",
                "topic": topic,
                "tags": tags or "",
            })

        return self.add_documents(
            config.RAG_COLLECTION_KNOWLEDGE,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def index_expenses(
        self,
        db_path: Path | str | None = None,
        student_id: int | None = None,
    ) -> int:
        """
        Read the ``expenses`` table (optionally filtered to one student)
        and index each transaction as a natural-language sentence into the
        ``expense_documents`` collection.

        Returns the number of expenses indexed.
        """
        db_path = Path(db_path or config.DB_PATH)
        student_id = student_id or config.DEFAULT_STUDENT_ID

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            rows = conn.execute(
                """
                SELECT e.id, e.txn_date, e.amount, e.merchant,
                       c.name AS category, e.description, e.payment_mode
                FROM   expenses e
                JOIN   categories c ON e.category_id = c.id
                WHERE  e.student_id = ?
                ORDER  BY e.txn_date DESC
                """,
                (student_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning(
                "No expenses found for student %d — nothing to index.",
                student_id,
            )
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for expense_id, txn_date, amount, merchant, category, description, payment_mode in rows:
            ids.append(f"exp_{expense_id}")
            # Natural-language sentence — this is what the LLM will see
            # when the RAG engine retrieves context.
            sentence = (
                f"On {txn_date}, spent {config.CURRENCY_SYMBOL}{amount:.2f} "
                f"at {merchant} on {category}"
            )
            if description:
                sentence += f" ({description})"
            sentence += f" via {payment_mode}."
            documents.append(sentence)
            metadatas.append({
                "source": "expenses",
                "txn_date": txn_date,
                "amount": float(amount),
                "merchant": merchant,
                "category": category,
                "payment_mode": payment_mode,
            })

        return self.add_documents(
            config.RAG_COLLECTION_EXPENSES,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
