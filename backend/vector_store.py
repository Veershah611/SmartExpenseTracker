"""
backend/vector_store.py
=======================
Semantic search over expense summaries and curated finance advice.

Resilience is the whole point of this module
--------------------------------------------
A hackathon demo runs on whatever laptop is in the room. So every external
dependency here degrades instead of failing:

Embeddings (``OllamaEmbedder``):
    1. ``nomic-embed-text`` via Ollama          -- best quality
    2. the chat model via Ollama                -- works with no extra download
    3. a deterministic lexical hash vectoriser  -- works with no Ollama at all

Index (``get_vector_store``):
    1. ChromaDB if it imports                   -- proper vector database
    2. a NumPy cosine-similarity index          -- ~200 documents, so a brute
       force scan is genuinely fast enough

The fallbacks are not decoration: ChromaDB has no reliable wheel on Python 3.13
at the time of writing, and ``nomic-embed-text`` is a separate 270 MB pull.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Dimensionality of the lexical fallback vectors. 512 buckets is ample for a
# vocabulary this small and keeps the whole index well under a megabyte.
LEXICAL_DIM = 512


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class OllamaEmbedder:
    """
    Turns text into vectors, degrading through three strategies.

    The chosen strategy is resolved once, lazily, on first use and then cached
    in :attr:`mode` -- the UI displays it so a demo operator can see at a glance
    whether real embeddings or the lexical fallback are in play.
    """

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.OLLAMA_EMBED_MODEL
        self.mode: str | None = None      # 'embed_model' | 'chat_model' | 'lexical'
        self._resolved = False

    # -- strategy resolution ------------------------------------------------ #
    def _resolve(self) -> None:
        """Pick the best available strategy exactly once."""
        if self._resolved:
            return
        self._resolved = True

        available = self._list_models()
        if available is None:
            # Ollama is not reachable at all.
            self.mode = "lexical"
            return

        # Ollama reports "nomic-embed-text:latest" for a "nomic-embed-text" pull,
        # so compare on the bare name before the tag.
        bare = {name.split(":")[0] for name in available}
        if self.model.split(":")[0] in bare:
            self.mode = "embed_model"
            return

        if config.OLLAMA_CHAT_MODEL.split(":")[0] in bare:
            # Any causal LM can emit embeddings. Quality is below a dedicated
            # embedding model but far better than lexical matching, and needs
            # no extra download.
            self.model = config.OLLAMA_CHAT_MODEL
            self.mode = "chat_model"
            return

        self.mode = "lexical"

    def _list_models(self) -> list[str] | None:
        """Return installed model names, or ``None`` if Ollama is unreachable."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            return None

    # -- public API --------------------------------------------------------- #
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings into an L2-normalised ``(n, dim)`` float32 array.

        Normalising here means cosine similarity is a plain dot product later,
        which keeps the NumPy index simple.
        """
        if not texts:
            return np.zeros((0, LEXICAL_DIM), dtype=np.float32)

        self._resolve()

        if self.mode in ("embed_model", "chat_model"):
            vectors = self._embed_via_ollama(texts)
            if vectors is not None:
                return vectors
            # A mid-session Ollama failure should not take the app down.
            self.mode = "lexical"

        return np.vstack([self._lexical_vector(text) for text in texts])

    def _embed_via_ollama(self, texts: list[str]) -> np.ndarray | None:
        """Call Ollama's embedding endpoint. Returns ``None`` on any failure."""
        try:
            # /api/embed is the current batch endpoint.
            response = requests.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            if response.status_code == 404:
                # Older Ollama builds only expose /api/embeddings, one at a time.
                return self._embed_legacy(texts)
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if not vectors:
                return None
            return self._normalise(np.asarray(vectors, dtype=np.float32))
        except (requests.RequestException, ValueError, KeyError):
            return None

    def _embed_legacy(self, texts: list[str]) -> np.ndarray | None:
        """Fallback to the single-input /api/embeddings endpoint."""
        collected = []
        try:
            for text in texts:
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=config.LLM_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                collected.append(response.json()["embedding"])
        except (requests.RequestException, ValueError, KeyError):
            return None
        return self._normalise(np.asarray(collected, dtype=np.float32))

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise row-wise, guarding against zero vectors."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    @staticmethod
    def _lexical_vector(text: str) -> np.ndarray:
        """
        Deterministic bag-of-words hash vector -- the no-Ollama fallback.

        Not semantic: "cheap" will not match "affordable". But it reliably
        matches shared keywords, which for a query like "how much on food" is
        enough to retrieve the food summaries. A working keyword search beats a
        broken semantic one.
        """
        vector = np.zeros(LEXICAL_DIM, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            if len(token) < 3:          # skip 'a', 'of', 'in' -- pure noise
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % LEXICAL_DIM
            vector[bucket] += 1.0

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


# --------------------------------------------------------------------------- #
# Store interface
# --------------------------------------------------------------------------- #
class VectorStore(Protocol):
    """The contract both backends implement, so callers never branch on type."""

    backend: str

    def add(self, ids: list[str], texts: list[str],
            metadatas: list[dict[str, Any]]) -> None: ...

    def query(self, text: str, top_k: int = 5,
              where: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class NumpyVectorStore:
    """
    Brute-force cosine similarity over an in-memory matrix, persisted to disk.

    With roughly 200 documents a full scan takes well under a millisecond, so
    there is nothing to gain from an approximate index here.
    """

    backend = "numpy"

    def __init__(self, embedder: OllamaEmbedder, collection: str) -> None:
        self.embedder = embedder
        self.collection = collection
        self.path = config.VECTOR_STORE_PATH / f"{collection}_numpy.npz"

        self._vectors: np.ndarray = np.zeros((0, LEXICAL_DIM), dtype=np.float32)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._load()

    # -- persistence -------------------------------------------------------- #
    def _load(self) -> None:
        """Restore a previous index; silently start empty if it is unreadable."""
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                self._vectors = archive["vectors"]
                payload = json.loads(str(archive["payload"]))
            self._ids = payload["ids"]
            self._texts = payload["texts"]
            self._metadatas = payload["metadatas"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A corrupt or stale-format cache is not worth crashing over --
            # the caller will simply rebuild it.
            self.reset()

    def _save(self) -> None:
        try:
            payload = json.dumps({
                "ids": self._ids,
                "texts": self._texts,
                "metadatas": self._metadatas,
            })
            np.savez_compressed(self.path, vectors=self._vectors, payload=payload)
        except OSError:
            pass  # An unwritable cache degrades speed, not correctness.

    # -- VectorStore -------------------------------------------------------- #
    def add(self, ids: list[str], texts: list[str],
            metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        vectors = self.embedder.embed(texts)

        # Dimensionality changes when the embedding strategy changes between
        # runs (e.g. Ollama was down last time). Rebuild rather than mix.
        if self._vectors.size and self._vectors.shape[1] != vectors.shape[1]:
            self.reset()

        self._vectors = (
            vectors if not self._vectors.size
            else np.vstack([self._vectors, vectors])
        )
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._save()

    def query(self, text: str, top_k: int = 5,
              where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self._ids:
            return []

        query_vector = self.embedder.embed([text])[0]
        if query_vector.shape[0] != self._vectors.shape[1]:
            return []  # Index built with a different embedder; caller rebuilds.

        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = self._vectors @ query_vector

        candidates = range(len(self._ids))
        if where:
            candidates = [
                i for i in candidates
                if all(self._metadatas[i].get(k) == v for k, v in where.items())
            ]
            if not candidates:
                return []

        ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)
        return [
            {
                "id": self._ids[i],
                "text": self._texts[i],
                "metadata": self._metadatas[i],
                "score": round(float(scores[i]), 4),
            }
            for i in ranked[:top_k]
        ]

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._vectors = np.zeros((0, LEXICAL_DIM), dtype=np.float32)
        self._ids, self._texts, self._metadatas = [], [], []
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


class ChromaVectorStore:
    """
    ChromaDB-backed store using our own embeddings.

    Chroma's default embedding function would pull ONNX runtime models on first
    use; passing vectors in explicitly keeps the embedding strategy consistent
    with the NumPy backend and avoids a surprise download mid-demo.
    """

    backend = "chromadb"

    def __init__(self, embedder: OllamaEmbedder, collection: str) -> None:
        import chromadb  # imported lazily -- may not be installed

        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_PATH))
        self._collection_name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], texts: list[str],
            metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        vectors = self.embedder.embed(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors.tolist(),
        )

    def query(self, text: str, top_k: int = 5,
              where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        query_vector = self.embedder.embed([text])[0].tolist()
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            where=where or None,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] or {},
                # Chroma returns cosine *distance*; convert for a consistent API.
                "score": round(1.0 - float(distances[i]), 4),
            }
            for i in range(len(documents))
        ]

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_vector_store(collection: str,
                     embedder: OllamaEmbedder | None = None) -> VectorStore:
    """
    Return the best available store for ``collection``.

    Tries ChromaDB and falls back to NumPy. The fallback covers both "not
    installed" (ImportError) and "installed but broken on this Python build",
    which is the more common failure on 3.13.
    """
    embedder = embedder or OllamaEmbedder()
    try:
        return ChromaVectorStore(embedder, collection)
    except Exception:  # noqa: BLE001 -- ImportError, RuntimeError, ONNX errors, ...
        return NumpyVectorStore(embedder, collection)


def describe_backend() -> dict[str, str]:
    """Diagnostics for the UI's status panel."""
    embedder = OllamaEmbedder()
    embedder._resolve()  # noqa: SLF001 -- deliberate: resolve for reporting

    try:
        import chromadb  # noqa: F401
        index_backend = "ChromaDB"
    except Exception:  # noqa: BLE001
        index_backend = "NumPy (fallback)"

    labels = {
        "embed_model": f"Ollama: {config.OLLAMA_EMBED_MODEL}",
        "chat_model": f"Ollama: {config.OLLAMA_CHAT_MODEL} (no embed model installed)",
        "lexical": "Lexical hashing (Ollama unavailable)",
    }
    return {
        "index": index_backend,
        "embeddings": labels.get(embedder.mode or "lexical", "unknown"),
        "mode": embedder.mode or "lexical",
    }
