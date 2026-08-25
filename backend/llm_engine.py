"""
backend/llm_engine.py
=====================
Thin, project-wide wrapper around the **Ollama** daemon.

Every module that needs LLM generation or embedding calls this file —
never the ``ollama`` library directly. That keeps the coupling in one
place: if the team later swaps Ollama for LM Studio or a cloud API,
only this file changes.

Public API
----------
- ``check_health()``       → ``bool``
- ``ensure_model(model)``  → ``None`` (raises on failure)
- ``generate(...)``        → ``str``
- ``generate_stream(...)`` → ``Iterator[str]``
- ``embed(text)``          → ``list[float]``
- ``embed_batch(texts)``   → ``list[list[float]]``
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

try:
    import ollama as _ollama_sdk
except ImportError:  # pragma: no cover
    _ollama_sdk = None  # type: ignore[assignment]

import config

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Custom exception
# ────────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _get_client() -> _ollama_sdk.Client:  # type: ignore[name-defined]
    """Return an ``ollama.Client`` pointed at the configured host."""
    if _ollama_sdk is None:
        raise LLMError(
            "The 'ollama' Python package is not installed. "
            "Run:  pip install ollama>=0.4.4"
        )
    return _ollama_sdk.Client(host=config.OLLAMA_HOST)


# ────────────────────────────────────────────────────────────────────
# Health & model verification
# ────────────────────────────────────────────────────────────────────

def check_health() -> bool:
    """
    Return ``True`` if the Ollama daemon is reachable, ``False`` otherwise.

    Uses a plain HTTP GET to ``/api/tags`` so it works even if the
    ``ollama`` SDK is not installed.
    """
    try:
        resp = requests.get(
            f"{config.OLLAMA_HOST}/api/tags",
            timeout=5,
        )
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_local_models() -> list[str]:
    """Return names of models already pulled on the Ollama host."""
    try:
        resp = requests.get(
            f"{config.OLLAMA_HOST}/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ensure_model(model: str | None = None) -> None:
    """
    Verify that *model* (default: the chat model from config) is available
    on the Ollama host. Raises ``LLMError`` with a human-friendly message
    if it is not.
    """
    model = model or config.OLLAMA_CHAT_MODEL
    if not check_health():
        raise LLMError(
            f"Ollama is not running at {config.OLLAMA_HOST}. "
            "Start it with:  ollama serve"
        )
    local = list_local_models()
    # Ollama may return fully-qualified names ("llama3.2:3b") or short
    # names ("llama3.2").  Accept a match on either prefix or full name.
    if not any(model == m or model == m.split(":")[0] for m in local):
        raise LLMError(
            f"Model '{model}' is not pulled. Run:  ollama pull {model}"
        )


# ────────────────────────────────────────────────────────────────────
# Text generation
# ────────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> str:
    """
    Single-shot (non-streaming) text generation.

    Parameters
    ----------
    prompt : str
        The user / instruction text.
    system : str
        Optional system prompt prepended to the conversation.
    model : str | None
        Ollama model name. Falls back to ``config.OLLAMA_CHAT_MODEL``.
    temperature : float | None
        Sampling temperature. Falls back to ``config.LLM_TEMPERATURE``.
    timeout : int | None
        Request timeout in seconds. Falls back to ``config.LLM_TIMEOUT_SECONDS``.

    Returns
    -------
    str
        The complete generated text.

    Raises
    ------
    LLMError
        On any connectivity or generation failure.
    """
    model = model or config.OLLAMA_CHAT_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _get_client()
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=False,
        )
        # The SDK returns a ChatResponse object; pull the text out safely.
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        return response.message.content  # type: ignore[union-attr]
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Generation failed ({model}): {exc}") from exc


def generate_stream(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> Iterator[str]:
    """
    Streaming text generation — yields tokens one at a time.

    Same parameters as :func:`generate`. The caller should iterate over
    the return value and concatenate the tokens.
    """
    model = model or config.OLLAMA_CHAT_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _get_client()
        stream = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            if isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = chunk.message.content  # type: ignore[union-attr]
            if token:
                yield token
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Streaming generation failed ({model}): {exc}") from exc


def generate_with_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> str:
    """
    Generate a response from a full message list (system + history + user).

    This is the multi-turn variant used by :mod:`rag_engine` where the
    conversation history is already assembled.
    """
    model = model or config.OLLAMA_CHAT_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS

    try:
        client = _get_client()
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=False,
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        return response.message.content  # type: ignore[union-attr]
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Generation failed ({model}): {exc}") from exc


def generate_stream_with_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> Iterator[str]:
    """
    Streaming variant of :func:`generate_with_messages`.

    Yields tokens one at a time for live UI rendering.
    """
    model = model or config.OLLAMA_CHAT_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

    try:
        client = _get_client()
        stream = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            if isinstance(chunk, dict):
                token = chunk.get("message", {}).get("content", "")
            else:
                token = chunk.message.content  # type: ignore[union-attr]
            if token:
                yield token
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Streaming generation failed ({model}): {exc}") from exc


# ────────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────────

def embed(text: str, *, model: str | None = None) -> list[float]:
    """
    Generate an embedding vector for a single piece of text.

    Parameters
    ----------
    text : str
        The text to embed. Leading/trailing whitespace is stripped.
    model : str | None
        Embedding model. Falls back to ``config.OLLAMA_EMBED_MODEL``.

    Returns
    -------
    list[float]
        A vector of length ``config.EMBEDDING_DIM`` (768 for nomic-embed-text).
    """
    model = model or config.OLLAMA_EMBED_MODEL
    text = text.strip()
    if not text:
        raise LLMError("Cannot embed an empty string.")

    try:
        client = _get_client()
        response = client.embed(model=model, input=text)
        # The SDK returns {"embeddings": [[...]]}
        if isinstance(response, dict):
            embeddings = response.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            raise LLMError("Ollama returned empty embeddings.")
        # Fallback for object-style response
        return response.embeddings[0]  # type: ignore[union-attr]
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Embedding failed ({model}): {exc}") from exc


def embed_batch(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Embed multiple texts efficiently.

    Sends texts in batches to the Ollama embed endpoint. The batch size
    is capped to avoid overwhelming the daemon on a laptop.

    Parameters
    ----------
    texts : list[str]
        Texts to embed. Empty strings are skipped (a zero-vector is
        returned in their position).
    model : str | None
        Embedding model. Falls back to ``config.OLLAMA_EMBED_MODEL``.
    batch_size : int
        Number of texts per request. Default 32.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text, in the same order.
    """
    model = model or config.OLLAMA_EMBED_MODEL
    results: list[list[float]] = []
    zero_vec = [0.0] * config.EMBEDDING_DIM

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Filter out empty strings, preserving indices.
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for j, t in enumerate(batch):
            stripped = t.strip()
            if stripped:
                non_empty_indices.append(j)
                non_empty_texts.append(stripped)

        if not non_empty_texts:
            results.extend([zero_vec] * len(batch))
            continue

        try:
            client = _get_client()
            response = client.embed(model=model, input=non_empty_texts)
            if isinstance(response, dict):
                embeddings = response.get("embeddings", [])
            else:
                embeddings = response.embeddings  # type: ignore[union-attr]
        except Exception as exc:
            raise LLMError(f"Batch embedding failed ({model}): {exc}") from exc

        # Reconstruct the full batch with zero vectors for skipped slots.
        batch_results: list[list[float]] = [zero_vec] * len(batch)
        for idx, emb_idx in enumerate(non_empty_indices):
            if idx < len(embeddings):
                batch_results[emb_idx] = embeddings[idx]
        results.extend(batch_results)

    return results
