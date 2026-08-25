"""
backend/llm_engine.py
=====================
The single LLM connection for the whole team.  **Owned by the Core Integrator.**

Every teammate's feature needs the model -- the Broke Alert, the Receipt
Splitter, the Quick Log parser, the chat assistant. If each of them opened their
own connection we would get four different timeout policies, four different
error messages, and four different ways to fail in front of a judge.

So they all call :func:`chat` or :func:`chat_json` and get:

* one provider probe shared across the app (Ollama, then LM Studio)
* one timeout and retry policy
* structured errors instead of raw tracebacks
* JSON-mode helpers, because three of the four creative features need the model
  to return parseable JSON rather than prose

Provider support
----------------
Ollama    -- native ``/api/chat``
LM Studio -- OpenAI-compatible ``/v1/chat/completions``

``LLM_PROVIDER=auto`` probes both, so a teammate running either one needs no
config change.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


class LLMUnavailableError(RuntimeError):
    """
    Raised when no local LLM runtime can be reached.

    Callers are expected to catch this and degrade -- show the rule-based
    insight, skip the AI summary -- rather than let the tab crash. The whole
    dashboard must stay usable with the model offline.
    """


# --------------------------------------------------------------------------- #
# Provider status
# --------------------------------------------------------------------------- #
@dataclass
class LLMStatus:
    """Snapshot of the runtime, rendered in the sidebar's status panel."""

    available: bool
    provider: str = "none"          # 'ollama' | 'lmstudio' | 'none'
    host: str = ""
    model: str = ""
    models: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def badge(self) -> str:
        """Short human label for the sidebar."""
        if not self.available:
            return "Offline"
        return f"{self.provider.title()} - {self.model}"


# Probing costs a network round trip and Streamlit reruns the script on every
# widget interaction, so the result is cached for a few seconds.
_STATUS_CACHE: dict[str, Any] = {"status": None, "checked_at": 0.0}
_STATUS_TTL_SECONDS = 15.0


def _probe_ollama(host: str, timeout: float = 3.0) -> list[str] | None:
    """Return Ollama's installed models, or ``None`` if it is not running."""
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def _probe_lmstudio(host: str, timeout: float = 3.0) -> list[str] | None:
    """Return LM Studio's loaded models, or ``None`` if it is not running."""
    try:
        response = requests.get(f"{host.rstrip('/')}/v1/models", timeout=timeout)
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def get_status(force_refresh: bool = False) -> LLMStatus:
    """
    Detect the active runtime, honouring ``config.LLM_PROVIDER``.

    Cached for a few seconds so a dashboard with a dozen widgets does not issue
    a dozen probes per rerun.
    """
    now = time.monotonic()
    cached = _STATUS_CACHE["status"]
    if (
        not force_refresh
        and cached is not None
        and now - _STATUS_CACHE["checked_at"] < _STATUS_TTL_SECONDS
    ):
        return cached

    preference = config.LLM_PROVIDER.lower()
    status = LLMStatus(available=False, detail="No local LLM runtime detected.")

    if preference in ("auto", "ollama"):
        models = _probe_ollama(config.OLLAMA_HOST)
        if models is not None:
            status = LLMStatus(
                available=True,
                provider="ollama",
                host=config.OLLAMA_HOST,
                model=_pick_model(models, config.OLLAMA_CHAT_MODEL),
                models=models,
                detail=f"{len(models)} model(s) installed.",
            )

    if not status.available and preference in ("auto", "lmstudio"):
        models = _probe_lmstudio(config.LMSTUDIO_HOST)
        if models is not None:
            status = LLMStatus(
                available=True,
                provider="lmstudio",
                host=config.LMSTUDIO_HOST,
                model=_pick_model(models, config.OLLAMA_CHAT_MODEL),
                models=models,
                detail=f"{len(models)} model(s) loaded.",
            )

    _STATUS_CACHE["status"] = status
    _STATUS_CACHE["checked_at"] = now
    return status


def _pick_model(available: list[str], preferred: str) -> str:
    """
    Choose a model, tolerating tag differences.

    A ``llama3.2`` pull is listed as ``llama3.2:latest``, so an exact-match-only
    check would fail on a machine that actually has the right model. Falls back
    to whatever is installed rather than erroring -- a demo with the wrong model
    beats a demo with no model.
    """
    if not available:
        return preferred
    if preferred in available:
        return preferred

    bare = preferred.split(":")[0]
    for name in available:
        if name.split(":")[0] == bare:
            return name

    # Prefer a general chat model over an embedding-only one.
    for name in available:
        if "embed" not in name.lower():
            return name
    return available[0]


def is_available() -> bool:
    """Cheap boolean for UI guards."""
    return get_status().available


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    timeout: int | None = None,
) -> str:
    """
    Send a conversation and return the reply text.

    ``messages`` uses the standard ``[{"role": ..., "content": ...}]`` shape,
    which both providers accept.

    Raises :class:`LLMUnavailableError` on any failure -- callers degrade.
    """
    status = get_status()
    if not status.available:
        raise LLMUnavailableError(
            "No local LLM is running.\n"
            "Start Ollama (`ollama serve`) or LM Studio's local server, then retry."
        )

    model = model or status.model
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    timeout = timeout or config.LLM_TIMEOUT_SECONDS

    try:
        if status.provider == "ollama":
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if json_mode:
                payload["format"] = "json"
            response = requests.post(
                f"{status.host}/api/chat", json=payload, timeout=timeout
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            f"{status.host}/v1/chat/completions", json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.Timeout as exc:
        raise LLMUnavailableError(
            f"The model took longer than {timeout}s to respond. "
            "A smaller model (llama3.2:3b) will be faster on a laptop."
        ) from exc
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise LLMUnavailableError(f"LLM request failed: {exc}") from exc


def chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """
    Yield reply tokens as they arrive, for ``st.write_stream``.

    Streaming matters for the demo: a 3B model takes several seconds to finish,
    and watching tokens appear reads as "working" where a frozen spinner reads
    as "broken".
    """
    status = get_status()
    if not status.available:
        raise LLMUnavailableError("No local LLM is running.")

    model = model or status.model
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    try:
        if status.provider == "ollama":
            url = f"{status.host}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            }
        else:
            url = f"{status.host}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }

        with requests.post(
            url, json=payload, stream=True, timeout=config.LLM_TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")

                if status.provider == "lmstudio":
                    # OpenAI-compatible SSE frames are prefixed with "data: ".
                    if not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line.strip() == "[DONE]":
                        break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if status.provider == "ollama":
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                else:
                    choices = chunk.get("choices") or []
                    if choices:
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            yield token

    except requests.Timeout as exc:
        raise LLMUnavailableError("The model timed out mid-response.") from exc
    except requests.RequestException as exc:
        raise LLMUnavailableError(f"Streaming failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# JSON helper -- shared by the Receipt Splitter and the Quick Log parser
# --------------------------------------------------------------------------- #
def chat_json(
    messages: list[dict[str, str]],
    model: str | None = None,
    retries: int = 1,
) -> dict[str, Any] | list[Any]:
    """
    Chat and parse the reply as JSON.

    Small models wrap JSON in prose or markdown fences even when told not to, so
    the response is salvaged before giving up, and retried once with a blunter
    instruction. Provided centrally because three teammates need exactly this:

    * Receipt Splitting  -- ``{item: price}``
    * Natural Language Quick Log -- ``{amount, category, merchant}``
    * Broke Alert -- structured warning fields

    Raises :class:`LLMUnavailableError` if nothing parseable comes back.
    """
    attempt = 0
    last_reply = ""
    while attempt <= retries:
        current = messages if attempt == 0 else messages + [{
            "role": "user",
            "content": "Return ONLY valid JSON. No explanation, no markdown fences.",
        }]

        last_reply = chat(current, model=model, json_mode=True, temperature=0.1)
        parsed = extract_json(last_reply)
        if parsed is not None:
            return parsed
        attempt += 1

    raise LLMUnavailableError(
        f"Model did not return valid JSON after {retries + 1} attempts. "
        f"Last reply: {last_reply[:200]}"
    )


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """
    Pull a JSON object or array out of a possibly chatty reply.

    Tried in order: the whole string, a ```json fenced block, then the widest
    brace/bracket span. Returns ``None`` if nothing parses.
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def ask(prompt: str, system: str | None = None, **kwargs: Any) -> str:
    """One-shot helper for teammates who do not need multi-turn history."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)
