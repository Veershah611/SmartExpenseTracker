"""
backend/integration.py
======================
The seam between the app shell and the four teammate modules.
**Owned by the Core Integrator.**

The problem this solves
-----------------------
Four people are writing four modules in parallel, and none of them will be
finished at the same time. Without a seam, ``app.py`` imports a module that does
not exist yet and the entire app dies on a missing file -- so the Integrator
cannot build or demo anything until everyone else is done. That is the failure
mode that kills hackathon projects on the last night.

So every teammate module is loaded through :func:`_load_feature`, which:

* imports it if present, and reports precisely which required functions are missing
* never raises -- a broken teammate module degrades one tab, not the app
* falls back to a reference implementation where one exists

This means the shell is demoable **today**, and each teammate's work lights up
the moment they push a conforming module.

The contracts are declared in :data:`FEATURE_CONTRACTS` and documented for the
team in ``INTEGRATION.md``. If a signature changes, change it there.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureContract:
    """What one teammate must deliver for their tab to light up."""

    key: str
    owner: str                       # role name, for the status panel
    module: str                      # dotted import path they must create
    required: tuple[str, ...]         # functions the shell calls
    headline: str                     # their creative feature
    fallback: str | None = None       # reference module used until theirs lands
    # Maps a contracted name to the differently-named function in the fallback
    # module. Without this a reference implementation would have to adopt the
    # teammate's naming, polluting a module another role owns.
    fallback_aliases: dict[str, str] = field(default_factory=dict)


FEATURE_CONTRACTS: tuple[FeatureContract, ...] = (
    FeatureContract(
        key="data",
        owner="Data & Database Engineer",
        module="backend.database",
        required=("get_students", "get_student", "get_expenses", "add_expense"),
        headline="Subscription Ghost Hunter",
        fallback="backend.database",
    ),
    FeatureContract(
        key="ghost_hunter",
        owner="Data & Database Engineer",
        module="backend.ghost_hunter",
        required=("find_recurring_charges",),
        headline="Subscription Ghost Hunter",
        fallback="backend.adapters",       # wraps analytics.detect_recurring()
    ),
    FeatureContract(
        key="analytics",
        owner="Analytics & Forecasting Developer",
        module="backend.analytics",
        required=("kpi_summary", "category_breakdown", "monthly_trend"),
        headline="Predictive Broke Alert",
        fallback="backend.analytics",
    ),
    FeatureContract(
        key="broke_alert",
        owner="Analytics & Forecasting Developer",
        module="backend.forecasting",
        required=("predict_broke_alert",),
        headline="Predictive Broke Alert",
        fallback="backend.adapters",
    ),
    FeatureContract(
        key="charts",
        owner="Analytics & Forecasting Developer",
        module="frontend.components.charts",
        required=("render_category_chart", "render_trend_chart"),
        headline="Plotly/Altair dashboard charts",
        fallback=None,                     # shell uses native Streamlit charts
    ),
    FeatureContract(
        key="ocr",
        owner="Vision & OCR Specialist",
        module="backend.ocr_engine",
        required=("extract_text", "process_receipt"),
        headline="Smart Receipt Splitting",
        fallback=None,
    ),
    FeatureContract(
        key="rag",
        owner="RAG & NLP Developer",
        module="backend.rag_engine",
        required=("answer_question",),
        headline="Conversational Assistant",
        fallback="backend.adapters",
    ),
    FeatureContract(
        key="quick_log",
        owner="RAG & NLP Developer",
        module="backend.rag_engine",
        required=("parse_quick_log",),
        headline="Natural Language Quick Log",
        fallback="backend.adapters",
    ),
)

CONTRACTS_BY_KEY = {contract.key: contract for contract in FEATURE_CONTRACTS}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class FeatureStatus:
    """Load result for one contract, rendered in the Team Status panel."""

    key: str
    owner: str
    headline: str
    module_name: str
    state: str                        # 'ready' | 'fallback' | 'missing' | 'error'
    missing: tuple[str, ...] = ()
    error: str = ""
    module: Any = None
    using_fallback: bool = False
    _functions: dict[str, Callable] = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        """True when the shell can call this feature at all (own module or fallback)."""
        return self.state in ("ready", "fallback")

    @property
    def icon(self) -> str:
        return {"ready": "OK", "fallback": "STUB",
                "missing": "--", "error": "FAIL"}[self.state]

    @property
    def label(self) -> str:
        """
        Sidebar wording.

        'adapted' and 'stub' are both the fallback state but mean very different
        things: adapted is the teammate's real code reached through a signature
        bridge, stub is a reference implementation standing in because their
        module has not arrived. Showing both as "stub" would misreport delivered
        work as missing.
        """
        if self.state == "fallback":
            return "adapted" if self.module_name.endswith("adapters") else "stub"
        return {"ready": "live", "missing": "pending", "error": "error"}[self.state]

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke a contracted function.

        Errors inside a teammate's code are wrapped so a bug in their module
        surfaces as a message in their tab rather than a white screen.
        """
        function = self._functions.get(name)
        if function is None:
            raise FeatureUnavailable(
                f"{self.owner} has not delivered {self.module_name}.{name}() yet."
            )
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- teammate code, trust nothing
            raise FeatureError(
                f"{self.module_name}.{name}() raised {type(exc).__name__}: {exc}"
            ) from exc


class FeatureUnavailable(RuntimeError):
    """A contracted module or function has not been delivered yet."""


class FeatureError(RuntimeError):
    """A delivered module raised while the shell was calling it."""


def _load_feature(contract: FeatureContract) -> FeatureStatus:
    """
    Import one contract, checking the required functions exist.

    Deliberately catches bare ``Exception``: a half-written teammate module can
    fail at import time in any number of ways (SyntaxError, a missing package,
    a typo at module scope) and none of them should stop the app from starting.
    """
    def _bind(module: Any, use_aliases: bool = False) -> tuple[dict[str, Callable],
                                                              tuple[str, ...]]:
        """Bind contracted names to callables, optionally via the alias map."""
        functions, missing = {}, []
        for name in contract.required:
            lookup = (
                contract.fallback_aliases.get(name, name) if use_aliases else name
            )
            attribute = getattr(module, lookup, None)
            if callable(attribute):
                # Keyed by the CONTRACTED name, so callers never learn that a
                # fallback was substituted.
                functions[name] = attribute
            else:
                missing.append(name)
        return functions, tuple(missing)

    # 1. The teammate's own module.
    try:
        module = importlib.import_module(contract.module)
        functions, missing = _bind(module)
        if not missing:
            return FeatureStatus(
                key=contract.key, owner=contract.owner, headline=contract.headline,
                module_name=contract.module, state="ready",
                module=module, _functions=functions,
            )
    except ImportError:
        missing = contract.required          # not written yet -- expected
    except Exception as exc:  # noqa: BLE001
        return FeatureStatus(
            key=contract.key, owner=contract.owner, headline=contract.headline,
            module_name=contract.module, state="error",
            missing=contract.required,
            error=f"{type(exc).__name__}: {exc}",
        )

    # 2. The reference implementation, if this contract has one.
    if contract.fallback and contract.fallback != contract.module:
        try:
            module = importlib.import_module(contract.fallback)
            functions, still_missing = _bind(module, use_aliases=True)
            if not still_missing:
                return FeatureStatus(
                    key=contract.key, owner=contract.owner,
                    headline=contract.headline, module_name=contract.fallback,
                    state="fallback", module=module, _functions=functions,
                    using_fallback=True, missing=missing,
                )
        except Exception:  # noqa: BLE001
            pass

    return FeatureStatus(
        key=contract.key, owner=contract.owner, headline=contract.headline,
        module_name=contract.module, state="missing", missing=missing,
    )


# Loaded once per process. Streamlit reruns the script constantly, so importing
# eight modules on every keystroke would be wasteful; `reload_features()` exists
# for the sidebar button a teammate hits after pushing their file.
_FEATURES: dict[str, FeatureStatus] | None = None


def get_features(force_reload: bool = False) -> dict[str, FeatureStatus]:
    """Load (once) and return every feature status, keyed by contract key."""
    global _FEATURES
    if _FEATURES is None or force_reload:
        _FEATURES = {c.key: _load_feature(c) for c in FEATURE_CONTRACTS}
    return _FEATURES


def reload_features() -> dict[str, FeatureStatus]:
    """
    Re-import every teammate module, picking up files added since startup.

    Backs the sidebar's "Reload team modules" button, so integrating a
    teammate's push does not mean restarting Streamlit mid-demo.
    """
    for contract in FEATURE_CONTRACTS:
        for name in {contract.module, contract.fallback or contract.module}:
            module = sys.modules.get(name)
            if module is not None:
                try:
                    importlib.reload(module)
                except Exception:  # noqa: BLE001
                    sys.modules.pop(name, None)
    return get_features(force_reload=True)


def feature(key: str) -> FeatureStatus:
    """Fetch one feature status by contract key."""
    features = get_features()
    if key not in features:
        raise KeyError(f"Unknown feature {key!r}. Known: {sorted(features)}")
    return features[key]


def integration_summary() -> dict[str, int]:
    """Counts for the sidebar header, e.g. '5/8 modules ready'."""
    features = get_features()
    return {
        "total": len(features),
        "ready": sum(1 for f in features.values() if f.state == "ready"),
        "fallback": sum(1 for f in features.values() if f.state == "fallback"),
        "missing": sum(1 for f in features.values() if f.state == "missing"),
        "error": sum(1 for f in features.values() if f.state == "error"),
    }


def describe_environment() -> dict[str, Any]:
    """
    Everything the sidebar diagnostics panel needs, gathered defensively.

    Every lookup here is wrapped: the diagnostics panel exists to explain
    breakage, so it must not become the thing that breaks.
    """
    info: dict[str, Any] = {}

    try:
        from backend import llm_engine
        status = llm_engine.get_status()
        info["llm"] = {
            "available": status.available,
            "badge": status.badge,
            "provider": status.provider,
            "model": status.model,
            "detail": status.detail,
        }
    except Exception as exc:  # noqa: BLE001
        info["llm"] = {"available": False, "badge": "Error",
                       "detail": f"{type(exc).__name__}: {exc}"}

    try:
        from backend import database
        info["database"] = {
            "exists": database.database_exists(),
            "path": str(__import__("config").DB_PATH),
        }
    except Exception as exc:  # noqa: BLE001
        info["database"] = {"exists": False, "path": "-",
                            "detail": f"{type(exc).__name__}: {exc}"}

    try:
        from backend import adapters
        info["vectors"] = adapters.describe_backend()
    except Exception as exc:  # noqa: BLE001
        info["vectors"] = {"index": "unavailable", "embeddings": "-",
                           "mode": f"{type(exc).__name__}"}

    return info


def format_traceback(exc: BaseException, limit: int = 3) -> str:
    """Short traceback for the expandable error box in a degraded tab."""
    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit)
    )
