"""Optional bridge to the compiled Rust extension `papertilling_rs`.

The Python package stays pure-Python installable and works without a Rust toolchain (including in
standalone Google Colab notebooks produced by `tools/build_colab.py`). When `papertilling_rs` is
installed, callers retrieve it through `get()`, unless `PAPERTILLING_NO_RUST` is set to `1`, `true`,
or `yes` (checked at call time so tests can toggle it dynamically via `monkeypatch.setenv`).
"""
from __future__ import annotations

import os
from types import ModuleType

_UNSET = object()
_MOD: object = _UNSET


def _load() -> ModuleType | None:
    global _MOD
    if _MOD is _UNSET:
        try:
            import papertilling_rs as mod
        except ImportError:
            mod = None
        _MOD = mod
    return _MOD  # type: ignore[return-value]


def _disabled_by_env() -> bool:
    return os.environ.get("PAPERTILLING_NO_RUST", "").strip().lower() in ("1", "true", "yes")


def have_rust() -> bool:
    """Whether the compiled `papertilling_rs` extension can be imported."""
    return _load() is not None


def enabled() -> bool:
    """Whether `papertilling_rs` is importable and `PAPERTILLING_NO_RUST` is not set."""
    return not _disabled_by_env() and _load() is not None


def get() -> ModuleType | None:
    """Return the `papertilling_rs` module when enabled and available, else `None`."""
    if _disabled_by_env():
        return None
    return _load()
