"""Shared helpers for papertilling.noneuclidean2d (Rust bridge, small vector utils)."""
from __future__ import annotations

import numpy as np

from . import _rust

def _rs_fn(name: str):
    rs = _rust.get()
    return getattr(rs, name, None) if rs is not None else None


def _norm3(v):
    a = np.asarray(v, dtype=float)
    return (a / max(1e-15, float(np.linalg.norm(a)))).tolist()


def _stereo_safe(pts):
    arr = np.asarray(pts, dtype=float)
    cands = ([0.0, 0.0, 1.0], _norm3([0.23, 0.37, 0.90]), _norm3([0.41, -0.29, 0.86]))
    pole = next((np.array(p) for p in cands if np.all(arr @ p < 0.995)), np.array(cands[1]))
    e1 = np.array(_norm3(np.cross(pole, [1.0, 0.0, 0.0] if abs(pole[0]) < 0.9 else [0.0, 1.0, 0.0])))
    e2 = np.cross(pole, e1)
    den = np.maximum(1e-9, 1.0 - arr @ pole)
    return np.column_stack([(arr @ e1) / den, (arr @ e2) / den]).tolist()


def _check_p(z) -> float:
    if (r2 := z[0] * z[0] + z[1] * z[1]) >= 1.0:
        raise ValueError(f"point {z} outside Poincare disk")
    return r2


def _dihedral_canon(seq: list[int]) -> tuple[list[int], bool]:
    n, rev_s = len(seq), list(reversed(seq))
    cyc, rev = min(seq[r:] + seq[:r] for r in range(n)), min(rev_s[r:] + rev_s[:r] for r in range(n))
    return (cyc if cyc <= rev else rev), cyc != rev
