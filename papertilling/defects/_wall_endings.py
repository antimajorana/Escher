"""Algebraic classification of wall endings via (sigma * G) ∩ E+(2).

A straight wall separating reference domain L from sigma(L) can end at a point in an
orientable sheet iff the coset sigma * G meets E+(2) (proper Euclidean isometries).
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def _clean_vec(v: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    out = np.asarray(v, float).copy()
    out[np.abs(out) < tol] = 0.0
    return out


def _clean_mat(M: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    out = np.asarray(M, float).copy()
    out[np.abs(out) < tol] = 0.0
    return out


def _group_parts(lattice_or_cat):
    """Extract (A, Ainv, ops, chiral) from DefectCatalogue or LatticeData."""
    if hasattr(lattice_or_cat, "grp"):
        g = lattice_or_cat.grp
        return g.A, g.Ainv, g.ops, bool(g.chiral)
    return (
        lattice_or_cat.A,
        lattice_or_cat.Ainv,
        lattice_or_cat.ops,
        not bool(lattice_or_cat.ops_minus),
    )


def _norm_angle_deg(R: np.ndarray) -> float:
    ang = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    if ang <= -180.0 + 1e-6:
        return 180.0
    if ang > 180.0:
        return ang - 360.0
    return ang


def _best_partial_dislocation(A: np.ndarray, Ainv: np.ndarray, M_sig: np.ndarray, proper_trans: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    best = None
    for R_h, t_h in proper_trans:
        n0 = np.round(-Ainv @ (M_sig.T @ t_h)).astype(int)
        for dm, dn in itertools.product(range(-4, 5), repeat=2):
            lam = A @ np.array([n0[0] + dm, n0[1] + dn], float)
            t_cand = _clean_vec(t_h + M_sig @ lam)
            b_cand = _clean_vec(-t_cand)
            bn = float(math.hypot(float(b_cand[0]), float(b_cand[1])))
            if bn < 1e-9:
                b_cand = np.zeros(2, float)
                t_cand = np.zeros(2, float)
                bn = 0.0
                ang = 0.0
            else:
                ang = math.atan2(float(b_cand[1]), float(b_cand[0])) % (2.0 * math.pi)
            score = (round(bn, 9), round(ang, 9))
            if best is None or score < best[0]:
                best = (score, b_cand, t_cand, bn)
    _, b_best, t_best, bn = best
    f = Ainv @ b_best
    is_full = bool(np.all(np.abs(f - np.round(f)) < 1e-5))
    b_list = [float(b_best[0]), float(b_best[1])]
    if bn < 1e-7:
        summary = "full-lattice trivial ending (b = [0.0, 0.0], sigma in G)"
    elif is_full:
        summary = f"full-lattice dislocation ending b = {np.round(b_best, 4).tolist()} (|b| = {bn:.4f})"
    else:
        summary = f"partial dislocation ending b = {np.round(b_best, 4).tolist()} (|b| = {bn:.4f}, fractional)"
    return {
        "can_end_orientable": True,
        "ending_kind": "partial_dislocation",
        "is_full_lattice_burgers": is_full,
        "burgers": b_list,
        "burgers_norm": float(bn),
        "frank_angle_deg": None,
        "centre": None,
        "holonomy_linear": [[1.0, 0.0], [0.0, 1.0]],
        "holonomy_trans": [float(t_best[0]), float(t_best[1])],
        "summary": summary,
    }


def _best_partial_disclination(A: np.ndarray, M_sig: np.ndarray, proper_rots: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    best = None
    I = np.eye(2, dtype=float)
    for R_h, t_h in proper_rots:
        alpha = _norm_angle_deg(R_h)
        I_R = I - R_h
        c0 = np.linalg.solve(I_R, t_h)
        L_mat = np.linalg.solve(I_R, M_sig @ A)
        n0 = np.round(-np.linalg.solve(L_mat, c0)).astype(int)
        for dm, dn in itertools.product(range(-4, 5), repeat=2):
            lam = A @ np.array([n0[0] + dm, n0[1] + dn], float)
            t_cand = _clean_vec(t_h + M_sig @ lam)
            c_cand = _clean_vec(np.linalg.solve(I_R, t_cand))
            cn = float(math.hypot(float(c_cand[0]), float(c_cand[1])))
            ang_c = 0.0 if cn < 1e-9 else (math.atan2(float(c_cand[1]), float(c_cand[0])) % (2.0 * math.pi))
            score = (round(abs(alpha), 7), 0 if alpha > 0.0 else 1, round(cn, 7), round(ang_c, 7))
            if best is None or score < best[0]:
                best = (score, alpha, _clean_mat(R_h), t_cand, c_cand)
    _, alpha_b, R_b, t_b, c_b = best
    alpha_val = float(round(alpha_b, 9))
    c_list = [float(c_b[0]), float(c_b[1])]
    return {
        "can_end_orientable": True,
        "ending_kind": "partial_disclination",
        "is_full_lattice_burgers": False,
        "burgers": None,
        "burgers_norm": None,
        "frank_angle_deg": alpha_val,
        "centre": c_list,
        "holonomy_linear": [[float(R_b[0, 0]), float(R_b[0, 1])], [float(R_b[1, 0]), float(R_b[1, 1])]],
        "holonomy_trans": [float(t_b[0]), float(t_b[1])],
        "summary": f"partial disclination ending alpha = {alpha_val:+.4g} deg about {np.round(c_b, 4).tolist()}",
    }


def classify_wall_ending(lattice_or_cat, sigma, wall_kind: str = "wall") -> dict:
    """Classify how a wall `sigma = (M, t)` can terminate in an orientable sheet."""
    if wall_kind in ("mobius_seam", "c", "Wc"):
        return {
            "can_end_orientable": False,
            "ending_kind": "non_orientable_twist",
            "is_full_lattice_burgers": False,
            "burgers": None,
            "burgers_norm": None,
            "frank_angle_deg": None,
            "centre": None,
            "holonomy_linear": None,
            "holonomy_trans": None,
            "summary": "non-orientable twist (Mobius seam cannot terminate inside an orientable sheet)",
        }

    A, Ainv, ops, chiral = _group_parts(lattice_or_cat)
    M_sig = np.asarray(sigma[0], float)
    t_sig = np.asarray(sigma[1], float)
    if float(np.linalg.det(M_sig)) < 0.0 and chiral:
        return {
            "can_end_orientable": False,
            "ending_kind": "chirality_obstruction",
            "is_full_lattice_burgers": False,
            "burgers": None,
            "burgers_norm": None,
            "frank_angle_deg": None,
            "centre": None,
            "holonomy_linear": None,
            "holonomy_trans": None,
            "summary": "chirality obstruction (orientation-reversing wall cannot end in an orientable chiral sheet)",
        }

    I = np.eye(2, dtype=float)
    proper_trans = []
    proper_rots = []
    for M_g, t_g in ops:
        R_h = M_sig @ M_g
        if float(np.linalg.det(R_h)) <= 0.0:
            continue
        t_h = M_sig @ t_g + t_sig
        if np.allclose(R_h, I, atol=1e-6):
            proper_trans.append((I, t_h))
        else:
            proper_rots.append((R_h, t_h))

    if proper_trans:
        return _best_partial_dislocation(A, Ainv, M_sig, proper_trans)
    if proper_rots:
        return _best_partial_disclination(A, M_sig, proper_rots)
    return {
        "can_end_orientable": False,
        "ending_kind": "chirality_obstruction",
        "is_full_lattice_burgers": False,
        "burgers": None,
        "burgers_norm": None,
        "frank_angle_deg": None,
        "centre": None,
        "holonomy_linear": None,
        "holonomy_trans": None,
        "summary": "no proper coset element in (sigma * G) ∩ E+(2)",
    }


def nearest_ending_holonomy(lattice_or_cat, sigma, at, wall_kind: str = "wall") -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Return `(ending_info, (R_h, t_h))` with `(R_h, t_h) in (sigma * G) ∩ E+(2)` centred nearest `at`."""
    ending = classify_wall_ending(lattice_or_cat, sigma, wall_kind=wall_kind)
    if not ending["can_end_orientable"]:
        raise ValueError(ending["summary"])
    R_h = np.asarray(ending["holonomy_linear"], float)
    t_h = np.asarray(ending["holonomy_trans"], float)
    if ending["ending_kind"] == "partial_dislocation":
        return ending, (R_h, t_h)

    A, _, _, _ = _group_parts(lattice_or_cat)
    M_sig = np.asarray(sigma[0], float)
    at_v = np.asarray(at, float)
    I_R = np.eye(2, dtype=float) - R_h
    c0 = np.linalg.solve(I_R, t_h)
    L_mat = np.linalg.solve(I_R, M_sig @ A)
    n0 = np.round(np.linalg.solve(L_mat, at_v - c0)).astype(int)
    best = None
    for dm, dn in itertools.product(range(-3, 4), repeat=2):
        lam = A @ np.array([n0[0] + dm, n0[1] + dn], float)
        t_cand = _clean_vec(t_h + M_sig @ lam)
        c_cand = _clean_vec(np.linalg.solve(I_R, t_cand))
        dist = float(math.hypot(float(c_cand[0] - at_v[0]), float(c_cand[1] - at_v[1])))
        if best is None or dist < best[0] - 1e-9:
            best = (dist, t_cand)
    return ending, (R_h, best[1])
