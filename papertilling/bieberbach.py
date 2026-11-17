"""General dimension-n Bieberbach space-group cohomology and defect sector engine.

Computes exact H^2(P, Z^n), splitness [nu] == 0, abelianizations (G^+)^ab and G^ab,
Master Affine-Centralizer defect sectors Q_R // Z_P(R), finite-torus conjugacy counts,
and unbased fusion channels. Delegates to `papertilling_rs` when available and enabled;
falls back to exact pure-Python Fraction + integer HNF/SNF arithmetic otherwise.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import itertools
import json
import math
from typing import Any

from . import _rust
from .groups import _G

__all__ = [
    "fmt_abelian",
    "SpaceGroupAnalyzer",
    "from_wallpaper_group",
    "analyze_wallpaper_group",
    "analyze_space_group",
]


def fmt_abelian(torsion: list[int], free_rank: int = 0) -> str:
    """Format invariant factors as e.g. '0', 'Z^2', 'Z_2^3', 'Z x Z_2^2', 'Z_2 x Z_4'."""
    parts: list[str] = []
    for d in sorted(set(d for d in torsion if d > 1)):
        c = sum(1 for x in torsion if x == d)
        parts.append(f"Z_{d}" if c == 1 else f"Z_{d}^{c}")
    if free_rank == 1:
        parts.append("Z")
    elif free_rank > 1:
        parts.append(f"Z^{free_rank}")
    return " x ".join(parts) if parts else "0"


def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return (-old_r, -old_s, -old_t) if old_r < 0 else (old_r, old_s, old_t)


def _hnf_compress_rows_block(rows: list[list[int]], ncols: int) -> list[list[int]]:
    if not rows or ncols == 0:
        return []
    a = [list(r) for r in rows]
    m = len(a)
    r_piv = 0
    for c in range(ncols):
        if r_piv >= m:
            break
        while True:
            best_r = None
            best_abs = None
            for r in range(r_piv, m):
                v = abs(a[r][c])
                if v > 0 and (best_abs is None or v < best_abs):
                    best_abs = v
                    best_r = r
            if best_r is None:
                break
            a[r_piv], a[best_r] = a[best_r], a[r_piv]
            if a[r_piv][c] < 0:
                a[r_piv] = [-x for x in a[r_piv]]
            piv = a[r_piv][c]
            changed = False
            for r in range(r_piv + 1, m):
                val = a[r][c]
                if val != 0:
                    q = val // piv
                    if q:
                        row_p = a[r_piv]
                        row_r = a[r]
                        for k in range(c, ncols):
                            if row_p[k]:
                                row_r[k] -= q * row_p[k]
                    if a[r][c] != 0:
                        changed = True
            if not changed:
                break
        if a[r_piv][c] != 0:
            piv = a[r_piv][c]
            row_p = a[r_piv]
            for r in range(r_piv):
                val = a[r][c]
                if val != 0:
                    q = val // piv
                    if q:
                        row_r = a[r]
                        for k in range(c, ncols):
                            if row_p[k]:
                                row_r[k] -= q * row_p[k]
            r_piv += 1
    return [a[r] for r in range(r_piv) if any(a[r])]


def _hnf_compress_rows(rows: list[list[int]], ncols: int) -> list[list[int]]:
    if len(rows) <= 512:
        return _hnf_compress_rows_block(rows, ncols)
    chunk = max(256, ncols)
    acc: list[list[int]] = []
    for i in range(0, len(rows), chunk):
        acc.extend(rows[i : i + chunk])
        acc = _hnf_compress_rows_block(acc, ncols)
    return acc


def _eye(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _snf(mat: list[list[int]], track_uv: bool = True) -> tuple[list[int], list[list[int]], list[list[int]]]:
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    a = [list(r) for r in mat]
    u = _eye(rows) if track_uv else []
    v = _eye(cols) if track_uv else []
    kmax = min(rows, cols)
    t = 0
    while t < kmax:
        best = None
        for r in range(t, rows):
            for c in range(t, cols):
                val = abs(a[r][c])
                if val > 0 and (best is None or val < best[2]):
                    best = (r, c, val)
        if best is None:
            break
        br, bc, _ = best
        a[t], a[br] = a[br], a[t]
        for r in range(rows):
            a[r][t], a[r][bc] = a[r][bc], a[r][t]
        if track_uv:
            u[t], u[br] = u[br], u[t]
            for r in range(rows if False else cols):
                v[r][t], v[r][bc] = v[r][bc], v[r][t]
        if a[t][t] < 0:
            a[t] = [-x for x in a[t]]
            if track_uv:
                u[t] = [-x for x in u[t]]
        while True:
            touched = False
            for r in range(t + 1, rows):
                val = a[r][t]
                if val == 0:
                    continue
                touched = True
                piv = a[t][t]
                if val % piv == 0:
                    q = val // piv
                    for j in range(cols):
                        a[r][j] -= q * a[t][j]
                    if track_uv:
                        for j in range(rows):
                            u[r][j] -= q * u[t][j]
                else:
                    g, p, q = _ext_gcd(piv, val)
                    ac, dc = -val // g, piv // g
                    for j in range(cols):
                        x, y = a[t][j], a[r][j]
                        a[t][j], a[r][j] = p * x + q * y, ac * x + dc * y
                    if track_uv:
                        for j in range(rows):
                            x, y = u[t][j], u[r][j]
                            u[t][j], u[r][j] = p * x + q * y, ac * x + dc * y
            for c in range(t + 1, cols):
                val = a[t][c]
                if val == 0:
                    continue
                touched = True
                piv = a[t][t]
                if val % piv == 0:
                    q = val // piv
                    for r_i in range(rows):
                        a[r_i][c] -= q * a[r_i][t]
                    if track_uv:
                        for r_i in range(cols):
                            v[r_i][c] -= q * v[r_i][t]
                else:
                    g, p, q = _ext_gcd(piv, val)
                    ac, dc = -val // g, piv // g
                    for r_i in range(rows):
                        x, y = a[r_i][t], a[r_i][c]
                        a[r_i][t], a[r_i][c] = p * x + q * y, ac * x + dc * y
                    if track_uv:
                        for r_i in range(cols):
                            x, y = v[r_i][t], v[r_i][c]
                            v[r_i][t], v[r_i][c] = p * x + q * y, ac * x + dc * y
            if not touched:
                piv = a[t][t]
                bad = next(
                    ((r, c) for r in range(t + 1, rows) for c in range(t + 1, cols) if a[r][c] % piv != 0),
                    None,
                )
                if bad is not None:
                    br_bad = bad[0]
                    for j in range(cols):
                        a[t][j] += a[br_bad][j]
                    if track_uv:
                        for j in range(rows):
                            u[t][j] += u[br_bad][j]
                    touched = True
            if a[t][t] < 0:
                a[t] = [-x for x in a[t]]
                if track_uv:
                    u[t] = [-x for x in u[t]]
            if not touched:
                break
        t += 1
    diag = [a[i][i] for i in range(kmax) if a[i][i] > 0]
    return diag, u, v


def _mat_vec(a: list[list[Any]], x: list[Any]) -> list[Any]:
    return [sum(r[j] * x[j] for j in range(len(x))) for r in a]


def _mat_mul(a: list[list[Any]], b: list[list[Any]]) -> list[list[Any]]:
    m, p, q = len(a), len(b[0]), len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(q)) for j in range(p)] for i in range(m)]


def _solve_int_linear(a: list[list[int]], b: list[int]) -> list[int] | None:
    cols = len(a[0]) if a else 0
    if cols == 0:
        return [] if all(v == 0 for v in b) else None
    diag, u, v = _snf(a, True)
    y = _mat_vec(u, b)
    r = len(diag)
    if any(yi != 0 for yi in y[r:]):
        return None
    z = [0] * cols
    for i in range(r):
        if y[i] % diag[i] != 0:
            return None
        z[i] = y[i] // diag[i]
    return _mat_vec(v, z)


def _rat_inv(mat: list[list[Fraction]]) -> list[list[Fraction]]:
    n = len(mat)
    aug = [list(mat[i]) + [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            raise ValueError("singular matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [aug[r][k] - fac * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def _int_inv_unimodular(u: list[list[int]]) -> list[list[int]]:
    res = _rat_inv([[Fraction(x) for x in r] for r in u])
    return [[int(x) for x in r] for r in res]


def _det(mat: list[list[int]]) -> int:
    n = len(mat)
    if n == 1:
        return mat[0][0]
    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    if n == 3:
        m = mat
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
    raise ValueError("only dim <= 3 supported")


def _to_frac(v: Any) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    return Fraction(float(v)).limit_denominator(24)


class SpaceGroupAnalyzer:
    """Dimension-n Bieberbach cohomology, abelianization, and defect sector analyzer."""

    def __init__(self, dim: int, number: int, symbol: str, gens_conv: list[tuple[Any, Any]], tb_rows: list[list[Any]]):
        self.dim = dim
        self.number = number
        self.symbol = symbol
        tb = [[_to_frac(x) for x in row] for row in tb_rows]
        self.b_mat = [[tb[j][i] for j in range(dim)] for i in range(dim)]
        self.b_inv = _rat_inv(self.b_mat)
        self.gens_prim = []
        for rc, tc in gens_conv:
            rc_rat = [[Fraction(int(x)) for x in row] for row in rc]
            rp_rat = _mat_mul(self.b_inv, _mat_mul(rc_rat, self.b_mat))
            rp = [[int(x) for x in row] for row in rp_rat]
            if any(x.denominator != 1 for row in rp_rat for x in row) or abs(_det(rp)) != 1:
                raise ValueError(f"{symbol}: invalid primitive rotation matrix")
            tp = _mat_vec(self.b_inv, [_to_frac(x) for x in tc])
            self.gens_prim.append((rp, tp))

        id_mat = _eye(dim)
        self.p_ops: list[list[list[int]]] = [id_mat]
        self.tau: list[list[Fraction]] = [[Fraction(0)] * dim]
        lookup = {tuple(tuple(r) for r in id_mat): 0}
        q = deque([0])
        while q:
            i1 = q.popleft()
            r1, t1 = self.p_ops[i1], self.tau[i1]
            for rg, tg in self.gens_prim:
                r2 = _mat_mul(r1, rg)
                k2 = tuple(tuple(r) for r in r2)
                t2 = [t1[c] + sum(r1[c][k] * tg[k] for k in range(dim)) for c in range(dim)]
                t2_red = [x - math.floor(x) for x in t2]
                if k2 not in lookup:
                    lookup[k2] = len(self.p_ops)
                    self.p_ops.append(r2)
                    self.tau.append(t2_red)
                    q.append(lookup[k2])
                elif self.tau[lookup[k2]] != t2_red:
                    raise ValueError(f"{symbol}: inconsistent translation remainder")

        p_len = len(self.p_ops)
        self.mul_table = [[lookup[tuple(tuple(r) for r in _mat_mul(self.p_ops[i], self.p_ops[j]))] for j in range(p_len)] for i in range(p_len)]
        self.inv_idx = [next(j for j in range(p_len) if self.mul_table[i][j] == 0) for i in range(p_len)]
        self.det_sign = [_det(r) for r in self.p_ops]
        self.nu = [
            [
                [
                    int(self.tau[i][c] + sum(self.p_ops[i][c][k] * self.tau[j][k] for k in range(dim)) - self.tau[self.mul_table[i][j]][c])
                    for c in range(dim)
                ]
                for j in range(p_len)
            ]
            for i in range(p_len)
        ]

    def verify_2_cocycle(self) -> bool:
        p_len, n = len(self.p_ops), self.dim
        for i in range(p_len):
            r1 = self.p_ops[i]
            for j in range(p_len):
                ij = self.mul_table[i][j]
                n12 = self.nu[i][j]
                for k in range(p_len):
                    jk = self.mul_table[j][k]
                    n23 = self.nu[j][k]
                    n12_3 = self.nu[ij][k]
                    n1_23 = self.nu[i][jk]
                    for r in range(n):
                        if sum(r1[r][c] * n23[c] for c in range(n)) - n12_3[r] + n1_23[r] - n12[r] != 0:
                            return False
        return True

    def compute_cohomology_h2(self) -> tuple[list[int], bool]:
        p_len, n = len(self.p_ops), self.dim
        if p_len <= 1:
            return [], True
        m1 = p_len - 1
        n1 = n * m1
        d1_rows: list[list[int]] = []
        aug_rows: list[list[int]] = []
        for i in range(1, p_len):
            i0, r1 = i - 1, self.p_ops[i]
            for j in range(1, p_len):
                j0 = j - 1
                ij = self.mul_table[i][j]
                val = self.nu[i][j]
                for p in range(n):
                    row = [0] * n1
                    row[i0 * n + p] += 1
                    for q in range(n):
                        row[j0 * n + q] += r1[p][q]
                    if ij != 0:
                        row[(ij - 1) * n + p] -= 1
                    d1_rows.append(row)
                    aug_rows.append(row + [val[p]])
        comp_d1 = _hnf_compress_rows(d1_rows, n1)
        torsion = [d for d in _snf(comp_d1, False)[0] if d > 1] if comp_d1 else []
        comp_aug = _hnf_compress_rows(aug_rows, n1 + 1)
        comp_a = [r[:n1] for r in comp_aug]
        comp_b = [r[n1] for r in comp_aug]
        is_split = _solve_int_linear(comp_a, comp_b) is not None
        return torsion, is_split

    def compute_abelianization(self, sub_indices: list[int] | None = None):
        if sub_indices is None:
            sub_indices = list(range(len(self.p_ops)))
        n = self.dim
        non_id = [i for i in sub_indices if i != 0]
        pos_map = {ri: idx for idx, ri in enumerate(non_id)}
        num_gens = n + len(non_id)
        rel_rows: list[list[int]] = []
        for ri in sub_indices:
            rm = self.p_ops[ri]
            for j in range(n):
                row = [(1 if i == j else 0) - rm[i][j] if i < n else 0 for i in range(num_gens)]
                if any(row):
                    rel_rows.append(row)
        for i1 in non_id:
            for i2 in non_id:
                i12 = self.mul_table[i1][i2]
                nu12 = self.nu[i1][i2]
                row = [-nu12[i] if i < n else 0 for i in range(num_gens)]
                row[n + pos_map[i1]] += 1
                row[n + pos_map[i2]] += 1
                if i12 != 0 and i12 in pos_map:
                    row[n + pos_map[i12]] -= 1
                if any(row):
                    rel_rows.append(row)
        if not rel_rows:
            return [], n, lambda b, r_idx: tuple(int(x) for x in b)
        h_row = _hnf_compress_rows(rel_rows, num_gens)
        h_col = [[h_row[r][c] for r in range(len(h_row))] for c in range(num_gens)]
        diag, u_mat, _ = _snf(h_col, True)
        r = len(diag)
        keep_slots = [(i, diag[i] if i < r else 0) for i in range(num_gens) if (diag[i] if i < r else 0) != 1]
        torsion = [d for _, d in keep_slots if d > 1]
        free_rk = sum(1 for _, d in keep_slots if d == 0)

        def chi(b: list[int], r_idx: int) -> tuple[int, ...]:
            v = [int(x) for x in b] + [0] * len(non_id)
            if r_idx != 0 and r_idx in pos_map:
                v[n + pos_map[r_idx]] = 1
            w = _mat_vec(u_mat, v)
            return tuple((w[i] % d) if d > 0 else w[i] for i, d in keep_slots)

        return torsion, free_rk, chi

    def count_torus_conjugacy_classes(self, n_scale: int) -> int:
        p_len, n = len(self.p_ops), self.dim
        visited = [False] * p_len
        reps = []
        for r_idx in range(p_len):
            if not visited[r_idx]:
                for s in range(p_len):
                    visited[self.mul_table[self.mul_table[s][r_idx]][self.inv_idx[s]]] = True
                reps.append(r_idx)
        total = 0
        for r_idx in reps:
            z_r = [s for s in range(p_len) if self.mul_table[s][r_idx] == self.mul_table[r_idx][s]]
            deltas = {s: [self.nu[s][r_idx][k] - self.nu[r_idx][s][k] for k in range(n)] for s in z_r}
            rm = self.p_ops[r_idx]
            m_col = [[(1 if i == j else 0) - rm[i][j] for j in range(n)] + [n_scale if i == j else 0 for j in range(n)] for i in range(n)]
            diag, u_mat, _ = _snf(m_col, True)
            u_inv = _int_inv_unimodular(u_mat)
            seen = set()
            for ck in itertools.product(*[range(d) for d in diag]):
                if ck in seen:
                    continue
                b = [sum(ck[i] * u_inv[row][i] for i in range(n) if diag[i] != 1) for row in range(n)]
                orb = {
                    tuple(_mat_vec(u_mat, [sum(self.p_ops[s][k][j] * b[j] for j in range(n)) + deltas[s][k] for k in range(n)])[i] % diag[i] for i in range(n))
                    for s in z_r
                }
                seen.update(orb)
                total += 1
        return total

    def summarize(self, torus_ns: tuple[int, ...] = (2, 3, 4, 6)) -> dict[str, Any]:
        h2_t, is_split = self.compute_cohomology_h2()
        p_plus = [i for i, d in enumerate(self.det_sign) if d == 1]
        ab_pt, ab_pf, _ = self.compute_abelianization(p_plus)
        ab_ft, ab_ff, _ = self.compute_abelianization(list(range(len(self.p_ops))))
        tor = [(n, self.count_torus_conjugacy_classes(n)) for n in torus_ns]
        return {
            "dim": self.dim,
            "number": self.number,
            "symbol": self.symbol,
            "point_order": len(self.p_ops),
            "h2_torsion": h2_t,
            "h2_label": fmt_abelian(h2_t),
            "is_split": is_split,
            "ab_plus_torsion": ab_pt,
            "ab_plus_free": ab_pf,
            "ab_plus_label": fmt_abelian(ab_pt, ab_pf),
            "ab_full_torsion": ab_ft,
            "ab_full_free": ab_ff,
            "ab_full_label": fmt_abelian(ab_ft, ab_ff),
            "torus_counts": tor,
        }


_WALLPAPER_ORDER = [
    "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
    "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m",
]


def from_wallpaper_group(group: str) -> SpaceGroupAnalyzer:
    G = _G(group)
    gens = [(op[0], op[1]) for op in G.ops] + [([[1, 0], [0, 1]], [1, 0]), ([[1, 0], [0, 1]], [0, 1])]
    tb = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(-1, 2), Fraction(1, 2)]] if G.centred else [[1, 0], [0, 1]]
    num = _WALLPAPER_ORDER.index(G.symbol) + 1 if G.symbol in _WALLPAPER_ORDER else 0
    return SpaceGroupAnalyzer(2, num, G.symbol, gens, tb)


def analyze_wallpaper_group(group: str) -> dict[str, Any]:
    """Return Bieberbach H^2, splitness, (G^+)^ab, G^ab, and torus conjugacy counts for a wallpaper group."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "bieberbach_wallpaper_json"):
        return json.loads(rs.bieberbach_wallpaper_json(group))
    return from_wallpaper_group(group).summarize()


def analyze_space_group(spec: dict[str, Any]) -> dict[str, Any]:
    """Analyze a dimension-n space group specification dict `(dim, number, symbol, ops, tb_rows)`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "bieberbach_analyze_json"):
        return json.loads(rs.bieberbach_analyze_json(json.dumps(spec)))
    sg = SpaceGroupAnalyzer(
        int(spec["dim"]),
        int(spec.get("number", 0)),
        str(spec.get("symbol", "custom")),
        [(op["r"], op["t"]) for op in spec["ops"]],
        spec["tb_rows"],
    )
    return sg.summarize()
