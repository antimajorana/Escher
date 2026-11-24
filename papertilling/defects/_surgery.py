"""Cone-safe Volterra surgical primitives (`_Builder`, 360-deg dislocation cuts, and same-point superposition)."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..legacy import fusion_framework as ff

TWO_PI = 2.0 * math.pi
_DISLOC_CUT_DEGS = (0.0, 25.0, -25.0, 50.0, -50.0, 72.0, -72.0, 180.0, 155.0, -155.0, 130.0, -130.0, 108.0, -108.0)


def _norm(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _insertion_rule(L, n_verts, ts, dup, Y, b):
    """Bonds allowed by the four overlapping charts of an inserted half-strip."""
    rule = {}
    for chart in range(4):
        here = {}
        for i in n_verts:
            t, s = ts[i]
            if chart == 0:
                if s <= 1e-9:
                    here[L.locate(Y[i])] = (i, 0)
            elif chart == 1:
                if t < -1e-9 or (s > 1e-9 and t < 1.0 - 1e-9):
                    here[L.locate(Y[i])] = (i, 0)
            elif chart == 2:
                if s <= 1e-9 and t >= -1e-9:
                    here[L.locate(Y[i])] = (i, 0)
                elif i in dup:
                    here[L.locate(Y[i])] = (dup[i], 0)
                elif s > 1e-9 and t >= 1.0 - 1e-9:
                    here[L.locate(Y[i])] = (i, 0)
            elif s > 1e-9:
                if t < 1.0 - 1e-9:
                    here[L.locate(Y[i])] = (i, 0)
                if i in dup:
                    k_sh = L.locate(Y[i] + b)
                    if k_sh is not None:
                        here[k_sh] = (dup[i], 1)
                elif t >= 1.0 - 1e-9:
                    k_sh = L.locate(Y[i] + b)
                    if k_sh is not None:
                        here[k_sh] = (i, 1)
        for img, (u, su) in here.items():
            if img is None:
                continue
            p_img = L.pos(img)
            for k2, kind in L.nbrs(img):
                hit = here.get(k2)
                if hit is not None and hit[0] != u:
                    v, sv = hit
                    rule[(u, v)] = (L.pos(k2) - p_img, su - sv, kind)
    return rule


class _Builder(ff._ConeSafeBuilder):
    """Cone-safe builder with symmetric 360-deg dislocation insertion and disclination/core-spur fixes."""

    _pad = None

    def add_dislocation(self, b, p, cut=None):
        b = np.asarray(b, float)
        p = np.asarray(p, float)
        lb = _norm(b)
        fb = self.Ainv @ b
        if lb < 1e-9 or abs(float(fb[0]) - round(float(fb[0]))) > 1e-5 or abs(float(fb[1]) - round(float(fb[1]))) > 1e-5:
            raise ff.ConfigError(f"b = {np.round(b, 4).tolist()} is not a non-zero lattice translation")
        c = np.array([-b[1], b[0]]) if cut is None else np.asarray(cut, float)
        lc = _norm(c)
        if lc < 1e-9:
            raise ValueError("cut direction must be non-zero")
        cr = float(c[0]) * float(b[1]) - float(c[1]) * float(b[0])
        if abs(cr) <= 1e-6 * lc * lb:
            raise ValueError("cut direction cannot be parallel to Burgers vector b")
        if cr < 0.0:
            return super().add_dislocation(b, p, c)
        self._add_dislocation_insertion(b, p, c, lb, lc)

    def _add_dislocation_insertion(self, b: np.ndarray, p: np.ndarray, c: np.ndarray, lb: float, lc: float):
        """Volterra half-strip insertion (`cross2(cut, b) > 0`) via 4 developed reference charts."""
        if self.clear(p) < lb + 2.5 * self.lmax:
            raise ValueError("dislocation point is too close to patch rim; increase R")
        ch = c / lc
        nh = np.array([-ch[1], ch[0]])
        bn, ba, m = float(b @ nh), float(b @ ch), 1.05 * self.lmax
        inside = lambda y: (-m <= float((y - p) @ nh) <= bn + m) and (float((y - p) @ ch) >= min(0.0, ba) - m)
        base = self.base_site(p, inside)
        D, Y, _ = self.develop(base, inside)
        n_verts = sorted(D)
        binv = np.linalg.inv(np.column_stack([b, c]))
        ts = {i: (binv @ (Y[i] - p)).tolist() for i in n_verts}
        dup, fresh = {}, set()
        for i in n_verts:
            t, s = ts[i]
            if s > 1e-9 and -1e-9 <= t < 1.0 - 1e-9:
                nv = self.new_copy(i)
                dup[i] = nv
                fresh.add(nv)
                D[nv] = D[i]
        dup_left = set(dup)
        kept = list(n_verts) + [dup[i] for i in n_verts if i in dup]
        rule = _insertion_rule(self.L, n_verts, ts, dup, Y, b)
        Q = self.local_rotation(base)
        xb, yb = np.array(self.X[base], float), np.array(Y[base], float)
        self._apply(rule, kept, D, lambda j: (np.eye(2), j * b), fresh)
        core = xb + Q @ (p + 0.5 * b - yb)
        bq, dq = Q @ b, Q @ ch
        for i in range(len(self.X)):
            if self.alive[i]:
                xi = self.X[i]
                if i in fresh:
                    frac = 0.0
                elif i in dup_left:
                    frac = 1.0
                else:
                    r = xi - core
                    frac = (math.atan2(float(dq[0] * r[1] - dq[1] * r[0]), float(r @ dq)) % TWO_PI) / TWO_PI
                self.X[i] = xi + (0.5 - frac) * bq
        self.touched |= set(kept)
        self.records.append(
            dict(
                kind="dislocation", b=b, at=p, cut=ch, removed=0, inserted=len(dup),
                frame_of=self.label[base], core_site=None, double_bonds=0, self_bonds=0,
            )
        )

    def add_disclination(self, angle, p, cut=None):
        """FIX 1: enlarge disconnected seam-band neighbourhood round face-centred disclinations."""
        p = np.asarray(p, float)
        snap = self.snapshot()
        super().add_disclination(angle, p, cut)
        if self.records[-1]["removed"] or self.records[-1]["inserted"]:
            return
        self.restore(snap)
        near = min(_norm(self.ref[i] - p) for i in range(len(self.ref)) if self.alive[i])
        self._pad = (p, near + 1.05 * self.lmax)
        try:
            super().add_disclination(angle, p, cut)
        finally:
            self._pad = None
        if not (self.records[-1]["removed"] or self.records[-1]["inserted"]):
            raise ValueError("the disclination step changed nothing; try another cut direction")

    def develop(self, base, inside):
        if self._pad is not None:
            q, rho = self._pad
            inner = inside
            inside = lambda y: inner(y) or _norm(y - q) <= rho
        return super().develop(base, inside)

    def prune_rim_spurs(self):
        """FIX 2: remove only 0-bond sites everywhere, then drop islands outside the largest component."""
        protected = set(self.cone_sites)
        for i in range(len(self.label)):
            if self.alive[i] and i not in protected and not self.nb[i]:
                self.alive[i] = False
        live = [i for i in range(len(self.label)) if self.alive[i]]
        if not live:
            raise ValueError("all crystal sites were removed")
        comp, seen = [], set()
        for s in live:
            if s in seen:
                continue
            c, dq = {s}, deque([s])
            seen.add(s)
            while dq:
                u = dq.popleft()
                for v in self.nb[u]:
                    if v not in seen:
                        seen.add(v)
                        c.add(v)
                        dq.append(v)
            comp.append(c)
        main = max(comp, key=len)
        for c in comp:
            if c is main:
                continue
            if c & protected:
                raise ValueError("a cut disconnected a disclination core from the crystal")
            for i in c:
                for v in list(self.nb[i]):
                    self.nb[v].pop(i, None)
                self.nb[i] = {}
                self.alive[i] = False


def candidate_cuts_360(d: dict, others: list[dict], disloc: bool, lmax: float | None = None) -> list[np.ndarray]:
    """Full-360-deg candidate dislocation cuts (removal + insertion) maximizing clearance from other cores."""
    if not disloc:
        return ff._candidate_cuts(d, others, False)
    b = np.asarray(d["b"], float)
    at = np.asarray(d["at"], float)
    lb = max(_norm(b), 1e-30)
    c0 = np.array([-b[1] / lb, b[0] / lb])
    cand = []
    for deg in _DISLOC_CUT_DEGS:
        r = math.radians(deg)
        cand.append(np.array([math.cos(r) * c0[0] - math.sin(r) * c0[1], math.sin(r) * c0[0] + math.cos(r) * c0[1]]))
    anchor = at + 0.5 * b
    other_pts = []
    for o in others:
        oa = np.asarray(o["at"], float)
        if o["kind"] == "dislocation":
            ob = np.asarray(o["b"], float)
            if _norm(oa + 0.5 * ob - anchor) > 1e-6:
                for t in (0.0, 0.5, 1.0):
                    other_pts.append(oa + t * ob)
        elif _norm(oa - anchor) > 1e-6:
            other_pts.append(oa)
    if not other_pts:
        return cand
    strip = [at + t * b for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    clr_cap = 2.2 * float(lmax) if lmax is not None and lmax > 0.0 else None

    def _score(u: np.ndarray) -> float:
        clr = min(ff._ray_distance(pt, sp, u) for pt in other_pts for sp in strip)
        eff_clr = min(clr, clr_cap) if clr_cap is not None else clr
        obliq = 1.0 - abs(float(u @ c0))
        away = min(-float((pt - anchor) @ u) / max(_norm(pt - anchor), 1e-9) for pt in other_pts)
        return eff_clr - 0.05 * obliq + 1e-4 * away

    return sorted(cand, key=_score, reverse=True)


def run_eff_with_fallback(make_builder, eff: list[dict], lat) -> object:
    """Apply `eff` in caller order; if that fails for 2..=4 defects, retry prioritized/reversed orders & bounded cut DFS."""
    def _cut_lists(ordered: list[dict]) -> list[list[np.ndarray]]:
        return [
            [np.asarray(d["cut"], float)] if d.get("cut") is not None
            else candidate_cuts_360(d, [o for o in ordered if o is not d], d["kind"] == "dislocation", lmax=lat.lmax)
            for d in ordered
        ]

    def _apply_one(B, d: dict, cut: np.ndarray) -> None:
        if d["kind"] == "dislocation":
            B.add_dislocation(d["b"], d["at"], cut)
        else:
            B.add_disclination(d["angle"], d["at"], cut)

    def _try_order(ordered: list[dict]):
        B = make_builder()
        cuts_per = _cut_lists(ordered)
        used = []
        first_err = None
        for idx, (d, cuts) in enumerate(zip(ordered, cuts_per), 1):
            err = None
            for cut in cuts:
                snap = B.snapshot()
                try:
                    _apply_one(B, d, cut)
                    used.append(np.asarray(cut, float))
                    err = None
                    break
                except ValueError as e:
                    B.restore(snap)
                    err = e
            if err is not None:
                first_err = ValueError(f"defect {idx} ({d['kind']}): {err}")
                break
        if first_err is None:
            for d, c in zip(ordered, used):
                d["cut_used"] = c
            return B, None
        # Bounded backtracking fallback (matching Rust surgery.rs)
        if 2 <= len(ordered) <= 4:
            B = make_builder()
            budget = [160]
            dfs_used: list[np.ndarray] = []

            def _dfs(pos: int) -> bool:
                if pos == len(ordered):
                    return True
                for cut in cuts_per[pos]:
                    if budget[0] <= 0:
                        return False
                    budget[0] -= 1
                    snap = B.snapshot()
                    try:
                        _apply_one(B, ordered[pos], cut)
                    except ValueError:
                        B.restore(snap)
                        continue
                    dfs_used.append(np.asarray(cut, float))
                    if _dfs(pos + 1):
                        return True
                    dfs_used.pop()
                    B.restore(snap)
                return False

            if _dfs(0):
                for d, c in zip(ordered, dfs_used):
                    d["cut_used"] = c
                return B, None
        return None, first_err

    B, caller_err = _try_order(eff)
    if caller_err is None:
        return B
    if 2 <= len(eff) <= 4:
        n = len(eff)
        def _prio(i: int) -> tuple[int, int]:
            d = eff[i]
            if d["kind"] == "disclination":
                return (0 if lat.L.locate(np.asarray(d["at"], float)) is not None else 1, i)
            return (2, i)
        orders = [sorted(range(n), key=_prio), list(reversed(range(n)))]
        seen = {tuple(range(n))}
        for ord_idx in orders:
            t = tuple(ord_idx)
            if t in seen:
                continue
            seen.add(t)
            B2, err2 = _try_order([eff[i] for i in ord_idx])
            if err2 is None:
                return B2
    raise caller_err from None


def _resolve_coincident_cuts(grp: list[dict]) -> np.ndarray | None:
    cuts = [np.asarray(g["cut"], float) for g in grp if g.get("cut") is not None]
    if not cuts:
        return None
    u0 = cuts[0] / max(_norm(cuts[0]), 1e-30)
    for c in cuts[1:]:
        uc = c / max(_norm(c), 1e-30)
        if float(u0 @ uc) < 1.0 - 1e-5:
            raise ValueError("coincident defects specify conflicting cut directions")
    return cuts[0]


def coalesce_coincident_specs(specs: list[dict]) -> list[dict]:
    """Superpose same-kind validated specs sharing the same position (`<= 1e-6`), dropping exact cancellations."""
    groups: list[list[dict]] = []
    for s in specs:
        at = np.asarray(s["at"], float)
        for grp in groups:
            if grp[0]["kind"] == s["kind"] and _norm(np.asarray(grp[0]["at"], float) - at) <= 1e-6:
                grp.append(s)
                break
        else:
            groups.append([s])
    out = []
    for grp in groups:
        if len(grp) == 1:
            out.append(dict(grp[0]))
            continue
        cut = _resolve_coincident_cuts(grp)
        kind = grp[0]["kind"]
        if kind == "disclination":
            ang_sum = float(sum(float(g["angle"]) for g in grp))
            if abs(ang_sum) <= 1e-9:
                continue
            if ang_sum >= 360.0 - 1e-6:
                raise ValueError(f"total Frank angle {ang_sum:+g} deg >= 360 deg: an open patch cannot close into a cone")
            if ang_sum <= -360.0 + 1e-6:
                raise ValueError(
                    f"coincident disclinations sum to {ang_sum:+g} deg <= -360 deg at a single apex; separate them to build a multi-turn sheet"
                )
            merged = dict(grp[0], angle=ang_sum)
            merged["cut"] = cut
            out.append(merged)
        else:
            b_sum = sum((np.asarray(g["b"], float) for g in grp), np.zeros(2))
            if _norm(b_sum) <= 1e-9:
                continue
            merged = dict(grp[0], b=b_sum)
            merged["cut"] = cut
            out.append(merged)
    total = sum(float(s["angle"]) for s in out if s["kind"] == "disclination")
    if total >= 360.0 - 1e-6:
        raise ValueError(f"total Frank angle {total:+g} deg >= 360 deg: an open patch cannot close into a cone")
    return out
