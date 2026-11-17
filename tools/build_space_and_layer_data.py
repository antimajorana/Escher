"""Build and verify `papertilling/data/space_groups_230.json` and `layer_groups_80.json` from spglib."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import spglib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from papertilling.bieberbach import SpaceGroupAnalyzer, fmt_abelian

DATA_DIR = ROOT / "papertilling" / "data"

CENT_TB: dict[str, list[list[list[int]]]] = {
    "P": [[[1, 1], [0, 1], [0, 1]], [[0, 1], [1, 1], [0, 1]], [[0, 1], [0, 1], [1, 1]]],
    "A": [[[1, 1], [0, 1], [0, 1]], [[0, 1], [1, 2], [1, 2]], [[0, 1], [-1, 2], [1, 2]]],
    "B": [[[1, 2], [0, 1], [1, 2]], [[0, 1], [1, 1], [0, 1]], [[-1, 2], [0, 1], [1, 2]]],
    "C": [[[1, 2], [1, 2], [0, 1]], [[-1, 2], [1, 2], [0, 1]], [[0, 1], [0, 1], [1, 1]]],
    "I": [[[-1, 2], [1, 2], [1, 2]], [[1, 2], [-1, 2], [1, 2]], [[1, 2], [1, 2], [-1, 2]]],
    "F": [[[0, 1], [1, 2], [1, 2]], [[1, 2], [0, 1], [1, 2]], [[1, 2], [1, 2], [0, 1]]],
    "R": [[[2, 3], [1, 3], [1, 3]], [[-1, 3], [1, 3], [1, 3]], [[-1, 3], [-2, 3], [1, 3]]],
}


def _frac_pair(x: float) -> list[int]:
    f = Fraction(float(x) % 1.0).limit_denominator(24)
    return [f.numerator, f.denominator]


def _crystal_system(num: int) -> str:
    if num <= 2:
        return "triclinic"
    if num <= 15:
        return "monoclinic"
    if num <= 74:
        return "orthorhombic"
    if num <= 142:
        return "tetragonal"
    if num <= 167:
        return "trigonal"
    if num <= 194:
        return "hexagonal"
    return "cubic"


def build_space_groups_230() -> dict:
    by_num = {}
    for hall in range(1, 531):
        st = spglib.get_spacegroup_type(hall)
        if st is not None and st.number not in by_num:
            by_num[st.number] = st
    assert len(by_num) == 230

    groups_out = []
    split_count = 0
    nonsym_count = 0

    for num in range(1, 231):
        st = by_num[num]
        cent = st.hall_symbol.lstrip("-")[0]
        tb_pairs = CENT_TB[cent]
        tb_frac = [[Fraction(n, d) for n, d in row] for row in tb_pairs]

        sym = spglib.get_symmetry_from_database(st.hall_number)
        ops_by_r = {}
        for R, t in zip(sym["rotations"], sym["translations"]):
            r_mat = [[int(x) for x in row] for row in R.tolist()]
            key = tuple(tuple(row) for row in r_mat)
            t_pairs = [_frac_pair(v) for v in t.tolist()]
            if key not in ops_by_r:
                ops_by_r[key] = {"r": r_mat, "t": t_pairs}

        ops = list(ops_by_r.values())
        gens = [
            (op["r"], [Fraction(n, d) for n, d in op["t"]])
            for op in ops
        ] + [
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [Fraction(row[0][0], row[0][1]), Fraction(row[1][0], row[1][1]), Fraction(row[2][0], row[2][1])])
            for row in tb_pairs
        ]
        sg = SpaceGroupAnalyzer(3, num, st.international_short, gens, tb_frac)
        assert sg.verify_2_cocycle(), f"d2(nu) != 0 in #{num} {st.international_short}"
        h2_t, is_split = sg.compute_cohomology_h2()
        ab_ft, ab_ff, _ = sg.compute_abelianization()
        if is_split:
            split_count += 1
        else:
            nonsym_count += 1
        if num == 123:
            assert h2_t == [2, 2, 2, 2] and is_split and fmt_abelian(ab_ft, ab_ff) == "Z_2^5"
        if num == 136:
            assert h2_t == [2, 2, 2, 2] and (not is_split) and fmt_abelian(ab_ft, ab_ff) == "Z_2^2 x Z_4"

        groups_out.append({
            "number": num,
            "hall_number": int(st.hall_number),
            "symbol": st.international_short,
            "full_symbol": st.international_full,
            "schoenflies": st.schoenflies,
            "pointgroup": st.pointgroup_international,
            "system": _crystal_system(num),
            "centring": cent,
            "tb_rows": tb_pairs,
            "point_order": len(ops),
            "is_split": is_split,
            "h2_label": fmt_abelian(h2_t),
            "ab_full_label": fmt_abelian(ab_ft, ab_ff),
            "ops": ops,
        })

    assert split_count == 73 and nonsym_count == 157, f"got {split_count} split + {nonsym_count} non-split"
    return {"count": 230, "split_count": split_count, "nonsymmorphic_count": nonsym_count, "groups": groups_out}


def build_layer_groups_80() -> dict:
    motif = [
        (1, np.array([0.113, 0.271, 0.061])),
        (2, np.array([0.149, 0.263, 0.057])),
        (3, np.array([0.108, 0.312, 0.064])),
        (4, np.array([0.117, 0.276, 0.089])),
    ]
    lats = [
        ("oblique", np.array([[1.0, 0.0, 0.0], [0.23, 1.31, 0.0], [0.0, 0.0, 6.0]])),
        ("rectangular", np.array([[1.0, 0.0, 0.0], [0.0, 1.37, 0.0], [0.0, 0.0, 6.0]])),
        ("square", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 6.0]])),
        ("hexagonal", np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 6.0]])),
    ]
    found: dict[int, dict] = {}
    for hall in range(1, 531):
        sym = spglib.get_symmetry_from_database(hall)
        keep = []
        for R, t in zip(sym["rotations"], sym["translations"]):
            if R[2, 0] == 0 and R[2, 1] == 0 and R[0, 2] == 0 and R[1, 2] == 0 and abs(t[2] - round(t[2])) < 1e-6:
                keep.append((R.copy(), np.array([t[0] % 1.0, t[1] % 1.0, 0.0])))
        if not keep:
            continue
        st = spglib.get_spacegroup_type(hall)
        n = st.number
        cands = lats[:2] if n <= 15 else ([lats[1]] if n <= 74 else ([lats[2]] if n <= 142 else ([lats[3]] if n <= 194 else [])))
        for sys_lbl, lat in cands:
            L = lat.T
            Linv = np.linalg.inv(L)
            if not all(np.allclose((L @ R @ Linv).T @ (L @ R @ Linv), np.eye(3), atol=1e-5) for R, _ in keep):
                continue
            pts, types = [], []
            for sp_t, seed in motif:
                for R, t in keep:
                    q = R @ seed + t
                    q[0] %= 1.0
                    q[1] %= 1.0
                    pts.append(q)
                    types.append(sp_t)
            lg = spglib.get_layergroup((lat, np.array(pts), types), aperiodic_dir=2, symprec=1e-4)
            if lg is not None and lg.number not in found and len(lg.rotations) == len(keep):
                centred = any(
                    np.array_equal(R, np.eye(3, dtype=int))
                    and abs((t[0] % 1.0) - 0.5) < 1e-4
                    and abs((t[1] % 1.0) - 0.5) < 1e-4
                    for R, t in keep
                )
                ops_by_r = {}
                for R, t in keep:
                    r_mat = [[int(x) for x in row] for row in R.tolist()]
                    key = tuple(tuple(row) for row in r_mat)
                    if key not in ops_by_r:
                        ops_by_r[key] = {"r": r_mat, "t2": [_frac_pair(t[0]), _frac_pair(t[1])]}
                ops = list(ops_by_r.values())
                found[lg.number] = {
                    "number": int(lg.number),
                    "hall_source": int(hall),
                    "symbol": str(lg.international),
                    "pointgroup": str(lg.pointgroup),
                    "system": sys_lbl,
                    "centred": centred,
                    "point_order": len(ops),
                    "ops": ops,
                }
        if len(found) == 80:
            break

    assert len(found) == 80, f"missing layer groups: {sorted(set(range(1, 81)) - set(found))}"
    return {"count": 80, "groups": [found[k] for k in range(1, 81)]}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sg230 = build_space_groups_230()
    lg80 = build_layer_groups_80()
    (DATA_DIR / "space_groups_230.json").write_text(json.dumps(sg230, separators=(",", ":")), encoding="utf-8")
    (DATA_DIR / "layer_groups_80.json").write_text(json.dumps(lg80, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote space_groups_230.json ({sg230['split_count']} split + {sg230['nonsymmorphic_count']} non-split)")
    print(f"Wrote layer_groups_80.json ({lg80['count']} layer groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
