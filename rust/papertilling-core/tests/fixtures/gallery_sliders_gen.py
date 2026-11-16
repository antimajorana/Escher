"""Python reference for the Rust Gallery sliders: writes gallery_sliders.json next to this file.

Run from the repository root, with the pure-Python package (no Rust extension), e.g.
    PAPERTILLING_NO_RUST=1 python rust/papertilling-core/tests/fixtures/gallery_sliders_gen.py

Everything is computed by papertilling (tuning.allowed_interval, tuning.special_points, the slider logic of
widgets.explore: widgets._ranges, widgets._inward, the snap of on_move, lattice.check_family, groups.label_family).
Read by rust/papertilling-core/tests/gallery_sliders.rs and papertilling-app (gallery tests).

Cases: for every rule (points, sheet, bonds) and group, the entries 0, 1 and the last of MinimalResult.all, plus the
21 named lattices (sheet.named, recipe + frame). For each case: the slider limits, forbidden segments and special
values at the default values, the group check there, one coupled move (the first site parameter moved halfway to
its lower slider limit, as tests/test_widgets.py), and the group check at the first special value inside the
limits. "entries" lists, for EVERY entry of .all, the occupied Wyckoff letters and the symmetry index.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PAPERTILLING_NO_RUST", "1")

from papertilling import kernels as KN  # noqa: E402
from papertilling.groups import label_family  # noqa: E402
from papertilling.lattice import check_family, load_catalogue  # noqa: E402
from papertilling.sheet import SHEET, VARIANTS, named  # noqa: E402
from papertilling.tuning import allowed_interval, special_points  # noqa: E402
from papertilling.widgets import _inward, _ranges  # noqa: E402

N_SCAN = 401
R = lambda x: None if x is None else round(float(x), 9)  # noqa: E731


def _limits(fam, loci, vals):
    """widgets.explore update_limits, without ipywidgets."""
    out = {}
    for name, (lo, hi, st) in _ranges(fam).items():
        r = allowed_interval(fam, name, lo, hi, n=N_SCAN, values=vals)
        sp = special_points(fam, name, lo, hi, values=vals, loci=loci)
        a, b = r["interval"] if r["current_ok"] else (lo, hi)
        a, b = _inward(a, b, vals[name], st) if r["edges"] != (None, None) else (a, b)
        out[name] = dict(range=[lo, hi, st], limits=[R(a), R(b)], interval=[R(x) for x in r["interval"]],
                         edges=[R(e) for e in r["edges"]], current_ok=bool(r["current_ok"]),
                         segments=[[R(s["lo"]), R(s["hi"])] for s in r["segments"]],
                         special=[[R(d["value"]), d["group"], int(d["dim"])] for d in sp])
    return out


def _status(fam, key, vals):
    try:
        st = check_family(fam, key=key, **vals)
    except Exception as err:  # pragma: no cover - reported, not hidden
        return dict(error=f"{type(err).__name__}: {err}")
    return dict(group=st["group"], exact=bool(st["exact"]), planar=bool(st["planar"]),
                sites_coincide=bool(st["sites_coincide"]), bond_through_site=bool(st["bond_through_site"]),
                same_key=st["same_key"])


def _case(L, fam, tag):
    try:
        loci = fam.supergroup_loci()
    except Exception:
        loci = []
    vals = dict(fam.values)
    params = _limits(fam, loci, vals)
    case = dict(tag, recipe=fam.recipe(), names=list(fam.names), values={k: R(v) for k, v in vals.items()},
                params=params, status=_status(fam, L.key, {}))
    site = [nm for nm in fam.site_names]
    if site:                                   # coupled move, as test_widget_limits_depend_on_other_parameters
        nm = site[0]
        lo_lim = params[nm]["limits"][0]
        new = round(0.5 * (vals[nm] + lo_lim), 3)
        st = _ranges(fam)[nm][2]
        for v, *_ in params[nm]["special"]:    # on_move snap: within half a step and inside the limits
            if abs(new - v) <= 0.5 * st + 1e-12 and params[nm]["limits"][0] <= v <= params[nm]["limits"][1]:
                new = v
                break
        moved = dict(vals, **{nm: new})
        after = _limits(fam, loci, moved)
        case["move"] = dict(name=nm, value=R(new), limits={k: p["limits"] for k, p in after.items()},
                            special={k: p["special"] for k, p in after.items()}, status=_status(fam, L.key, moved))
    for nm in site:                            # group at the first special value inside the slider limits
        a, b = params[nm]["limits"]
        hit = [d for d in params[nm]["special"] if a <= d[0] <= b]
        if hit:
            v = hit[0][0]
            case["at_special"] = dict(name=nm, value=v, group=hit[0][1], status=_status(fam, L.key, {nm: v}))
            break
    return case


def main(out_path):
    t0 = time.time()
    cases, entries = [], []
    for rule in ("points", "sheet", "bonds"):
        for group, res in load_catalogue(rule=rule).items():
            n = len(res.all)
            for k, L in enumerate(res.all):
                f = L.family
                letters = sorted({d["letter"] for d in label_family(f)})
                entries.append([rule, group, k, "".join(letters), int(L.symmetry_index), list(f.names)])
            for k in sorted({0, min(1, n - 1), n - 1}):
                L = res.all[k]
                cases.append(_case(L, L.family, dict(rule=rule, group=group, index=k, named=None)))
            print(f"{rule} {group}: {n} entries, {time.time() - t0:.1f} s", flush=True)
    for _nm, key, _grp in SHEET + [(a, b, c) for a, b, c, _ in VARIANTS]:
        L = named(key)
        f = L.family
        Q, o0 = f.frame
        tag = dict(rule="named", group=f.group, index=None, named=key,
                   frame=dict(q=[[R(x) for x in row] for row in Q], origin=[R(x) for x in o0]))
        cases.append(_case(L, f, tag))
        print(f"named {key}: {time.time() - t0:.1f} s", flush=True)
    data = dict(note="generated by gallery_sliders_gen.py from the pure-Python papertilling (see its docstring)",
                numba=bool(KN.HAVE_NUMBA), n_scan=N_SCAN, cases=cases, entries=entries)
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(data, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {out_path}: {len(cases)} cases, {len(entries)} entries, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "gallery_sliders.json"))
