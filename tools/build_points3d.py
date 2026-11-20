"""Sweep of the 3D points rule over the 230 space groups (C1 WP-P1). Python only: no cargo, no lock.

    python -X utf8 tools/build_points3d.py [numbers...]   # default 1..230; a group whose shard exists is skipped
    python -X utf8 tools/build_points3d.py --merge        # only rebuild the merged file and the coverage report
    options: --limit S (per-group wall-clock budget, default 280 s, checked between Wyckoff choices)
             --out DIR (default papertilling/data), --force (recompute existing shards)

Shards `<out>/points3d/<nnn>.json` are the checkpoints (written atomically). After each group the coverage report
`<out>/points3d_coverage.json` is rewritten; when all 230 shards exist the merged `<out>/points3d.json` is written
too (also by --merge, flagged `partial` if groups are missing). Status: [COMPUTATIONAL EVIDENCE] (48 samples per
choice, seed 0; a phase missed by every sample is not found). The duplicate key is not an exact net identity (plan Q10).
"""
import json
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from papertilling.points3d import points3d_lattices                           # noqa: E402

N_GROUPS = 230
NOTE = ("Duplicate key = space group, sites per primitive cell, per-orbit coordination sequences to depth 12 "
        "(plan Q10): two non-isomorphic nets with equal sequences would be merged; collisions found are listed.")


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def shard_path(out, number):
    return os.path.join(out, "points3d", f"{number:03d}.json")


def read_shard(out, number):
    with open(shard_path(out, number), encoding="utf8") as f:
        return json.load(f)


def write_atomic(path, obj, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, default=_json_default, **kw)
    os.replace(tmp, path)


def coverage_row(d):
    """Per-group facts for the app's 'k of 7 (search stopped at tier t)'."""
    done = [t for t in d["tiers"] if t["status"] == "complete"]
    deepest = [done[-1]["k"], done[-1]["m"]] if done else None
    refused = {}
    for t in d["tiers"]:
        for why, c in t["counts"].items():
            if why.startswith("refused") or why in ("rare_phases", "same_lattice"):
                refused[why] = refused.get(why, 0) + c
    return dict(group=d["group"], symbol=d["symbol"], entries=len(d["entries"]), deepest_complete_tier=deepest,
                timed_out=any(t["status"] != "complete" for t in d["tiers"]), collisions=len(d["collisions"]),
                tiers=[[t["k"], t["m"], t["status"]] for t in d["tiers"]], counts=refused, seconds=d["seconds"])


def merge(out):
    have = [n for n in range(1, N_GROUPS + 1) if os.path.exists(shard_path(out, n))]
    groups = {str(n): read_shard(out, n) for n in have}
    cov = dict(note=NOTE, rule="plan T4/Q1: shells until connected; exact group by spglib; 48 samples, seed 0",
               status="COMPUTATIONAL EVIDENCE", partial=len(have) < N_GROUPS, n_done=len(have),
               total_seconds=round(sum(g["seconds"] for g in groups.values()), 1),
               entries_histogram={}, groups=[coverage_row(groups[str(n)]) for n in have])
    for r in cov["groups"]:
        h = cov["entries_histogram"]; h[str(r["entries"])] = h.get(str(r["entries"]), 0) + 1
    write_atomic(os.path.join(out, "points3d_coverage.json"), cov, indent=0)
    if not cov["partial"] or "--merge" in sys.argv:
        write_atomic(os.path.join(out, "points3d.json"), dict(note=NOTE, partial=cov["partial"], groups=groups))
    return cov


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = os.path.join(ROOT, "papertilling", "data")
    limit = 280.0
    for opt in ("--limit", "--out"):
        if opt in argv:
            i = argv.index(opt)
            v = argv[i + 1]
            del argv[i:i + 2]
            if opt == "--limit":
                limit = float(v)
            else:
                out = os.path.abspath(v)
    nums = [int(a) for a in argv if a.isdigit()] or list(range(1, N_GROUPS + 1))
    if "--merge" not in argv:
        for n in nums:
            if os.path.exists(shard_path(out, n)) and "--force" not in argv:
                print(f"# skip {n:3d} (shard exists)", flush=True)
                continue
            t = time.time()
            try:
                d = points3d_lattices(n, n=7, samples=48, seed=0, time_limit=limit)
            except Exception as e:                       # one bad group must not stop the sweep; it is retried on resume
                print(f"# ERROR {n:3d} {type(e).__name__}: {e}", flush=True)
                continue
            d["build_seconds"] = round(time.time() - t, 1)
            write_atomic(shard_path(out, n), d)
            r = coverage_row(d)
            print(f"# time {n:3d} {d['symbol']:10s} {d['build_seconds']:7.1f} s  {r['entries']} entries, deepest tier "
                  f"{r['deepest_complete_tier']}, timed_out {r['timed_out']}, collisions {r['collisions']}", flush=True)
            merge(out)
    cov = merge(out)
    print(f"# coverage: {cov['n_done']}/{N_GROUPS} groups, {cov['total_seconds']} s, histogram {cov['entries_histogram']}", flush=True)
