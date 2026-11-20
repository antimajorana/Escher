"""tools/build_points3d.py (C1 WP-P1): resumable shards, coverage report, merge.

Known answers, by hand: group 2 (P-1) with the inversion centre 1a is a Bravais lattice; its shells-until-connected
net has the bonds +-a, +-b, +-c (degree 6, the pcu net) and coordination sequence 4n^2+2 (points with
|x|+|y|+|z| = n in Z^3: 6, 18, 38, 66, ...), counted by hand for n = 1, 2. In P1 one or two sites per cell always
have an inversion centre (plan Q1 note), so the first P1 entry has 3 sites.
"""
import json
import os
import subprocess
import sys

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "build_points3d.py")


def run(out, *args):
    env = dict(os.environ, PAPERTILLING_NO_RUST="1")
    r = subprocess.run([sys.executable, "-X", "utf8", TOOL, *args, "--out", str(out)], capture_output=True, text=True,
                       env=env, timeout=250)
    assert r.returncode == 0, r.stderr
    return r.stdout


def load(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def test_sweep_is_resumable_and_reports_coverage(tmp_path):
    out1 = run(tmp_path, "2")
    shard = tmp_path / "points3d" / "002.json"
    d = load(shard)
    first = d["entries"][0]
    assert first["wyckoff"] == ["1a"] and first["sites"] == 1 and first["degrees"] == [6]
    assert first["cs"][0][:4] == [6, 18, 38, 66]
    assert "1/230" in out1
    m1 = shard.stat().st_mtime_ns
    assert "skip   2" in run(tmp_path, "2")                       # checkpoint honoured: nothing recomputed
    assert shard.stat().st_mtime_ns == m1
    cov = load(tmp_path / "points3d_coverage.json")
    row = cov["groups"][0]
    assert cov["partial"] and cov["n_done"] == 1 and row["group"] == 2 and row["symbol"] == "P-1"
    assert row["entries"] == len(d["entries"]) and row["timed_out"] is False and row["deepest_complete_tier"] is not None
    assert not list((tmp_path / "points3d").glob("*.tmp"))        # atomic writes leave no temp file
    assert not (tmp_path / "points3d.json").exists()              # partial: no merged file until asked
    run(tmp_path, "--merge")
    merged = load(tmp_path / "points3d.json")
    assert merged["partial"] and list(merged["groups"]) == ["2"]


def test_time_limit_is_reported_as_open_tier(tmp_path):
    run(tmp_path, "3", "--limit", "0.01")
    row = load(tmp_path / "points3d_coverage.json")["groups"][0]
    assert row["timed_out"] is True and any(t[2].startswith("OPEN") for t in row["tiers"])


def test_p1_needs_three_sites(tmp_path):
    run(tmp_path, "1")
    d = load(tmp_path / "points3d" / "001.json")
    assert all(e["sites"] >= 3 for e in d["entries"]) and d["entries"][0]["sites"] == 3
