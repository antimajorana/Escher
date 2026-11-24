"""Stage 3 defect fields: known answers (hand counts, closed forms), the Rust core against the Python
`gsv_elastic` / `burgers_circuit` reference code, and the same tests on the pure-Python fallback."""
import io
import math

import numpy as np
import pytest

from papertilling import _rust
from papertilling import defect_fields as dfm
from papertilling.defects.build import build
from papertilling.defects.circuit import burgers_circuit

S3 = math.sqrt(3.0)
# the Rust leg needs a papertilling_rs built with the Stage 3 bindings (an older build silently falls back to Python)
HAVE_S3_RUST = _rust.enabled() and hasattr(_rust.get(), "detect_defects_coords_json")


@pytest.fixture(params=["python", pytest.param("rust", marks=pytest.mark.skipif(
        not HAVE_S3_RUST, reason="papertilling_rs with Stage 3 bindings not available or PAPERTILLING_NO_RUST set"))])
def mode(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    assert (_rust.get() is not None) == (request.param == "rust")
    return request.param


def grid(a1, a2, basis, n):
    return np.array([np.add(b, i * np.asarray(a1) + j * np.asarray(a2)) for i in range(-n, n + 1)
                     for j in range(-n, n + 1) for b in basis], float)


def test_wigner_seitz_vacancy_and_interstitial(mode):
    ref = grid((1, 0), (0, 1), [(0, 0)], 3)
    o = int(np.argmin(np.linalg.norm(ref, axis=1)))
    ws = dfm.wigner_seitz_occupancy(ref, np.delete(ref, o, axis=0))
    assert ws["vacancies"] == [o] and ws["interstitial_hosts"] == []
    ws = dfm.wigner_seitz_occupancy(ref, np.vstack([ref, [0.3, 0.2]]))
    assert ws["vacancies"] == [] and ws["interstitial_hosts"] == [o] and ws["occupancy"][o] == 2


def test_psi_n_and_cna_hand_counts(mode):
    sq, tri = grid((1, 0), (0, 1), [(0, 0)], 4), grid((1, 0), (0.5, S3 / 2), [(0, 0)], 4)
    hc = grid((S3, 0), (S3 / 2, 1.5), [(0, 0), (0, 1)], 4)
    inner = lambda P, r: np.linalg.norm(P, axis=1) < r  # noqa: E731
    for P, n in [(sq, 4), (tri, 6), (hc, 3), (hc, 6)]:
        psi = dfm.psi_n_order(P, n, cutoff=1.1)
        assert np.allclose(np.abs(psi[inner(P, 3.0)]), 1.0, atol=1e-12)
    psi = dfm.psi_n_order([[0, 0], [1, 0], [5, 5]], 6, cutoff=1.1)
    assert psi[2] == 0
    # triangular bond: 2 common neighbours, not bonded to each other; square: none
    for P, want in [(tri, (2, 0, 0)), (sq, (0, 0, 0))]:
        cna = dfm.cna_signatures(P, cutoff=1.1)
        ok = inner(P, 2.0)
        sel = [s for (i, j), s in zip(cna["edges"], cna["signatures"]) if ok[i] and ok[j]]
        assert sel and all(tuple(s) == want for s in sel)


def test_strain_and_vitek_of_affine_maps(mode):
    ref = grid((1, 0), (0.5, S3 / 2), [(0, 0)], 2)
    g = 0.1
    cur = ref @ np.array([[1, 0], [g, 1]]) + [0.3, -0.2]          # x' = x + g y
    out = dfm.compute_nye_and_strain(cur, "triangular", ref_pos=ref, edges=None, cutoff=1.1, burgers=(1.0, 0.0))
    assert np.allclose(out["eps"], [[0, g / 2], [g / 2, 0]], atol=1e-12)
    assert np.allclose(out["green"], [[0, g / 2], [g / 2, g * g / 2]], atol=1e-12)
    v = out["vitek"]
    along = np.asarray(v["along_b"])
    assert np.allclose(np.asarray(v["wrapped"]), along - np.ceil(along - 0.5), atol=1e-12)
    two = dfm.compute_nye_and_strain([[0.2, 0.1], [1.9, 0.1]], "square", ref_pos=[[0, 0], [1, 0]], edges=[(0, 1)],
                                     burgers=(1.0, 0.0))
    assert abs(two["vitek"]["wrapped"][0] + 0.3) < 1e-12          # 0.7 |b| wraps to -0.3 |b|
    assert two["nye_total"] is None and "no bounded faces" in two["reason"]   # one bond encloses no area: no Nye field


def test_moduli_square_by_hand_and_python_parity(mode):
    from papertilling.defects.gsv_elastic import barnett_lothe_tensor, cauchy_born_tensor
    out = dfm.barnett_lothe_stroh("square", (1.0, 0.0))
    want = np.zeros((2, 2, 2, 2))
    want[0, 0, 0, 0] = want[1, 1, 1, 1] = 2.0
    want[0, 1, 0, 1] = want[0, 1, 1, 0] = want[1, 0, 0, 1] = want[1, 0, 1, 0] = 1.0
    assert np.allclose(out["C"], want, atol=1e-12) and np.allclose(out["B"], np.eye(2), atol=1e-12)
    assert abs(out["K"] - 1 / (2 * math.pi)) < 1e-12
    for name in ["square", "triangular", "honeycomb", "kagome"]:
        o = dfm.barnett_lothe_stroh(name, (1.0, 0.0))
        C = cauchy_born_tensor(name)
        assert np.allclose(o["C"], C, rtol=0, atol=1e-10), name
        assert np.allclose(o["B"], barnett_lothe_tensor(C), rtol=0, atol=1e-10), name
        assert o["elastic_class"] == "Isotropic", name                # hexagonal / this square model: isotropic


def test_stroh_field_jump_and_equilibrium(mode):
    rect = dict(a1=[1, 0], a2=[0, 1.5], basis=[[0, 0]], bonds=[[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, -1]])
    for lat in [rect, "honeycomb"]:
        for b in [(1.0, 0.0), (0.3, -0.8)]:
            o = dfm.barnett_lothe_stroh(lat, b, [[-0.8, 1e-9], [-0.8, -1e-9]])
            assert np.allclose(o["u"][0] - o["u"][1], b, atol=1e-6)    # +b counter-clockwise across the cut
            h, x = 1e-5, np.array([0.7, 0.4])
            pts = [x + [h, 0], x - [h, 0], x + [0, h], x - [0, h]]
            s = dfm.barnett_lothe_stroh(lat, b, pts)["stress"]
            div = (s[0][:, 0] - s[1][:, 0] + s[2][:, 1] - s[3][:, 1]) / (2 * h)
            assert np.all(np.abs(div) < 1e-5)
    assert dfm.barnett_lothe_stroh(rect, (1, 0))["elastic_class"] != "Isotropic"


def _dislocation(name, b):
    c = build(name, dict(kind="dislocation", b=b, at=(0.37, 0.41)), R=7.0)
    return c, np.array(c.defect.positions, float)              # keep nothing but the coordinates


@pytest.mark.parametrize("name,b,uv,weber", [("square", (1.0, 0.0), [1, 0], None),
                                             ("honeycomb", (S3, 0.0), [1, 0], "1/3[2 -1 -1 0]")])
def test_detect_dislocation_round_trip(mode, name, b, uv, weber):
    c, P = _dislocation(name, b)
    det = dfm.detect_defects_coords(P, name)
    ds = [d for d in det["defects"] if d["kind"] != "neutral"]
    assert len(ds) == 1 and ds[0]["kind"] == "dislocation"
    d = ds[0]
    assert d["uv"] == uv and d["weber"] == weber and np.allclose(d["burgers"], b, atol=1e-9)
    # parity with the Python Burgers circuit run on the same loop of the built crystal
    assert np.allclose(burgers_circuit(c, d["loop_path"])["closure_vector"], d["burgers"], atol=1e-9)
    nye = dfm.compute_nye_and_strain(P, name)
    assert nye["net_frank_deg"] == 0 and np.allclose(nye["nye_total"], b, atol=1e-9)
    assert np.allclose(nye["boundary_closure"], b, atol=1e-9)


def test_nye_sums_burgers_vectors_of_two_disconnected_crystals(mode):
    _, pos = _dislocation("square", (1.0, 0.0))
    pair = np.vstack([pos, pos + [30.0, 0.0]])
    nye = dfm.compute_nye_and_strain(pair, "square")
    assert nye["boundary_walk"] == []
    assert np.allclose(nye["boundary_closure"], [2.0, 0.0], atol=1e-9)
    assert np.allclose(nye["nye_total"], [2.0, 0.0], atol=1e-9)


def test_detect_separated_disclination_dipole(mode):
    c1 = np.array([S3 / 2, 0.5])                                       # hexagon centres of the honeycomb cell
    d_vec = np.array([2 * S3, 0.0])
    c = build("honeycomb", dict(kind="disclination", angle=60, at=tuple(c1)),
              dict(kind="disclination", angle=-60, at=tuple(c1 + d_vec)), R=8.0)
    P = np.array(c.defect.positions, float)
    det = dfm.detect_defects_coords(P, "honeycomb")
    charged = sorted(cl["frank_deg"] for cl in det["clusters"] if cl["frank_deg"] != 0)
    assert charged == [-60.0, 60.0]
    sizes = sorted(len(det["faces"][f]) for cl in det["clusters"] for f in cl["faces"])
    assert sizes == [5, 7]                                               # pentagon +60, heptagon -60
    ds = [d for d in det["defects"] if d["kind"] == "dipole"]
    assert len(ds) == 1 and ds[0]["frank_deg"] == 0
    bvec = np.asarray(ds[0]["burgers"])
    # translation part of R(+60, c1) R(-60, c1 + d) is +-(I - R(60)) d: |b| = 2 sin(30 deg) |d| = |d|, b.d = |d|^2 / 2
    assert abs(np.linalg.norm(bvec) - np.linalg.norm(d_vec)) < 1e-9 and abs(abs(bvec @ d_vec) - 6.0) < 1e-9
    # burgers_circuit writes the closure in the chart of the walk's first vertex, which is rotated by 60 deg inside the
    # wedge between the two cuts; the detector transports its frame from loop_path[0], so b is the closure of its own
    # walk and hence one of those basepoint values
    assert np.allclose(burgers_circuit(c, ds[0]["loop_path"])["closure_vector"], bvec, atol=1e-9)
    core = ds[0]["loop_path"][:-1]
    closures = [burgers_circuit(c, core[k:] + core[:k] + [core[k]])["closure_vector"] for k in range(len(core))]
    assert any(np.allclose(cv, bvec, atol=1e-9) for cv in closures)
    assert all(abs(np.linalg.norm(cv) - np.linalg.norm(d_vec)) < 1e-9 for cv in closures)


def test_lone_disclination_has_no_burgers_vector(mode):
    c = build("honeycomb", dict(kind="disclination", angle=60, at=(S3 / 2, 0.5)), R=6.0)
    det = dfm.detect_defects_coords(np.array(c.defect.positions, float), "honeycomb")
    (d,) = det["defects"]
    assert d["kind"] == "disclination" and d["frank_deg"] == 60 and d["burgers"] is None
    assert d["reason"].startswith("[PROVED]")
    assert dfm.compute_nye_and_strain(np.array(c.defect.positions, float), "honeycomb")["nye_total"] is None


def test_motion_gif(mode, tmp_path):
    from PIL import Image
    start = grid((1, 0), (0, 1), [(0, 0)], 2)
    frames = dfm.interpolate_crystal_frames(start, start + [[0.5, 0.0]] * len(start), 4)
    assert np.allclose(frames[2], start + [1 / 3, 0])
    edges = [(i, j) for i in range(len(start)) for j in range(i) if np.linalg.norm(start[i] - start[j]) < 1.1]
    data = dfm.export_motion_gif(frames, edges, tmp_path / "m.gif", width=64, height=48, delay_cs=5)
    assert data[:6] == b"GIF89a" and data[-1:] == b"\x3b" and b"NETSCAPE2.0" in data
    assert (tmp_path / "m.gif").read_bytes() == data
    im = Image.open(io.BytesIO(data))                                  # independent decoder
    assert (im.size, im.n_frames) == ((64, 48), 4)
    for k in range(4):
        im.seek(k)
        px = np.asarray(im.convert("RGB"))
        assert (px == [40, 40, 40]).all(-1).any() and (px == [200, 30, 30]).all(-1).any()


def test_screw_field_known_answers(mode):
    # K = mu I: Volterra screw u_z = b theta / (2 pi) (Hirth & Lothe 1982 ch. 3), grad = b (-y, x) / (2 pi r^2)
    mu, bz = 0.7, 1.3
    pts = [[1, 0], [0, 2], [-1, 1e-9], [-1, -1e-9], [0.3, -0.4]]
    o = dfm.screw_dislocation_field([[mu, 0], [0, mu]], bz, pts)
    assert np.allclose(o["uz"], [0, bz / 4, bz / 2, -bz / 2, bz * math.atan2(-0.4, 0.3) / (2 * math.pi)], atol=1e-8)
    g = bz / (2 * math.pi * 0.25) * np.array([0.4, 0.3])
    assert np.allclose(o["grad"][4], g, atol=1e-12) and np.allclose(o["stress"][4], mu * g, atol=1e-12)
    K = np.array([[2.0, 0.5], [0.5, 1.0]])                              # anisotropic: jump b, K_ab u_ab = 0
    u = lambda P: dfm.screw_dislocation_field(K, bz, np.atleast_2d(P))["uz"][0]  # noqa: E731
    assert abs(u([-0.8, 1e-9]) - u([-0.8, -1e-9]) - bz) < 1e-6
    p, h, E = np.array([0.6, 0.7]), 1e-3, np.eye(2)
    d2 = lambda a, b: (u(p + h * (a + b)) - u(p + h * (a - b)) - u(p - h * (a - b)) + u(p - h * (a + b))) / (4 * h * h)  # noqa: E731
    assert abs(sum(K[i, j] * d2(E[i], E[j]) for i in range(2) for j in range(2))) < 1e-5
    with pytest.raises(ValueError, match="REFUSED"):
        dfm.screw_dislocation_field([[1, 0], [0, -1]], 1.0, pts)


def test_voronoi_charge_and_fs_rh_sign(mode):
    tri, sq = grid((1, 0), (0.5, S3 / 2), [(0, 0)], 6), grid((1, 0), (0, 1), [(0, 0)], 4)
    for P, z in [(tri, 6), (sq, 4)]:                                     # hand counts: z = 6 / z = 4 inside
        v = dfm.voronoi_coordination(P, z_ideal=z)
        inner = np.flatnonzero(np.linalg.norm(P, axis=1) < 3.0)
        assert all(v["z"][i] == z and v["charge"][i] == 0 for i in inner) and any(v["rim"])
    # Volterra edge field (jump +b counter-clockwise) on the triangular crystal: a 5-fold (+1) and a 7-fold (-1) site
    # at the core, net charge 0; the detector closes to -b, i.e. the FS/RH Burgers vector (Hirth & Lothe) is b
    x0 = np.array([0.37, 0.41])
    P = tri + dfm.barnett_lothe_stroh("triangular", (1.0, 0.0), tri - x0)["u"]
    v = dfm.voronoi_coordination(P)
    core = sorted(q for q, d in zip(v["charge"], np.linalg.norm(tri - x0, axis=1)) if d < 4.0 and q)
    assert core == [-1, 1]
    (d,) = [d for d in dfm.detect_defects_coords(P, "triangular")["defects"] if d["kind"] == "dislocation"]
    assert np.allclose(d["burgers"], [-1, 0], atol=1e-9) and np.allclose(d["burgers_fs_rh"], [1, 0], atol=1e-9)


def test_diagnostics_are_stamped(mode):
    o = dfm.barnett_lothe_stroh("square", (1.0, 0.0))
    assert o["elastic_class_status"].startswith("[COMPUTATIONAL EVIDENCE]")
    c, P = _dislocation("square", (1.0, 0.0))
    for d in dfm.detect_defects_coords(P, "square")["defects"]:
        assert d["reason"].startswith(("[PROVED]", "REFUSED"))


def test_oriented_darts_of_a_square_annulus_and_two_islands():
    faces = [[v, v + 1, v + 5, v + 4] for v in [0, 1, 2, 4, 5, 6, 8, 9, 10]]
    left = {(u, v): i for i, f in enumerate(faces) for u, v in zip(f, f[1:] + f[:1])}

    def area(walk):
        return sum((u % 4) * (v // 4) - (v % 4) * (u // 4) for u, v in zip(walk, walk[1:])) / 2

    ring = dfm._boundaries(faces, left, set(range(9)) - {4})
    assert sorted(map(area, ring)) == [-1.0, 9.0]
    for selected in [{0, 8}, {0, 4}]:  # disjoint squares, then two squares sharing one vertex
        islands = dfm._boundaries(faces, left, selected)
        assert sorted(map(area, islands)) == [1.0, 1.0]
