"""Sliders for the free parameters of a lattice (Notebook 1), with the group and the tiling re-checked live.

    from papertilling.widgets import explore
    box = explore(L)        # L: papertilling.lattice.Lattice

Each site-parameter slider can only reach values where the lattice stays the same planar lattice (no two
sites coincide, no bond through a site, no crossing, same rotation system = same tiling): its min/max are
tuning.allowed_interval, recomputed for every slider after every move, because the allowed interval of
one parameter depends on the others. The cell parameters b, gamma never break planarity (proved result,
tuning.py) and keep their full range. Under each slider a bar (plain HTML/CSS, no JavaScript) shows the
whole nominal range: forbidden parts hatched, the current value as a black mark, and a tick with the group
name at each special value where the symmetry group jumps up (tuning.special_points). A slider within half
a step of a special value is snapped onto it.

Every move also re-runs check_family: the group (legacy route and spglib), planarity, a bond through a site,
and whether the D-symbol key is still the entry's key (rule B); a change is shown in red, and on a special
value the status line names the group there. Returns the box: box.values (dict), box.lattice() (Lattice at
the slider values), box.limits (name -> (a, b) slider limits), box.special (name -> list of special
points), box.sliders, box.bars, box.status.
"""
from __future__ import annotations

import math
import warnings

LEGEND = ("<span style='font-size:11px;color:#555'>Bars: hatched = forbidden (bonds cross, a bond through a "
          "site, sites coincide or another tiling); black mark = current value; blue tick = special value where "
          "the group is larger (group named).</span>")
BAR_W = 400


def _ranges(fam):
    out = {}
    for name in fam.names:
        v = fam.values[name]
        if name == "b":
            out[name] = (0.4, 3.0, 0.01)
        elif name == "gamma":
            out[name] = (45.0, 135.0, 0.5)
        else:
            out[name] = (v - 0.5, v + 0.5, 0.002)
    return out


def _inward(a, b, x0, st):
    """Slider limits on the step grid (multiples of st), inside [a, b], always containing x0."""
    a2 = math.ceil(a / st - 1e-9) * st
    b2 = math.floor(b / st + 1e-9) * st
    a2 = min(round(a2, 10), x0)
    b2 = max(round(b2, 10), x0)
    return a2, b2


def bar_html(lo, hi, value, segments=(), special=(), width=BAR_W):
    """The range bar of one slider as an HTML string (inline CSS only)."""
    pc = lambda x: 100.0 * min(max((x - lo) / (hi - lo), 0.0), 1.0)
    parts = [f"<div style='position:relative;width:{width}px;height:30px;margin:0 0 4px 92px;font-size:10px;"
             "line-height:12px'>",
             "<div style='position:absolute;left:0;right:0;top:14px;height:9px;background:#dcedc8;"
             "border:1px solid #9e9e9e'></div>"]
    for s in segments:
        a, b = pc(s["lo"]), pc(s["hi"])
        parts.append(f"<div title='{s['reason']}' style='position:absolute;left:{a:.2f}%;width:{max(b - a, 0.4):.2f}%;"
                     "top:14px;height:11px;background:repeating-linear-gradient(45deg,#c62828 0,#c62828 2px,"
                     "#ffcdd2 2px,#ffcdd2 5px)'></div>")
    for d in special:
        x = pc(d["value"])
        parts.append(f"<div style='position:absolute;left:{x:.2f}%;top:10px;width:2px;height:17px;background:#1565c0'></div>"
                     f"<div style='position:absolute;left:{x:.2f}%;top:0;transform:translateX(-50%);color:#1565c0;"
                     f"white-space:nowrap'>{d['group']}</div>")
    parts.append(f"<div style='position:absolute;left:{pc(value):.2f}%;top:12px;width:3px;height:15px;background:#000;"
                 "transform:translateX(-1px)'></div>")
    parts.append(f"<div style='position:absolute;left:-40px;top:13px;color:#555'>{lo:.3g}</div>"
                 f"<div style='position:absolute;right:-40px;top:13px;color:#555'>{hi:.3g}</div></div>")
    return "".join(parts)


def _set_limits(s, a, b):
    """Set slider limits without ever having min > max (ipywidgets raises) and without moving the value."""
    if a > s.max:
        s.max = b; s.min = a
    else:
        s.min = a; s.max = b


def explore(lattice, n=2, figsize=(6.5, 6.5)):
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:                                   # pragma: no cover
        print("ipywidgets is not installed: use lattice.set(name=value) and lattice.plot() instead")
        return None
    import matplotlib.pyplot as plt
    from .lattice import check_family
    from .tuning import allowed_interval, special_points

    from .kernels import HAVE_NUMBA
    n_scan = 401 if HAVE_NUMBA else 101                   # grid of the scan (pure-Python kernels are ~40x slower)
    fam = lattice.family
    try:
        loci = fam.supergroup_loci()
    except Exception:                                     # e.g. a disconnected network: no special points
        loci = []
    ranges = _ranges(fam)
    sliders, bars = {}, {}
    for name, (lo, hi, st) in ranges.items():
        sliders[name] = W.FloatSlider(value=fam.values[name], min=lo, max=hi, step=st, description=name,
                                      continuous_update=False, readout_format=".3f",
                                      layout=W.Layout(width="520px"))
        bars[name] = W.HTML()
    status = W.HTML()
    out = W.Output()
    children = []
    for name in sliders:
        children += [sliders[name], bars[name]]
    box = W.VBox(children + ([W.HTML(LEGEND)] if sliders else []) + [status, out])
    box.values = dict(fam.values)
    box.lattice = lambda: lattice.set(**box.values)
    box.limits, box.special, box.scan = {}, {}, {}
    box.sliders, box.bars, box.status = sliders, bars, status
    busy = [False]

    def update_limits(vals):
        for name, s in sliders.items():
            lo, hi, st = ranges[name]
            r = allowed_interval(fam, name, lo, hi, n=n_scan, values=vals)
            sp = special_points(fam, name, lo, hi, values=vals, loci=loci)
            a, b = r["interval"] if r["current_ok"] else (lo, hi)
            a, b = _inward(a, b, vals[name], st) if r["edges"] != (None, None) else (a, b)
            _set_limits(s, a, b)
            box.limits[name], box.special[name], box.scan[name] = (s.min, s.max), sp, r
            bars[name].value = bar_html(lo, hi, vals[name], r["segments"], sp)
        return box.limits

    def redraw(vals):
        f = fam.set(**vals)
        try:
            st = check_family(f, key=lattice.key)
        except Exception as err:                          # legacy group route can fail within ~1e-6 of a locus
            status.value = f"<b style='color:#b00020'>check failed here ({type(err).__name__}: {err}); move the slider</b>"
            box.check = None
            return
        on = [(nm, d) for nm, lst in box.special.items() for d in lst if abs(d["value"] - vals[nm]) < 1e-9]
        bad = []
        if st["sites_coincide"]:
            bad.append("two sites coincide")
        elif not st["exact"]:
            where = f" at the special value {on[0][0]} = {on[0][1]['value']:.6g} ({', '.join(on[0][1]['equations'])})" if on else ""
            bad.append(f"group is now {st['group']}{where}: a lattice of a larger group (special locus)")
        if not st["planar"]:
            bad.append("bonds cross")
        if st["bond_through_site"]:
            bad.append("a bond passes through a site")
        if st["key"] is not None and st["same_key"] is False and st["planar"]:
            bad.append("different tiling (D-symbol changed)")
        colour = "#b00020" if bad else "#1b5e20"
        text = "; ".join(bad) if bad else f"same lattice: group {st['group']} (spglib {st['spglib']}), planar, same D-symbol"
        note = ""
        if lattice.meta.get("rule") == "points":           # points-only rule: the bonds should be the Voronoi neighbours
            from .points import is_voronoi
            box.voronoi = is_voronoi(f)
            note = ("<br><span style='color:#555'>the bonds are the Voronoi-neighbour bonds of these points</span>" if box.voronoi else
                    "<br><b style='color:#e65100'>the Voronoi neighbours have changed here: these bonds are no longer the "
                    "nearest-neighbour network of the points (by the points-only rule this is another network)</b>")
        status.value = f"<b style='color:{colour}'>{text}</b>{note}"
        box.check = st
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.plot(n=n, ax=ax, title=True)
                plt.show()
            plt.close(fig)

    def on_move(change=None):
        if busy[0]:
            return
        busy[0] = True
        try:
            vals = {nm: float(s.value) for nm, s in sliders.items()}
            if change is not None:                         # snap onto a special value within half a step
                nm = change["owner"].description
                for d in box.special.get(nm, []):
                    if abs(vals[nm] - d["value"]) <= 0.5 * ranges[nm][2] + 1e-12 and sliders[nm].min <= d["value"] <= sliders[nm].max:
                        vals[nm] = d["value"]; sliders[nm].value = d["value"]
                        break
            box.values = dict(vals)
            update_limits(vals)
            redraw(vals)
        finally:
            busy[0] = False

    if not sliders:
        redraw({})
        status.value = ("<b>no free parameter: this lattice is rigid inside its group (only the scale a)</b><br>"
                        + status.value)
    else:
        on_move()
        for s in sliders.values():
            s.observe(on_move, names="value")
    display(box)
    return box
