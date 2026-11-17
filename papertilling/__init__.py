"""papertilling: planar periodic lattices of the 17 wallpaper groups and their defects.

    import papertilling as pt
    res = pt.load_catalogue()["p4g"]; res.show()        # the most Bravais-like lattices of a group (stored lists)
    res = pt.minimal_lattices("p6m", n=12)              # or search now; rule="points" (default), "sheet" or "bonds"
    L = pt.named("Kagomé"); L.plot()                    # the named lattices (pt.named_lattices(), pt.gallery())
    cat = pt.DefectCatalogue(L); cat.summary()          # dislocations, disclinations, walls of a lattice
    c = pt.build(L, dict(kind="dislocation", b=cat.dislocations[0]["b"], at=(0.37, 0.41)))   # put defects on it

The names below are imported on first use, so `import papertilling` stays fast.
"""
__version__ = "0.2.0"

_EXPORTS = {
    "Lattice": "lattice", "minimal_lattices": "lattice", "load_catalogue": "lattice", "check_family": "lattice",
    "points_lattices": "points", "voronoi_bonds": "points",
    "named": "sheet", "named_lattices": "sheet", "gallery": "sheet",
    "wyckoff_table": "groups", "wyckoff_markdown": "groups",
    "network_key": "dsymbol",
    "explore": "widgets",
    "spectrum": "spectrum",
    "DefectCatalogue": "defects.catalogue", "build": "defects.build",
    "equivalent_single": "defects.config", "compare": "defects.config", "classify": "defects.config",
    "space_group_extension": "defects.gauge", "wyckoff_rotocentres": "defects.gauge",
    "fusion_channels": "defects.gauge", "ribbon_faces": "defects.gauge",
    "plaquette_holonomies": "defects.gauge", "twisted_coboundary": "defects.gauge",
    "relax_elastic_2d": "defects.gsv_elastic", "eval_gsv_energy": "defects.gsv_elastic",
    "gsv_hessian_2d": "defects.gsv_elastic", "gsv_pairs": "defects.gsv_elastic",
}
__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
