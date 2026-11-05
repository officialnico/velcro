"""Attack on ``geometry.kind = "vmec"``: Velcro's tube against GX's own pyvmec module on decks E7a never tried.

GX_PYVMEC = GX's geometry_modules/pyvmec/gx_geo_vmec.py at the cards' commit 3865a53 (staged as an input), run here as
GX runs it (``python gx_geo_vmec.py DECK.in OUT.eik.nc``, toml from PyPI); VELCRO_WOUT_DIR = the wouts.  Every array of
GX's eik.nc against :func:`velcro.vmec_tube.vmec_tube_eik` of the same deck, E7a's bands (1e-4 of each array's maximum,
1e-5 on the scalars).  Untested inputs: α ≠ 0, other s₀ (off VMEC's full grid, near the axis / edge), npol 2, the
'exact periodic' (gds21) and 'continuous drifts' (gbdrift0) cuts, x0 ≠ y0, a deck jtwist, which_crossing 0 / −2,
'linked' (no cut), the QH reactor-scale wout (ι < 0), W7-X off α = 0, ntheta 96.

GX_ZSHEAR_DIR (optional): GX's production zshear npz of test E parts 4–6 (bmag, gds22, shat, kx of the out.nc GX ran):
the α = π/2 QA, α = π/5 W7-X and QH tubes against the native tube.

Findings encoded here (T2_ATTACK.md §8): on decks off VMEC's full radial grid the codes' radial interpolation differs
(GX's combined-grid ι spline) — asserted as exactly an ŝ difference plus a small residual (``OFF_GRID``); the box length
x0 ∝ 1/fac is ill-conditioned where the cut's twist factor fac → 0 (W7-X α = π/5), so its band scales as 1/|fac|."""
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

WOUT_DIR = os.environ.get("VELCRO_WOUT_DIR", "/nonexistent")
GX = os.environ.get("GX_PYVMEC", "")
ZDIR = os.environ.get("GX_ZSHEAR_DIR", "")
QA = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QA_lowres.nc")
W7X = os.path.join(WOUT_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")
QH = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc")
CUT = {"fix aspect": "fix aspect", "exact periodic": "gds21", "continuous drifts": "gbdrift0", "linked": "none"}
ARR = ("bmag", "gradpar", "grho", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "cvdrift", "cvdrift0")


def _vt(path):
    try:
        import booz_xform  # noqa: F401
        import netCDF4  # noqa: F401
    except ImportError:
        pytest.skip("booz_xform / netCDF4 not installed")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available (VELCRO_WOUT_DIR)")
    from velcro import vmec_tube as vt
    if not hasattr(vt, "_attack_cache"):
        orig, cache = vt._splines, {}

        def cached(w, mb, nb):
            if (w, mb, nb) not in cache:
                cache[(w, mb, nb)] = orig(w, mb, nb)
            return cache[(w, mb, nb)]
        vt._splines, vt._attack_cache = cached, cache
    return vt


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _gx(tmp_path, wout, s0, alpha, npol, boundary, ntheta, y0, x0, jtwist, which):
    lines = ["[Dimensions]", f" ntheta = {ntheta}", "[Domain]", f" y0 = {y0}", f' boundary = "{boundary}"']
    if x0 is not None:
        lines.append(f" x0 = {x0}")
    if jtwist is not None:
        lines.append(f" jtwist = {jtwist}")
    lines += ["[Physics]", " beta = 0.0", "[Geometry]", ' geo_option = "vmec"', f' vmec_file = "{wout}"',
              f" alpha = {alpha}", f" npol = {npol}", f" desired_normalized_toroidal_flux = {s0}", " betaprim = 0.0"]
    if which is not None:
        lines.append(f" which_crossing = {which}")
    deck = tmp_path / "deck.in"
    deck.write_text("\n".join(lines) + "\n")
    out = tmp_path / "deck.eik.nc"
    r = subprocess.run([sys.executable, GX, str(deck), str(out)], capture_output=True, text=True, cwd=tmp_path,
                       timeout=1500)
    return r, out


CASES = {
    "qa_base": dict(wout=QA),
    "qa_a90": dict(wout=QA, alpha=1.5707963),
    "qa_a03": dict(wout=QA, alpha=0.3),
    "qa_s03": dict(wout=QA, s0=0.3),
    "qa_s0505": dict(wout=QA, s0=0.505),
    "qa_s005": dict(wout=QA, s0=0.05),
    "qa_s098": dict(wout=QA, s0=0.98),
    "qa_s0995": dict(wout=QA, s0=0.995),
    "qa_npol2": dict(wout=QA, npol=2),
    "qa_gds21": dict(wout=QA, boundary="exact periodic"),
    "qa_gbdrift0": dict(wout=QA, boundary="continuous drifts"),
    "qa_x0half": dict(wout=QA, x0=10.5),
    "qa_jt1": dict(wout=QA, jtwist=1),
    "qa_which0": dict(wout=QA, which=0),
    "qa_whichm2": dict(wout=QA, which=-2),
    "qa_linked": dict(wout=QA, boundary="linked"),
    "qh_base": dict(wout=QH),
    "qh_a45": dict(wout=QH, alpha=0.7853982),
    "qh_s03": dict(wout=QH, s0=0.3),
    "w7x_base": dict(wout=W7X),
    "w7x_a36": dict(wout=W7X, alpha=0.6283185),
    "w7x_linked": dict(wout=W7X, boundary="linked"),
    "w7x_s03": dict(wout=W7X, s0=0.3),
    "w7x_nz96": dict(wout=W7X, ntheta=96),
}

# Off VMEC's full radial grid the two codes interpolate radially in different ways: Velcro splines ι, p and the Boozer
# harmonics on VMEC's half grid, GX 3865a53 ι on the combined half + full grid, whose dι/ds zig-zags between nodes
# (test_attack_vmec_followup.py::test_gx_combined_iota_spline_oscillates; T2_ATTACK.md §8).  On those decks the
# shear-dependent difference must be exactly the ŝ difference — gds21, gbdrift0, cvdrift0 ∝ ŝ and gds22 ∝ ŝ² agree once
# rescaled by ŝ_GX/ŝ_Velcro — and everything else agrees to the residual of the harmonics' radial interpolation: band
# per deck (measured ≤ 6e-4 inside the grid, 1.1e-3 at s 0.995 beyond the last half-grid node).  s 0.5 is a full-grid
# node of the QA and W7-X wouts, not of the QH one.
OFF_GRID = {"qa_s03": 1e-3, "qa_s0505": 1e-3, "qa_s005": 1e-3, "qa_s098": 1e-3, "qa_s0995": 2.5e-3, "qh_base": 1e-3,
            "qh_a45": 1e-3, "qh_s03": 1e-3, "w7x_s03": 1e-3}
S_POWER = {"gds21": 1, "gbdrift0": 1, "cvdrift0": 1, "gds22": 2}


@pytest.mark.parametrize("case", list(CASES))
def test_velcro_tube_equals_gx_pyvmec(case, tmp_path):
    c = dict(s0=0.5, alpha=0.0, npol=1, boundary="fix aspect", ntheta=48, y0=21.0, x0=None, jtwist=None, which=None)
    c.update(CASES[case])
    vt = _vt(c["wout"])
    if not GX or not os.path.exists(GX):
        pytest.skip("GX_PYVMEC not staged")
    r, out = _gx(tmp_path, **c)
    gx_ok = r.returncode == 0 and out.exists()
    v_err = None
    try:
        e = vt.vmec_tube_eik(c["wout"], c["ntheta"], s0=c["s0"], alpha=c["alpha"], npol=c["npol"], cut=CUT[c["boundary"]],
                             x0_over_y0=(c["x0"] or c["y0"]) / c["y0"], jtwist=c["jtwist"],
                             which_crossing=-1 if c["which"] is None else c["which"])
    except Exception as ex:                                     # noqa: BLE001
        e, v_err = None, repr(ex)
    gxlog = [ln for ln in r.stdout.splitlines() if any(t in ln for t in ("iota =", "shat =", "d iota", "d pressure",
                                                                       "Final (unscaled)", "domain_scaling"))]
    print(f"\n[{case}] GX exit {r.returncode}; " + " | ".join(gxlog))
    if not gx_ok:
        print(f"[{case}] GX stderr tail: " + " / ".join(r.stderr.strip().splitlines()[-4:]))
        print(f"[{case}] Velcro: {'error ' + v_err if v_err else 'built, cut ' + str(e['theta_cut'])}")
        pytest.xfail(f"GX's own module fails on this deck: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")
    assert v_err is None, f"GX builds this tube, Velcro raises {v_err}"
    from netCDF4 import Dataset
    with Dataset(str(out)) as ds:
        ds.set_auto_mask(False)
        g = {k: np.asarray(ds.variables[k][:], float) for k in ARR + ("theta", "theta_PEST", "Rplot", "Zplot")}
        sc = {k: float(np.asarray(ds.variables[k][:]).reshape(-1)[0]) for k in ("shat", "q", "drhodpsi", "Rmaj",
                                                                                "kxfac", "scale")}
    errs = {k: _rel(e[k], g[k]) for k in ARR}
    errs["R"], errs["Z"], errs["theta"] = _rel(e["R"], g["Rplot"]), _rel(e["Z"], g["Zplot"]), _rel(e["theta"], g["theta"])
    vsc = dict(shat=e["shat"], q=e["q"], drhodpsi=e["drhodpsi"], Rmaj=e["rmaj"], kxfac=e["kxfac"], scale=e["domain_scaling"])
    serr = {k: abs(vsc[k] - sc[k]) / max(abs(sc[k]), 1e-300) for k in sc}
    cut_gx, cut_v = float(g["theta_PEST"][-1]), float(e["theta_b_range"])
    fac_gx = 2 * sc["shat"] * g["gds21"][0] / g["gds22"][0]
    print(f"[{case}] cut theta_B: GX {cut_gx:+.7f} Velcro {cut_v:+.7f}; twist fac GX {fac_gx:+.5f} Velcro "
          f"{2 * e['shat'] * e['gds21'][0] / e['gds22'][0]:+.5f}")
    print(f"[{case}] arrays (rel. to max): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    print(f"[{case}] scalars (rel): " + ", ".join(f"{k} {serr[k]:.1e} ({sc[k]:.7g})" for k in serr))
    if case in OFF_GRID:
        rs = sc["shat"] / vsc["shat"]
        res = {**errs, **{k: _rel(e[k] * rs ** p, g[k]) for k, p in S_POWER.items()},
               **{k: v for k, v in serr.items() if k != "shat"}}
        worst = max(res, key=res.get)
        print(f"[{case}] off the full grid: ŝ_GX/ŝ_Velcro − 1 = {rs - 1:+.2e}; with the ŝ arrays rescaled the largest "
              f"difference is {res[worst]:.1e} ({worst}), band {OFF_GRID[case]:.0e}")
        assert res[worst] < OFF_GRID[case], res
        return
    bad = {k: v for k, v in errs.items() if v > 1e-4}
    bad.update({k: v for k, v in serr.items() if v > 1e-5})
    assert not bad, f"Velcro differs from GX's pyvmec beyond E7a's bands: {bad}"


ZS = {"SP_p5_qa_a90_t2_s7_zshear.npz": (QA, 1.5707963), "SP_p5_w7x_a36_t2_s7_zshear.npz": (W7X, 0.6283185),
      "SP_p6_qh_t2_s7_zshear.npz": (QH, 0.0), "GXSSP_qa_t2_s7_zshear.npz": (QA, 0.0)}


@pytest.mark.parametrize("fname", list(ZS))
def test_production_gx_tubes_parts_4_to_6(fname):
    """The tubes GX actually ran in test E parts 4–6 (its out.nc Geometry as gx_zshear saved it, float32) against the native
    tube of the same deck (s 0.5, 'fix aspect', y0 = x0 = 21, ntheta 48)."""
    wout, alpha = ZS[fname]
    vt = _vt(wout)
    p = os.path.join(ZDIR, fname)
    if not os.path.exists(p):
        pytest.skip(f"{p} not staged")
    z = np.load(p)
    m = vt.vmec_tube_gx_arrays(wout, 48, s0=0.5, alpha=alpha, cut="fix aspect")
    kx = np.sort(np.asarray(z["kx"], float))
    x0_gx = 1.0 / np.min(np.diff(kx))
    fac = m["twist_shift_geo_fac"]
    jt = int(round(fac)) or 1
    x0_v = 21.0 * abs(jt) / abs(fac)
    e_b = _rel(m["bmag"], z["bmag"])
    sh_gx = float(np.asarray(z["shat"]).reshape(-1)[0]) if "shat" in z.files else m["shat"]
    off_grid = wout == QH                    # s 0.5 is not a full-grid node of the QH wout: the ι-spline difference
    rs = sh_gx / m["shat"] if off_grid else 1.0
    e_22 = _rel(m["gds22"] * rs ** 2, z["gds22"]) if "gds22" in z.files else float("nan")
    # x0 = y0 |jtwist| / |fac| with fac = 2ŝ gds21/gds22 at the tube's end, a ratio of end values: ill-conditioned as
    # fac → 0 (W7-X α = π/5: fac +0.058, where the arrays' 1e-5 agreement allows 2e-4 in x0) — band 1e-4 / min(1, |fac|)
    x0_band = 1e-4 / min(1.0, abs(fac))
    print(f"\n[{fname}] bmag {e_b:.1e}, gds22 {e_22:.1e}{' (ŝ-rescaled)' if off_grid else ''}, shat GX {sh_gx:+.7f} "
          f"Velcro {m['shat']:+.7f}, x0 GX {x0_gx:.5f} Velcro {x0_v:.5f} (rel {abs(x0_v - x0_gx) / x0_gx:.1e}, band "
          f"{x0_band:.1e}; fac {fac:+.5f}, jtwist {jt}); npz keys {sorted(z.files)[:8]}...")
    assert e_b < 1e-4 and not e_22 > 1e-4
    if off_grid:
        assert abs(rs - 1) < 1e-3
    else:
        assert abs(m["shat"] - sh_gx) < 1e-5 * max(abs(sh_gx), 1e-3)
    assert abs(x0_v - x0_gx) < x0_band * x0_gx
