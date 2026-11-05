"""Quasisymmetry ⇒ no bounce-averaged radial drift: a physics check of vmec_tube's geodesic curvature (B×κ·∇ψ).

In a quasisymmetric field B = B(s, Mθ_B − Nφ_B) the radial magnetic drift of every trapped particle averages to zero
over its bounce orbit (Boozer, Phys. Fluids 26, 1288 (1983); Helander, Rep. Prog. Phys. 77, 087001 (2014) §5): along a
field line, (G ∂_θ − I ∂_φ) B ∝ dB/dℓ, so ∮ f(B) v_d·∇ψ dt is the integral of a total derivative.  The Landreman & Paul
QA and QH equilibria (PRL 128, 035001 (2022); the low-resolution wouts of simsopt's tests) are quasisymmetric to the
level of their Boozer spectrum's symmetry-breaking modes, ε_QS.  Velcro's tube computes the drift from its own formula
for κ_g (vmec_tube.boozer_fieldline: Bxk_psi = κ_g B², κ_g from the √g_B harmonics), so

    |∫ (1 − λB/2) (Bxk_psi/B²) dθ_B / (B √(1 − λB))| / ∫ |…| dθ_B / (B √(1 − λB))     (dℓ ∝ dθ_B/B in Boozer angles)

must be of order ε_QS for every trapped λ, while the binormal drift (Bxk_alpha, which drives the precession) keeps an
O(1) bounce average for most λ (it passes through zero at the precession reversal, so only its median is tested).
The field lines are generic (α = 0.37, 1.9): on α = 0 a stellarator-symmetric equilibrium makes B even and the radial
drift odd about the well's centre, so the bounce average vanishes by symmetry alone (1e-11 in QA, QH and W7-X alike,
job 2004) and tests nothing.  W7-X (quasi-isodynamic, β 5 %) is the contrast.  Needs booz_xform, netCDF4 and the wouts
(VELCRO_WOUT_DIR).
"""
import os

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import roots_legendre

WOUT_DIR = os.environ.get("VELCRO_WOUT_DIR", "/nonexistent")
QA = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QA_lowres.nc")
QH = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc")
W7X = os.path.join(WOUT_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")


def _vt(path):
    try:
        import booz_xform  # noqa: F401
        import netCDF4  # noqa: F401
    except ImportError:
        pytest.skip("booz_xform / netCDF4 not installed")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available (VELCRO_WOUT_DIR)")
    from velcro import vmec_tube as vt
    return vt


def _qs_error(vs, s):
    """ε_QS: the symmetry-breaking Boozer modes of |B| relative to the symmetric non-constant ones, the helicity taken
    from the largest non-constant mode (n/m = N/M)."""
    b = np.array([f(s) for f in vs.bmnc_b])
    xm, xn = vs.xm_b, vs.xn_b
    nonconst = ~((xm == 0) & (xn == 0))
    k = np.argmax(np.abs(b) * nonconst)
    ratio = xn[k] / xm[k] if xm[k] != 0 else np.inf
    sym = nonconst & (np.isclose(xn, ratio * xm) if np.isfinite(ratio) else (xm == 0))
    return float(np.sqrt(np.sum(b[nonconst & ~sym] ** 2)) / np.sqrt(np.sum(b[sym] ** 2))), float(ratio)


def _bounce_ratios(vt, path, s0=0.5, alpha=0.0, nlam=24):
    vs = vt._splines(path, None, None)
    th = np.linspace(-np.pi, np.pi, 8001)
    fl = vt.boozer_fieldline(vs, s0, alpha, th)
    B = CubicSpline(th, fl["modB"])
    kpsi = CubicSpline(th, fl["Bxk_psi"] / fl["modB"] ** 2)
    kal = CubicSpline(th, fl["Bxk_alpha"] / fl["modB"] ** 2)
    i0 = int(np.argmin(fl["modB"]))
    t0, Bmin = th[i0], fl["modB"][i0]
    # the well around the line's minimum, bounded inside the domain: λ with both bounce points found
    Bl, Br = fl["modB"][:i0 + 1].max(), fl["modB"][i0:].max()
    lam_lo = 1.0 / min(Bl, Br)
    lams = 1.0 / Bmin - (1.0 / Bmin - lam_lo) * (np.arange(1, nlam + 1) / (nlam + 1)) ** 1.5
    xg, wg = roots_legendre(64)
    out_psi, out_al = [], []
    for lam in lams:
        g = lambda t: B(t) - 1.0 / lam
        gv = fl["modB"] - 1.0 / lam                                  # < 0 at the minimum (λ < 1/B_min: trapped)
        j = i0
        while gv[j] < 0:                                             # walk left to the first sample above 1/λ
            j -= 1
        k = i0
        while gv[k] < 0:
            k += 1
        t1 = brentq(g, th[j], th[j + 1])
        t2 = brentq(g, th[k - 1], th[k])
        sg = 0.5 * np.pi * (xg + 1)                                  # θ = t1 + (t2 − t1)(1 − cos σ)/2, σ ∈ (0, π)
        tt = t1 + 0.5 * (t2 - t1) * (1 - np.cos(sg))
        jac = 0.5 * (t2 - t1) * np.sin(sg) * 0.5 * np.pi * wg
        Bt = B(tt)
        w = jac / (Bt * np.sqrt(np.maximum(1 - lam * Bt, 1e-300)))
        e = 1 - 0.5 * lam * Bt
        for kk, out in ((kpsi, out_psi), (kal, out_al)):
            f = e * kk(tt)
            out.append(abs(np.sum(w * f)) / np.sum(w * np.abs(f)))
    return np.array(out_psi), np.array(out_al), vs


@pytest.mark.parametrize("alpha", [0.37, 1.9])
@pytest.mark.parametrize("name,path", [("QA", QA), ("QH", QH)])
def test_quasisymmetric_fields_have_no_bounce_averaged_radial_drift(name, path, alpha):
    vt = _vt(path)
    r_psi, r_al, vs = _bounce_ratios(vt, path, alpha=alpha)
    eps_qs, hel = _qs_error(vs, 0.5)
    wx = _vt(W7X)
    w_psi, _, _ = _bounce_ratios(wx, W7X, alpha=alpha) if os.path.exists(W7X) else (np.array([np.nan]), None, None)
    print(f"{name} α {alpha}: ε_QS {eps_qs:.2e} (helicity n/m {hel:g}); bounce-averaged radial drift / its absolute "
          f"average: max {r_psi.max():.2e}, median {np.median(r_psi):.2e}; binormal median {np.median(r_al):.2f}; "
          f"W7-X on the same α: radial max {np.nanmax(w_psi):.2e} median {np.nanmedian(w_psi):.2e}")
    # "of order ε_QS": the median over λ is 1–2.3 ε_QS (job 2013); the largest relative residual sits at the most deeply
    # trapped λ, whose tiny wells make the absolute average small (QA: 20 ε_QS, QH: 1.9 ε_QS)
    assert np.median(r_psi) < 5 * eps_qs and r_psi.max() < 30 * eps_qs
    assert np.median(r_al) > 0.3                       # the precession does not average away
    if os.path.exists(W7X):
        assert np.median(w_psi) > 30 * np.median(r_psi)   # W7-X keeps a radial drift two orders larger on the same lines


def test_alpha0_is_symmetric_for_any_field():
    """The degenerate case: on α = 0 the stellarator symmetry alone cancels the bounce average around a minimum at θ = 0."""
    vt = _vt(QA)
    r_psi, _, _ = _bounce_ratios(vt, QA, alpha=0.0)
    assert r_psi.max() < 1e-8
