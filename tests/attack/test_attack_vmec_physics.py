"""Attack on ``geometry.kind = "vmec"`` (velcro/vmec_tube.py): physics identities and conventions, independent of GX.

Card only (never on the Mac): ``VELCRO_WOUT_DIR=$IN`` with the QA, QH and W7-X wouts staged; ``pytest -s`` prints the
numbers.  What is checked:

* the field-line curvature from EXACT (analytic) θ-derivatives of the Boozer-reconstructed line r(θ) (no finite
  differences), against the Hegna–Nakajima form of vmec_tube (GX 3865a53: + β ∂_φ√g), the pre-2026 GX sign
  (− β ∂_φ√g), the form derived in REPORT_vmec.md (+ β (ι∂_θ√g + ∂_φ√g) = −β b·∇B/B²), no β term, and the
  force-balance metric form μ₀p'/B² + ∇ψ·∇B/(B|∇ψ|²);
* the discrimination of the existing finite-difference b·∇b test (what its bands accept);
* the dual relations / Jacobian convention (booz_xform's gmnc_b against the geometric Jacobian of the line);
* the default L_ref of from_vmec_tube (GX's VMEC 'Rmaj' is R in metres, not R/a);
* the ends of α ≠ 0 tubes (one-ended twist, bmag jump, spectral bgrad on a non-periodic bmag);
* Boozer resolution convergence; the half-grid ι and p splines against GX's combined half+full grids (3865a53).
"""
import os

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

WOUT_DIR = os.environ.get("VELCRO_WOUT_DIR", "/nonexistent")
QA = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QA_lowres.nc")
W7X = os.path.join(WOUT_DIR, "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc")
QH = os.path.join(WOUT_DIR, "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc")
NAME = {QA: "QA", W7X: "W7X", QH: "QH"}
MU0 = 4e-7 * np.pi


def _vt(path):
    try:
        import booz_xform  # noqa: F401
        import netCDF4  # noqa: F401
    except ImportError:
        pytest.skip("booz_xform / netCDF4 not installed")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available (VELCRO_WOUT_DIR)")
    from velcro import vmec_tube as vt
    if not hasattr(vt, "_attack_cache"):                 # one booz_xform run per (wout, mboz, nboz) in this session
        orig, cache = vt._splines, {}

        def cached(w, mb, nb):
            if (w, mb, nb) not in cache:
                cache[(w, mb, nb)] = orig(w, mb, nb)
            return cache[(w, mb, nb)]
        vt._splines, vt._attack_cache = cached, cache
    return vt


def _splines(vt, wout, mboz=None, nboz=None):
    return vt._splines(wout, mboz, nboz)


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _line(vt, vs, s, alpha, th):
    """vmec_tube's field line plus an independent reconstruction: partial derivatives r_s, r_θ, r_φ and the exact
    along-line derivatives r', r'' (d/dθ at fixed α: every harmonic is cos/sin((m − n/ι)θ + nα/ι))."""
    fl = vt.boozer_fieldline(vs, s, alpha, th)
    c = lambda spl, d=0: np.array([f.derivative()(s) if d else f(s) for f in spl])
    rmnc, drmnc = c(vs.rmnc_b), c(vs.rmnc_b, 1)
    zmns, dzmns = c(vs.zmns_b), c(vs.zmns_b, 1)
    numns, dnumns = c(vs.numns_b), c(vs.numns_b, 1)
    gmnc = c(vs.gmnc_b)
    bmnc, dbmnc = c(vs.bmnc_b), c(vs.bmnc_b, 1)
    iota = float(vs.iota(s))
    xm, xn = vs.xm_b, vs.xn_b
    ph = (th - alpha) / iota
    ang = xm[:, None] * th[None, :] - xn[:, None] * ph[None, :]
    C, S = np.cos(ang), np.sin(ang)
    m, n = xm[:, None], xn[:, None]
    k = m - n / iota
    R = rmnc @ C
    R_s, R_t, R_p = drmnc @ C, -(rmnc @ (m * S)), rmnc @ (n * S)
    Z_s, Z_t, Z_p = dzmns @ S, zmns @ (m * C), -(zmns @ (n * C))
    nu = numns @ S
    nu_s, nu_t, nu_p = dnumns @ S, numns @ (m * C), -(numns @ (n * C))
    R1, R2 = -(rmnc @ (k * S)), -(rmnc @ (k ** 2 * C))
    Z1, Z2 = zmns @ (k * C), -(zmns @ (k ** 2 * S))
    nu1, nu2 = numns @ (k * C), -(numns @ (k ** 2 * S))
    phc = ph - nu
    cp, sp = np.cos(phc), np.sin(phc)
    p1, p2 = 1.0 / iota - nu1, -nu2
    r1 = np.stack([R1 * cp - R * sp * p1, R1 * sp + R * cp * p1, Z1])
    r2 = np.stack([R2 * cp - 2 * R1 * sp * p1 - R * cp * p1 ** 2 - R * sp * p2,
                   R2 * sp + 2 * R1 * cp * p1 - R * sp * p1 ** 2 + R * cp * p2, Z2])
    cart = lambda Ra, Za, pa: np.stack([Ra * cp - R * sp * pa, Ra * sp + R * cp * pa, Za])
    r_s, r_t, r_p = cart(R_s, Z_s, -nu_s), cart(R_t, Z_t, -nu_t), cart(R_p, Z_p, 1.0 - nu_p)
    sqrtg = gmnc @ C
    out = dict(fl=fl, iota=iota, ph=ph, r1=r1, r2=r2, r_s=r_s, r_t=r_t, r_p=r_p, sqrtg=sqrtg,
               sqrtg_t=-(gmnc @ (m * S)), sqrtg_p=gmnc @ (n * S), modB=bmnc @ C, B_s=dbmnc @ C,
               B_t=-(bmnc @ (m * S)), B_p=bmnc @ (n * S), G=float(vs.Gfun(s)), I=float(vs.Ifun(s)),
               diota=float(vs.iota.derivative()(s)), dp_ds=float(vs.pressure.derivative()(s)))
    return out


def _curvature_table(vt, vs, s, alpha, nth=2001):
    th = np.linspace(-np.pi, np.pi, nth)
    L = _line(vt, vs, s, alpha, th)
    fl = L["fl"]
    psi_e, iota, modB = fl["psi_edge"], L["iota"], L["modB"]
    gpsi, galpha = fl["grad_psi"], fl["grad_alpha"]
    gpp = np.sum(gpsi ** 2, 0)
    gap = np.sum(galpha * gpsi, 0)
    L1 = gap / gpp
    zeta_c = -alpha / iota
    L0 = -(gap / gpp + L["diota"] * (L["ph"] - zeta_c) / psi_e)
    GI = L["G"] + iota * L["I"]
    sg, sg_t, sg_p = L["sqrtg"], L["sqrtg_t"], L["sqrtg_p"]
    beta_b = fl["beta_b"]
    base = (modB * L["B_s"] + MU0 * L["dp_ds"]) / (modB ** 2 * psi_e)
    Lterm = L0 * (L["G"] * sg_t - L["I"] * sg_p) / (2 * sg * GI)
    bphi = beta_b * sg_p / (2 * sg * GI)
    bth = beta_b * iota * sg_t / (2 * sg * GI)
    kg = (L["G"] * sg_t - L["I"] * sg_p) / (2 * sg * GI)
    # the force-balance metric form, with vmec_tube's own dual-relation gradients
    gth = np.cross(L["r_p"], L["r_s"], axis=0) / (sg * psi_e)
    gph = np.cross(L["r_s"], L["r_t"], axis=0) / (sg * psi_e)
    gradB = L["B_s"] * gpsi / psi_e + L["B_t"] * gth + L["B_p"] * gph
    kn_metric = MU0 * L["dp_ds"] / psi_e / modB ** 2 + np.sum(gradB * gpsi, 0) / (modB * gpp)
    variants = {"code (+b*dphi)": base + bphi + Lterm, "pre-2026 GX (-b*dphi)": base - bphi + Lterm,
                "derived +b(iota*dth+dphi)": base + bphi + bth + Lterm, "derived, minus": base - bphi - bth + Lterm,
                "no beta term": base + Lterm, "metric mu0p'/B^2+gradpsi.gradB": kn_metric}
    # exact curvature of the reconstructed line, B along the representation B = grad psi x grad alpha
    r1, r2 = L["r1"], L["r2"]
    l1 = np.linalg.norm(r1, axis=0)
    t = r1 / l1
    kap = (r2 - np.sum(r2 * t, 0) * t) / l1 ** 2
    Brep = np.cross(gpsi, galpha, axis=0)
    sigma = np.sign(np.sum(Brep * r1, 0))
    bxk = sigma * np.cross(t, kap, axis=0)
    num_a = modB * np.sum(bxk * galpha, 0)
    num_p = modB * np.sum(bxk * gpsi, 0)
    kn_num = np.sum(kap * gpsi, 0) / gpp
    kg_num = num_p / modB ** 2
    res = dict(sigma=(sigma.min(), sigma.max()), kappa_n_code_vs_variant=_rel(fl["kappa_n"], variants["code (+b*dphi)"]),
               kg_err=_rel(kg, kg_num), beta_scale=float(np.abs(bphi).max() / np.abs(kn_num).max()),
               bth_scale=float(np.abs(bth).max() / np.abs(kn_num).max()), rows={})
    for name, kn in variants.items():
        Bxk_a = (kn + kg * L1) * modB ** 2
        res["rows"][name] = (_rel(kn, kn_num), _rel(Bxk_a, num_a))
    res["Bxk_psi_err"] = _rel(kg * modB ** 2, num_p)
    res["code_Bxk_alpha_err"] = _rel(fl["Bxk_alpha"], num_a)
    # the existing test's finite-difference curvature on the same line (its 4001 points and edge trimming)
    th2 = np.linspace(-np.pi, np.pi, 4001)
    fl2 = vt.boozer_fieldline(vs, s, alpha, th2)
    phc = fl2["phi_b"] - fl2["nu"]
    r = np.stack([fl2["R"] * np.cos(phc), fl2["R"] * np.sin(phc), fl2["Z"]])
    dth = th2[1] - th2[0]
    r_t = np.gradient(r, dth, axis=1, edge_order=2)
    lt = np.linalg.norm(r_t, axis=0)
    b = r_t / lt
    kfd = np.gradient(b, dth, axis=1, edge_order=2) / lt
    bxk2 = np.cross(b, kfd, axis=0)
    sl = slice(50, -50)
    num_a2 = fl2["modB"] * np.sum(bxk2 * fl2["grad_alpha"], 0)
    res["fd_test_err_a"] = float(np.abs(fl2["Bxk_alpha"] - num_a2)[sl].max() / np.abs(num_a2[sl]).max())
    return res


def _print_table(tag, res):
    print(f"\n[{tag}] B·(grad psi x grad alpha) orientation sigma along r' = {res['sigma']}; code kappa_n = "
          f"variant 'code' to {res['kappa_n_code_vs_variant']:.1e}")
    print(f"[{tag}] |beta dphi term|/|kappa_n| = {res['beta_scale']:.3e}, |beta iota dth term|/|kappa_n| = "
          f"{res['bth_scale']:.3e}; kappa_g (Boozer) vs exact {res['kg_err']:.3e}; Bxk.grad psi {res['Bxk_psi_err']:.3e}")
    print(f"[{tag}] {'kappa_n form':38s} {'err kappa_n':>12s} {'err Bxk.grad alpha':>19s}")
    for name, (ekn, ea) in res["rows"].items():
        print(f"[{tag}] {name:38s} {ekn:12.3e} {ea:19.3e}")
    print(f"[{tag}] existing FD test metric err_a (code form) = {res['fd_test_err_a']:.3e}; exact = "
          f"{res['code_Bxk_alpha_err']:.3e}")


# ------------------------------------------------------------------ curvature: which normal-curvature form is b·∇b?
@pytest.mark.parametrize("wout,s0,alpha", [(QA, 0.5, 0.0), (QA, 0.5, 0.7), (QH, 0.5, 0.0),
                                           (W7X, 0.5, 0.0), (W7X, 0.25, 0.0), (W7X, 0.75, 0.0), (W7X, 0.5, 0.6283185)])
def test_normal_curvature_forms_against_exact_line_curvature(wout, s0, alpha):
    vt = _vt(wout)
    vs = _splines(vt, wout)
    res = _curvature_table(vt, vs, s0, alpha)
    _print_table(f"{NAME[wout]} s{s0} a{alpha:.3f}", res)
    assert res["kappa_n_code_vs_variant"] < 1e-12
    assert np.isfinite(res["code_Bxk_alpha_err"])


@pytest.mark.parametrize("wout,mb", [(W7X, (30, 30)), (QA, (24, 24))])
def test_normal_curvature_forms_at_higher_boozer_resolution(wout, mb):
    """The same table at higher (mboz, nboz): a residual that is Boozer truncation shrinks, a formula error does not."""
    import time
    vt = _vt(wout)
    t0 = time.time()
    vs = _splines(vt, wout, *mb)
    print(f"\n[{NAME[wout]} mboz,nboz={mb}] booz_xform + splines {time.time() - t0:.1f} s")
    res = _curvature_table(vt, vs, 0.5, 0.0)
    _print_table(f"{NAME[wout]} s0.5 mboz{mb}", res)
    assert np.isfinite(res["code_Bxk_alpha_err"])


@pytest.mark.parametrize("wout", [QA, W7X])
def test_existing_fd_test_bands_discriminate(wout):
    """What the bands of tests/test_vmec_tube.py::test_boozer_curvature_is_b_dot_grad_b_of_the_field_line accept:
    its metric for deliberately wrong curvature formulas (QA 3e-3/6e-3, W7-X 1.5e-2/6e-2)."""
    vt = _vt(wout)
    vs = _splines(vt, wout)
    th = np.linspace(-np.pi, np.pi, 4001)
    L = _line(vt, vs, 0.5, 0.0, th)
    fl = L["fl"]
    phc = fl["phi_b"] - fl["nu"]
    r = np.stack([fl["R"] * np.cos(phc), fl["R"] * np.sin(phc), fl["Z"]])
    dth = th[1] - th[0]
    r_t = np.gradient(r, dth, axis=1, edge_order=2)
    lt = np.linalg.norm(r_t, axis=0)
    b = r_t / lt
    kap = np.gradient(b, dth, axis=1, edge_order=2) / lt
    bxk = np.cross(b, kap, axis=0)
    num_a = fl["modB"] * np.sum(bxk * fl["grad_alpha"], 0)
    num_p = fl["modB"] * np.sum(bxk * fl["grad_psi"], 0)
    sl = slice(50, -50)
    err = lambda a, nref: float(np.abs(a - nref)[sl].max() / np.abs(nref[sl]).max())
    gpsi, galpha = fl["grad_psi"], fl["grad_alpha"]
    gpp = np.sum(gpsi ** 2, 0)
    L1 = np.sum(galpha * gpsi, 0) / gpp
    B2 = fl["modB"] ** 2
    kn, kg = fl["kappa_n"], fl["kappa_g"]
    GI = L["G"] + L["iota"] * L["I"]
    zeta_c = 0.0
    L0 = -(L1 + L["diota"] * (L["ph"] - zeta_c) / fl["psi_edge"])
    Lterm = L0 * (L["G"] * L["sqrtg_t"] - L["I"] * L["sqrtg_p"]) / (2 * L["sqrtg"] * GI)
    wrong = {"as coded": (kn, kg), "no L0 term": (kn - Lterm, kg), "kappa_g x 0.9": (kn, 0.9 * kg),
             "kappa_n x 0.95": (0.95 * kn, kg), "B_s term x 0.9": (kn - 0.1 * L["B_s"] / (L["modB"] * fl["psi_edge"]), kg),
             "kappa_g sign": (kn, -kg)}
    tag = NAME[wout]
    for name, (a, g) in wrong.items():
        print(f"[{tag} FD-test bands] {name:16s} err_a {err((a + g * L1) * B2, num_a):.3e}  err_psi {err(g * B2, num_p):.3e}")
    assert err((kn + kg * L1) * B2, num_a) < 0.02


# ------------------------------------------------------------------ dual relations and the Jacobian convention
@pytest.mark.parametrize("wout", [QA, QH, W7X])
@pytest.mark.parametrize("s0", [0.1, 0.5, 0.9])
def test_dual_relations_and_jacobian_convention(wout, s0):
    """vmec_tube (as GX) builds ∇ψ, ∇θ, ∇φ by the dual relations with booz_xform's √g = (G+ιI)/B² and ψ_e = −Φ_e/2π.
    They are the gradients of the coordinates only if J_s = r_s·(r_θ×r_φ) equals ψ_e √g_B, and B = ∇ψ×∇α has |B| =
    bmag and points along the line."""
    vt = _vt(wout)
    vs = _splines(vt, wout)
    th = np.linspace(-np.pi, np.pi, 801)
    L = _line(vt, vs, s0, 0.0, th)
    fl = L["fl"]
    psi_e = fl["psi_edge"]
    J = np.sum(L["r_s"] * np.cross(L["r_t"], L["r_p"], axis=0), 0)
    rho_J = J / (psi_e * L["sqrtg"])
    gpsi_mine = np.cross(L["r_t"], L["r_p"], axis=0) / L["sqrtg"]
    rec = _rel(gpsi_mine, fl["grad_psi"])
    drt_err = _rel(L["r1"], L["r_t"] + L["r_p"] / L["iota"])
    Brep = np.cross(fl["grad_psi"], fl["grad_alpha"], axis=0)
    Bline = L["iota"] / L["sqrtg"] * L["r1"]
    cosang = np.sum(Brep * L["r1"], 0) / (np.linalg.norm(Brep, axis=0) * np.linalg.norm(L["r1"], axis=0))
    b_dot_gpsi = np.abs(np.sum(L["r1"] * fl["grad_psi"], 0)) / (np.linalg.norm(L["r1"], axis=0) *
                                                               np.linalg.norm(fl["grad_psi"], axis=0))
    tag = f"{NAME[wout]} s{s0}"
    print(f"\n[{tag}] iota {L['iota']:+.5f} psi_e {psi_e:+.5e} G {L['G']:+.5e} I {L['I']:+.4e} sqrtg_B in "
          f"[{L['sqrtg'].min():+.4e}, {L['sqrtg'].max():+.4e}] J_s in [{J.min():+.4e}, {J.max():+.4e}]")
    print(f"[{tag}] J_s/(psi_e sqrtg_B): min {rho_J.min():.6f} max {rho_J.max():.6f}  (1 = consistent duals)")
    print(f"[{tag}] |grad psi x grad alpha|/bmag: [{(np.linalg.norm(Brep, axis=0) / L['modB']).min():.6f}, "
          f"{(np.linalg.norm(Brep, axis=0) / L['modB']).max():.6f}];  |iota/sqrtg_B r'|/bmag: "
          f"[{(np.linalg.norm(Bline, axis=0) / L['modB']).min():.6f}, {(np.linalg.norm(Bline, axis=0) / L['modB']).max():.6f}]")
    print(f"[{tag}] cos(grad psi x grad alpha, r'): [{cosang.min():+.6f}, {cosang.max():+.6f}]; max |b.grad psi|/|grad psi| "
          f"{b_dot_gpsi.max():.2e}; reconstruction of grad psi {rec:.1e}, r' = r_t + r_p/iota to {drt_err:.1e}")
    assert rec < 1e-10 and drt_err < 1e-10
    assert np.abs(rho_J - 1).max() < 5e-2, ("dual relations inconsistent", rho_J.min(), rho_J.max())
    assert np.abs(np.abs(cosang) - 1).max() < 1e-3


# ------------------------------------------------------------------ the default L_ref of geometry.kind = "vmec"
@pytest.mark.parametrize("wout", [QA, QH])
def test_default_lref_is_r_over_a(wout):
    """from_vmec_tube's default L_ref = rmaj·a with rmaj = (R_max + R_min)/2 of the tube, which GX writes in METRES for
    VMEC.  So the default L_ref depends on the machine's absolute size.  Expected: rmaj ≈ R/a (Rmajor_p/Aminor_p)."""
    vt = _vt(wout)
    from netCDF4 import Dataset
    with Dataset(wout) as nc:
        R0, a = float(nc.variables["Rmajor_p"][:]), float(nc.variables["Aminor_p"][:])
    geom = vt.from_vmec_tube(wout, Nx=8, Lx=30.0, Ny=6, Nz=48, Ly=30.0, s0=0.5, cut="fix aspect")
    rmaj = geom.extra["rmaj"]
    print(f"\n[{NAME[wout]}] default Lref_over_a (the tube's rmaj) = {rmaj:.5f}; Rmajor_p = {R0:.4f} m, Aminor_p = {a:.4f} m, "
          f"R/a = {R0 / a:.4f}; gpar[0] = {float(geom.gpar[0, 0, 0]):+.5f}")
    print(f"[{NAME[wout]}] tests/test_vmec_tube._cfg's omega_T = 18 (meant as R/L_T with R/a = 6) is then a/L_T = "
          f"{18 / rmaj:.3f}, not 3")
    assert abs(rmaj - R0 / a) < 0.1 * R0 / a, f"default L_ref = {rmaj:.3f} a (metres of R), R/a = {R0 / a:.3f}"


# ------------------------------------------------------------------ the ends of α ≠ 0 tubes and the spectral bgrad
@pytest.mark.parametrize("wout,alpha", [(QA, 0.0), (QA, 1.5707963), (QA, 0.3), (W7X, 0.0), (W7X, 0.6283185),
                                        (QH, 0.0), (QH, 0.7853982)])
def test_tube_ends_twist_and_bgrad(wout, alpha):
    vt = _vt(wout)
    from velcro.gx_geometry import _spectral_bgrad
    try:
        e = vt.vmec_tube_eik(wout, 48, s0=0.5, alpha=alpha, cut="fix aspect")
    except ValueError as ex:
        pytest.skip(f"no 'fix aspect' cut: {ex}")
    sh = e["shat"]
    g21, g22, B = e["gds21"], e["gds22"], e["bmag"]
    facL = 2 * sh * g21[0] / g22[0]
    facR = -2 * sh * g21[-1] / g22[-1]
    fac2 = -sh * (g21[-1] / g22[-1] - g21[0] / g22[0])
    jumpB = abs(B[-1] - B[0]) / B.mean()
    jump22 = abs(g22[-1] - g22[0]) / g22.max()
    jumpgb = abs(e["gbdrift"][-1] - e["gbdrift"][0]) / np.abs(e["gbdrift"]).max()
    jumpgb0 = abs(e["gbdrift0"][-1] + e["gbdrift0"][0]) / np.abs(e["gbdrift0"]).max()
    gp = float(e["gradpar"][0])
    spec = _spectral_bgrad(B[:-1], gp)
    fd = gp * np.gradient(np.log(B), e["theta"], edge_order=2)[:-1]
    # a fine tube of the same line as the reference derivative
    ef = vt.vmec_tube_eik(wout, 480, s0=0.5, alpha=alpha, cut="fix aspect")
    fine = float(ef["gradpar"][0]) * np.gradient(np.log(ef["bmag"]), ef["theta"], edge_order=2)
    fine_on = np.interp(e["theta"][:-1], ef["theta"], fine)
    tag = f"{NAME[wout]} alpha {alpha:.4f}"
    print(f"\n[{tag}] cut theta_B = {e['theta_cut']:.6f}, shat {sh:+.6f}, q {e['q']:.5f}, iota {e['iota']:+.5f}")
    print(f"[{tag}] twist fac: left end (GX, Velcro) {facL:+.5f}, right end {facR:+.5f}, two-ended {fac2:+.5f}")
    print(f"[{tag}] end jumps: bmag {jumpB:.3e}, gds22 {jump22:.3e}, gbdrift {jumpgb:.3e}, gbdrift0 (odd) {jumpgb0:.3e}")
    print(f"[{tag}] bgrad: spectral (GX/Velcro) vs fine-tube FD max err {_rel(spec, fine_on):.3e}; 48-pt FD vs fine "
          f"{_rel(fd, fine_on):.3e}; spectral at the two ends {spec[0]:+.4e} vs fine {fine_on[0]:+.4e}")
    assert np.isfinite(facL)


# ------------------------------------------------------------------ Boozer resolution
@pytest.mark.parametrize("wout,mb", [(QA, (24, 24)), (W7X, (30, 30)), (QH, (24, 24))])
def test_boozer_resolution_convergence(wout, mb):
    vt = _vt(wout)
    base = vt.vmec_tube_eik(wout, 48, s0=0.5, alpha=0.0, cut="none")
    hi = vt.vmec_tube_eik(wout, 48, s0=0.5, alpha=0.0, cut="none", mboz=mb[0], nboz=mb[1])
    bc = vt.vmec_tube_eik(wout, 48, s0=0.5, alpha=0.0, cut="fix aspect")
    hc = vt.vmec_tube_eik(wout, 48, s0=0.5, alpha=0.0, cut="fix aspect", mboz=mb[0], nboz=mb[1])
    keys = ("bmag", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "cvdrift", "grho")
    tag = f"{NAME[wout]} default vs mboz,nboz {mb}"
    print(f"\n[{tag}] cut none : " + ", ".join(f"{k} {_rel(hi[k], base[k]):.2e}" for k in keys) +
          f"; shat {hi['shat']:.7f}/{base['shat']:.7f}")
    print(f"[{tag}] fix aspect: cut {bc['theta_cut']:.6f} -> {hc['theta_cut']:.6f}; " +
          ", ".join(f"{k} {_rel(hc[k], bc[k]):.2e}" for k in keys))
    assert np.isfinite(_rel(hi["bmag"], base["bmag"]))


# ------------------------------------------------------------------ ι and p: half grid (Velcro) vs half + full (GX)
@pytest.mark.parametrize("wout", [QA, QH, W7X])
def test_iota_pressure_splines_half_vs_gx_combined(wout):
    vt = _vt(wout)
    from netCDF4 import Dataset
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    vs = _splines(vt, wout)
    with Dataset(wout) as nc:
        v = lambda k: np.asarray(nc.variables[k][:], float)
        ns = int(v("ns"))
        iotas, iotaf, pres, presf = v("iotas"), v("iotaf"), v("pres"), v("presf")
    sf = np.linspace(0, 1, ns)
    sh = 0.5 * (sf[:-1] + sf[1:])
    sc = np.concatenate([sh, sf])
    o = np.argsort(sc)
    gi = IUS(sc[o], np.concatenate([iotas[1:], iotaf])[o])
    gp = IUS(sc[o], np.concatenate([pres[1:], presf])[o])
    avg = np.abs(iotaf[1:-1] - 0.5 * (iotas[1:-1] + iotas[2:])).max()
    tag = NAME[wout]
    print(f"\n[{tag}] ns {ns}; max |iotaf - mean of neighbouring iotas| (interior) {avg:.2e} (0: iotaf is VMEC's average)")
    for s0 in (0.02, 0.1, 0.3, 0.5, 0.505, 0.75, 0.98, 0.995):
        hi_, gi_ = float(vs.iota(s0)), float(gi(s0))
        hd, gd = float(vs.iota.derivative()(s0)), float(gi.derivative()(s0))
        sv, sg = -2 * s0 / hi_ * hd, -2 * s0 / gi_ * gd
        pv, pg = float(vs.pressure.derivative()(s0)), float(gp.derivative()(s0))
        dp = abs(pv - pg) / max(abs(pg), 1e-300) if pg != 0 else abs(pv)
        print(f"[{tag}] s0 {s0:5.3f}: iota rel {abs(hi_ - gi_) / abs(gi_):.1e}; shat Velcro {sv:+.6e} GX {sg:+.6e} rel "
              f"{abs(sv - sg) / max(abs(sg), 1e-300):.1e}; dp/ds rel {dp:.1e}")
    assert np.isfinite(avg)
