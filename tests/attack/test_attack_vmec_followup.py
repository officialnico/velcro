"""Attack on ``geometry.kind = "vmec"``, second round (card only; VELCRO_WOUT_DIR=$IN with the QA, QH, W7-X wouts).

* A negative twist (S < 0: W7-X, QH, QA npol 2 or the 2nd crossing) was never run against anything.  An up-down
  symmetric tube A and the same tube with its odd arrays (gds21, gbdrift0, cvdrift0) negated, B, are the same physics
  under (z, v∥) → (−z, −v∥): B(z) = A(−z) and S_B = −S_A.  Adiabatic electrons (FD4, the twisted ghost cells): the
  mapped state must evolve into the mapped state, to rounding.  Kinetic electrons on GX's chains: γ(k_y) must agree.
* GX 3865a53 splines ι and p through VMEC's half AND full grids; VMEC's full-grid iotaf is the mean of its two half-grid
  neighbours, so the interpolant zig-zags and dι/ds oscillates between nodes; the half-grid spline (Velcro) does not.
* The normal curvature: GX's form against the force-balance metric form μ₀p'/B² + ∇ψ·∇B/(B|∇ψ|²) directly.
* N_z = 48 against N_z = 480 (GX's discretisation of the same tube: the cut, the equal-arc remap, bgrad).
* A kind = "vmec" Simulation with an L_x that is not jtwist/(|S| k_y,min) is accepted (no guard).
"""
import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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


# ------------------------------------------------------------------ negative twist = z-reflection of a positive one
def _mirror_files(tmp_path, vt, zero_end_drift=True):
    """A (the QA α = 0 'fix aspect' tube) and B (its odd arrays negated).  The grid point z = −π belongs to the left end
    of A but to the right end of the mirrored tube: the two ends agree on every coefficient except gbdrift0 (odd, and
    ≠ 0 at a 'fix aspect' cut), whose jump across the link is GX's model.  ``zero_end_drift`` sets gbdrift0(−π) = 0 in
    both, so that A and B are exactly the same discrete physics and only the twist's sign is tested."""
    from velcro.gx_geometry import twist_shear, write_gx_eik
    gx = dict(vt.vmec_tube_gx_arrays(QA, 48, s0=0.5, alpha=0.0, cut="fix aspect"))
    Nz = len(gx["theta"])
    print(f"\n[mirror] gbdrift0(-pi)/max|gbdrift0| = {abs(gx['gbdrift0'][0]) / np.abs(gx['gbdrift0']).max():.3e} "
          f"(the radial drift's jump across a twist link is 2 k_x gbdrift0(-pi)/shat); zeroed: {zero_end_drift}")
    if zero_end_drift:
        for k in ("gbdrift0", "cvdrift0"):
            a = np.array(gx[k], float)
            a[0] = 0.0
            gx[k] = a
    for k in ("bmag", "gds2", "gds22", "gbdrift", "grho"):                       # even about θ = 0 on the grid
        a = np.asarray(gx[k])
        assert np.abs(a[1:] - a[1:][::-1]).max() <= 1e-12 * np.abs(a).max(), k
    for k in ("gds21", "gbdrift0"):                                              # odd
        a = np.asarray(gx[k])
        assert np.abs(a[1:] + a[1:][::-1]).max() <= 1e-12 * np.abs(a).max() + 1e-14, k
    gb = dict(gx)
    for k in ("gds21", "gbdrift0", "cvdrift0"):
        gb[k] = -np.asarray(gx[k])
    pa, pb = str(tmp_path / "A.eik.out"), str(tmp_path / "B.eik.out")
    write_gx_eik(pa, gx)
    write_gx_eik(pb, gb)
    return pa, pb, twist_shear(gx), twist_shear(gb), Nz


def _cfg(path, Lx, Ly, kinetic, Nx=16, Ny=6, L=4, M=2, chains=True, omega_n=1.0, omega_T=3.0):
    from velcro.config import VelcroConfig
    physics = dict(omega_n=dict(value=omega_n), omega_T=dict(value=omega_T), nonlinear=False, Ti_profile=False,
                   n_profile=False, nu=0.0)
    numerics = dict(hyper_x=0.0, hyper_z=0.0, hyper_v=0.0, krook_rate=0.0, krook_width=0.0, integrator="rk4", cfl=0.5)
    if kinetic:
        physics.update(electrons="kinetic", beta=0.0, electron=dict(mass=2.7e-4))
        numerics.update(parallel_derivative="fourier", field_solver="fft", radial_drift="fourier", bracket_dx="fourier")
        if chains:
            numerics.update(x_dealias=True, parallel_bc="gx_chains", gx_nyquist=False, gx_damp_ends=0.0)
    return VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=Lx, Ny=Ny, Nz=48, L=L, M=M),
        geometry=dict(kind="gx_file", file=path, Ly=Ly, Lref_over_a=1.0, rho_star=1e-5),
        physics=physics, numerics=numerics, init=dict(kind="noise", amplitude=1e-3, seed=4),
        run=dict(t_end=1.0, precision="fp64")))


def _box(S):
    Ly = 2 * np.pi / 0.3                                         # k_y,min 0.3 (k_y rho_i 0.21)
    jt = int(round(2 * np.pi * S)) or 1
    return abs(jt) / (abs(S) * 2 * np.pi / Ly), Ly, jt


@pytest.mark.parametrize("zero_end_drift,band", [(True, None), (False, None), (True, 1)])
def test_negative_twist_adiabatic_is_the_exact_mirror(tmp_path, zero_end_drift, band):
    """``band``: the initial state keeps only |m| ≤ band of the FFT-ordered x modes.  The twist e^{−i k_y τ(x)} is a
    circular shift of the discrete k_x spectrum by n·jtwist, so a full-band state wraps around at the Nyquist mode, where
    the circulant field/gyro symbols of A and B are evaluated at different k_x; a band-limited state does not reach the
    wrap within the run, and the map must then be exact to rounding."""
    vt = _vt(QA)
    from velcro.simulation import Simulation
    pa, pb, SA, SB, Nz = _mirror_files(tmp_path, vt, zero_end_drift)
    Lx, Ly, jt = _box(SA)
    A, B = Simulation(_cfg(pa, Lx, Ly, False)), Simulation(_cfg(pb, Lx, Ly, False))
    x, ky = np.asarray(A.geom.x), np.asarray(A.ky)
    print(f"\n[mirror, adiabatic] S_A {SA:+.6f} S_B {SB:+.6f} jtwist {jt} Lx {Lx:.4f}; twist phase B = -A: "
          f"{np.abs(np.asarray(B.geom.twist_phase) + np.asarray(A.geom.twist_phase)).max():.1e}; band {band}")
    GA = np.asarray(A.G)
    if band is not None:
        Fx = np.fft.fft(GA, axis=2)
        m = np.abs(np.fft.fftfreq(GA.shape[2], d=1.0 / GA.shape[2]))
        Fx[:, :, m > band] = 0.0
        GA = np.fft.ifft(Fx, axis=2)
        GA[:, :, :, 0, :] = GA[:, :, :, 0, :].real
    GA0 = jnp.asarray(GA, dtype=A.G.dtype)
    L = A.cfg.grid.L
    par = ((-1.0) ** (np.arange(GA.shape[0]) % L))[:, None, None, None, None]
    tw = np.exp(-1j * ky[None, :] * np.asarray(A.geom.twist_phase)[:, None])    # e^{−i k_y τ_A(x)}

    def P(G):
        G = np.asarray(G)
        out = np.empty_like(G)
        out[..., 1:] = G[..., 1:][..., ::-1]
        out[..., 0] = G[..., 0] * tw
        return out * par
    GB0 = jnp.asarray(P(GA), dtype=A.G.dtype)
    dt = min(A.dt_from_cfl(GA0), B.dt_from_cfl(GB0))
    n = 400
    GA1, GB1 = A._nsteps(GA0, dt, n), B._nsteps(GB0, dt, n)
    ra = _rel(P(np.asarray(GA1)), np.asarray(GB1))
    phA, phB = np.asarray(A._solve_fields(GA1)[0]), np.asarray(B._solve_fields(GB1)[0])
    mph = np.empty_like(phA)
    mph[..., 1:] = phA[..., 1:][..., ::-1]
    mph[..., 0] = phA[..., 0] * tw
    rp = _rel(mph, phB)
    gA = np.log(np.sum(np.abs(phA) ** 2, axis=(0, 2)))
    gB = np.log(np.sum(np.abs(phB) ** 2, axis=(0, 2)))
    print(f"[mirror, adiabatic, gbdrift0(-pi) zeroed {zero_end_drift}, band {band}] after {n} steps of {dt:.3e} "
          f"(t = {n * dt:.2f}): |P G_A - G_B|/|G_B| = {ra:.2e}, phi {rp:.2e}; log|phi|^2 per k_y A {np.round(gA, 5)} "
          f"B {np.round(gB, 5)}")
    if zero_end_drift and band is not None:
        assert ra < 1e-9 and rp < 1e-9


@pytest.mark.parametrize("chains", [True, False])
def test_negative_twist_kinetic_growth_rates(tmp_path, chains):
    """A strongly driven ITG (a/L_T 9, a/L_n 3) so that one eigenmode dominates within the run; γ of A and of its mirror
    B (S < 0) from the last third, with the middle third printed as the convergence check."""
    vt = _vt(QA)
    from velcro.simulation import Simulation
    pa, pb, SA, SB, Nz = _mirror_files(tmp_path, vt)
    Lx, Ly, jt = _box(SA)
    kw = dict(chains=chains, omega_n=3.0, omega_T=9.0)
    A, B = Simulation(_cfg(pa, Lx, Ly, True, **kw)), Simulation(_cfg(pb, Lx, Ly, True, **kw))
    dt = min(A.dt_from_cfl(A.G), B.dt_from_cfl(B.G))
    T, nseg = 30.0, 30
    n = int(np.ceil(T / nseg / dt))
    ga, gb = A.G, B.G
    la, lb = [], []
    for _ in range(nseg):
        ga, gb = A._nsteps(ga, dt, n), B._nsteps(gb, dt, n)
        pa_, pb_ = np.asarray(A._solve_fields(ga)[0]), np.asarray(B._solve_fields(gb)[0])
        la.append(np.log(np.sum(np.abs(pa_) ** 2, axis=(0, 2))))
        lb.append(np.log(np.sum(np.abs(pb_) ** 2, axis=(0, 2))))
        na, nb = float(np.abs(np.asarray(ga)).max()), float(np.abs(np.asarray(gb)).max())
        if na > 1e50 or nb > 1e50:
            break
    la, lb = np.array(la), np.array(lb)
    tt = np.arange(1, len(la) + 1) * n * dt
    third = len(la) // 3
    gam = lambda l, sl: 0.5 * np.polyfit(tt[sl], l[sl], 1)[0]
    rows = range(1, la.shape[1])
    gA = np.array([gam(la[:, j], slice(2 * third, None)) for j in rows])
    gB = np.array([gam(lb[:, j], slice(2 * third, None)) for j in rows])
    gA2 = np.array([gam(la[:, j], slice(third, 2 * third)) for j in rows])
    gB2 = np.array([gam(lb[:, j], slice(third, 2 * third)) for j in rows])
    print(f"\n[mirror, kinetic, gx_chains={chains}] S_A {SA:+.5f} S_B {SB:+.5f} jtwist {jt}, dt {dt:.3e}, t {tt[-1]:.1f}: "
          f"gamma (last third) A {np.round(gA, 5)} B {np.round(gB, 5)}; (middle third) A {np.round(gA2, 5)} "
          f"B {np.round(gB2, 5)}")
    assert np.allclose(gA, gB, rtol=1e-2, atol=1e-3), (gA, gB)


# ------------------------------------------------------------------ ι: GX's half+full spline against the half grid
@pytest.mark.parametrize("wout,s_lo,s_hi", [
    (QA, 0.28, 0.32),
    pytest.param(QH, 0.08, 0.14, marks=pytest.mark.xfail(strict=True, reason=(
        "documents GX's own behaviour, not Velcro's: near the axis of the QH wout GX's pyvmec spline of ι on the combined "
        "half/full grid zig-zags between VMEC nodes (6 sign changes of dι/ds, Velcro's half-grid spline 1)"))),
    (QH, 0.48, 0.52)])
def test_gx_combined_iota_spline_oscillates(wout, s_lo, s_hi):
    vt = _vt(wout)
    from netCDF4 import Dataset
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    vs = vt._splines(wout, None, None)
    with Dataset(wout) as nc:
        v = lambda k: np.asarray(nc.variables[k][:], float)
        ns = int(v("ns"))
        iotas, iotaf = v("iotas"), v("iotaf")
    sf = np.linspace(0, 1, ns)
    sh = 0.5 * (sf[:-1] + sf[1:])
    sc = np.concatenate([sh, sf])
    o = np.argsort(sc)
    gi = IUS(sc[o], np.concatenate([iotas[1:], iotaf])[o])
    s = np.linspace(s_lo, s_hi, 41)
    dv, dg = vs.iota.derivative()(s), gi.derivative()(s)
    sv, sg = -2 * s / vs.iota(s) * dv, -2 * s / gi(s) * dg
    wig = lambda a: int(np.sum(np.diff(np.sign(np.diff(a))) != 0))
    print(f"\n[{NAME[wout]} s in [{s_lo}, {s_hi}], h = {1 / (ns - 1):.4f}] shat Velcro {sv.min():+.5e}..{sv.max():+.5e} "
          f"(turning points {wig(sv)}); GX {sg.min():+.5e}..{sg.max():+.5e} (turning points {wig(sg)})")
    print(f"[{NAME[wout]}] s: " + " ".join(f"{a:.3f}" for a in s[::4]))
    print(f"[{NAME[wout]}] Velcro shat: " + " ".join(f"{a:+.4e}" for a in sv[::4]))
    print(f"[{NAME[wout]}] GX     shat: " + " ".join(f"{a:+.4e}" for a in sg[::4]))
    assert wig(sg) <= wig(sv) + 2, "GX's combined-grid derivative zig-zags between VMEC nodes"


# ------------------------------------------------------------------ normal curvature: GX's form vs force balance
@pytest.mark.parametrize("s0,alpha", [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.5, 0.6283185)])
def test_gx_normal_curvature_against_force_balance_metric_form(s0, alpha):
    vt = _vt(W7X)
    vs = vt._splines(W7X, None, None)
    th = np.linspace(-np.pi, np.pi, 1201)
    fl = vt.boozer_fieldline(vs, s0, alpha, th)
    c = lambda spl, d=0: np.array([f.derivative()(s0) if d else f(s0) for f in spl])
    rmnc, drmnc, zmns, dzmns = c(vs.rmnc_b), c(vs.rmnc_b, 1), c(vs.zmns_b), c(vs.zmns_b, 1)
    numns, dnumns, gmnc = c(vs.numns_b), c(vs.numns_b, 1), c(vs.gmnc_b)
    bmnc, dbmnc = c(vs.bmnc_b), c(vs.bmnc_b, 1)
    iota, diota = float(vs.iota(s0)), float(vs.iota.derivative()(s0))
    dp = float(vs.pressure.derivative()(s0))
    G, I = float(vs.Gfun(s0)), float(vs.Ifun(s0))
    xm, xn = vs.xm_b[:, None], vs.xn_b[:, None]
    ph = (th - alpha) / iota
    ang = xm * th[None] - xn * ph[None]
    C, S = np.cos(ang), np.sin(ang)
    R = rmnc @ C
    nu = numns @ S
    phc = ph - nu
    cp, sp = np.cos(phc), np.sin(phc)
    cart = lambda Ra, Za, pa: np.stack([Ra * cp - R * sp * pa, Ra * sp + R * cp * pa, Za])
    r_s = cart(drmnc @ C, dzmns @ S, -(dnumns @ S))
    r_t = cart(-(rmnc @ (xm * S)), zmns @ (xm * C), -(numns @ (xm * C)))
    r_p = cart(rmnc @ (xn * S), -(zmns @ (xn * C)), 1 - (-(numns @ (xn * C))))
    sg, sg_t, sg_p = gmnc @ C, -(gmnc @ (xm * S)), gmnc @ (xn * S)
    B, B_s, B_t, B_p = bmnc @ C, dbmnc @ C, -(bmnc @ (xm * S)), bmnc @ (xn * S)
    pe = fl["psi_edge"]
    gpsi = fl["grad_psi"]
    gth = np.cross(r_p, r_s, axis=0) / (sg * pe)
    gph = np.cross(r_s, r_t, axis=0) / (sg * pe)
    gpp = np.sum(gpsi ** 2, 0)
    kn_metric = MU0 * dp / pe / B ** 2 + np.sum((B_s * gpsi / pe + B_t * gth + B_p * gph) * gpsi, 0) / (B * gpp)
    GI = G + iota * I
    beta = fl["beta_b"]
    kn_gx = fl["kappa_n"]
    kn_full = kn_gx + beta * iota * sg_t / (2 * sg * GI)
    kn_old = kn_gx - 2 * beta * sg_p / (2 * sg * GI)
    scale = np.abs(kn_metric).max()
    e_gx, e_full, e_old = (float(np.abs(k - kn_metric).max() / scale) for k in (kn_gx, kn_full, kn_old))
    # the same through B×κ·∇α (what cvdrift carries): (κ_n + κ_g L1) B²
    L1 = np.sum(fl["grad_alpha"] * gpsi, 0) / gpp
    kg = fl["kappa_g"]
    ba = lambda kn: (kn + kg * L1) * B ** 2
    sa = np.abs(ba(kn_metric)).max()
    print(f"\n[W7X s{s0} a{alpha:.3f}] kappa_n vs the force-balance metric form (max err / max|kappa_n|): GX 3865a53 {e_gx:.2e}, "
          f"with the iota*d_theta term {e_full:.2e}, pre-2026 sign {e_old:.2e};  Bxk.grad alpha: GX "
          f"{np.abs(ba(kn_gx) - ba(kn_metric)).max() / sa:.2e}, full {np.abs(ba(kn_full) - ba(kn_metric)).max() / sa:.2e}")
    assert e_full < e_gx


# ------------------------------------------------------------------ N_z = 48 against a fine tube
@pytest.mark.parametrize("wout,alpha", [(QA, 0.0), (QH, 0.0), (W7X, 0.0), (QA, 1.5707963)])
def test_ntheta48_against_fine_tube(wout, alpha):
    vt = _vt(wout)
    from velcro.gx_geometry import _spectral_bgrad
    e, f = (vt.vmec_tube_eik(wout, n, s0=0.5, alpha=alpha, cut="fix aspect") for n in (48, 960))
    keys = ("bmag", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "grho")
    on = lambda a: np.interp(e["theta"], f["theta"], a)
    errs = {k: _rel(e[k], on(f[k])) for k in keys}
    gp = float(e["gradpar"][0])
    bg48 = _spectral_bgrad(e["bmag"][:-1], gp)
    bgf = float(f["gradpar"][0]) * np.gradient(np.log(f["bmag"]), f["theta"], edge_order=2)
    facs = [2 * x["shat"] * x["gds21"][0] / x["gds22"][0] for x in (e, f)]
    print(f"\n[{NAME[wout]} a{alpha:.3f} N_z 48 vs 960] cut {e['theta_cut']:.6f} vs {f['theta_cut']:.6f}; gradpar "
          f"{gp:.6f} vs {float(f['gradpar'][0]):.6f}; twist fac {facs[0]:+.5f} vs {facs[1]:+.5f}")
    print(f"[{NAME[wout]} a{alpha:.3f} N_z 48 vs 960] " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()) +
          f"; bgrad (spectral, 48) {_rel(bg48, on(bgf)[:-1]):.2e}")
    assert np.isfinite(errs["bmag"])


# ------------------------------------------------------------------ L_x not consistent with the twist
def test_vmec_simulation_accepts_a_non_integer_twist():
    vt = _vt(QA)
    from velcro.config import VelcroConfig
    from velcro.simulation import Simulation
    Ly = 2 * np.pi / 0.3
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=12, Lx=30.0, Ny=6, Nz=48, L=4, M=2),
        geometry=dict(kind="vmec", file=QA, Ly=Ly, s0=0.5, vmec_cut="fix aspect", Lref_over_a=1.0, rho_star=1e-5),
        physics=dict(omega_n=dict(value=1.0), omega_T=dict(value=3.0), nonlinear=False, Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0.0, hyper_z=0.0, krook_rate=0.0, krook_width=0.0, integrator="rk4"),
        init=dict(kind="noise", amplitude=1e-3, seed=1), run=dict(t_end=0.1, precision="fp64")))
    # since the fix (simulation.py): a flux-tube kind refuses a box whose twist S k_y,min L_x is not an integer
    with pytest.raises(ValueError, match="twist"):
        Simulation(cfg)
    S = vt.from_vmec_tube(QA, Nx=12, Lx=30.0, Ny=6, Nz=48, Ly=Ly, s0=0.5, cut="fix aspect", Lref_over_a=1.0).extra["twist_shat"]
    good = VelcroConfig.model_validate(dict(cfg.model_dump(), grid=dict(cfg.grid.model_dump(), Lx=3.0 / (S * 0.3))))
    sim = Simulation(good)
    w = float(sim.geom.extra["twist_shat"]) * 0.3 * sim.cfg.grid.Lx
    print(f"\n[twist] refused at S k_y,min L_x = {S * 0.3 * 30.0:.5f}; built at {w:.6f}")
    assert abs(w - 3.0) < 1e-9
