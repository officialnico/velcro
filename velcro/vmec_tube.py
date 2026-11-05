"""A flux tube of a VMEC equilibrium in Boozer coordinates (``geometry.kind = "vmec"``): GX's ``pyvmec`` geometry
(``geometry_modules/pyvmec/gx_geo_vmec.py``, Gaur & Mandell, after Landreman's simsopt ``vmec_fieldlines``) re-implemented
on the Boozer representation of ``booz_xform`` (Landreman), producing GX's arrays on GX's grid, mapped onto Velcro's
:class:`~velcro.geometry.Geometry` by :func:`velcro.gx_geometry.from_gx_arrays` — so a Velcro tube and a GX tube of the
same field line run the same equations in the same geometry (``tests/test_vmec_tube.py``).

The construction (no self-consistent shear or pressure variation; ``sfac = pfac = 1`` of GX's module):

* ``booz_xform`` gives, on VMEC's half grid, the Boozer harmonics R_mn, Z_mn, ν_mn (φ_cyl = φ_B − ν), the Jacobian
  √g_B = (G + ιI)/B² harmonics g_mn, |B| harmonics b_mn, and the covariant currents G(s), I(s); each is splined in s;
* the field line α = θ_B − ι φ_B at s₀ is followed in the Boozer poloidal angle θ_B ∈ [−npol π, npol π] (``ntheta`` + 1
  points), φ_B = (θ_B − α)/ι; the Cartesian derivatives of (R, Z, φ_cyl) give ∇ψ, ∇θ_B, ∇φ_B by the dual relations and
  ∇α = ∇θ_B − ι∇φ_B − (φ_B − ζ_c) ι′ ∇ψ/ψ′_edge (ζ_c = −α/ι, GX's ``shift_grad_alpha``);
* the normal and geodesic curvatures from the Boozer form of B·∇B and the √g_B harmonics (Hegna–Nakajima form, as
  GX): κ_n, κ_g; B×κ·∇α = (κ_n + κ_g L₁)B², B×κ·∇ψ = κ_g B², L₁ = ∇α·∇ψ/|∇ψ|² the integrated local shear;
* GX's arrays: bmag = B/B_ref, gds2 = |∇α|² L_ref² s, gds21 = (∇α·∇ψ) ŝ/B_ref, gds22 = |∇ψ|² ŝ²/(L_ref² B_ref² s),
  gbdrift, gbdrift0, cvdrift, grho, with B_ref = 2|ψ_edge|/L_ref² and L_ref = the VMEC minor radius ``Aminor_p``,
  ŝ = −(2s/ι) dι/ds, ψ_edge = −Φ_edge/2π (GX's sign);
* the flux-tube cut of GX's ``boundary``: ``"fix aspect"`` ends the tube where jtwist = 2ŝ (gds21/gds22)(x0/y0) is an
  integer (cubic-spline roots, GX's search over ±30 or the deck's jtwist; the last crossing by default), ``"gds21"`` and
  ``"gbdrift0"`` where those vanish, ``"none"`` keeps the full ±npol π; the arrays are resampled on the cut by cubic
  splines (GX's ``InterpolatedUnivariateSpline``);
* the equal-arc angle θ ∈ [−π, π]: gradpar constant = 2π/∫dθ_B/gradpar(θ_B), the arrays interpolated linearly onto
  the uniform grid (GX's ``np.interp``), drhodpsi = 1/(2√s ψ_edge), kxfac = 1, Rmaj = (R_max + R_min)/2 of the tube.

GX's interpolation choices are kept on purpose: the tube is then GX's own, and Velcro's arrays agree with GX's
``eik.nc`` of the same deck to the precision of ``booz_xform`` and of the splines.  An independent
check of the Boozer construction is the axisymmetric limit and the agreement of |B| along the line with the VMEC
(PEST) evaluation of :class:`velcro.geometry.FourierEquilibrium`.

Needs ``booz_xform`` and ``netCDF4`` (``uv pip install booz_xform netCDF4``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from velcro.geometry import Geometry
from velcro.gx_geometry import _from_eik_points, from_gx_arrays

MU0 = 4e-7 * np.pi


class _Struct:
    pass


def _splines(wout: str, mboz: Optional[int], nboz: Optional[int]):
    """booz_xform of the wout and the radial splines of GX's ``vmec_splines``."""
    try:
        import booz_xform as bx
        from netCDF4 import Dataset
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    except ImportError as e:  # pragma: no cover
        raise ImportError("geometry.kind = 'vmec' needs booz_xform and netCDF4 (uv pip install booz_xform netCDF4)") from e
    b = bx.Booz_xform()
    b.verbose = 0
    b.read_wout(wout)
    b.mboz = int(2 * b.mpol) if mboz is None else int(mboz)
    b.nboz = int(2 * b.ntor) if nboz is None else int(nboz)
    b.run()
    nc = Dataset(wout)
    v = lambda k: np.asarray(nc.variables[k][:])
    # booz_xform's stellarator-asymmetric harmonics are not used below, and the dual relations assume VMEC's usual
    # left-handed (signgs = −1) coordinates: refuse what would silently be wrong (VMEC review)
    if "lasym__logical__" in nc.variables and int(v("lasym__logical__")) != 0:
        nc.close()
        raise ValueError(f"{wout}: lasym = true (stellarator-asymmetric) wouts are not supported by geometry.kind = 'vmec'")
    if "signgs" in nc.variables and int(v("signgs")) != -1:
        nc.close()
        raise ValueError(f"{wout}: signgs = {int(v('signgs'))}; geometry.kind = 'vmec' assumes VMEC's signgs = −1")
    ns = int(v("ns"))
    s_full = np.linspace(0, 1, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    r = _Struct()
    arrs = dict(rmnc_b=np.asarray(b.rmnc_b).T, zmns_b=np.asarray(b.zmns_b).T, numns_b=np.asarray(b.numns_b).T,
                gmnc_b=np.asarray(b.gmnc_b).T, bmnc_b=np.asarray(b.bmnc_b).T)
    for k, a in arrs.items():
        if a.shape[0] != len(s_half):
            raise ValueError(f"booz_xform {k} has {a.shape[0]} surfaces, the wout half grid {len(s_half)}")
        setattr(r, k, [IUS(s_half, a[:, j]) for j in range(a.shape[1])])
    r.Gfun = IUS(s_half, np.asarray(b.Boozer_G))
    r.Ifun = IUS(s_half, np.asarray(b.Boozer_I))
    r.pressure = IUS(s_half, v("pres")[1:])
    r.iota = IUS(s_half, v("iotas")[1:])
    r.phiedge = float(v("phi")[-1])
    r.Aminor_p = float(v("Aminor_p"))
    r.nfp = int(v("nfp"))
    r.raxis_cc = v("raxis_cc")
    r.mpol, r.ntor = int(v("mpol")), int(v("ntor"))
    r.xm_b, r.xn_b = np.asarray(b.xm_b, float), np.asarray(b.xn_b, float)
    r.mnboz = int(b.mnboz)
    nc.close()
    return r


def boozer_fieldline(vs, s0: float, alpha: float, theta_b: np.ndarray) -> dict:
    """GX's ``vmec_fieldlines`` for one surface and one field line, without the shear / pressure variation: the eikcoefs
    on the Boozer angle grid ``theta_b`` (GX's names, GX's normalisation)."""
    s = float(s0)
    iota = float(vs.iota(s))
    diota = float(vs.iota.derivative()(s))
    dp_ds = float(vs.pressure.derivative()(s))
    shat = -2 * s / iota * diota
    sqrt_s = np.sqrt(s)
    zeta_center = -alpha / iota
    L = vs.Aminor_p
    psi_edge = -vs.phiedge / (2 * np.pi)                                   # edge_toroidal_flux_over_2pi
    tsign = np.sign(psi_edge)
    B_ref = 2 * abs(psi_edge) / L ** 2
    xm, xn = vs.xm_b, vs.xn_b
    G, dG = float(vs.Gfun(s)), float(vs.Gfun.derivative()(s))
    I, dI = float(vs.Ifun(s)), float(vs.Ifun.derivative()(s))
    coef = lambda spl, d=0: np.array([f.derivative()(s) if d else f(s) for f in spl])
    rmnc, drmnc = coef(vs.rmnc_b), coef(vs.rmnc_b, 1)
    zmns, dzmns = coef(vs.zmns_b), coef(vs.zmns_b, 1)
    numns, dnumns = coef(vs.numns_b), coef(vs.numns_b, 1)
    gmnc = coef(vs.gmnc_b)
    bmnc, dbmnc = coef(vs.bmnc_b), coef(vs.bmnc_b, 1)
    th = np.asarray(theta_b, float)
    ph = (th - alpha) / iota
    ang = xm[:, None] * th[None, :] - xn[:, None] * ph[None, :]
    c, sn = np.cos(ang), np.sin(ang)
    mc, ncn, ms, nsn = xm[:, None] * c, xn[:, None] * c, xm[:, None] * sn, xn[:, None] * sn
    R = rmnc @ c; R_s = drmnc @ c; R_t = -(rmnc @ ms); R_p = rmnc @ nsn
    Z = zmns @ sn; Z_s = dzmns @ sn; Z_t = zmns @ mc; Z_p = -(zmns @ ncn)
    nu = numns @ sn; nu_s = dnumns @ sn; nu_t = numns @ mc; nu_p = -(numns @ ncn)
    sqrtg = gmnc @ c; sqrtg_t = -(gmnc @ ms); sqrtg_p = gmnc @ nsn
    modB = bmnc @ c; B_s = dbmnc @ c
    Vprime = gmnc[0]
    delmnc = np.zeros_like(gmnc); delmnc[1:] = gmnc[1:] / Vprime
    denom = xm[1:] * iota - xn[1:]
    betamns = np.zeros_like(gmnc)
    betamns[1:] = delmnc[1:] / psi_edge * MU0 * dp_ds * Vprime / denom
    beta_b = betamns @ sn
    # Cartesian derivatives (X = R cos φ_cyl, Y = R sin φ_cyl, φ_cyl = φ_B − ν)
    phc = ph - nu
    cp, sp = np.cos(phc), np.sin(phc)
    X_t = R_t * cp - R * sp * (-nu_t); X_p = R_p * cp - R * sp * (1 - nu_p); X_s = R_s * cp - R * sp * (-nu_s)
    Y_t = R_t * sp + R * cp * (-nu_t); Y_p = R_p * sp + R * cp * (1 - nu_p); Y_s = R_s * sp + R * cp * (-nu_s)
    # dual relations
    gpsi = np.stack([(Y_t * Z_p - Z_t * Y_p), (Z_t * X_p - X_t * Z_p), (X_t * Y_p - Y_t * X_p)]) / sqrtg
    gth = np.stack([(Y_p * Z_s - Z_p * Y_s), (Z_p * X_s - X_p * Z_s), (X_p * Y_s - Y_p * X_s)]) / (sqrtg * psi_edge)
    gph = np.stack([(Y_s * Z_t - Z_s * Y_t), (Z_s * X_t - X_s * Z_t), (X_s * Y_t - Y_s * X_t)]) / (sqrtg * psi_edge)
    galpha = -(ph - zeta_center) * diota * gpsi / psi_edge + gth - iota * gph
    gpp = np.sum(gpsi ** 2, axis=0)
    gap = np.sum(galpha * gpsi, axis=0)
    L0 = -(gap / gpp + diota * (ph - zeta_center) / psi_edge)
    L1 = gap / gpp
    GI = G + iota * I
    # The Pfirsch–Schlüter (β_mn) part of the normal curvature enters with the sign that b·∇b gives when the field line
    # is differentiated directly (tests/test_vmec_tube.py; W7-X at β 5 %: 3e-2 against 0.17 with the other sign).  GX's
    # module at commit 3865a53 has this sign (its W7-X eik.nc is reproduced to 1e-5); the December-2024
    # revision of gx_geo_vmec.py carries the opposite one.
    kappa_n = (1.0 / modB ** 2 * (modB * B_s + MU0 * dp_ds) / psi_edge + beta_b / (2 * sqrtg * GI) * sqrtg_p
               + L0 * (G * sqrtg_t - I * sqrtg_p) / (2 * sqrtg * GI))
    kappa_g = (G * sqrtg_t - I * sqrtg_p) / (2 * sqrtg * GI)
    Bxk_alpha = (kappa_n + kappa_g * L1) * modB ** 2
    Bxk_psi = kappa_g * modB ** 2
    gaa = modB ** 2 / gpp + gpp * L1 ** 2
    gap_b = gpp * L1
    out = dict(theta_b=th, phi_b=ph, R=R, Z=Z, bmag=modB / B_ref, gradpar=np.abs(-L / modB / sqrtg * iota),
               gds2=gaa * L ** 2 * s, gds21=gap_b * shat / B_ref, gds22=gpp * shat ** 2 / (L ** 2 * B_ref ** 2 * s),
               grho=np.sqrt(gpp / (L ** 2 * B_ref ** 2 * s)),
               gbdrift0=-Bxk_psi * 2 * shat / (modB ** 2 * sqrt_s) * tsign,
               cvdrift=-2 * B_ref * L ** 2 * sqrt_s * Bxk_alpha / modB ** 2 * tsign)
    out["gbdrift"] = out["cvdrift"] + 2 * B_ref * L ** 2 * sqrt_s * MU0 * dp_ds * tsign / (psi_edge * modB ** 2)
    out["cvdrift0"] = out["gbdrift0"].copy()
    out.update(iota=iota, shat=shat, q=abs(1.0 / iota), psi_edge=psi_edge, L_ref=L, B_ref=B_ref, nfp=vs.nfp,
               dpsidrho=2 * sqrt_s * psi_edge, theta_pest=th - iota * nu, modB=modB, nu=nu, grad_alpha=galpha,
               grad_psi=gpsi, Bxk_alpha=Bxk_alpha, Bxk_psi=Bxk_psi, kappa_n=kappa_n, kappa_g=kappa_g, beta_b=beta_b,
               beta_term=beta_b / (2 * sqrtg * GI) * sqrtg_p)
    return out


def _cut(theta: np.ndarray, fl: dict, cut: str, x0_over_y0: float, jtwist: Optional[int], which: int) -> np.ndarray:
    """GX's flux-tube cut: the half-length θ_c of the tube in the Boozer angle (None = no cut)."""
    from scipy.interpolate import CubicSpline, PPoly, splrep
    if cut == "none":
        return None
    if cut in ("gds21", "gbdrift0"):
        pp = PPoly.from_spline(splrep(theta, fl[cut], s=0))
        roots = np.sort(pp.roots(extrapolate=False))
        return float(roots[which])
    if cut == "fix aspect":
        jt = 2.0 * fl["shat"] * fl["gds21"] / fl["gds22"] * x0_over_y0
        spl = CubicSpline(theta, jt)
        vals = np.array([-jtwist, jtwist]) if jtwist is not None else np.arange(-30, 30)
        vals = vals[(vals < -0.1) | (vals > 0.1)]
        cr = np.sort(np.concatenate([spl.solve(float(v), extrapolate=False) for v in vals]))
        if cr.size == 0:
            raise ValueError("vmec 'fix aspect' cut: no θ_B where jtwist is an integer")
        return float(cr[which])
    raise ValueError(f"unknown vmec cut {cut!r}")


def vmec_tube_eik(wout: str, Nz: int, s0: float, alpha: float = 0.0, npol: int = 1, cut: str = "none",
                  x0_over_y0: float = 1.0, jtwist: Optional[int] = None, which_crossing: int = -1,
                  mboz: Optional[int] = None, nboz: Optional[int] = None) -> dict:
    """GX's ``eik.nc`` content of the tube (N_z + 1 points on the equal-arc θ ∈ [−π, π], drifts unhalved) plus the
    scalars, exactly as GX's pyvmec module writes them."""
    from scipy.integrate import cumulative_trapezoid as ctrap
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    vs = _splines(wout, mboz, nboz)
    theta = np.linspace(-npol * np.pi, npol * np.pi, Nz + 1)
    fl = boozer_fieldline(vs, s0, alpha, theta)
    keys = ("bmag", "gradpar", "cvdrift", "cvdrift0", "gbdrift", "gbdrift0", "gds2", "gds21", "gds22", "grho", "R", "Z")
    arr = {k: np.asarray(fl[k], float) for k in keys}
    thc = _cut(theta, fl, cut, x0_over_y0, jtwist, which_crossing)
    if thc is not None:
        tc = np.linspace(-thc, thc, Nz + 1)
        arr = {k: IUS(theta, a)(tc) for k, a in arr.items()}
        theta = tc
    inv = ctrap(1.0 / arr["gradpar"], theta, initial=0)
    gradpar_eq = 2 * np.pi / inv[-1]
    th_eq = gradpar_eq * inv - np.pi
    th_gx = np.linspace(-np.pi, np.pi, Nz + 1)
    out = {k: np.interp(th_gx, th_eq, a) for k, a in arr.items() if k != "gradpar"}
    out["gradpar"] = np.full(Nz + 1, gradpar_eq)
    out["theta"] = th_gx
    out.update(shat=fl["shat"], q=fl["q"], iota=fl["iota"], drhodpsi=abs(1.0 / fl["dpsidrho"]), kxfac=1.0,
               rmaj=0.5 * (out["R"].max() + out["R"].min()), theta_cut=thc, theta_b_range=float(theta[-1]),
               domain_scaling=float(theta[-1] / th_eq[-1]), nfp=fl["nfp"], L_ref=fl["L_ref"], B_ref=fl["B_ref"],
               twist_shift_geo_fac=2.0 * fl["shat"] * out["gds21"][0] / out["gds22"][0])
    return out


def vmec_tube_gx_arrays(wout: str, Nz: int, **kw) -> dict:
    """GX-internal arrays (drifts halved, N_z points, spectral bgrad, jacobian) of :func:`vmec_tube_eik`."""
    e = vmec_tube_eik(wout, Nz, **kw)
    arrs = {k: e[k] for k in ("bmag", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "cvdrift", "cvdrift0", "grho")}
    sc = dict(drhodpsi=e["drhodpsi"], kxfac=e["kxfac"], shat=e["shat"], rmaj=e["rmaj"], q=e["q"])
    gx = _from_eik_points(e["theta"], arrs, e["gradpar"], sc, f"vmec:{wout}")
    gx.update(R=e["R"][:Nz], Z=e["Z"][:Nz], iota=e["iota"], theta_cut=e["theta_cut"], nfp=e["nfp"],
              twist_shift_geo_fac=e["twist_shift_geo_fac"], L_ref_m=e["L_ref"])
    return gx


def from_vmec_tube(wout: str, Nx: int, Lx: float, Ny: int, Nz: int, Ly: float, s0: float, alpha: float = 0.0,
                   npol: int = 1, cut: str = "none", x0_over_y0: float = 1.0, jtwist: Optional[int] = None,
                   which_crossing: int = -1, mboz: Optional[int] = None, nboz: Optional[int] = None,
                   rho_star: float = 1e-5, Lref_over_a: Optional[float] = None, x_offset: float = 0.0,
                   drift_rtol: float = 1e-4) -> Geometry:
    """The Velcro flux-tube geometry of the VMEC field line (``Ly``, ``Lx``, ``Lref_over_a`` as
    :func:`velcro.gx_geometry.from_gx_arrays`; L_ref = R = (R_max + R_min)/2 of the tube by default, i.e. ``Lref_over_a``
    = R/Aminor_p — GX writes rmaj in metres for VMEC, which as a ratio would make L_ref depend on the machine's size).  A finite-β equilibrium has
    cvdrift ≠ gbdrift (the curvature drift carries μ₀p′), which Velcro's single drift coefficient (2v̂² + μ̂) cannot
    represent: refused beyond ``drift_rtol`` of the drift's maximum, as ``gx_file`` does (vacuum fields pass)."""
    gx = vmec_tube_gx_arrays(wout, Nz, s0=s0, alpha=alpha, npol=npol, cut=cut, x0_over_y0=x0_over_y0, jtwist=jtwist,
                             which_crossing=which_crossing, mboz=mboz, nboz=nboz)
    if Lref_over_a is None:            # GX's VMEC rmaj is in metres: the default L_ref is the tube's R, i.e. R/a in units of a
        Lref_over_a = float(gx["rmaj"]) / float(gx["L_ref_m"])
    geom = from_gx_arrays(gx, Nx, Lx, Ny, Nz, Ly, rho_star=rho_star, Lref_over_a=Lref_over_a, x_offset=x_offset,
                          drift_rtol=drift_rtol)
    cv, gb = np.asarray(gx["cvdrift"]), np.asarray(gx["gbdrift"])
    geom.extra.update(vmec=dict(file=wout, s0=s0, alpha=alpha, npol=npol, cut=cut, iota=gx["iota"], nfp=gx["nfp"],
                                theta_cut=gx["theta_cut"]), cvdrift_minus_gbdrift_max=float(np.max(np.abs(cv - gb))),
                      R=gx["R"], Z=gx["Z"])
    geom.name = "vmec-tube"
    return geom
