"""Native Miller local equilibrium (``geometry.kind = "miller"``): the flux-tube arrays of a shaped tokamak surface from
the Miller parameters, in GX's array convention, mapped onto Velcro's :class:`~velcro.geometry.Geometry` by
:func:`velcro.gx_geometry.from_gx_arrays` (the calibrated term-by-term map, so a native Miller tube and a ``gx_file`` tube
of the same surface run the same equations).

The model (Miller, Chu, Greene, Lin-Liu & Waltz 1998 PoP 5, 973; the local-equilibrium closure in the Bishop / Mercier–Luc
form of GS2's ``leq`` module, which GX's ``geometry_modules/miller/gx_geo.py`` follows):

* surface shape  R(ρ, t) = R₀(ρ) + ρ cos(t + arcsin δ(ρ) sin t),  Z(ρ, t) = κ(ρ) ρ sin t,  ρ = r/a, with R₀' = −|shift|,
  κ' = ``akappri``, δ' = ``tripri`` (derivatives in ρ; GX's convention, R₀ decreasing outward for shift > 0);
* B = F ∇φ + ∇φ × ∇ψ with F = R_geo B_ref a (normalised F = ``R_geo``), ψ'(ρ) fixed by q through
  q = (F/2π) ∮ J/(R ψ') dt (J = R_ρ Z_t − Z_ρ R_t), with GX's sign ψ' < 0;
* the radial variation of the equilibrium that the drifts and the local shear need (F' = dF/dψ, the Shafranov-shift
  consistency) follows from the Grad–Shafranov equation on the surface: the integrated local shear is
  a_s F' + b_s p' − c_s (Bishop's a, b, c integrals along the surface), and ŝ fixes F' through Miller's eq. (21);
* the arrays: bmag, gradpar (equal-arc θ, constant), gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift (= gbdrift + the
  2p'/B² piece), grho, drhodpsi = |1/ψ'|, kxfac = |q ψ'/ρ|, rmaj = (R_max + R_min)/2 — GS2's definitions, GX's names.

Numerics: everything is evaluated on a fine uniform grid of the surface parameter t over the full period with analytic
ρ-derivatives of the shape (R_ρ, Z_ρ, R_tρ, Z_tρ, R_tt, Z_tt), spectral (FFT) antiderivatives for the surface integrals
and a periodic cubic spline for the equal-arc resampling (4th order in ``n_fine``: 1.7e-6 at 128, 4e-9 at 512, 1e-11 from
2048).  An independent Grad–Shafranov reference (tests/attack/miller_gs_ref.py: no Bishop integrals, no Miller eq. 21)
reproduces every array to ≤ 4.3e-9 on ten surfaces (shift, R_geo, ŝ < 0, δ 0.5, β′, low aspect ratio).  The differences
from GX's module (second-order differences on N_z/2 + 1 points, three surfaces δρ = 10⁻³ apart, linear interpolation onto
the equal-arc grid) are GX's discretisation errors: 1e-2 to 2e-1 of each array's maximum at ntheta 24 (the test tube: gds2 3.8e-2,
gbdrift 9.3e-2; δ 0.5: gbdrift 0.20), falling as ntheta⁻² onto these arrays with no floor (bgrad only as ntheta⁻¹: GX
differentiates a linearly interpolated bmag) — GX's own module run to ntheta 768 (the validation notes,
tests/test_miller.py).  q ≤ 0 and surfaces reaching R ≤ 0 are refused (the |q| tube is the physical
one).  Up-down asymmetric shapes are not parametrised (Miller's are symmetric).

β' ≠ 0 makes cvdrift ≠ gbdrift (the curvature drift carries the pressure gradient, cv_p = 2B p'/|ψ'| B³ … above; β'
also enters F' and so aprime, gds2, gds21 and gbdrift through Miller's relation): ``from_gx_arrays`` carries the excess as
``Ky_cv`` (on the v∥² part of the drift only), as for a ``gx_file`` tube.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from velcro.geometry import Geometry
from velcro.gx_geometry import _spectral_bgrad, from_gx_arrays

ARRAYS_EIK = ("bmag", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "cvdrift", "cvdrift0", "grho")


def _antiderivative(f: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """∫_0^t f dt' on a uniform periodic grid (N points over ``period``), spectrally: the mean part linearly, the rest by
    division by ik.  Returns F with F[0] = 0."""
    N = len(f)
    t = period * np.arange(N) / N
    mean = f.mean()
    g = f - mean
    k = 2 * np.pi * np.fft.fftfreq(N, d=period / N)
    gh = np.fft.fft(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gh = np.where(k != 0, gh / (1j * k), 0.0)
    G = np.real(np.fft.ifft(Gh))
    G = G - G[0]
    return G + mean * t


def miller_surface(t: np.ndarray, rhoc: float, Rmaj: float, shift: float, akappa: float, akappri: float, tri: float,
                   tripri: float) -> dict:
    """R, Z and their analytic derivatives (ρ, t, tt, tρ) on the parameter grid ``t``."""
    x = np.arcsin(tri)
    xr = tripri / np.sqrt(1.0 - tri ** 2)
    s_, c_ = np.sin(t), np.cos(t)
    th = t + x * s_
    S, C = np.sin(th), np.cos(th)
    dR0 = -abs(shift)
    R = Rmaj + rhoc * C
    Z = akappa * rhoc * s_
    R_t = -rhoc * S * (1.0 + x * c_)
    Z_t = akappa * rhoc * c_
    R_r = dR0 + C - rhoc * S * s_ * xr
    Z_r = (akappri * rhoc + akappa) * s_
    R_tt = -rhoc * (C * (1.0 + x * c_) ** 2 - S * x * s_)
    Z_tt = -akappa * rhoc * s_
    R_tr = -S * (1.0 + x * c_) - rhoc * (C * s_ * xr * (1.0 + x * c_) + S * c_ * xr)
    Z_tr = (akappri * rhoc + akappa) * c_
    return dict(R=R, Z=Z, R_t=R_t, Z_t=Z_t, R_r=R_r, Z_r=Z_r, R_tt=R_tt, Z_tt=Z_tt, R_tr=R_tr, Z_tr=Z_tr)


def miller_fine(rhoc: float, Rmaj: float, R_geo: float, q: float, shat: float, shift: float = 0.0, akappa: float = 1.0,
                akappri: float = 0.0, tri: float = 0.0, tripri: float = 0.0, betaprim: float = 0.0,
                n_fine: int = 4096) -> dict:
    """The GS2/GX eikcoefs of the surface on the fine parameter grid, as functions of the straight-field-line angle
    θ_f ∈ [0, 2π) (t = 0 the outboard midplane) and of the equal-arc angle θ_eq; plus the surface scalars."""
    if not q > 0:
        # GX's formulas (after GS2's) fix ψ' < 0 with F > 0; a negative q only flips gbdrift0 and the aprime part of
        # gbdrift (gds2/gds21/gds22 stay), a tube no relabelling maps onto the q > 0 one — while the current-reversed
        # tokamak's tube is exactly the q > 0 one (tests/attack/test_attack_miller.py, the Grad–Shafranov reference)
        raise ValueError(f"geometry.kind = 'miller' needs q > 0 (got {q}): the flux tube of a current-reversed "
                         "up-down symmetric tokamak is that of |q|; pass |q|")
    N = int(n_fine)
    t = 2 * np.pi * np.arange(N) / N
    s = miller_surface(t, rhoc, Rmaj, shift, akappa, akappri, tri, tripri)
    R, Z = s["R"], s["Z"]
    if np.min(R) <= 0:
        raise ValueError(f"the Miller surface reaches R ≤ 0 (min R = {np.min(R):.4g}): rhoc too large for Rmaj")
    R_t, Z_t, R_r, Z_r = s["R_t"], s["Z_t"], s["R_r"], s["Z_r"]
    J = R_r * Z_t - Z_r * R_t                                   # (ρ, t) → (R, Z) Jacobian
    if np.any(J <= 0):
        raise ValueError("Miller surfaces intersect (Jacobian ≤ 0): the shift, κ′ or δ′ is too large for this ρ")
    l_t = np.sqrt(R_t ** 2 + Z_t ** 2)                          # dℓ/dt
    grad_rho = l_t / J                                          # |∇ρ|
    F = float(R_geo)
    # ψ' from q (GX's sign: ψ' < 0): q = (F/2π) ∮ J/(R ψ') dt
    dpsidrho = -(F / abs(2 * np.pi * q)) * abs(2 * np.pi * (J / R).mean())      # ∮ = 2π × the mean on the periodic grid
    drhodpsi = 1.0 / dpsidrho
    dpdpsi = 0.5 * betaprim * drhodpsi
    B_p = abs(dpsidrho) * grad_rho / R
    B_t = F / R
    B = np.sqrt(B_p ** 2 + B_t ** 2)
    dpsi_dr = dpsidrho * grad_rho                               # sign(ψ') |∇ψ|
    # straight-field-line angle: θ_f(t) = 2π ∫_0^t (J/R) / ∮ (J/R)
    theta_f = 2 * np.pi * _antiderivative(J / R) / (2 * np.pi * (J / R).mean())
    dthf_dt = (J / R) / (J / R).mean()
    dthf_dl = dthf_dt / l_t
    gradpar = -(1.0 / (R * B)) * dpsi_dr * dthf_dl              # b·∇θ_f (> 0)
    sin_u = Z_t / l_t
    curv = (R_t * s["Z_tt"] - Z_t * s["R_tt"]) / l_t ** 3     # 1/R_c, > 0 for a convex surface
    # Bishop's integrals in θ_f (dθ_f = dthf_dt dt)
    cum = lambda f: _antiderivative(f * dthf_dt)
    a_s = -(2 * q / F * theta_f + 2 * F * q * cum(1.0 / (R ** 2 * B_p ** 2)))
    b_s = -2 * q * cum(1.0 / B_p ** 2)
    c_s = 2 * q * cum((2 * sin_u / R - 2 * curv) / (R * B_p))
    full = lambda f: 2 * np.pi * np.mean(f * dthf_dt)           # the full-period value, ∮ … dθ_f
    a_pi, b_pi, c_pi = -(2 * q / F * 2 * np.pi + 2 * F * q * full(1.0 / (R ** 2 * B_p ** 2))), -2 * q * full(1.0 / B_p ** 2), \
        2 * q * full((2 * sin_u / R - 2 * curv) / (R * B_p))
    # Miller's relation for F′ from ŝ, over the full period (GX writes it at θ_f = π with the half-period integrals, the
    # same equation): a_s F′ + b_s p′ − c_s = −2π q ŝ/(ρ ψ′) at 2π ↔ the half-period form at π
    dFdpsi = (-shat / (rhoc * dpsidrho / (2 * np.pi * q)) * 2.0 - (b_pi * dpdpsi - c_pi)) / a_pi
    aprime = -R * B_p * (a_s * dFdpsi + b_s * dpdpsi - c_s) / (2 * abs(drhodpsi))
    dBdr = B_p / B * (-B_p * curv + dpdpsi * R - F ** 2 * sin_u / (R ** 3 * B_p))
    qprime = shat * q / rhoc
    dqdr = qprime * grad_rho
    B2 = B ** 2
    dB2_dl = np.real(np.fft.ifft(1j * 2 * np.pi * np.fft.fftfreq(N, d=2 * np.pi / N) * np.fft.fft(B2))) / l_t
    gds22 = qprime ** 2 * dpsi_dr ** 2
    gds21 = qprime * (-dpsi_dr) * aprime
    dtdr_st = (aprime * drhodpsi - dqdr * theta_f) / q
    gds2 = dpsidrho ** 2 * (1.0 / R ** 2 + (dqdr * theta_f) ** 2 + q ** 2 * (dtdr_st ** 2 + dthf_dl ** 2)
                            + 2 * q * dqdr * theta_f * dtdr_st)
    gbdrift0 = (1.0 / B2 ** 2) * dpsidrho * F / R * (dqdr * dB2_dl)
    # gbdrift = gb_a + aprime · gb_b: gb_a, gb_b periodic, aprime secular (∝ q′θ_f + periodic)
    gb_a = 1.0 / abs(drhodpsi * B ** 3) * (2 * B2 * dBdr / dpsi_dr)
    gb_b = 1.0 / abs(drhodpsi * B ** 3) * (drhodpsi * F / R * dB2_dl / B)
    gbdrift = gb_a + aprime * gb_b
    cv_p = 1.0 / abs(drhodpsi * B ** 3) * (2 * B * dpdpsi)
    cvdrift = cv_p + gbdrift
    # GS2's Coriolis-drift arrays (geometry.f90 'drift': cdrift = −4 ψ′ Ẑ·∇α/B, cdrift0 = −4 ψ′ q′ Ẑ·∇ρ/B; not halved),
    # in THIS module's (GX's) sign convention.  ∇α in the (ê_ρ = ∇ρ/|∇ρ|, ê_l, ê_φ) basis of gds2/gds21 above:
    # (−aprime/ψ′, −q dθ_f/dl, 1/R); Ẑ·ê_ρ = −R_t/ℓ_t, Ẑ·ê_l = Z_t/ℓ_t.  GX's drift arrays carry the opposite overall sign
    # to GS2's formulas written with these symbols (at large aspect ratio gbdrift0 above is +2ψ′q′Ẑ·∇ρ/R, GS2's
    # use_large_aspect form −2ψ′q′Ẑ·∇ρ/R), so cdrift takes that sign too: GS2's own ratio cdrift/cvdrift = 2R/B at large
    # aspect ratio then holds here (tests/test_coriolis.py), and the Coriolis : magnetic drift ratio is GS2's
    cd_a = 4 * dpsidrho * (-q * dthf_dl * Z_t / l_t) / B
    cd_b = 4 * dpsidrho * (drhodpsi * R_t / l_t) / B
    cdrift = cd_a + aprime * cd_b
    cdrift0 = 4 * dpsidrho * qprime * (-R_t / J) / B
    grho = drhodpsi * dpsi_dr
    # equal-arc angle: b·∇θ_eq constant = 2π/∮ dθ_f/gradpar
    inv = 1.0 / gradpar
    gradpar_eq = 2 * np.pi / full(inv)
    theta_eq = gradpar_eq * cum(inv)
    return dict(t=t, theta_f=theta_f, theta_eq=theta_eq, bmag=B, gradpar=gradpar, gradpar_eq=float(gradpar_eq),
                gds2=gds2, gds21=gds21, gds22=gds22, gbdrift=gbdrift, gbdrift0=gbdrift0, cvdrift=cvdrift,
                cvdrift0=gbdrift0, grho=grho, R=R, Z=Z, aprime=aprime, dBdr=dBdr, cdrift=cdrift, cdrift0=cdrift0,
                # the periodic ingredients of the secular arrays (for the resampling in θ_eq)
                dtdr_st=dtdr_st, dqdr=dqdr, dthf_dl=dthf_dl, gb_a=gb_a, gb_b=gb_b, cv_p=cv_p, cd_a=cd_a, cd_b=cd_b,
                dpsidrho=float(dpsidrho), drhodpsi_abs=float(abs(drhodpsi)), kxfac=float(abs(q * dpsidrho / rhoc)),
                rmaj=float(0.5 * (R.max() + R.min())), q=float(q), shat=float(shat), dFdpsi=float(dFdpsi),
                dpdpsi=float(dpdpsi), F=F)


def miller_gx_arrays(Nz: int, rhoc: float, Rmaj: float, R_geo: float, q: float, shat: float, shift: float = 0.0,
                     akappa: float = 1.0, akappri: float = 0.0, tri: float = 0.0, tripri: float = 0.0,
                     betaprim: float = 0.0, n_fine: int = 4096) -> dict:
    """GX-internal arrays (as :func:`velcro.gx_geometry.read_gx_geometry` returns them: drifts halved, N_z points on
    θ_k = −π + 2πk/N_z of the equal-arc angle, spectral bgrad, jacobian) of the Miller surface.

    The periodic quantities are resampled in θ_eq by periodic cubic splines; the secular ones (gds2, gds21, gbdrift,
    cvdrift carry q′θ_f and q′²θ_f² parts) are rebuilt from their periodic ingredients and θ_f(θ_eq), continued
    secularly, so the arrays at θ = ±π are the two ends of one poloidal turn (gds21(−π) = −gds21(π), as GX's)."""
    f = miller_fine(rhoc, Rmaj, R_geo, q, shat, shift, akappa, akappri, tri, tripri, betaprim, n_fine)
    th_eq = f["theta_eq"]                                       # increasing from 0 towards 2π over the period
    z = -np.pi + 2 * np.pi * np.arange(Nz) / Nz
    zz = np.mod(z, 2 * np.pi)
    x = np.append(th_eq, 2 * np.pi)
    sp = lambda y: CubicSpline(x, np.append(y, y[0]), bc_type="periodic")(zz)
    out = {k: sp(f[k]) for k in ("bmag", "gds22", "gbdrift0", "grho", "R", "Z", "dtdr_st", "dqdr", "dthf_dl", "gb_a",
                                 "gb_b", "cv_p", "cd_a", "cd_b", "cdrift0")}
    thf = z + sp(f["theta_f"] - th_eq)                          # θ_f at the samples, secular (θ_f − θ_eq is periodic)
    dpsi = f["dpsidrho"]
    qprime = shat * q / rhoc
    dq, dtdr, dl_, Rz = out["dqdr"], out["dtdr_st"], out["dthf_dl"], out["R"]
    aprime = (q * dtdr + dq * thf) * dpsi                       # from dtdr_st = (aprime/ψ′ − q′|∇ρ| θ_f)/q
    gds2 = dpsi ** 2 * (1.0 / Rz ** 2 + (dq * thf) ** 2 + q ** 2 * (dtdr ** 2 + dl_ ** 2) + 2 * q * dq * thf * dtdr)
    gds21 = qprime * (-dpsi * out["grho"]) * aprime
    gbdrift = out["gb_a"] + aprime * out["gb_b"]
    cvdrift = out["cv_p"] + gbdrift
    gx = dict(bmag=out["bmag"], gds2=gds2, gds21=gds21, gds22=out["gds22"], grho=out["grho"],
              gbdrift=0.5 * gbdrift, gbdrift0=0.5 * out["gbdrift0"], cvdrift=0.5 * cvdrift, cvdrift0=0.5 * out["gbdrift0"],
              # GS2's Coriolis arrays, NOT halved (GX has none): from_gx_arrays maps them like gbdrift / gbdrift0
              cdrift=out["cd_a"] + aprime * out["cd_b"], cdrift0=out["cdrift0"])
    gx.update(theta=z, gradpar=f["gradpar_eq"], shat=float(shat), q=float(q), rmaj=f["rmaj"], kxfac=f["kxfac"],
              drhodpsi=f["drhodpsi_abs"], nperiod=1, source="miller", R=out["R"], Z=out["Z"], theta_f=thf)
    gx["bgrad"] = _spectral_bgrad(gx["bmag"], gx["gradpar"])
    gx["jacobian"] = 1.0 / np.abs(gx["drhodpsi"] * gx["gradpar"] * gx["bmag"])
    return gx


def from_miller(Nx: int, Lx: float, Ny: int, Nz: int, Ly: float, rhoc: float, Rmaj: float, q: float, shat: float,
                R_geo: float | None = None, shift: float = 0.0, akappa: float = 1.0, akappri: float = 0.0,
                tri: float = 0.0, tripri: float = 0.0, betaprim: float = 0.0, rho_star: float = 1e-5,
                Lref_over_a: float | None = None, x_offset: float = 0.0, n_fine: int = 4096) -> Geometry:
    """The Velcro flux-tube geometry of a Miller surface (see the module docstring; ``Ly``, ``Lx``, ``Lref_over_a`` as
    :func:`velcro.gx_geometry.from_gx_arrays`; L_ref = Rmaj·a by default)."""
    gx = miller_gx_arrays(Nz, rhoc, Rmaj, Rmaj if R_geo is None else R_geo, q, shat, shift, akappa, akappri, tri, tripri,
                          betaprim, n_fine)
    geom = from_gx_arrays(gx, Nx, Lx, Ny, Nz, Ly, rho_star=rho_star, Lref_over_a=Rmaj if Lref_over_a is None else Lref_over_a,
                          x_offset=x_offset)
    geom.name = "miller"
    geom.extra.update(miller=dict(rhoc=rhoc, Rmaj=Rmaj, R_geo=Rmaj if R_geo is None else R_geo, q=q, shat=shat, shift=shift,
                                  akappa=akappa, akappri=akappri, tri=tri, tripri=tripri, betaprim=betaprim),
                      R=gx["R"], Z=gx["Z"])
    return geom
