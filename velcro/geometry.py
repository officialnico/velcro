"""Magnetic geometry in field-aligned coordinates (§2.3 of the design document).

Coordinates: ``x = a ρ_tor`` (radial, measured in ρ_ref), ``y = C_y (q θ* − φ)``
(field-line label, in ρ_ref), ``z = θ*`` (PEST poloidal angle).  The equilibrium
field is ``B = C(x) ∇x × ∇y``.

All quantities are dimensionless in the Velcro normalisation:

* perpendicular lengths in ``ρ_ref``, parallel lengths in ``L_ref``;
* ``ρ* = ρ_ref / L_ref`` is an explicit parameter (global code);
* ``B`` in ``B_ref``; ``gpar = L_ref b·∇z``; ``K_x = L_ref (b×∇ln B)·∇x``,
  ``K_y = L_ref (b×∇ln B)·∇y``; ``omega_B = -L_ref ∂_x ln B``.

Three constructors are provided:

* :func:`slab` — uniform field, for Landau-damping and slab-ITG tests;
* :func:`s_alpha` — analytic large-aspect-ratio circular tokamak with global
  ``q(x)``, ``B(x,z)`` variation (Cyclone Base Case);
* :class:`FourierEquilibrium` — general 3D equilibrium given by Fourier
  surfaces ``R_mn(s), Z_mn(s), λ_mn(s)`` and ``ι(s)`` (VMEC/GVEC output),
  metric computed numerically.  :func:`from_gvec` / :func:`from_vmec_wout`
  load those coefficients from equilibrium-solver output.

GX's own flux-tube arrays (``geometry.kind = "gx_file"``: s-α, Miller, VMEC as GX computed them) are mapped by
:mod:`velcro.gx_geometry`.

Geometry arrays carry a leading ``(Nx, Ny, Nz)`` shape.  Axisymmetric
geometries use ``Ny = 1`` (broadcastable), so the linear operators can be
applied directly in ``k_y`` space; 3D geometries carry the real-space ``y``
dependence and the solver applies them pseudo-spectrally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton


@dataclass
class Geometry:
    """Container of geometry coefficient fields on the ``(x, y, z)`` grid."""

    x: np.ndarray            # (Nx,) radial coordinate in ρ_ref, centred on 0
    y: np.ndarray            # (Ny,) real-space binormal grid in ρ_ref
    z: np.ndarray            # (Nz,) parallel angle, uniform on [-π, π)
    rho_star: float          # ρ_ref / L_ref
    Ly: float                # y period in ρ_ref
    B: np.ndarray            # (Nx,Ny,Nz)  |B| / B_ref
    gpar: np.ndarray         # (Nx,Ny,Nz)  L_ref b·∇z
    dlnBdz: np.ndarray       # (Nx,Ny,Nz)  ∂_z ln B  (b·∇ln B = gpar dlnBdz)
    omega_B: np.ndarray      # (Nx,Ny,Nz)  -L_ref ∂_x ln B
    gxx: np.ndarray          # (Nx,Ny,Nz)
    gxy: np.ndarray
    gyy: np.ndarray
    Kx: np.ndarray           # (Nx,Ny,Nz)  L_ref (b×∇ln B)·∇x
    Ky: np.ndarray           # (Nx,Ny,Nz)  L_ref (b×∇ln B)·∇y
    Cinv: np.ndarray         # (Nx,1,1)    B_ref / C(x): E×B bracket coefficient
    twist_phase: np.ndarray  # (Nx,)       2π q(x) C_y  (phase per unit k_y, in ρ_ref)
    jac: np.ndarray          # (Nx,Ny,Nz)  Jacobian ∝ 1/(B gpar), for flux-surface averages
    axisymmetric: bool = True
    name: str = "geometry"
    extra: dict = field(default_factory=dict)
    # β' ≠ 0: the curvature drift's excess over the ∇B drift, L_ref (b×κ − b×∇ln B)·∇x and ·∇y (Nx,Ny,Nz).  It acts on
    # the 2v̂² part of the drift only (the ∇B drift, Kx/Ky, acts on 2v̂² + μ̂); None = curvature drift = ∇B drift
    Kx_cv: Optional[np.ndarray] = None
    Ky_cv: Optional[np.ndarray] = None
    # GS2's Coriolis drift geometry (native Miller): L_ref-scaled maps of cdrift0/cdrift like Kx/Ky, per unit GS2 mach
    # (Ω_φ a/v_t,ref); the drift is mach ½K_co/B · k · v̂ (T/Z)/v̂t_s on H (physics.coriolis_mach).  None elsewhere
    Kx_co: Optional[np.ndarray] = None
    Ky_co: Optional[np.ndarray] = None

    @property
    def shape(self):
        return (len(self.x), len(self.y), len(self.z))

    def fs_weights(self) -> np.ndarray:
        """Flux-surface-average weights ``w(x, z)`` (y-averaged Jacobian), Σ_z w = 1."""
        w = self.jac.mean(axis=1)
        return w / w.sum(axis=1, keepdims=True)

    def y_averaged(self, arr: np.ndarray) -> np.ndarray:
        return arr.mean(axis=1, keepdims=True)


def _grids(Nx: int, Lx: float, Ny: int, Ly: float, Nz: int, x_offset: float = 0.0):
    x = (np.arange(Nx) - Nx / 2 + x_offset) * (Lx / Nx) if x_offset else (np.arange(Nx) - Nx / 2) * (Lx / Nx)
    y = np.arange(Ny) * (Ly / Ny)
    z = -np.pi + 2 * np.pi * np.arange(Nz) / Nz
    return x, y, z


# --------------------------------------------------------------------------
def slab(Nx: int, Lx: float, Ny: int, Ly: float, Nz: int, rho_star: float = 1e-3,
         Lz_over_Lref: float = 1.0) -> Geometry:
    """Uniform magnetic field ``B = B_ref ẑ``; ``z`` spans ``2π`` over a length
    ``Lz_over_Lref L_ref`` (so ``gpar = 2π / Lz``); periodic parallel BC."""
    x, y, z = _grids(Nx, Lx, Ny, Ly, Nz)
    shp = (Nx, 1, Nz)
    one = np.ones(shp)
    zero = np.zeros(shp)
    return Geometry(
        x=x, y=y, z=z, rho_star=rho_star, Ly=Ly,
        B=one, gpar=one * (2 * np.pi / Lz_over_Lref), dlnBdz=zero, omega_B=zero,
        gxx=one, gxy=zero, gyy=one, Kx=zero, Ky=zero,
        Cinv=np.ones((Nx, 1, 1)), twist_phase=np.zeros(Nx), jac=one,
        axisymmetric=True, name="slab",
    )


# --------------------------------------------------------------------------
def s_alpha(Nx: int, Lx: float, Ny: int, Nz: int, rho_star: float, r0: float,
            q0: float, shat: float, n0: int = 1, alpha_mhd: float = 0.0,
            q_profile=None, Lref_is_R0: bool = True, gpar_model: str = "velcro", x_offset: float = 0.0) -> Geometry:
    """Global analytic circular tokamak (``s–α`` model, ``α_MHD`` optional).

    ``L_ref = R_0``; ``r0`` is the reference minor radius in units of ``R_0``;
    ``q(r) = q0 (1 + ŝ (r − r0)/r0)`` unless ``q_profile(r)`` is given.
    ``B = 1/(1 + r cos z)``, ``b·∇z = 1/(q R)``, ``C_y = r0/q0``.
    The ``y`` box is ``L_y = 2π C_y / n0`` (``n0`` = minimum toroidal mode number).
    ``gpar_model = "constant"``: ``b·∇z = 1/(qR₀)`` (GX's s-α model) instead of ``1/(qR₀(1 + r cos z))``; only ``gpar``
    (hence the mirror coefficient and the Jacobian) changes.  ``x_offset``: the radial grid shifted by that fraction of Δx.
    """
    Cy = r0 / q0 / rho_star                      # in ρ_ref
    Ly = 2 * np.pi * Cy / n0
    x, y, z = _grids(Nx, Lx, Ny, Ly, Nz, x_offset)
    r = r0 + rho_star * x                        # in R0 units
    if r.min() <= 0.02 * r0:
        raise ValueError(f"radial box too wide for rho_star={rho_star}: r reaches {r.min():.3f} R0 "
                         f"(r0={r0}); reduce Lx or rho_star")
    if q_profile is None:
        q = q0 * (1 + shat * (r - r0) / r0)
        dq = q0 * shat / r0 * np.ones_like(r)
        if q.min() <= 0.2 * q0:
            raise ValueError("q(x) approaches zero inside the box; reduce Lx, rho_star or shat")
    else:
        q = q_profile(r)
        dq = np.gradient(q, r)
    R, Z = r[:, None], z[None, :]                # broadcast (Nx, Nz)
    cz, sz = np.cos(Z), np.sin(Z)
    B = 1.0 / (1.0 + R * cz)
    # With B = C ∇x×∇y, C > 0 and y = C_y(qθ − φ): B ∝ −(q/r) φ̂ − (1/R) θ̂, so b·∇θ = −1/(qR).
    gpar = -1.0 / (q[:, None] * (1.0 + R * cz)) * np.ones_like(B)
    if gpar_model == "constant":
        gpar = -1.0 / q[:, None] * np.ones_like(B)
    elif gpar_model != "velcro":
        raise ValueError(f"unknown s-alpha gpar model {gpar_model!r}")
    dlnBdz = R * sz / (1.0 + R * cz)
    omega_B = cz / (1.0 + R * cz)                # -R0 ∂_r ln B
    # metric with ∇x = ∇r, ∇y = C_y (q' z ∇r + q ∇θ − ∇φ)
    Cy_phys = r0 / q0
    shear_local = Cy_phys * dq                   # dimensionless (= ŝ at r0)
    # Shafranov-shift (α) term of the s–α model
    gxy = (shear_local[:, None] * Z - alpha_mhd * sz) * np.ones_like(B)
    gxx = np.ones_like(B)
    gyy = 1.0 + gxy**2
    # b×∇ln B with ∇ln B = -(cos z, sin z)/R in the (r, θ) basis at large aspect ratio.
    # Orientation: B = C ∇x×∇y with C > 0 and y = C_y(qθ − φ) puts b along −φ̂, so
    # b×∇ln B = −Ẑ/R and both drift coefficients are negative on the outboard side.  With the
    # ion diamagnetic frequency also negative in this convention (ω_*i ∝ ∂_x p / C), the two
    # drifts are aligned at the outboard midplane (bad curvature), as required for the ITG.
    Kx = -sz / (1.0 + R * cz) * np.ones_like(B)
    Ky = -(cz + gxy * sz) / (1.0 + R * cz)
    Cinv = (q * Cy_phys / r)[:, None, None] * np.ones((1, 1, 1))  # B_ref/C = q C_y / r ≈ 1
    twist = 2 * np.pi * q * Cy                   # phase per unit k_y
    jac = 1.0 / (B * gpar)
    add = lambda a: a[:, None, :]
    return Geometry(
        x=x, y=y, z=z, rho_star=rho_star, Ly=Ly,
        B=add(B), gpar=add(gpar), dlnBdz=add(dlnBdz), omega_B=add(omega_B),
        gxx=add(gxx), gxy=add(gxy), gyy=add(gyy), Kx=add(Kx), Ky=add(Ky),
        Cinv=Cinv, twist_phase=twist, jac=add(jac), axisymmetric=True, name="s-alpha",
        extra=dict(q=q, r=r, Cy=Cy, n0=n0),
    )


# --------------------------------------------------------------------------
class FourierEquilibrium:
    """General 3D MHD equilibrium from Fourier flux surfaces.

    Parameters
    ----------
    s : (Ns,) normalised toroidal flux grid of the coefficient arrays.
    xm, xn : (Nmn,) poloidal / toroidal mode numbers (``n`` includes the field-period factor).
    rmnc, zmns, lmns : (Ns, Nmn) cosine/sine Fourier coefficients of ``R``, ``Z``, ``λ``
        with ``R = Σ rmnc cos(mθ − nφ)``, ``Z = Σ zmns sin(mθ − nφ)``, ``λ = Σ lmns sin(mθ − nφ)``
        (stellarator-symmetric convention; non-symmetric terms ``rmns, zmnc, lmnc`` optional).
    iota : (Ns,) rotational transform.
    phi_edge : toroidal flux at the boundary (Wb); ``ψ = phi_edge s / 2π``.
    nfp : number of field periods.
    a : minor radius (m); ``x = a sqrt(s)``.
    B_ref, L_ref : reference field (T) and length (m).
    """

    def __init__(self, s, xm, xn, rmnc, zmns, lmns, iota, phi_edge, nfp, a, B_ref, L_ref,
                 rmns=None, zmnc=None, lmnc=None):
        self.s = np.asarray(s, float)
        self.xm = np.asarray(xm, float)
        self.xn = np.asarray(xn, float)
        self.nfp = int(nfp)
        self.a, self.B_ref, self.L_ref = float(a), float(B_ref), float(L_ref)
        self.phi_edge = float(phi_edge)
        # Coefficients are splined in ρ = sqrt(s): odd-m modes scale as ρ^m near the axis,
        # so this keeps the radial derivatives of |B| (a near-cancellation) accurate.
        rho = np.sqrt(self.s)
        bc = "not-a-knot"
        mk = lambda arr: CubicSpline(rho, np.asarray(arr, float), axis=0, bc_type=bc)
        zero = np.zeros_like(np.asarray(rmnc, float))
        self._R, self._Z, self._Lam = mk(rmnc), mk(zmns), mk(lmns)
        self._Rs = mk(zero if rmns is None else rmns)
        self._Zc = mk(zero if zmnc is None else zmnc)
        self._Lc = mk(zero if lmnc is None else lmnc)
        self._iota_rho = CubicSpline(rho, np.asarray(iota, float), bc_type=bc)

    def _iota(self, s, d=0):
        """ι(s) and dι/ds from the ρ-spline."""
        rho = np.sqrt(np.asarray(s, float))
        if d == 0:
            return self._iota_rho(rho)
        return self._iota_rho(rho, 1) / (2 * rho)

    def _coef(self, spl, s):
        """Coefficient array and its s-derivative from a ρ-spline."""
        rho = np.sqrt(np.asarray(s, float))
        return spl(rho), spl(rho, 1) / (2 * rho[..., None])

    # -- Fourier evaluation ------------------------------------------------
    def _eval(self, s, th, ph, d=0):
        """Return R, Z, λ and their (s, θ, φ) derivatives at points (broadcast arrays)."""
        s, th, ph = np.broadcast_arrays(np.asarray(s, float), np.asarray(th, float), np.asarray(ph, float))
        ang = self.xm * th[..., None] - self.xn * ph[..., None]
        c, sn = np.cos(ang), np.sin(ang)
        out = {}
        for name, Cc, Ss, cpart, spart in (
            ("R", self._R, self._Rs, c, sn), ("Z", self._Zc, self._Z, c, sn), ("L", self._Lc, self._Lam, c, sn)):
            A, dA = self._coef(Cc, s)    # cos coefficients and s-derivative
            Bc, dB = self._coef(Ss, s)   # sin coefficients
            out[name] = (A * cpart + Bc * spart).sum(-1)
            out[name + "_s"] = (dA * cpart + dB * spart).sum(-1)
            out[name + "_t"] = (self.xm * (-A * spart + Bc * cpart)).sum(-1)
            out[name + "_p"] = (-self.xn * (-A * spart + Bc * cpart)).sum(-1)
        return out

    def theta_from_theta_star(self, s, th_star, ph):
        """Invert ``θ* = θ + λ(s, θ, φ)`` by Newton iteration (vectorised)."""
        s, th_star, ph = np.broadcast_arrays(np.asarray(s, float), np.asarray(th_star, float), np.asarray(ph, float))
        th = th_star.copy()
        for _ in range(30):
            e = self._eval(s, th, ph)
            f = th + e["L"] - th_star
            th = th - f / (1.0 + e["L_t"])
            if np.max(np.abs(f)) < 1e-13:
                break
        return th

    def modB(self, s, th_star, ph):
        return self._fields(s, th_star, ph)["modB"]

    def _fields(self, s, th_star, ph):
        """Compute B, metric and basis vectors at field-aligned points."""
        s, th_star, ph = np.broadcast_arrays(np.asarray(s, float), np.asarray(th_star, float), np.asarray(ph, float))
        th = self.theta_from_theta_star(s, th_star, ph)
        e = self._eval(s, th, ph)
        R, Z = e["R"], e["Z"]
        cp, sp = np.cos(ph), np.sin(ph)
        def vec(a, b, c):
            return np.stack([a, b, c], axis=-1)
        e_s = vec(e["R_s"] * cp, e["R_s"] * sp, e["Z_s"])
        e_t = vec(e["R_t"] * cp, e["R_t"] * sp, e["Z_t"])
        e_p = vec(e["R_p"] * cp - R * sp, e["R_p"] * sp + R * cp, e["Z_p"])
        sqrtg = np.einsum("...i,...i->...", e_s, np.cross(e_t, e_p))
        grad_s = np.cross(e_t, e_p) / sqrtg[..., None]
        grad_t = np.cross(e_p, e_s) / sqrtg[..., None]
        grad_p = np.cross(e_s, e_t) / sqrtg[..., None]
        grad_ts = (1 + e["L_t"])[..., None] * grad_t + e["L_s"][..., None] * grad_s + e["L_p"][..., None] * grad_p
        iota, diota = self._iota(s), self._iota(s, 1)
        grad_alpha = grad_ts - (diota * ph)[..., None] * grad_s - iota[..., None] * grad_p
        dpsi_ds = self.phi_edge / (2 * np.pi)
        # orientation: make B·∇φ have the sign of phi_edge regardless of the handedness of (s,θ,φ)
        Bvec = np.sign(sqrtg)[..., None] * dpsi_ds * np.cross(grad_s, grad_alpha)
        modB = np.linalg.norm(Bvec, axis=-1)
        return dict(Bvec=Bvec, modB=modB, grad_s=grad_s, grad_t=grad_t, grad_p=grad_p,
                    grad_ts=grad_ts, grad_alpha=grad_alpha, iota=iota, diota=diota, th=th, R=R, Z=Z)

    # -- Build the Velcro geometry -----------------------------------------
    def build(self, Nx: int, Lx: float, Ny: int, Nz: int, s0: float, n0: Optional[int] = None,
              rho_ref: Optional[float] = None, rho_star: Optional[float] = None) -> Geometry:
        """Evaluate geometry on a Velcro grid centred on flux surface ``s0``.

        Exactly one of ``rho_ref`` (m) or ``rho_star`` must be given.  ``n0``
        defaults to ``nfp`` (one field period in ``y``); use ``n0=1`` for the full torus.
        """
        if rho_star is None:
            rho_star = rho_ref / self.L_ref
        n0 = self.nfp if n0 is None else int(n0)
        L = self.L_ref
        x0 = self.a * np.sqrt(s0)                              # m
        iota0 = float(self._iota(s0))
        q0 = 1.0 / iota0
        Cy_phys = x0 / q0                                      # m, so |∇y| ≈ 1 at s0
        Cy = Cy_phys / (rho_star * L)                          # ρ_ref
        Ly = 2 * np.pi * Cy / n0
        x, y, z = _grids(Nx, Lx, Ny, Ly, Nz)
        x_phys = x0 + rho_star * L * x                         # m
        s = (x_phys / self.a) ** 2
        if np.any(s <= 0) or np.any(s > 1):
            raise ValueError("radial domain leaves the equilibrium (s outside (0, 1])")
        X, Y, Zg = np.meshgrid(x_phys, y * rho_star * L, z, indexing="ij")   # m, m, rad
        S = (X / self.a) ** 2
        iota = self._iota(S)
        q = 1.0 / iota
        # toroidal angle from the field-line label:  y = C_y (q θ* − φ)
        PH = q * Zg - Y / Cy_phys
        f = self._fields(S, Zg, PH)
        modB = f["modB"]
        dx_ds = self.a / (2 * np.sqrt(S))
        grad_x = f["grad_s"] * dx_ds[..., None]                # ∇x (dimensionless)
        dq_ds = -f["diota"] / iota**2
        alpha = Zg - iota * PH
        grad_y = Cy_phys * (dq_ds[..., None] * alpha[..., None] * f["grad_s"] + q[..., None] * f["grad_alpha"])
        dot = lambda u, v: np.einsum("...i,...i->...", u, v)
        gxx, gxy, gyy = dot(grad_x, grad_x), dot(grad_x, grad_y), dot(grad_y, grad_y)
        b = f["Bvec"] / modB[..., None]
        gpar = L * dot(f["Bvec"], f["grad_ts"]) / modB
        cross_xy = np.cross(grad_x, grad_y)
        # B = C ∇x×∇y with C > 0 (design document eq. 3): orient B accordingly.  The overall sign
        # of B is immaterial for electrostatic ion physics; what matters is that gpar, K_x, K_y and
        # the E×B coefficient 1/C are mutually consistent.
        C = dot(f["Bvec"], cross_xy) / np.sum(cross_xy**2, axis=-1)   # T, signed
        orient = np.sign(C.mean())
        C = C * orient
        b = b * orient
        gpar = gpar * orient
        # ∇|B| by central differences in (s, θ*, φ)
        def dB(ds, dt, dp):
            return (self.modB(S + ds, Zg + dt, PH + dp) - self.modB(S - ds, Zg - dt, PH - dp))
        hs, ht, hp = 1e-4 * np.ones_like(S), 1e-4, 1e-4
        dB_ds = dB(hs, 0, 0) / (2 * hs)
        dB_dt = dB(0, ht, 0) / (2 * ht)
        dB_dp = dB(0, 0, hp) / (2 * hp)
        gradB = dB_ds[..., None] * f["grad_s"] + dB_dt[..., None] * f["grad_ts"] + dB_dp[..., None] * f["grad_p"]
        gradlnB = gradB / modB[..., None]
        bxg = np.cross(b, gradlnB)
        Kx = L * dot(bxg, grad_x)
        Ky = L * dot(bxg, grad_y)
        dlnBdz = dB_dt / modB
        # -L ∂_x ln B at fixed (y, z): x-derivative along the grid
        lnB = np.log(modB)
        dx_m = (x[1] - x[0]) * rho_star * L
        omega_B = -L * np.gradient(lnB, dx_m, axis=0)
        Bn = modB / self.B_ref
        Cn = C / self.B_ref
        Cinv = (1.0 / Cn.mean(axis=(1, 2)))[:, None, None]
        twist = 2 * np.pi * (1.0 / self._iota(s)) * Cy
        jac = 1.0 / (Bn * gpar)
        axi = (self.xn == 0).all()
        return Geometry(
            x=x, y=y, z=z, rho_star=rho_star, Ly=Ly,
            B=Bn, gpar=gpar, dlnBdz=dlnBdz, omega_B=omega_B,
            gxx=gxx, gxy=gxy, gyy=gyy, Kx=Kx, Ky=Ky, Cinv=Cinv, twist_phase=twist, jac=jac,
            axisymmetric=bool(axi), name="fourier-equilibrium",
            extra=dict(q=1.0 / self._iota(s), s=s, Cy=Cy, n0=n0, C_over_Bref=Cn, R=f["R"], Z=f["Z"]),
        )


def circular_tokamak_fourier(R0: float, a: float, B0: float, q_of_s, ns: int = 32) -> FourierEquilibrium:
    """Analytic circular concentric-surface tokamak as a :class:`FourierEquilibrium`.

    ``R = R0 + r cos θ``, ``Z = r sin θ`` with ``r = a sqrt(s)``.  ``λ`` is chosen
    so that ``θ*`` is the PEST straight-field-line angle for a large-aspect-
    ratio ``B_θ ∝ 1/R``-corrected field: ``λ = -2 (r/R0) sin θ`` to leading order
    in ``r/R0``, which makes ``B·∇θ*`` constant on a surface at that order.
    """
    s = np.linspace(1e-3, 1.0, ns)
    r = a * np.sqrt(s)
    xm = np.array([0.0, 1.0, 1.0])
    xn = np.zeros(3)
    rmnc = np.stack([R0 * np.ones(ns), r, np.zeros(ns)], axis=1)
    zmns = np.stack([np.zeros(ns), r, np.zeros(ns)], axis=1)
    # λ = -(r/R0) sin θ … first-order correction making θ* ≈ straight-field-line
    lmns = np.stack([np.zeros(ns), -(r / R0), np.zeros(ns)], axis=1)
    iota = 1.0 / q_of_s(s)
    phi_edge = np.pi * a**2 * B0
    return FourierEquilibrium(s, xm, xn, rmnc, zmns, lmns, iota, phi_edge, nfp=1, a=a, B_ref=B0, L_ref=R0)


def from_vmec_wout(path: str, L_ref: Optional[float] = None) -> FourierEquilibrium:
    """Load a VMEC ``wout`` NetCDF file (also produced by GVEC's VMEC-compatible writer)."""
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as f:
        v = lambda k: np.array(f.variables[k].data)          # ``[:]`` fails on the scalar variables (ns, nfp, …)
        ns = int(v("ns"))
        s = np.linspace(0, 1, ns)
        xm, xn = v("xm"), v("xn")
        rmnc, zmns, lmns = v("rmnc"), v("zmns"), v("lmns")
        iotaf = v("iotaf") if "iotaf" in f.variables else v("iotas")
        phi = v("phi")
        nfp = int(v("nfp"))
        a = float(v("Aminor_p"))
        R0 = float(v("Rmajor_p"))
        B0 = float(np.abs(v("b0"))) if "b0" in f.variables else 1.0
        lasym = bool(v("lasym__logical__")) if "lasym__logical__" in f.variables else False
        kw = {}
        if lasym:
            kw = dict(rmns=v("rmns"), zmnc=v("zmnc"), lmnc=v("lmnc"))
    # drop the s=0 axis point (singular Jacobian) — λ is on the half mesh in VMEC; average to full
    lmns_full = 0.5 * (lmns[1:] + np.roll(lmns, -1, axis=0)[1:])
    lmns_full[-1] = lmns[-1]
    return FourierEquilibrium(s[1:], xm, xn, rmnc[1:], zmns[1:], lmns_full, iotaf[1:], phi[-1], nfp, a,
                              B0, R0 if L_ref is None else L_ref, **{k: val[1:] for k, val in kw.items()})


def from_gvec(state_file: str, **kwargs) -> FourierEquilibrium:
    """Build from a GVEC state via its Python bindings (``pip install gvec``).

    GVEC can export a VMEC-compatible ``wout`` file; when the bindings expose
    Fourier coefficients directly we read them, otherwise we fall back to the
    ``wout`` route.  See §2.3 of the design document.
    """
    try:
        import gvec  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional package
        raise ImportError("gvec python package is not installed; use from_vmec_wout on a GVEC-exported wout file") from e
    if state_file.endswith(".nc"):
        return from_vmec_wout(state_file, **kwargs)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:  # pragma: no cover
        out = os.path.join(d, "wout_gvec.nc")
        gvec.State(state_file).to_vmec(out) if hasattr(gvec, "State") else gvec.convert_to_vmec(state_file, out)
        return from_vmec_wout(out, **kwargs)
