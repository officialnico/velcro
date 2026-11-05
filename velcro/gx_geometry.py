"""GX's own flux-tube geometry arrays as a Velcro :class:`~velcro.geometry.Geometry` (``geometry.kind = "gx_file"``).

Velcro and GX then run on identical geometry: whatever GX's geometry module produced (s-α, Miller, VMEC, …) enters
Velcro unchanged, in the flux-tube limit (every coefficient independent of x; Velcro's global radial variation, O(ρ* x),
is dropped — which is exactly GX's model).

Sources (:func:`read_gx_geometry`)
----------------------------------
* GX's output ``RUN.out.nc``, group ``Geometry`` (+ ``Grids/theta``): the arrays GX actually used, after its reader's
  conventions (drifts halved, interpolated to its uniform θ grid, ``bgrad`` = its spectral ∂_θ ln B · gradpar).  Preferred.
* GX's geometry-module files, before any run: the text ``RUN.eik.out`` (Miller, GS2 format) and ``RUN.eik.nc`` (VMEC,
  GX's ``geo_nc`` format).  GX halves their ``gbdrift, gbdrift0, cvdrift, cvdrift0`` on reading and interpolates the
  N_z + 1 points (both ends of [−π, π]) linearly onto θ_k = −π + 2πk/N_z; here the grid must already be uniform (the
  interpolation is then the identity) or the file is refused.  ``bgrad`` is recomputed as GX does it (FFT derivative).

Normalisation (GX: a, ρ_i = √(T_i m_i)/(eB_ref), v_ti; Velcro: L_ref = R₀, ρ_ref = √2ρ_i, v_tref = √2v_ti)
------------------------------------------------------------------------------------------------------------
Velcro's L_ref is ``R = Lref_over_a · a`` (``rmaj``, default the file's; a GX deck translation passes the deck's
Rmaj so that its rate factor R/(√2a) and its gradients R/L use the same R).  The GX equation (linear.cu, device_funcs.cu
rhs_linear) and Velcro's (vlasov.rhs) then map term by term, with Velcro's k_x = −kxfac·k_x,GX (so x_V = −x_GX/kxfac;
for kxfac = 1 this is the sign flip of our notes) and the same k_y:

====================  ============================================  ==========================================
Velcro field          from GX                                         GX term it reproduces
====================  ============================================  ==========================================
B                     bmag                                            b_s = k⊥²ρ²/B² (k⊥² carries bmagInv²)
gpar  (L_ref b·∇z)    −rmaj · gradpar                                 v_t gradpar ∂_z (GX takes |gradpar|)
dlnBdz                bgrad / gradpar                                 mirror force v_t bgrad
gyy                   gds2                                            k⊥² = k_y² gds2
gxy                   −gds21 / (kxfac ŝ)                              + 2 k_y (k_x/ŝ) gds21
gxx                   gds22 / (kxfac ŝ)²                              + (k_x/ŝ)² gds22
Ky                    −rmaj · B · gbdrift                             ω_d = k_y gb + (k_x/ŝ) gb0  (c_Ky = ½Ky/B)
Kx                    rmaj · B · gbdrift0 / (kxfac ŝ)                 (c_Kx = ½Kx/B)
Ky_co / Kx_co         −rmaj·B·cdrift / rmaj·B·cdrift0/(kxfac ŝ)        GS2's Coriolis drift (native Miller only; GX has
                                                                      none): c_co = mach ½K_co/B on v̂ (T/Z)/v̂t_s
Ky_cv                 −rmaj · B · (cvdrift − gbdrift)                 β' ≠ 0: the curvature drift's excess, on v∥²
Kx_cv                 rmaj · B · (cvdrift0 − gbdrift0) / (kxfac ŝ)    only (GX: cv_d v∥² + gb_d μB; Velcro: c_K (2v̂² + μ̂)
                                                                      + c_K,cv 2v̂²); None when cv = gb (β' = 0)
Cinv                  1                                               drive i k_y ω_* (no geometric factor);
                                                                      bracket kxfac{·,·} (absorbed by x_V)
jac                   1/(B gpar)  (∝ GX's 1/|drhodpsi gradpar bmag|)  flux-surface / volume weights
twist_phase           2π S x,  S = kxfac ŝ gds21[0]/(π gds22[0])       GX's linked BC: k_x shift per k_y is
                                                                      twist_shift_geo_fac = 2ŝ gds21[0]/gds22[0]
omega_B               0 (not in GX; used only by drive_gradB)
====================  ============================================  ==========================================

The signs and the factor rmaj·B in K_x, K_y come from the calibration on s-α: GX's S_alpha_geo arrays (bmag =
1/(1+ε cos θ), gradpar = 1/(q rmaj), gbdrift = (cos θ + ŝθ sin θ)/rmaj, gbdrift0 = −ŝ sin θ/rmaj, gds2 = 1 + ŝ²θ²,
gds21 = −ŝ²θ, gds22 = ŝ²) reproduce :func:`velcro.geometry.s_alpha` with ``gpar_model = "constant"`` at x = 0 exactly
(tests/test_gx_geometry.py; on GX's own s-α output to float32 round-off, 2e-7).  Every GX geometry module writes the
same arrays in the same convention (GX's equations do not depend on the module), so the map carries over.  For an
axisymmetric tube gds21[0]/gds22[0] = π (θ = −π), so S = kxfac ŝ (GX's circular Miller benchmark: S = 0.8000000 at
ŝ = 0.8).  Velcro's chain shift S k_y,min L_x = jtwist holds with L_x = jtwist/(|S| k_y,min) [ρ_ref], which is GX's
2π x0 = 2π y0 jtwist/|twist_shift_geo_fac| [ρ_i] divided by √2|kxfac|.

Refused: N_z ≠ GX's ntheta (the arrays are GX's, not re-interpolated: the secular gds2/gds21 are not periodic, so a
Fourier interpolation would not be the same geometry, and GX itself requires ntheta to match its geometry file);
nperiod > 1 or zp ≠ 1 (Velcro's z is one poloidal turn [−π, π)); ŝ = 0 (GX's periodic zero-shear box); a non-constant
gradpar.  cvdrift ≠ gbdrift (β' ≠ 0) is carried by Ky_cv / Kx_cv (above); equal to ``drift_rtol`` of the array's
maximum they are taken as equal (None: the β' = 0 operator, bitwise).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from velcro.geometry import Geometry, _grids

ARRAYS = ("bmag", "bgrad", "gds2", "gds21", "gds22", "gbdrift", "gbdrift0", "cvdrift", "cvdrift0", "grho", "jacobian")
_HALVED = ("gbdrift", "gbdrift0", "cvdrift", "cvdrift0")      # GX's eik readers multiply these by 0.5


# ------------------------------------------------------------------------------------------------ readers
def _nc_tree(path: str, groups=("Geometry", "Grids")) -> dict:
    """Top-level variables and those of ``groups`` of a netCDF file, as numpy arrays (netCDF4, else h5py; netCDF-3 via
    scipy)."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:3] == b"CDF":
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as f:
            return {k: np.array(v[:]) for k, v in f.variables.items()}
    try:
        import netCDF4
    except ImportError:
        netCDF4 = None
    if netCDF4 is not None:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            out = {k: np.array(v[:]) for k, v in ds.variables.items() if v.dtype != str}
            for gname in groups:
                if gname in ds.groups:
                    out[gname] = {k: np.array(v[:]) for k, v in ds.groups[gname].variables.items() if v.dtype != str}
            return out
    try:
        import h5py
    except ImportError as e:
        raise ImportError(f"reading {path} needs netCDF4 or h5py") from e

    def read(grp):
        res = {}
        for k, v in grp.items():
            if isinstance(v, h5py.Dataset):
                try:
                    res[k] = np.array(v[()])
                except (TypeError, ValueError, OSError):
                    pass
        return res
    with h5py.File(path, "r") as f:
        out = read(f)
        for gname in groups:
            if gname in f:
                out[gname] = read(f[gname])
        return out


def _scalar(tree: dict, *names, default=None):
    for n in names:
        if n in tree:
            return float(np.asarray(tree[n]).reshape(-1)[0])
    if default is None:
        raise KeyError(f"none of {names} in the GX geometry file")
    return float(default)


def _spectral_bgrad(bmag: np.ndarray, gradpar: float) -> np.ndarray:
    """GX's calculate_bgrad: gradpar · ∂_θ ln B with ∂_θ by FFT on the periodic N_z grid (Nyquist dropped)."""
    Nz = len(bmag)
    k = np.fft.fftfreq(Nz, d=1.0 / Nz)
    if Nz % 2 == 0:
        k[Nz // 2] = 0.0
    dB = np.real(np.fft.ifft(1j * k * np.fft.fft(bmag)))
    return gradpar * dB / bmag


def _from_eik_points(theta: np.ndarray, arrs: dict, gradpar: np.ndarray, sc: dict, source: str) -> dict:
    """GX's eik/geo_nc reader on N_z + 1 points: halve the drifts, check gradpar, map onto θ_k = −π + 2πk/N_z."""
    theta = np.asarray(theta, float)
    Nz = len(theta) - 1
    gp = np.asarray(gradpar, float).reshape(-1)
    if not np.allclose(gp, gp[0], rtol=1e-6, atol=0.0):
        raise ValueError(f"{source}: gradpar is not constant (GX needs an equal-arc θ)")
    z = -np.pi + 2 * np.pi * np.arange(Nz) / Nz
    dz = 2 * np.pi / Nz
    if abs(theta[-1] - theta[0] - 2 * np.pi) > 1e-6 * dz or np.max(np.abs(theta[:Nz] - z)) > 1e-6 * dz:
        raise ValueError(f"{source}: θ is not the uniform grid [−π, π] (nperiod > 1, zp ≠ 1, or a non-uniform grid GX "
                         "would interpolate): pass GX's RUN.out.nc instead")
    out = {k: np.asarray(arrs[k], float)[:Nz] * (0.5 if k in _HALVED else 1.0) for k in arrs}
    out.update(sc)
    out["gradpar"] = float(gp[0])
    out["theta"] = z
    out["bgrad"] = _spectral_bgrad(out["bmag"], out["gradpar"])
    out["jacobian"] = 1.0 / np.abs(out["drhodpsi"] * out["gradpar"] * out["bmag"])
    out["nperiod"] = 1
    out["source"] = source
    return out


def _read_eik_text(path: str) -> dict:
    """GX's Eik_geo reader (geometry.cu) for the GS2-format ``eik.out`` text file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    head = lines[1].split()
    nperiod, ntheta = int(head[1]), int(head[2])
    if nperiod != 1:
        raise ValueError(f"{path}: nperiod = {nperiod}; Velcro's z is one poloidal turn (nperiod = 1)")
    sc = dict(drhodpsi=float(head[3]), rmaj=float(head[4]), shat=float(head[5]), kxfac=float(head[6]), q=float(head[7]))
    n = ntheta + 1

    def block(i0, ncol):
        rows = np.array([[float(v) for v in lines[i0 + j].split()[:ncol]] for j in range(n)])
        return rows.T
    gb, gp, grho, th = block(3, 4)                  # the columns GX reads, in its order
    cv, gds2, bmag = block(4 + n, 3)
    gds21, gds22 = block(5 + 2 * n, 2)
    cv0, gb0 = block(6 + 3 * n, 2)
    arrs = dict(gbdrift=gb, grho=grho, cvdrift=cv, gds2=gds2, bmag=bmag, gds21=gds21, gds22=gds22, cvdrift0=cv0,
                gbdrift0=gb0)
    return _from_eik_points(th, arrs, gp, sc, path)


def read_gx_geometry(path: str) -> dict:
    """GX's geometry arrays in GX's internal convention on its N_z-point grid θ_k = −π + 2πk/N_z: the keys of
    ``ARRAYS`` plus ``theta, gradpar, shat, q, rmaj, kxfac, drhodpsi, nperiod, source``."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:3] != b"CDF" and magic != b"\x89HDF":
        return _read_eik_text(path)
    tree = _nc_tree(path)
    if "Geometry" in tree:                                   # GX output RUN.out.nc: the arrays as GX used them
        g = tree["Geometry"]
        theta = np.asarray(tree.get("Grids", {}).get("theta", []), float)
        out = {k: np.asarray(g[k], float) for k in ARRAYS}
        if theta.size != out["bmag"].size:
            raise ValueError(f"{path}: Grids/theta missing or not on the Geometry grid")
        out.update(theta=theta, gradpar=_scalar(g, "gradpar"), shat=_scalar(g, "shat"), q=_scalar(g, "q"),
                   rmaj=_scalar(g, "rmaj", "Rmaj"), kxfac=_scalar(g, "kxfac"), drhodpsi=_scalar(g, "drhodpsi"),
                   nperiod=int(_scalar(g, "nperiod", default=1)), source=path)
        return out
    if "bmag" in tree:                                       # GX's geo_nc input format (VMEC eik.nc, gs2 eik.out.nc)
        sc = dict(drhodpsi=_scalar(tree, "drhodpsi"), kxfac=_scalar(tree, "kxfac"), shat=_scalar(tree, "shat"),
                  rmaj=_scalar(tree, "Rmaj", "rmaj"), q=_scalar(tree, "q"))
        arrs = {k: tree[k] for k in ARRAYS if k not in ("bgrad", "jacobian")}
        return _from_eik_points(tree["theta"], arrs, tree["gradpar"], sc, path)
    raise ValueError(f"{path}: neither a GX output (Geometry group) nor a GX geometry file")


def write_gx_eik(path: str, gx: dict) -> None:
    """Write GX-internal arrays on N_z points (as :func:`read_gx_geometry` returns them) as a GS2-format ``eik.out``
    that GX's Eik_geo reads back to the same arrays: N_z + 1 points (θ = +π closes the line with the secular parts of
    gds2/gds21 continued and the periodic arrays repeated), drifts doubled.  For tests and fixtures."""
    th = np.append(np.asarray(gx["theta"], float), np.pi)
    Nz = len(th) - 1

    def ext(k):
        a = np.asarray(gx[k], float)
        if k == "gds21":
            return np.append(a, -a[0])
        return np.append(a, a[0])
    a = {k: ext(k) * (2.0 if k in _HALVED else 1.0) for k in ("gbdrift", "grho", "cvdrift", "gds2", "bmag", "gds21",
                                                             "gds22", "cvdrift0", "gbdrift0")}
    a["gbdrift0"][-1] *= -1.0                          # odd in θ: its +π value is −(its −π value)
    a["cvdrift0"][-1] *= -1.0
    gp = np.full(Nz + 1, float(gx["gradpar"]))
    f = lambda v: f"{v: .16e}"
    L = ["ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q",
         f"{Nz // 2} 1 {Nz} {f(gx['drhodpsi'])} {f(gx['rmaj'])} {f(gx['shat'])} {f(gx['kxfac'])} {f(gx['q'])}",
         "gbdrift gradpar grho tgrid"]
    L += [" ".join(f(v) for v in r) for r in zip(a["gbdrift"], gp, a["grho"], th)]
    L += ["cvdrift gds2 bmag tgrid"]
    L += [" ".join(f(v) for v in r) for r in zip(a["cvdrift"], a["gds2"], a["bmag"], th)]
    L += ["gds21 gds22 tgrid"]
    L += [" ".join(f(v) for v in r) for r in zip(a["gds21"], a["gds22"], th)]
    L += ["cvdrift0 gbdrift0 tgrid"]
    L += [" ".join(f(v) for v in r) for r in zip(a["cvdrift0"], a["gbdrift0"], th)]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def gx_s_alpha_arrays(Nz: int, eps: float, rmaj: float, q: float, shat: float, shift: float = 0.0) -> dict:
    """GX's S_alpha_geo arrays (geometry.cu) on θ_k = −π + 2πk/N_z, in float64 — the calibration geometry."""
    th = -np.pi + 2 * np.pi * np.arange(Nz) / Nz
    gradpar = abs(1.0 / (q * rmaj))
    bmag = 1.0 / (1.0 + eps * np.cos(th))
    sec = shat * th - shift * np.sin(th)
    gb = (np.cos(th) + sec * np.sin(th)) / rmaj
    gb0 = -shat * np.sin(th) / rmaj
    return dict(theta=th, bmag=bmag, bgrad=gradpar * eps * np.sin(th) * bmag, gds2=1.0 + sec ** 2, gds21=-shat * sec,
                gds22=np.full(Nz, shat ** 2), gbdrift=gb, cvdrift=gb.copy(), gbdrift0=gb0, cvdrift0=gb0.copy(),
                grho=np.ones(Nz), jacobian=1.0 / (gradpar * bmag), gradpar=gradpar, shat=shat, q=q, rmaj=rmaj,
                kxfac=1.0, drhodpsi=1.0, nperiod=1, source="gx_s_alpha_arrays")


# ------------------------------------------------------------------------------------------------ the map
def twist_shear(gx: dict) -> float:
    """S = kxfac ŝ gds21[0]/(π gds22[0]): Velcro's twist e^{−i k_y 2π S x} (= ŝ for an axisymmetric tube); GX's
    twist_shift_geo_fac = 2ŝ gds21[0]/gds22[0] = 2πS/kxfac."""
    return float(gx["kxfac"]) * float(gx["shat"]) * float(np.asarray(gx["gds21"])[0]) / (np.pi * float(np.asarray(gx["gds22"])[0]))


def from_gx_arrays(gx: dict, Nx: int, Lx: float, Ny: int, Nz: int, Ly: float, rho_star: float = 1e-5,
                   Lref_over_a: Optional[float] = None, x_offset: float = 0.0, drift_rtol: float = 1e-4) -> Geometry:
    """The Velcro flux-tube geometry of GX-internal arrays ``gx`` (see the module docstring for the map).

    ``Ly`` is the binormal box in ρ_ref (2π/k_y,min = 2π y0/√2 for GX's y0); ``Lx`` the radial box in ρ_ref (for GX's
    linked boundary L_x = jtwist/(S k_y,min), S = :func:`twist_shear`); ``Lref_over_a`` Velcro's L_ref in units of GX's
    a (default: the file's rmaj)."""
    theta = np.asarray(gx["theta"], float)
    if len(theta) != Nz:
        raise ValueError(f"grid.Nz = {Nz} but the GX geometry has ntheta = {len(theta)} points: run with N_z = GX's "
                         "ntheta (the arrays are used as GX used them, not re-interpolated)")
    z_ref = -np.pi + 2 * np.pi * np.arange(Nz) / Nz
    if np.max(np.abs(theta - z_ref)) > 1e-5:
        raise ValueError("the GX geometry's θ grid is not [−π, π) with N_z points (nperiod > 1 or zp ≠ 1?)")
    if int(gx.get("nperiod", 1)) > 1:
        raise ValueError("GX nperiod > 1: Velcro's z is one poloidal turn")
    shat, kxfac = float(gx["shat"]), float(gx["kxfac"])
    if abs(shat) < 1e-6:
        raise ValueError("ŝ = 0 (GX's periodic zero-shear box) is not supported by the gx_file geometry")
    if kxfac == 0.0:
        raise ValueError("kxfac = 0 in the GX geometry")
    arr = {k: np.asarray(gx[k], float) for k in ARRAYS}
    differ = lambda a, b: np.max(np.abs(arr[a] - arr[b])) > drift_rtol * max(np.max(np.abs(arr[b])), 1e-300)
    beta_prime = differ("cvdrift", "gbdrift") or differ("cvdrift0", "gbdrift0")
    rmaj = float(gx["rmaj"]) if Lref_over_a is None else float(Lref_over_a)
    gradpar = float(gx["gradpar"])
    B = arr["bmag"]
    ks = kxfac * shat
    gpar = -rmaj * gradpar * np.ones(Nz)
    fields = dict(B=B, gpar=gpar, dlnBdz=arr["bgrad"] / gradpar, omega_B=np.zeros(Nz),
                  gxx=arr["gds22"] / ks ** 2, gxy=-arr["gds21"] / ks, gyy=arr["gds2"],
                  Kx=rmaj * B * arr["gbdrift0"] / ks, Ky=-rmaj * B * arr["gbdrift"], jac=1.0 / (B * gpar))
    if "cdrift" in gx:              # GS2's Coriolis arrays (native Miller only, unhalved): mapped like gbdrift/gbdrift0, per
        # unit GS2 mach and per unit (T/Z)/v̂t_s; used only with physics.coriolis_mach ≠ 0
        fields.update(Kx_co=rmaj * B * np.asarray(gx["cdrift0"], float) / ks, Ky_co=-rmaj * B * np.asarray(gx["cdrift"], float))
    if beta_prime:                  # the curvature drift's excess over the ∇B drift, on the v∥² part (module docstring)
        fields.update(Kx_cv=rmaj * B * (arr["cvdrift0"] - arr["gbdrift0"]) / ks, Ky_cv=-rmaj * B * (arr["cvdrift"] - arr["gbdrift"]))
    x, y, z = _grids(Nx, Lx, Ny, Ly, Nz, x_offset)
    S = twist_shear(gx)
    add = lambda a: np.ascontiguousarray(np.broadcast_to(a[None, None, :], (Nx, 1, Nz)))
    return Geometry(
        x=x, y=y, z=z, rho_star=rho_star, Ly=Ly,
        Cinv=np.ones((Nx, 1, 1)), twist_phase=2 * np.pi * S * x, axisymmetric=True, name="gx-file",
        extra=dict(q=float(gx["q"]), shat=shat, twist_shat=S, twist_shift_geo_fac=2 * np.pi * S / kxfac, rmaj=rmaj,
                   rmaj_file=float(gx["rmaj"]), kxfac=kxfac, drhodpsi=float(gx["drhodpsi"]), gradpar=gradpar,
                   grho=arr["grho"], gx_jacobian=arr["jacobian"], source=gx.get("source"),
                   # GX divides every flux by ⟨|∇ρ|⟩ (geometry.cu: flux_fac = jacobian/Σ jacobian·grho); Velcro's
                   # diagnostics do not, so Q_GX = 2√2 (a/L_ref)² Q / grho_avg (1 for s-α, 0.815 for the Miller test tube)
                   grho_avg=float(np.sum(arr["jacobian"] * arr["grho"]) / np.sum(arr["jacobian"]))),
        **{k: add(v) for k, v in fields.items()},
    )


def from_gx_geometry(path: str, Nx: int, Lx: float, Ny: int, Nz: int, Ly: float, rho_star: float = 1e-5,
                     Lref_over_a: Optional[float] = None, x_offset: float = 0.0, drift_rtol: float = 1e-4) -> Geometry:
    """:func:`from_gx_arrays` of :func:`read_gx_geometry` (``path``: GX's RUN.out.nc, RUN.eik.out or RUN.eik.nc)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GX geometry file {path!r} not found")
    return from_gx_arrays(read_gx_geometry(path), Nx, Lx, Ny, Nz, Ly, rho_star=rho_star, Lref_over_a=Lref_over_a,
                          x_offset=x_offset, drift_rtol=drift_rtol)
