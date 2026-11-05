"""Right-hand side of the Hermite–Laguerre gyrokinetic Vlasov equation (§2.1, §3).

Normalisation: perpendicular lengths in ρ_ref, parallel in L_ref, time in
L_ref/v_t,ref with v_t = sqrt(2T/m), φ̂ = (eφ/T_ref)(L_ref/ρ_ref), Ĝ = (G/F0)(L_ref/ρ_ref).
Single ion species (Z = 1), adiabatic electrons, electrostatic.  With
``Φ_{ℓm} = δ_{ℓ0} 𝒥_m φ̂`` (gyroaveraged potential as a Laguerre field) and
``H = G + Φ / T̂_i``:

    ∂_t G = − √T̂ gpar  V ∂_z H                                  (streaming + E∥)
            + √T̂ gpar ∂_z ln B  Mirror H                        (mirror force)
            − ½ B⁻¹ (2V² + Mu) [K_x ∂_x (T̂ G + Φ) + i k_y K_y (T̂ G + Φ)]   (∇B/curvature drift)
            − ½ C⁻¹ i k_y [ω_n + ω_T (v̂² + μ̂ − 3/2) − ω_B μ̂] Φ   (−v_E·∇F0 drive)
            − ½ C⁻¹ {Φ, G}_{xy}                                  (E×B nonlinearity)
            + ν C_coll G − hyperdiffusion − Krook buffers − sources

Velocity-space operators are the exact matrices of ``velcro.basis``; the
nonlinearity is evaluated pseudo-spectrally in μ on Gauss–Laguerre nodes
(exact Galerkin projection with N_μ = M nodes) and in (x, y) by the Orszag
method with the 2/3 rule.  Basis-variation terms of order (ρ/L)² relative to
the retained terms (∂_x of the local normalisation acting on G and Φ) are
dropped; the full ∇F0 (including ∇B) is kept in the drive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from velcro import basis
from velcro.collisions import collision_matrix
from tests.reference.ref_operators import apply_coef, bracket, ddx, ddz, hyper_x, hyper_z


class VlasovOperators(eqx.Module):
    """All precomputed arrays needed by :func:`rhs` (an Equinox pytree: arrays are
    traced, scalar/flag fields are static)."""
    # velocity-space matrices, reshaped to (L, M, L, M)
    V: Any; mirror: Any; energy: Any; drive_T: Any; mu: Any; coll: Any; hypv: Any
    # Laguerre node transforms
    lag_to_nodes: Any   # (Nmu, M)
    nodes_to_lag: Any   # (M, Nmu)
    # coefficient fields (broadcastable to (Nx, Ny|1, Nz))
    c_stream: Any; c_mirror: Any; c_Kx: Any; c_Ky: Any; c_Cinv: Any; c_omegaB: Any
    Ti: Any; inv_Ti: Any; omega_n: Any; omega_T: Any; nu: Any; krook: Any
    ky: Any; phase: Any
    fsw: Any  # (Nx, Nz) flux-surface weights
    kymax: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dz: float = eqx.field(static=True)
    Ny: int = eqx.field(static=True)
    axisymmetric: bool = eqx.field(static=True)
    hyper_x: float = eqx.field(static=True)
    hyper_z: float = eqx.field(static=True)
    hyper_ky: float = eqx.field(static=True)
    source_rate: float = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)


def _vs(mat: np.ndarray, L: int, M: int) -> jnp.ndarray:
    return jnp.asarray(mat.reshape(L, M, L, M))


def apply_vs(mat: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """Apply an (L,M,L,M) velocity-space matrix to G of shape (L,M,...)."""
    return jnp.einsum("lmpq,pq...->lm...", mat, G)


def build_operators(geom, prof: dict, cfg, ky: np.ndarray, twist_phase: np.ndarray, dtype) -> VlasovOperators:
    L, M = cfg.grid.L, cfg.grid.M
    Nx, Ny, Nz = len(geom.x), cfg.grid.Ny, len(geom.z)
    I_L, I_M = np.eye(L), np.eye(M)
    V, Mu = basis.hermite_v_matrix(L), basis.laguerre_mu_matrix(M)
    Vmat = np.kron(V, I_M)
    mirror = basis.mirror_matrix(L, M)
    energy = basis.energy_matrix(L, M)                               # 2v² + μ
    drive_T = np.kron(V @ V, I_M) + np.kron(I_L, Mu) - 1.5 * np.eye(L * M)  # v² + μ − 3/2
    mu_mat = np.kron(I_L, Mu)
    coll = collision_matrix(cfg.physics.collision_model, L, M)
    l = np.arange(L)[:, None] * np.ones((1, M)); m = np.ones((L, 1)) * np.arange(M)[None, :]
    hyp = (l / max(L - 1, 1)) ** 4 + (m / max(M - 1, 1)) ** 4
    hypv = cfg.numerics.hyper_v * np.diag(hyp.reshape(-1))
    # Laguerre nodes
    mu_nodes, w_nodes = basis.laguerre_quadrature(M)
    Lmat = np.array([basis.laguerre(mm, mu_nodes) for mm in range(M)])   # (M, Nmu)
    lag_to_nodes = Lmat.T                                              # (Nmu, M)
    nodes_to_lag = Lmat * w_nodes[None, :]                             # (M, Nmu)
    # coefficient fields
    sqrtT = np.sqrt(prof["Ti"])[:, None, None]
    c_stream = sqrtT * geom.gpar
    c_mirror = sqrtT * geom.gpar * geom.dlnBdz
    c_Kx = 0.5 * geom.Kx / geom.B
    c_Ky = 0.5 * geom.Ky / geom.B
    c_Cinv = 0.5 * geom.Cinv
    # Krook buffer profile
    x = geom.x
    Lx = cfg.grid.Lx
    wb = cfg.numerics.krook_width * Lx
    d = np.minimum(x - x.min(), x.max() - x)
    krook = cfg.numerics.krook_rate * 0.5 * (1 + np.cos(np.pi * np.clip(d / max(wb, 1e-12), 0, 1))) if wb > 0 else np.zeros_like(x)
    ky_j = jnp.asarray(ky)[None, :, None]
    phase = jnp.exp(-1j * jnp.asarray(ky)[None, :, None] * jnp.asarray(twist_phase)[:, None, None])
    a = lambda arr: jnp.asarray(np.asarray(arr), dtype=dtype)
    r = lambda arr: jnp.asarray(np.asarray(arr))
    return VlasovOperators(
        V=_vs(Vmat, L, M), mirror=_vs(mirror, L, M), energy=_vs(energy, L, M), drive_T=_vs(drive_T, L, M),
        mu=_vs(mu_mat, L, M), coll=_vs(coll, L, M), hypv=_vs(hypv, L, M),
        lag_to_nodes=r(lag_to_nodes), nodes_to_lag=r(nodes_to_lag),
        c_stream=r(c_stream), c_mirror=r(c_mirror), c_Kx=r(c_Kx), c_Ky=r(c_Ky), c_Cinv=r(c_Cinv),
        c_omegaB=r(geom.omega_B), Ti=r(prof["Ti"][:, None, None]), inv_Ti=r(1.0 / prof["Ti"][:, None, None]),
        omega_n=r(prof["omega_n"][:, None, None]), omega_T=r(prof["omega_T"][:, None, None]),
        nu=r(prof["nu"][:, None, None]), krook=r(krook[:, None, None]),
        ky=ky_j, kymax=float(ky.max()) if ky.max() > 0 else 1.0, phase=phase,
        dx=float(Lx / Nx), dz=float(2 * np.pi / Nz), Ny=Ny, axisymmetric=bool(geom.axisymmetric),
        fsw=r(geom.fs_weights()),
        hyper_x=cfg.numerics.hyper_x, hyper_z=cfg.numerics.hyper_z, hyper_ky=cfg.numerics.hyper_ky,
        source_rate=cfg.numerics.source_rate, nonlinear=cfg.physics.nonlinear,
    )


def rhs(G: jnp.ndarray, phi: jnp.ndarray, Phi: jnp.ndarray, op: VlasovOperators) -> jnp.ndarray:
    """``∂_t G`` for ``G`` (L,M,Nx,Nky,Nz), ``φ`` (Nx,Nky,Nz), ``Φ`` gyroaveraged (M,Nx,Nky,Nz)."""
    L, M = G.shape[:2]
    ky = op.ky
    ax = op.axisymmetric
    coef = lambda c, f: apply_coef(c, f, op.Ny, ax)
    # Φ as a full (L, M, ...) field with ℓ = 0 only
    PhiL = jnp.zeros_like(G).at[0].set(Phi)
    H = G.at[0].add(op.inv_Ti * Phi)

    # --- parallel streaming + E∥ :  −√T gpar V ∂_z H
    dHdz = ddz(H, op.dz, op.phase)
    out = -coef(op.c_stream, apply_vs(op.V, dHdz))
    # --- mirror force: +√T gpar ∂_z lnB  Mirror H
    out = out + coef(op.c_mirror, apply_vs(op.mirror, H))
    # --- magnetic drift: −½ B⁻¹ (2v²+μ) [Kx ∂_x(T G + Φ) + i ky Ky (T G + Φ)]
    TG = op.Ti * G + PhiL
    dTGdx = op.Ti * ddx(G, op.dx) + ddx(PhiL, op.dx)
    drift = coef(op.c_Kx, dTGdx) + coef(op.c_Ky, 1j * ky * TG)
    out = out - apply_vs(op.energy, drift)
    # --- drive: −½ C⁻¹ i ky [ω_n + ω_T (v²+μ−3/2) − ω_B μ] Φ
    drv = op.omega_n * PhiL + op.omega_T * apply_vs(op.drive_T, PhiL) - coef(op.c_omegaB, apply_vs(op.mu, PhiL))
    out = out - op.c_Cinv * (1j * ky) * drv
    # --- E×B nonlinearity: −½ C⁻¹ {Φ, G}, pseudo-spectral in μ
    if op.nonlinear:
        Gn = jnp.einsum("jm,lm...->lj...", op.lag_to_nodes, G)            # (L, Nmu, ...)
        Pn = jnp.einsum("jm,m...->j...", op.lag_to_nodes, Phi)             # (Nmu, ...)
        br = bracket(Pn[None], Gn, ky, op.dx, op.Ny)                        # (L, Nmu, ...)
        out = out - op.c_Cinv * jnp.einsum("mj,lj...->lm...", op.nodes_to_lag, br)
    # --- collisions
    out = out + op.nu * apply_vs(op.coll, G)
    # --- numerical dissipation
    if op.hyper_x > 0:
        out = out - op.hyper_x * hyper_x(G)
    if op.hyper_z > 0:
        out = out - op.hyper_z * hyper_z(G, op.phase)
    if op.hyper_ky > 0:
        out = out - op.hyper_ky * (ky / op.kymax) ** 4 * G
    out = out - apply_vs(op.hypv, G)  # hyper-collisions (ν_h folded into hypv)
    # --- Krook buffers
    out = out - op.krook * G
    # --- optional profile-maintaining source on k_y = 0 flux-surface-averaged moments
    if op.source_rate > 0:
        G0 = G[:, :, :, 0, :]
        fsa = jnp.sum(G0 * op.fsw, axis=-1, keepdims=True)
        out = out.at[:, :, :, 0, :].add(-op.source_rate * fsa)
    return out


def cfl_rates(G, phi, Phi, op: VlasovOperators, L: int, M: int):
    """Rough maximum characteristic rates for the explicit time step."""
    vmax = np.sqrt(2.0 * L)
    stream = vmax * jnp.max(jnp.abs(op.c_stream)) / op.dz * 2.0  # ∂_z FD factor
    ky_max = op.kymax
    drift = jnp.max(jnp.abs(op.c_Ky)) * (2 * L + 2 * M) * ky_max + jnp.max(jnp.abs(op.c_Kx)) * (2 * L + 2 * M) / op.dx
    if op.nonlinear:
        Pr = jnp.max(jnp.abs(Phi))
        dPdx = jnp.max(jnp.abs(ddx(Phi, op.dx)))
        nl = jnp.max(op.c_Cinv) * (dPdx * ky_max + Pr * ky_max / op.dx) * 2.0
    else:
        nl = 0.0
    return stream + drift + nl
