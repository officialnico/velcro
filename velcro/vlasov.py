"""Right-hand side of the Hermite–Laguerre gyrokinetic Vlasov equation (§2.1, §3).

Normalisation: perpendicular lengths in ρ_ref, parallel in L_ref, time in
L_ref/v_t,ref with v_t = sqrt(2T/m), φ̂ = (eφ/T_ref)(L_ref/ρ_ref), Ĝ = (G/F0)(L_ref/ρ_ref).
Single ion species (Z = 1), adiabatic electrons, electrostatic.  With
``Φ_{ℓm} = δ_{ℓ0} 𝒥_m φ̂`` (gyroaveraged potential as a Laguerre field) and
``H = G + Φ / T̂_i``:

    ∂_t G = − √T̂ gpar  V ∂_z H                                  (streaming + E∥)
            + √T̂ gpar ∂_z ln B  Mirror H                        (mirror force)
            − ½ B⁻¹ (2V² + Mu) [K_x ∂_x (T̂ G + Φ) + i k_y K_y (T̂ G + Φ)]   (∇B/curvature drift)
            − ½ C⁻¹ i k_y [ω_n + ω_T (v̂² + μ̂ − 3/2)] Φ           (−v_E·∇F0 drive; − ω_B μ̂ only if physics.drive_gradB)
            − ½ C⁻¹ {Φ, G}_{xy}                                  (E×B nonlinearity, skew form)
            + ν C_coll H − hyperdiffusion − Krook buffers − sources       (C on G with physics.collisions_on = "g")

Velocity-space operators are the exact matrices of ``velcro.basis``; the
nonlinearity is evaluated pseudo-spectrally in μ on Gauss–Laguerre nodes
(exact Galerkin projection with N_μ = M nodes) and in (x, y) by the Orszag
method with the 2/3 rule.  Basis-variation terms of order (ρ/L)² relative to
the retained terms (∂_x of the local normalisation acting on G and Φ) are
dropped.  The ∇B part of ∇F0 at fixed μ (−ω_B μ̂ Φ) is cancelled by the v_E·μ∇B part of v̇_∥ ∂F0/∂v_∥
and is off by default (``physics.drive_gradB``); with it the growth rates are 12–15 % below GX.

Performance notes (all exact to roundoff, checked against the frozen baseline in
``tests/test_perf_equivalence.py``): the streaming operator ``V`` is tridiagonal in ℓ and is
applied as two shifted axpys instead of a dense ``(LM × LM)`` matmul; the hyper-collision
matrix is diagonal and applied element-wise; the collision matrix is split into its diagonal
and its few off-diagonal restoring entries; the mirror, magnetic-drift and drive velocity matrices (``Mirror − ½V``,
``Mirror``, ``2V² + Mu``, ``V² + Mu − 3/2``, ``Mu``) are banded in ``(ℓ, m)`` — at most five entries per row, ``Mu P``
being bidiagonal — and are applied as ``(ℓ, m)`` stencils (:func:`apply_banded`) instead of dense ``(LM × LM)`` matmuls
(2.6x per step in fp64 at (32,16), 1.5x in fp32); the drive's act on ``Φ`` zero-padded to an ``(L, M, …)`` field
(ℓ = 0 only), which XLA fuses away.  On the GPU no velocity-space operator goes to cuBLAS: the Laguerre ↔ node
transforms are fused multiply–adds (:func:`contract`) and the Lawson drift propagator is four fused passes in a
node-minor layout (:func:`propagate_nodes`); as einsums these small-K contractions (the drive's ``(L, M, M)`` slices
among them) were 25–27 % of an adiabatic step on the GPU.
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
from velcro.operators import (BRACKETS, apply_coef, bracket as skew_bracket, ddx, ddz, fourier_dx, fourier_kx, frame_bracket,
                              gql_bracket, hyper_x, hyper_x_sheared, hyper_z, kx_grid, real_fourier_dx, x_dealias_mask)


class VlasovOperators(eqx.Module):
    """All precomputed arrays needed by :func:`rhs` (an Equinox pytree: arrays are
    traced, scalar/flag fields are static)."""
    # velocity-space matrices, reshaped to (L, M, L, M)
    V: Any; mirror: Any; energy: Any; drive_T: Any; mu: Any; coll: Any; hypv: Any
    mirror_skew: Any    # (L,M,L,M)  Mirror − ½V: exactly antisymmetric
    jac: Any; inv_jac: Any     # (Nx,1,Nz)  Jacobian J and 1/J, for the skew parallel form
    # structured forms of the above used by rhs (derived in build_operators)
    V_lo: Any           # (L,1,1,1,1)  V[ℓ, ℓ−1]
    V_up: Any           # (L,1,1,1,1)  V[ℓ, ℓ+1]
    hypv_diag: Any      # (L,M,1,1,1)  diagonal of hypv
    coll_diag: Any      # (L,M,1,1,1)  diagonal of coll
    coll_off_vals: Any  # (n_off,)     off-diagonal entries of coll, indices in coll_off_idx
    drive_T0: Any       # (L,M,M)      drive_T[:, :, 0, :]  (acts on Φ, which lives at ℓ = 0)
    mu0: Any            # (L,M,M)      mu[:, :, 0, :]
    # Laguerre node transforms
    lag_to_nodes: Any   # (Nmu, M)
    nodes_to_lag: Any   # (M, Nmu)
    # coefficient fields (broadcastable to (Nx, Ny|1, Nz))
    c_stream: Any; c_mirror: Any; c_Kx: Any; c_Ky: Any; c_Cinv: Any; c_omegaB: Any
    Ti: Any; inv_Ti: Any; omega_n: Any; omega_T: Any; nu: Any; krook: Any
    ky: Any; phase: Any
    fsw: Any  # (Nx, Nz) flux-surface weights
    coll_off_idx: tuple = eqx.field(static=True)   # ((ℓ_i, m_i, ℓ_j, m_j), …) for coll_off_vals
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
    bracket_form: str = eqx.field(static=True, default="skew")
    # spectral radii of the velocity-space matrices, for cfl_rates_spectral
    rad_v: float = eqx.field(static=True, default=0.0)        # largest Gauss–Hermite root (= ρ(V))
    rad_energy: float = eqx.field(static=True, default=0.0)   # ρ(2v̂² + μ̂)
    rad_mirror: float = eqx.field(static=True, default=0.0)   # ‖Mirror‖₂
    parallel_skew: bool = eqx.field(static=True, default=False)
    gql_cutoff: int = eqx.field(static=True, default=-1)       # ≥ 0: generalised quasilinear E×B term (0 = quasilinear)
    split_drift_ky: bool = eqx.field(static=True, default=False)  # True: rhs omits −(2v̂²+μ̂) i k_y K_y T̂ G (Lawson)
    # velocity-node transforms for the drift integrating factor: 2V² = U_v diag(2 v_i²) U_vᵀ, Mu = U_mu diag(μ_j) U_muᵀ
    U_v: Any = None; U_mu: Any = None; lam_v: Any = None; lam_mu: Any = None
    # banded forms of mirror_skew, mirror and energy (see apply_banded): offsets are static, coefficients (L,M,1,1,1)
    mirror_skew_off: tuple = eqx.field(static=True, default=())
    mirror_off: tuple = eqx.field(static=True, default=())
    energy_off: tuple = eqx.field(static=True, default=())
    mirror_skew_band: Any = None; mirror_band: Any = None; energy_band: Any = None
    # banded forms of the drive's v̂² + μ̂ − 3/2 and of μ̂, applied to Φ zero-padded to (L, M, …) at ℓ = 0 (the kinetic path:
    # χ at ℓ = 0, 1): the (L, M, M) slices drive_T0 / mu0 as stencils instead of small-K contractions
    drive_off: tuple = eqx.field(static=True, default=())
    mu_off: tuple = eqx.field(static=True, default=())
    drive_band: Any = None; mu_band: Any = None
    # β' ≠ 0 (geometry Kx_cv/Ky_cv): the curvature drift's excess ½K_cv/B, acting through 2v̂² alone (banded vpar2_*);
    # None = curvature drift = ∇B drift, the operator unchanged
    c_Kx_cv: Any = None; c_Ky_cv: Any = None
    # physics.coriolis_mach ≠ 0: GS2's Coriolis drift, mach ½K_co/B (Kx_co/Ky_co of the geometry), acting through v̂ (the
    # Hermite V) on (T/Z)H / v̂t_s; None = off
    c_Kx_co: Any = None; c_Ky_co: Any = None
    coriolis_mach: float = eqx.field(static=True, default=0.0)
    vpar2_off: tuple = eqx.field(static=True, default=())
    vpar2_band: Any = None
    # physics.ext_zonal: the external zonal potential gyroaveraged at the μ nodes, (Nmu, Nx, 1, Nz), or None (off)
    ext_nodes: Any = None
    # physics.exb_shear (see config.ExBShearSpec): exb_x = γ_E x (Nx, 1, 1) for the local term −i k_y γ_E x G, or None;
    # exb_frame: the shearing frame (rhs takes ``theta``, (1, Nky, 1), and uses ∂_x − iθ in the radial drift);
    # pvg_T0 (L, M, M) = (2v̂)[:, :, ℓ′ = 0, :] and pvg_u = ω_u for the parallel-velocity-gradient drive, or None
    exb_x: Any = None
    exb_frame: bool = eqx.field(static=True, default=False)
    exb_hyper_grid: bool = eqx.field(static=True, default=False)   # hyper_x on the frame representation (GX's grid k_x)
    exb_xg: Any = None          # (Nx,) the radial grid, for the frame bracket's phases e^{−iθx} (operators.frame_bracket)
    pvg_T0: Any = None
    pvg_u: float = eqx.field(static=True, default=0.0)
    pvg_prof: Any = None        # (Nx, 1, Nz) 1/B(z) of physics.exb_shear.u_prime_over_B, or None (constant ω_u)
    # numerics.radial_drift / bracket_dx = "fourier" and numerics.x_dealias on the adiabatic path (the
    # kinetic path carries its own on KineticOperators): kx_sym (Nx, 1, 1) the symbol of the Fourier ∂_x (Nyquist 0),
    # x_mask (Nx, 1, 1) 1 on the retained |k_x| ≤ (N_x−1)//3 Δk_x, kx_bound the largest |k_x| the Fourier ∂_x returns on
    # the retained modes (the CFL bound in place of the stencil's 1.3722/Δx)
    kx_sym: Any = None
    x_mask: Any = None
    drift_fourier: bool = eqx.field(static=True, default=False)
    bracket_fourier: bool = eqx.field(static=True, default=False)
    x_dealias: bool = eqx.field(static=True, default=False)
    kx_bound: float = eqx.field(static=True, default=0.0)
    # physics.collision_model = "landau" / "gs2": the dense (ℓ, m) matrix is applied by apply_vs instead of the diagonal + the
    # few restoring entries; rad_coll = its spectral radius for the CFL bound
    coll_dense: bool = eqx.field(static=True, default=False)
    rad_coll: float = eqx.field(static=True, default=0.0)
    coll_on_h: bool = eqx.field(static=True, default=False)      # physics.collisions_on = "h" (the config default since
    #                                                              2026-10-03): C acts on H, not G
    # physics.collision_flr (velcro/collisions_flr.py): the gyrokinetic FLR increment C(b) − C(0) applied in k_x space
    # (x-periodic flux tube, symbols at the reference surface).  flr_kind "dougherty" / "lenard-bernstein" / "landau";
    # flr_b (Nkx, Nky, Nz); flr_vec (4, M, Nkx, Nky, Nz) the Dougherty restoring vectors; Landau: flr_K / flr_CF (L, M, L, M),
    # the field part's sectors flr_F0 (L, K₀, L, K₀) / flr_F1 (L, K₁, L, K₁) and their Bessel maps flr_q0 (K₀, M, Nkx, Nky,
    # Nz) / flr_q1 (K₁, M, …); rad_flr bounds ‖C(b) − C(0)‖₂
    coll_flr: bool = eqx.field(static=True, default=False)
    flr_kind: str = eqx.field(static=True, default="")
    flr_b: Any = None
    flr_vec: Any = None
    flr_K: Any = None
    flr_CF: Any = None
    flr_F0: Any = None
    flr_F1: Any = None
    flr_q0: Any = None
    flr_q1: Any = None
    rad_flr: float = eqx.field(static=True, default=0.0)
    # numerics.hyper_kz on the adiabatic path (the kinetic path carries its own): GX's |k∥|-scaled
    # Hermite hypercollisions −ν ℓ^p |k_z| G_ℓ for ℓ ≥ 3.  hkz = ν (v_t/R), hkz_ell (L−3, 1, 1, 1, 1) = ℓ^p, kz_abs
    # (Nx, Nky, Nz) = |n − θ/2π| on the Bloch modes of each field line and bloch the matching phase (kinetic._bloch_kz);
    # 0 / None = off (the default: the path is then bitwise the old one)
    hkz: float = eqx.field(static=True, default=0.0)
    hkz_ell: Any = None
    kz_abs: Any = None
    bloch: Any = None


def _vs(mat: np.ndarray, L: int, M: int) -> jnp.ndarray:
    return jnp.asarray(mat.reshape(L, M, L, M))


def apply_vs(mat: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """Apply an (L,M,L,M) velocity-space matrix to G of shape (L,M,...)."""
    return jnp.einsum("lmpq,pq...->lm...", mat, G)


def _parts(X):
    """(Re X, Im X) of a complex array, (X,) of a real one."""
    return (jnp.real(X), jnp.imag(X)) if jnp.iscomplexobj(X) else (X,)


def _whole(parts):
    return jax.lax.complex(*parts) if len(parts) == 2 else parts[0]


def _madd(terms):
    """Σ_k c_k ⊙ x_k over (c_k, x_k) pairs, x_k as :func:`_parts` (a fixed left-to-right order)."""
    acc = None
    for c, xs in terms:
        t = [c * x for x in xs]
        acc = t if acc is None else [a + b for a, b in zip(acc, t)]
    return acc


def contract(C, X, axis: int):
    """``Σ_k C[i, k] X[…, k, …]``: the real (I, K) matrix ``C`` applied along ``axis`` of ``X`` (size K there, I in the
    result), as K broadcast multiply–adds on the real and imaginary parts, which XLA fuses into one elementwise kernel
    with its neighbours.  As a jnp.einsum, these K = 4–16 contractions (the Laguerre ↔ node transforms) went to cuBLAS
    as 7–11 kernels per call at 40–60 % of DRAM bandwidth plus layout copies.
    Equal to the einsum up to the order of the K-term sums."""
    axis = axis % X.ndim
    shape = [1] * X.ndim
    shape[axis] = C.shape[0]
    xs = _parts(X)
    return _whole(_madd((C[:, k].reshape(shape), [jax.lax.index_in_dim(x, k, axis) for x in xs])
                        for k in range(X.shape[axis])))


def propagate_nodes(G, phase, Uv, Umu):
    """``U (phase ⊙ Uᵀ G)`` with ``U = U_v ⊗ U_mu`` for G (…, L, M, N_x, N_ky, N_z) and ``phase`` (…, N_x, N_ky, N_z,
    L, M), the node-minor layout of :func:`drift_phase`.  Four elementwise passes (U_vᵀ over ℓ, U_muᵀ over m with the
    phase, U_mu, U_v), each writing its node indices minor-most, so the outputs that share an input are neighbouring
    threads and every input is read from memory once (with the natural layout, the I outputs along the transformed axis
    re-read each input I times); then one transpose back to (…, L, M, N_x, N_ky, N_z).  Equal to the einsum form up to
    the order of the sums (the einsums were 21.5 % of the (16,8) adiabatic step)."""
    a = G.ndim - 5                                                                 # leading axes (species)
    L, M = G.shape[a], G.shape[a + 1]
    gs = _parts(G)
    # T1 (…, M, x, y, z, i) = Σ_ℓ U_v[ℓ, i] G[…, ℓ, m, x, y, z]
    T1 = _whole(_madd((Uv[l], [jax.lax.index_in_dim(g, l, a, keepdims=False)[..., None] for g in gs]) for l in range(L)))
    t1 = _parts(T1)
    # T2 (…, x, y, z, i, j) = phase ⊙ Σ_m U_mu[m, j] T1[…, m, x, y, z, i]
    T2 = phase * _whole(_madd((Umu[m], [jax.lax.index_in_dim(t, m, a, keepdims=False)[..., None] for t in t1])
                              for m in range(M)))
    # XLA may fuse a pass into the next (in this layout the recomputed inputs are shared by neighbouring threads, so one
    # level costs little), but never all four into one kernel: T2 is materialised
    t2 = _parts(jax.lax.optimization_barrier(T2))
    # T3 (…, x, y, z, i, m) = Σ_j U_mu[m, j] T2[…, x, y, z, i, j]
    T3 = _whole(_madd((Umu[:, j], [t[..., j, None] for t in t2]) for j in range(M)))
    t3 = _parts(T3)
    # D (…, x, y, z, ℓ, m) = Σ_i U_v[ℓ, i] T3[…, x, y, z, i, m]
    D = _whole(_madd((Uv[:, i, None], [t[..., i, None, :] for t in t3]) for i in range(L)))
    return jnp.transpose(D, tuple(range(a)) + (a + 3, a + 4, a, a + 1, a + 2))


def flr_increment_hat(Xh, kind: str, b, vec=None, K=None, CF=None, sectors=(), momentum: bool = True):
    """C(b) − C(0) of ``physics.collision_flr`` applied to ``Xh`` (L, M, Nkx, Nky, Nz), Fourier in x (rate ν = 1;
    velcro/collisions_flr.py, the validation notes): dougherty / lenard-bernstein −b Xh, landau −b K Xh; then the
    J-dressed restoring minus its k⊥ = 0 value (which C(0) already holds): dougherty 2(a∥a∥ᵀ − a∥⁰a∥⁰ᵀ) + 2a⊥a⊥ᵀ +
    (4/3)(a_E a_Eᵀ − a_E⁰a_E⁰ᵀ); landau Σₙ wₙ 𝒬ₙᵀF⁽ⁿ⁾𝒬ₙ − C_F over ``sectors`` = ((F⁽ⁿ⁾, 𝒬ₙ(b), wₙ), …) (n = 0, w 1;
    n = 1, w 2 for ±1).  ``momentum`` False drops the momentum restoring (the e–i Dougherty part, which restores no
    electron momentum)."""
    rdt = Xh.real.dtype
    b = b.astype(rdt)
    L, M = Xh.shape[:2]
    if kind == "landau":
        out = -b * apply_vs(K.astype(rdt), Xh) - apply_vs(CF.astype(rdt), Xh)
        for F, q, w in sectors:
            qq = q.astype(rdt)
            QX = jnp.einsum("pm...,lm...->lp...", qq, Xh)                  # 𝒬ₙ X: the sector components (L, Kₙ, …)
            out = out + w * jnp.einsum("pm...,lp...->lm...", qq, apply_vs(F.astype(rdt), QX))
        return out
    out = -b * Xh
    if kind != "dougherty":
        return out
    v = vec.astype(rdt)                                                     # (4, M, …): a∥, a⊥, a_E0, a_E2
    s2 = float(np.sqrt(2.0))
    if momentum:
        if L > 1:                                                           # ∥ on ℓ = 1, a∥⁰ = e₀/√2: 2a∥⁰a∥⁰ᵀ = e₀e₀ᵀ
            u = jnp.sum(v[0] * Xh[1], axis=0)
            out = out.at[1].add(2.0 * v[0] * u).at[1, 0].add(-Xh[1, 0])
        u = jnp.sum(v[1] * Xh[0], axis=0)                                   # ⊥ on ℓ = 0 (zero at b = 0)
        out = out.at[0].add(2.0 * v[1] * u)
    uE = jnp.sum(v[2] * Xh[0], axis=0)                                      # energy: a_E on ℓ = 0 and ℓ = 2
    if L > 2:
        uE = uE + jnp.sum(v[3] * Xh[2], axis=0)
    out = out.at[0].add((4.0 / 3.0) * v[2] * uE)
    if L > 2:
        out = out.at[2].add((4.0 / 3.0) * v[3] * uE)
    uE0 = (-Xh[0, 1] if M > 1 else 0.0) + (Xh[2, 0] / s2 if L > 2 else 0.0)   # a_E⁰ = −e_(0,1) + e_(2,0)/√2 (truncated)
    if M > 1:
        out = out.at[0, 1].add((4.0 / 3.0) * uE0)
    if L > 2:
        out = out.at[2, 0].add(-(4.0 / 3.0) / s2 * uE0)
    return out


def flr_collisions(X, op: "VlasovOperators"):
    """The FLR increment of the adiabatic path's collision operator (rate ν = 1) on X (L, M, Nx, Nky, Nz): FFT in x, the
    per-(k_x, k_y, z) operator of :func:`flr_increment_hat`, inverse FFT."""
    Xh = jnp.fft.fft(X, axis=2)
    sectors = ((op.flr_F0, op.flr_q0, 1.0), (op.flr_F1, op.flr_q1, 2.0)) if op.flr_kind == "landau" else ()
    out = flr_increment_hat(Xh, op.flr_kind, op.flr_b, op.flr_vec, op.flr_K, op.flr_CF, sectors)
    return jnp.fft.ifft(out, axis=2)


def flr_reference_b(geom, ky, Lx: float, Nx: int, rho2_scale: float) -> np.ndarray:
    """b(k_x, k_y, z) ≥ 0 at the reference surface (x index N_x/2), k_x in FFT order: ρ²/2 · k⊥² with
    ρ²/2 = ``rho2_scale``/B² (½T̂ for the adiabatic ions; ½T̂m̂/Z² per kinetic species), the y-averaged metric — the
    symbols of the flux-tube FFT field solvers."""
    i0 = Nx // 2
    kx = kx_grid(Nx, Lx)
    gxx, gxy, gyy, B = (np.asarray(geom.y_averaged(a))[i0, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    ky = np.asarray(ky, dtype=float)
    k2 = (gxx[None, None, :] * kx[:, None, None] ** 2 + 2 * gxy[None, None, :] * kx[:, None, None] * ky[None, :, None]
          + gyy[None, None, :] * ky[None, :, None] ** 2)
    return np.maximum(rho2_scale / B[None, None, :] ** 2 * k2, 0.0)


def apply_tridiag_l(lo: jnp.ndarray, up: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """``(A G)_ℓ = lo_ℓ G_{ℓ−1} + up_ℓ G_{ℓ+1}`` for a matrix tridiagonal in ℓ with zero diagonal
    and identity in m (the Hermite ``v̂`` recurrence)."""
    z = jnp.zeros_like(G[:1])
    down = jnp.concatenate([z, G[:-1]], axis=0)     # G_{ℓ−1}
    upsh = jnp.concatenate([G[1:], z], axis=0)      # G_{ℓ+1}
    return lo * down + up * upsh


def _band(mat: np.ndarray, L: int, M: int):
    """Banded form of an ``(LM × LM)`` matrix: ``(offsets, coefs)`` with ``(A G)_{ℓm} = Σ_i coefs[i][ℓ, m] G[ℓ+dℓ_i, m+dm_i]``
    (entries outside the range are zero); ``coefs[i]`` has shape ``(L, M, 1, 1, 1)``."""
    m4 = np.asarray(mat).reshape(L, M, L, M)
    li, mi, lj, mj = np.nonzero(m4)
    dl_all, dm_all = lj - li, mj - mi
    offsets, coefs = [], []
    for dl, dm in sorted(set(zip(dl_all.tolist(), dm_all.tolist()))):
        sel = (dl_all == dl) & (dm_all == dm)
        c = np.zeros((L, M))
        c[li[sel], mi[sel]] = m4[li[sel], mi[sel], lj[sel], mj[sel]]
        offsets.append((int(dl), int(dm))); coefs.append(c.reshape(L, M, 1, 1, 1))
    return tuple(offsets), tuple(coefs)


def apply_banded(offsets: tuple, coefs, G: jnp.ndarray) -> jnp.ndarray:
    """Apply a banded ``(ℓ, m)`` matrix (from :func:`_band`) to ``G`` of shape ``(L, M, ...)`` as a stencil:
    one shifted, zero-padded copy of ``G`` per band (a handful), no ``(LM × LM)`` matmul."""
    L, M = G.shape[:2]
    out = None
    for (dl, dm), c in zip(offsets, coefs):
        sl = G[max(dl, 0):L + min(dl, 0), max(dm, 0):M + min(dm, 0)]           # sl[ℓ, m] = G[ℓ+dℓ, m+dm]
        pad = [(max(-dl, 0), max(dl, 0)), (max(-dm, 0), max(dm, 0))] + [(0, 0)] * (G.ndim - 2)
        term = c * jnp.pad(sl, pad)
        out = term if out is None else out + term
    return jnp.zeros_like(G) if out is None else out


def _split_lm_matrix(mat: np.ndarray, L: int, M: int):
    """Diagonal of an ``(LM × LM)`` matrix as ``(L,M,1,1,1)`` plus its off-diagonal entries."""
    d = np.diag(mat).reshape(L, M, 1, 1, 1)
    off = mat - np.diag(np.diag(mat))
    ii, jj = np.nonzero(off)
    idx = tuple((int(i // M), int(i % M), int(j // M), int(j % M)) for i, j in zip(ii, jj))
    return d, idx, off[ii, jj]


def _node_transforms(V: np.ndarray, Mu: np.ndarray) -> dict:
    """Orthogonal eigen-decompositions of the Hermite ``v̂`` and Laguerre ``μ̂`` matrices.  Their eigenvalues are
    the Gauss–Hermite / Gauss–Laguerre nodes, so ``2v̂² + μ̂`` (the magnetic-drift velocity factor) is diagonal
    in the transformed space with entries ``2 v_i² + μ_j``; the eigenvectors are orthonormal (no ``e^{μ/2}``)."""
    v, Uv = np.linalg.eigh(V)
    mu, Um = np.linalg.eigh(Mu)
    return dict(U_v=jnp.asarray(Uv), U_mu=jnp.asarray(Um), lam_v=jnp.asarray(2.0 * v ** 2), lam_mu=jnp.asarray(mu))


def drift_phase(h, op: "VlasovOperators"):
    """``exp(−i h (2v_i²+μ_j) k_y K̂_y(x,z) T̂_i(x))``: the exact propagator of the binormal magnetic drift acting on
    ``G`` (the part :func:`rhs` omits when ``split_drift_ky``), in node space, with shape ``(Nx, Nky, Nz, L, M)`` (node
    indices minor-most: the layout :func:`propagate_nodes` reads; only :func:`apply_drift_propagator` consumes it).
    Axisymmetric geometry only (``K̂_y`` independent of ``y``).  A kinetic-electron operator (velcro/kinetic.py) gives
    its per-species phase, (S, Nx, Nky, Nz, L, M)."""
    if getattr(op, "kinetic", False):
        return op.drift_phase(h)
    theta = h * op.ky * op.Ti * op.c_Ky                                           # (Nx, Nky, Nz)
    lam = op.lam_v[:, None] + op.lam_mu[None, :]                                  # (L, M)
    arg = theta[..., None, None] * lam
    if op.c_Ky_cv is not None:      # β' ≠ 0: the curvature excess rotates the v∥² nodes only (2v_i², diagonal there too)
        arg = arg + (h * op.ky * op.Ti * op.c_Ky_cv)[..., None, None] * op.lam_v[:, None]
    return jnp.exp(-1j * arg)


def apply_drift_propagator(G, phase, op: "VlasovOperators"):
    """``U (phase ⊙ Uᵀ G)`` with the separable transform ``U = U_v ⊗ U_mu`` (an L×L and an M×M product, as fused
    passes: :func:`propagate_nodes`); species by species for a kinetic-electron operator."""
    if getattr(op, "kinetic", False):
        return op.propagate(G, phase)
    return propagate_nodes(G, phase, op.U_v, op.U_mu)


def hyper_kz_operators(cfg, geom, ky: np.ndarray, dtype) -> dict:
    """The :class:`VlasovOperators` entries of ``numerics.hyper_kz`` on the adiabatic path, empty
    when it is off.  GX's ``hypercollisions_kz`` (linear.cu: ν = nu_hyper_m (p+½)/(L−1)^(p+½) · 2.3 · v_t |∇∥z|, applied as
    −ν ℓ^p |k_z| g for ℓ ≥ 3) with the kinetic path's conversion to Velcro units (kinetic.build_kinetic_operators: a GX
    rate carries (R/a)/√2 and ∇∥z_GX = a/(qR), so ν = hyper_kz (p+½)/(L−1)^(p+½) · 2.3 · v̂t ⟨|b·∇z|⟩/√2, v̂t = 1 for the
    ions); |k_z| along each field line with Velcro's own parallel boundary, as the kinetic path's parallel_bc 'velcro'.
    It acts along z at fixed (x, k_y), so it commutes with the shearing frame's e^{−iθx}."""
    hk = cfg.numerics.hyper_kz
    if hk <= 0 or cfg.physics.electrons == "kinetic":
        return {}
    from velcro.kinetic import _bloch_kz            # (kinetic imports this module: imported at call time)
    L, p = cfg.grid.L, cfg.numerics.hyper_kz_p
    if L <= 3:
        raise ValueError("numerics.hyper_kz acts on the Hermite moments ℓ ≥ 3: it needs grid.L > 3")
    rdt = jnp.zeros((), dtype).real.dtype
    gref = float(np.mean(np.abs(np.asarray(geom.gpar))))                 # ⟨|b·∇z|⟩ (= 1/q in s-α)
    nu = hk * (p + 0.5) / max(L - 1, 1) ** (p + 0.5) * 2.3 * 1.0 * gref / np.sqrt(2.0)
    kz, bl = _bloch_kz(geom, np.asarray(ky, dtype=float), len(geom.z))
    ell = (np.arange(3, L, dtype=float) ** p).reshape(-1, 1, 1, 1, 1)
    return dict(hkz=float(nu), hkz_ell=jnp.asarray(ell, dtype=rdt), kz_abs=jnp.asarray(np.abs(kz), dtype=rdt),
                bloch=jnp.asarray(bl, dtype=dtype))


def build_operators(geom, prof: dict, cfg, ky: np.ndarray, twist_phase: np.ndarray, dtype) -> VlasovOperators:
    L, M = cfg.grid.L, cfg.grid.M
    Nx, Ny, Nz = len(geom.x), cfg.grid.Ny, len(geom.z)
    I_L, I_M = np.eye(L), np.eye(M)
    V, Mu = basis.hermite_v_matrix(L), basis.laguerre_mu_matrix(M)
    Vmat = np.kron(V, I_M)
    mirror = basis.mirror_matrix(L, M)
    energy = basis.energy_matrix(L, M)                               # 2v² + μ
    vpar2 = 2.0 * np.kron(V @ V, I_M)                                # 2v² (the curvature drift's velocity factor)
    nodes = _node_transforms(V, Mu)
    if cfg.numerics.gx_hermite_closure and cfg.physics.electrons != "kinetic":   # adiabatic path (kinetic.py applies its
        # own): GX's v̂² on the last Hermite row, (V V)[:L,:L] with V of L+1 modes
        A = (basis.hermite_v_matrix(L + 1) @ basis.hermite_v_matrix(L + 1))[:L, :L]
        energy = 2.0 * np.kron(A, I_M) + np.kron(I_L, Mu)
        vpar2 = 2.0 * np.kron(A, I_M)
        a, Ua = np.linalg.eigh(A)
        nodes.update(U_v=jnp.asarray(Ua), lam_v=jnp.asarray(2.0 * a))
    drive_T = np.kron(V @ V, I_M) + np.kron(I_L, Mu) - 1.5 * np.eye(L * M)  # v² + μ − 3/2
    mu_mat = np.kron(I_L, Mu)
    coll = collision_matrix(cfg.physics.collision_model, L, M, cfg.physics.gs2_collision_terms, cfg.physics.gs2_const_v)
    l = np.arange(L)[:, None] * np.ones((1, M)); m = np.ones((L, 1)) * np.arange(M)[None, :]
    pw = cfg.numerics.hyper_v_power
    hyp = (l / max(L - 1, 1)) ** pw + (m / max(M - 1, 1)) ** pw
    hypv = cfg.numerics.hyper_v * np.diag(hyp.reshape(-1))
    # structured forms
    V_lo = np.array([V[i, i - 1] if i >= 1 else 0.0 for i in range(L)])
    V_up = np.array([V[i, i + 1] if i + 1 < L else 0.0 for i in range(L)])
    V_tri = np.diag(V_lo[1:], -1) + np.diag(V_up[:-1], 1)
    assert np.array_equal(V_tri, V), "hermite_v_matrix is expected to be tridiagonal with zero diagonal"
    assert np.array_equal(np.diag(np.diag(hypv)), hypv), "hyper-collision matrix is expected to be diagonal"
    coll_dense = cfg.physics.collision_model in ("landau", "gs2")
    if coll_dense:                      # the dense matrix goes through apply_vs; the split form stays empty
        coll_diag, coll_off_idx, coll_off_vals = np.zeros((L, M, 1, 1, 1)), (), np.zeros(0)
    else:
        coll_diag, coll_off_idx, coll_off_vals = _split_lm_matrix(coll, L, M)
    mirror_skew_off, mirror_skew_band = _band(mirror - 0.5 * Vmat, L, M)
    mirror_off, mirror_band = _band(mirror, L, M)
    energy_off, energy_band = _band(energy, L, M)
    assert max(len(mirror_skew_off), len(mirror_off), len(energy_off)) <= 8, "velocity matrices are expected to be banded"
    drive_off, drive_band = _band(drive_T, L, M)
    mu_off, mu_band = _band(mu_mat, L, M)
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
    cv = {}
    mach = float(cfg.physics.coriolis_mach)
    if mach != 0.0:                                  # GS2's Coriolis drift (native Miller geometry carries its arrays)
        if getattr(geom, "Ky_co", None) is None:
            raise ValueError(f"physics.coriolis_mach = {mach} needs the Coriolis geometry (cdrift, cdrift0), which only "
                             f"geometry.kind = 'miller' computes (this tube: {geom.name!r})")
        cv.update(c_Kx_co=jnp.asarray(mach * 0.5 * geom.Kx_co / geom.B), c_Ky_co=jnp.asarray(mach * 0.5 * geom.Ky_co / geom.B),
                  coriolis_mach=mach)
    if getattr(geom, "Ky_cv", None) is not None:     # β' ≠ 0: the curvature drift's excess, on 2v̂² only
        vpar2_off, vpar2_band = _band(vpar2, L, M)
        cv.update(c_Kx_cv=jnp.asarray(0.5 * geom.Kx_cv / geom.B), c_Ky_cv=jnp.asarray(0.5 * geom.Ky_cv / geom.B),
                  vpar2_off=vpar2_off, vpar2_band=tuple(jnp.asarray(c) for c in vpar2_band))
    c_Cinv = 0.5 * geom.Cinv
    # Krook buffer profile
    x = geom.x
    Lx = cfg.grid.Lx
    wb = cfg.numerics.krook_width * Lx
    d = np.minimum(x - x.min(), x.max() - x)
    krook = cfg.numerics.krook_rate * 0.5 * (1 + np.cos(np.pi * np.clip(d / max(wb, 1e-12), 0, 1))) if wb > 0 else np.zeros_like(x)
    krook = krook[:, None, None]
    if not cfg.numerics.krook_zonal:                # buffers leave k_y = 0 alone → (Nx, Nky, 1)
        krook = krook * (ky != 0)[None, :, None]
    ky_j = jnp.asarray(ky)[None, :, None]
    phase = jnp.exp(-1j * jnp.asarray(ky)[None, :, None] * jnp.asarray(twist_phase)[:, None, None])
    a = lambda arr: jnp.asarray(np.asarray(arr), dtype=dtype)
    r = lambda arr: jnp.asarray(np.asarray(arr))
    exb = exb_operators(cfg, geom, V, M, r)
    fx = fourier_x_operators(cfg, Nx, float(Lx / Nx), r)
    flr = flr_operators(cfg, geom, prof, ky, L, M, r) if (cfg.physics.collision_flr and cfg.physics.electrons != "kinetic") else {}
    hkz = hyper_kz_operators(cfg, geom, ky, dtype)
    return VlasovOperators(**exb, **fx, **flr, **hkz, coll_dense=coll_dense, rad_coll=float(np.abs(np.linalg.eigvals(coll)).max()) if coll_dense else 0.0,
        coll_on_h=cfg.physics.collisions_on == "h" and cfg.physics.electrons != "kinetic",
        drive_off=drive_off, drive_band=tuple(r(c) for c in drive_band), mu_off=mu_off, mu_band=tuple(r(c) for c in mu_band),
        mirror_skew=_vs(mirror - 0.5 * Vmat, L, M), jac=r(geom.jac), inv_jac=r(1.0 / geom.jac),
        parallel_skew=cfg.numerics.parallel_form == "skew" and bool(geom.axisymmetric),
        gql_cutoff=-1 if cfg.physics.gql_cutoff is None else int(cfg.physics.gql_cutoff),
        V=_vs(Vmat, L, M), mirror=_vs(mirror, L, M), energy=_vs(energy, L, M), drive_T=_vs(drive_T, L, M),
        mu=_vs(mu_mat, L, M), coll=_vs(coll, L, M), hypv=_vs(hypv, L, M),
        V_lo=r(V_lo.reshape(L, 1, 1, 1, 1)), V_up=r(V_up.reshape(L, 1, 1, 1, 1)),
        hypv_diag=r(np.diag(hypv).reshape(L, M, 1, 1, 1)), coll_diag=r(coll_diag),
        coll_off_vals=r(coll_off_vals), coll_off_idx=coll_off_idx,
        drive_T0=r(drive_T.reshape(L, M, L, M)[:, :, 0, :]), mu0=r(mu_mat.reshape(L, M, L, M)[:, :, 0, :]),
        lag_to_nodes=r(lag_to_nodes), nodes_to_lag=r(nodes_to_lag),
        c_stream=r(c_stream), c_mirror=r(c_mirror), c_Kx=r(c_Kx), c_Ky=r(c_Ky), c_Cinv=r(c_Cinv),
        c_omegaB=r(geom.omega_B if cfg.physics.drive_gradB else 0.0 * geom.omega_B), Ti=r(prof["Ti"][:, None, None]), inv_Ti=r(1.0 / prof["Ti"][:, None, None]),
        omega_n=r(prof["omega_n"][:, None, None]), omega_T=r(prof["omega_T"][:, None, None]),
        nu=r(prof["nu"][:, None, None]), krook=r(krook),
        ky=ky_j, kymax=float(ky.max()) if ky.max() > 0 else 1.0, phase=phase,
        dx=float(Lx / Nx), dz=float(2 * np.pi / Nz), Ny=Ny, axisymmetric=bool(geom.axisymmetric),
        fsw=r(geom.fs_weights()),
        hyper_x=cfg.numerics.hyper_x, hyper_z=cfg.numerics.hyper_z, hyper_ky=cfg.numerics.hyper_ky,
        source_rate=cfg.numerics.source_rate, nonlinear=cfg.physics.nonlinear,
        bracket_form=cfg.numerics.bracket,
        rad_v=float(np.abs(np.linalg.eigvalsh(V)).max()), rad_energy=float(np.abs(np.linalg.eigvalsh(energy)).max()),
        rad_mirror=float(np.linalg.norm(mirror, 2)),
        split_drift_ky=cfg.numerics.integrator == "lawson" or (cfg.numerics.integrator == "imex"
                                                                and cfg.numerics.imex_lawson),
        mirror_skew_off=mirror_skew_off, mirror_skew_band=tuple(r(c) for c in mirror_skew_band),
        mirror_off=mirror_off, mirror_band=tuple(r(c) for c in mirror_band),
        energy_off=energy_off, energy_band=tuple(r(c) for c in energy_band),
        **nodes, **cv,
    )


def flr_operators(cfg, geom, prof: dict, ky, L: int, M: int, r) -> dict:
    """The :class:`VlasovOperators` entries of ``physics.collision_flr`` on the adiabatic path (the ions' b at the
    reference surface, b = ½T̂_i k⊥²/B²; velcro/collisions_flr.py)."""
    from velcro import collisions_flr as cf
    cf.check_config(cfg)
    model = cfg.physics.collision_model
    Nx = len(geom.x)
    b = flr_reference_b(geom, ky, cfg.grid.Lx, Nx, 0.5 * float(np.asarray(prof["Ti"])[Nx // 2]))
    d = cf.flr_data(model, L, M, b)
    # the CFL bound: the FLR increment, plus ρ(C(0)) for the models whose k⊥ = 0 radius the adiabatic rate does not count
    out = dict(coll_flr=True, flr_kind=model, flr_b=r(d["b"]), rad_flr=float(d["rad"]) + float(d.get("rad0", 0.0)))
    if model == "dougherty":
        out["flr_vec"] = r(d["vec"])
    elif model == "landau":
        out.update(flr_K=r(d["K"]), flr_CF=r(d["CF"]), flr_F0=r(d["F0"]), flr_F1=r(d["F1"]), flr_q0=r(d["q0"]),
                   flr_q1=r(d["q1"]))
    return out


def exb_frame_kind(cfg) -> str:
    """"" (off), "local" or "shearing" for ``physics.exb_shear`` (``auto``: shearing without Krook buffers)."""
    ex = cfg.physics.exb_shear
    if ex is None or (ex.gamma_E == 0.0 and ex.u_prime == 0.0):
        return ""
    if ex.frame != "auto":
        return ex.frame
    buffered = cfg.numerics.krook_rate > 0 and cfg.numerics.krook_width > 0
    return "local" if buffered else "shearing"


def exb_operators(cfg, geom, V: np.ndarray, M: int, r) -> dict:
    """The :class:`VlasovOperators` entries of ``physics.exb_shear`` (empty when off: the operators are the old ones)."""
    kind = exb_frame_kind(cfg)
    if not kind:
        return {}
    ex = cfg.physics.exb_shear
    out = {}
    if ex.gamma_E != 0.0:
        if kind == "local":
            out["exb_x"] = r((ex.gamma_E * np.asarray(geom.x))[:, None, None])
        else:
            if cfg.numerics.field_solver != "fft":
                raise ValueError("physics.exb_shear in the shearing frame needs numerics.field_solver = 'fft' (the "
                                 "symbols move with the frame every step); or frame = 'local' in a buffered box")
            if cfg.physics.nonlinear and (cfg.numerics.bracket != "skew" or cfg.physics.gql_cutoff is not None):
                raise ValueError("physics.exb_shear in the shearing frame needs numerics.bracket = 'skew' without "
                                 "physics.gql_cutoff (operators.frame_bracket)")
            out["exb_frame"] = True
            out["exb_xg"] = r(np.asarray(geom.x, dtype=float))
            out["exb_hyper_grid"] = bool(ex.gx_hyper_grid)
    if ex.u_prime != 0.0:
        L = V.shape[0]
        T2v = (2.0 * np.kron(V, np.eye(M))).reshape(L, M, L, M)[:, :, 0, :]
        out["pvg_T0"], out["pvg_u"] = r(T2v), float(ex.u_prime)
        if ex.u_prime_over_B:                       # ω_u ∝ I/B along the field line (purely toroidal rotation)
            out["pvg_prof"] = r((1.0 / np.asarray(geom.y_averaged(geom.B)))[:, :1, :])
    return out


def fourier_x_operators(cfg, Nx: int, dx: float, r) -> dict:
    """The :class:`VlasovOperators` entries of ``numerics.radial_drift = "fourier"``, ``numerics.bracket_dx = "fourier"``
    and ``numerics.x_dealias`` on the adiabatic path (empty when all are off or the electrons are kinetic, whose
    operators carry their own): the symbol ``i k_x`` with the Nyquist mode 0, the 2/3 mask, and the CFL bound
    max |k_x| over the retained modes."""
    nm = cfg.numerics
    if cfg.physics.electrons == "kinetic" or not (nm.radial_drift == "fourier" or nm.bracket_dx == "fourier" or nm.x_dealias):
        return {}
    if nm.bracket_dx == "fourier" and (nm.bracket != "skew" or cfg.physics.gql_cutoff is not None):
        raise ValueError("numerics.bracket_dx = 'fourier' needs numerics.bracket = 'skew' without physics.gql_cutoff")
    kx = fourier_kx(Nx, dx)
    mask = x_dealias_mask(Nx) if nm.x_dealias else np.ones(Nx)
    return dict(kx_sym=r(kx.reshape(-1, 1, 1)), x_mask=r(mask.reshape(-1, 1, 1)) if nm.x_dealias else None,
                drift_fourier=nm.radial_drift == "fourier", bracket_fourier=nm.bracket_dx == "fourier",
                x_dealias=bool(nm.x_dealias), kx_bound=float(np.abs(kx * mask).max()))


def exb_remap(G: jnp.ndarray, theta: jnp.ndarray, dkx: float, x0: float = 0.0, mask=None):
    """The shearing frame's remap: where |θ_ky| > Δk_x/2 the frame phase is moved by one k_x and the spectrum of p along x
    shifted by one mode (exact: e^{∓iΔk_x x} is periodic); the mode that would wrap around the band is dropped (GX's
    g_shift).  ``theta`` (Nky,), ``x0`` the first grid point (the DFT of a grid starting at x₀ carries e^{ijΔk_x x₀}
    on mode j, so the shifted coefficients are c_{j±1} e^{∓iΔk_x x₀}: on Velcro's centred grid, x₀ = −L_x/2, the
    factor is −1).  ``mask`` (Nx, 1, 1), the 2/3 mask of ``numerics.x_dealias``: the rolled spectrum is masked, so the
    mode that leaves the retained band is dropped as GX's g_shift drops the cell beyond |kxbar| > nakx/2.  Returns (G, θ)."""
    Nx = G.shape[2]
    sgn = jnp.where(theta > 0.5 * dkx, 1, jnp.where(theta < -0.5 * dkx, -1, 0))
    theta = theta - sgn * dkx
    Gh = jnp.fft.fft(G, axis=2)
    ph = jnp.exp(-1j * dkx * x0).astype(Gh.dtype)
    # θ′ = θ − Δk_x needs p′ = e^{−iΔk_x x} p, whose coefficient j is c_{j+1} e^{−iΔk_x x₀} (roll −1: the entry at the
    # highest frequency, index (N_x − 1)//2, would receive the lowest one: dropped); θ′ = θ + Δk_x needs p′ = e^{+iΔk_x x} p:
    # roll +1 times e^{+iΔk_x x₀}, the lowest frequency's entry, index (N_x + 1)//2, dropped (N_x/2 − 1 and N_x/2 for even
    # N_x; the even-only indices shifted a legitimate mode out and kept the wrapped one for odd N_x)
    dn = (jnp.roll(Gh, -1, axis=2) * ph).at[:, :, (Nx - 1) // 2].set(0.0)
    up = (jnp.roll(Gh, 1, axis=2) * jnp.conj(ph)).at[:, :, (Nx + 1) // 2].set(0.0)
    sg = sgn.reshape((1, 1, 1, -1, 1))
    Gh = jnp.where(sg == 1, dn, jnp.where(sg == -1, up, Gh))
    if mask is not None:
        Gh = Gh * mask
    return jnp.fft.ifft(Gh, axis=2), theta


def rhs(G: jnp.ndarray, phi: jnp.ndarray, Phi: jnp.ndarray, op: VlasovOperators, Pn=None, theta=None) -> jnp.ndarray:
    """``∂_t G`` for ``G`` (L,M,Nx,Nky,Nz), ``φ`` (Nx,Nky,Nz), ``Φ`` gyroaveraged (M,Nx,Nky,Nz); ``theta`` (1, Nky, 1)
    the shearing-frame phase (``op.exb_frame``), else ignored."""
    if op.exb_frame and theta is None:
        # without θ every frame term would silently be the θ = 0 operator: right only before the first time step
        raise ValueError("rhs: the shearing-frame operator (physics.exb_shear) needs the frame phase theta — pass "
                         "sim.exb_theta at call time, or refuse the frame")
    L, M = G.shape[:2]
    ky = op.ky
    ax = op.axisymmetric
    coef = lambda c, f: apply_coef(c, f, op.Ny, ax)
    PhiL = jnp.pad(Phi[None], ((0, L - 1),) + ((0, 0),) * Phi.ndim)  # Φ as an (L, M, …) field at ℓ = 0, for the drive's stencils
    H = G.at[0].add(op.inv_Ti * Phi)

    # --- parallel streaming + E∥ :  −√T gpar V ∂_z H   (V tridiagonal in ℓ)
    if op.parallel_skew:
        # c ∂_z f = (1/2J)[w ∂_z f + ∂_z(w f)] + ½ c ∂_z lnB f with w = J c (∝ 1/B): the bracket is exactly skew in the
        # J-weighted inner product for an antisymmetric stencil, and Mirror − ½V is an antisymmetric matrix
        VH = apply_tridiag_l(op.V_lo, op.V_up, H)
        out = -0.5 * (op.c_stream * ddz(VH, op.dz, op.phase) + op.inv_jac * ddz(op.jac * op.c_stream * VH, op.dz, op.phase))
        out = out + op.c_mirror * apply_banded(op.mirror_skew_off, op.mirror_skew_band, H)
    else:
        dHdz = ddz(H, op.dz, op.phase)
        out = -coef(op.c_stream, apply_tridiag_l(op.V_lo, op.V_up, dHdz))
        # --- mirror force: +√T gpar ∂_z lnB  Mirror H
        out = out + coef(op.c_mirror, apply_banded(op.mirror_off, op.mirror_band, H))
    # --- magnetic drift: −½ B⁻¹ (2v²+μ) [Kx ∂_x(T G + Φ) + i ky Ky (T G + Φ)]
    TG = (op.Ti * G).at[0].add(Phi)
    dxo = (lambda f: fourier_dx(f, op.kx_sym)) if op.drift_fourier else (lambda f: ddx(f, op.dx))   # numerics.radial_drift
    dTGdx = (op.Ti * dxo(G)).at[0].add(dxo(Phi))
    if op.exb_frame and theta is not None:          # ∂_x → ∂_x − iθ_ky on the frame representation p = e^{iθx} G
        dTGdx = dTGdx - 1j * theta * TG
    if op.split_drift_ky:   # the k_y drift of G is advanced exactly by the Lawson factor; keep its action on Φ
        drift = coef(op.c_Kx, dTGdx).at[0].add(coef(op.c_Ky, 1j * ky * Phi))
    else:
        drift = coef(op.c_Kx, dTGdx) + coef(op.c_Ky, 1j * ky * TG)
    out = out - apply_banded(op.energy_off, op.energy_band, drift)
    if op.c_Ky_cv is not None:  # β' ≠ 0: the curvature drift's excess over the ∇B drift, through 2v̂² only
        if op.split_drift_ky:
            drift = coef(op.c_Kx_cv, dTGdx).at[0].add(coef(op.c_Ky_cv, 1j * ky * Phi))
        else:
            drift = coef(op.c_Kx_cv, dTGdx) + coef(op.c_Ky_cv, 1j * ky * TG)
        out = out - apply_banded(op.vpar2_off, op.vpar2_band, drift)
    if op.c_Ky_co is not None:  # GS2's Coriolis drift: v̂ (T/Z)H/v̂t (T̂_i/√T̂_i = √T̂_i for the ions), on all of TG
        sT = jnp.sqrt(op.Ti)
        co = coef(op.c_Kx_co, sT * dTGdx) + coef(op.c_Ky_co, 1j * ky * (sT * TG))
        out = out - apply_tridiag_l(op.V_lo, op.V_up, co)
    # --- drive: −½ C⁻¹ i ky [ω_n + ω_T (v²+μ−3/2) − ω_B μ] Φ
    drv = (op.omega_T * apply_banded(op.drive_off, op.drive_band, PhiL)
           - coef(op.c_omegaB, apply_banded(op.mu_off, op.mu_band, PhiL))).at[0].add(op.omega_n * Phi)
    if op.pvg_T0 is not None:                       # parallel-velocity-gradient drive: + 2v̂ ω_u Φ (physics.exb_shear.u_prime)
        pv = op.pvg_u * (2.0 * apply_tridiag_l(op.V_lo, op.V_up, PhiL))
        drv = drv + (pv if op.pvg_prof is None else op.pvg_prof * pv)
    out = out - op.c_Cinv * (1j * ky) * drv
    if op.exb_x is not None:                        # the sheared equilibrium E×B flow, local form: −i k_y γ_E x G
        out = out - (1j * ky) * op.exb_x * G
    # --- E×B nonlinearity: −½ C⁻¹ {Φ, G}, pseudo-spectral in μ
    if op.nonlinear:
        Gn = contract(op.lag_to_nodes, G, 1)                               # (L, Nmu, ...)
        if Pn is None:   # truncated Laguerre sum; ``Pn`` given = exact J0 node values (numerics.node_gyro)
            Pn = contract(op.lag_to_nodes, Phi, 0)                         # (Nmu, ...)
        if op.ext_nodes is not None:   # imposed zonal E×B flow (physics.ext_zonal): it advects, nothing else sees it
            Pn = Pn.at[:, :, :1, :].add(op.ext_nodes)
        if op.exb_frame and theta is not None:     # the shearing frame: the bracket of the physical fields at any θ
            dxp = (lambda f: fourier_dx(f, op.kx_sym)) if op.bracket_fourier else (lambda f: ddx(f, op.dx))
            br = frame_bracket(Pn[None], Gn, ky, op.dx, op.Ny, theta, op.exb_xg, dxp)
        elif op.gql_cutoff >= 0:
            br = gql_bracket(op.bracket_form, Pn[None], Gn, ky, op.dx, op.Ny, op.gql_cutoff)
        elif op.bracket_fourier:    # numerics.bracket_dx = "fourier": i k_x by FFT in the skew bracket (GX's pseudo-spectral
            # form), outside the shearing frame (in the frame the branch above takes the bracket of the physical fields)
            br = skew_bracket(Pn[None], Gn, ky, op.dx, op.Ny, dxf=real_fourier_dx(op.kx_sym))
        else:
            br = BRACKETS[op.bracket_form](Pn[None], Gn, ky, op.dx, op.Ny)     # (L, Nmu, ...)
        out = out - op.c_Cinv * contract(op.nodes_to_lag, br, 1)
    # --- collisions: diagonal part + the few off-diagonal restoring entries (dense for the Landau operator), on the
    # non-adiabatic H (the gyrokinetic operator's argument; physics.collisions_on = "h", the default since 2026-10-03)
    # or, with collisions_on = "g", on G (every adiabatic run before 2026-10-03)
    X = H if op.coll_on_h else G
    if op.coll_dense:
        cG = apply_vs(op.coll.astype(G.real.dtype), X)
    else:
        cG = op.coll_diag * X
        for n, (li, mi, lj, mj) in enumerate(op.coll_off_idx):
            cG = cG.at[li, mi].add(op.coll_off_vals[n] * X[lj, mj])
    if op.coll_flr:                                 # physics.collision_flr: the gyrokinetic FLR increment C(b) − C(0)
        cG = cG + flr_collisions(X, op)
    out = out + op.nu * cG
    # --- numerical dissipation
    if op.hyper_x > 0:
        if op.exb_frame and theta is not None and not op.exb_hyper_grid:   # the frame: the physical field's k_x − θ
            out = out - op.hyper_x * hyper_x_sheared(G, theta, op.dx)
        else:
            out = out - op.hyper_x * hyper_x(G)
    if op.hyper_z > 0:
        out = out - op.hyper_z * hyper_z(G, op.phase)
    if op.hyper_ky > 0:
        out = out - op.hyper_ky * (ky / op.kymax) ** 4 * G
    out = out - op.hypv_diag * G  # hyper-collisions (ν_h folded into hypv, diagonal)
    if op.hkz > 0:      # numerics.hyper_kz: GX's |k∥|-scaled Hermite hypercollisions −ν ℓ^p |k_z| G_ℓ, ℓ ≥ 3
        akz = jnp.fft.ifft(jnp.fft.fft(G[3:] * op.bloch, axis=-1) * op.kz_abs, axis=-1) * jnp.conj(op.bloch)
        out = out.at[3:].add(-op.hkz * op.hkz_ell * akz)
    # --- Krook buffers
    out = out - op.krook * G
    # --- optional profile-maintaining source on k_y = 0 flux-surface-averaged moments
    if op.source_rate > 0:
        G0 = G[:, :, :, 0, :]
        fsa = jnp.sum(G0 * op.fsw, axis=-1, keepdims=True)
        out = out.at[:, :, :, 0, :].add(-op.source_rate * fsa)
    if op.x_dealias:                                  # numerics.x_dealias: the 2/3 rule in k_x (GX's retained set)
        out = jnp.fft.ifft(jnp.fft.fft(out, axis=2) * op.x_mask, axis=2)
    return out


def drift_abs(op, axis: str):
    """|c_K| bounding the drift's rate per unit k: c_K (2v̂² + μ̂) + c_K,cv 2v̂² has the eigenvalues
    (c_K + c_K,cv) 2v_i² + c_K μ_j (2v̂² and μ̂ commute, both ≥ 0), at most max(|c_K|, |c_K + c_K,cv|) ρ(2v̂² + μ̂)."""
    c = op.c_Ky if axis == "y" else op.c_Kx
    cv = op.c_Ky_cv if axis == "y" else op.c_Kx_cv
    return jnp.abs(c) if cv is None else jnp.maximum(jnp.abs(c), jnp.abs(c + cv))


def cfl_rates(G, phi, Phi, op: VlasovOperators, L: int, M: int):
    """Rough maximum characteristic rates for the explicit time step."""
    vmax = np.sqrt(2.0 * L)
    stream = vmax * jnp.max(jnp.abs(op.c_stream)) / op.dz * 2.0  # ∂_z FD factor
    ky_max = op.kymax
    drift = jnp.max(drift_abs(op, "y")) * (2 * L + 2 * M) * ky_max + jnp.max(drift_abs(op, "x")) * (2 * L + 2 * M) / op.dx
    if op.c_Ky_co is not None:
        drift = drift + vmax * jnp.max(jnp.sqrt(op.Ti)) * (jnp.max(jnp.abs(op.c_Ky_co)) * ky_max + jnp.max(jnp.abs(op.c_Kx_co)) * 2 / op.dx)
    if op.nonlinear:
        Pr = jnp.max(jnp.abs(Phi))
        dPdx = jnp.max(jnp.abs(ddx(Phi, op.dx)))
        if op.ext_nodes is not None:                  # the imposed zonal flow adds to v_y
            dPdx = dPdx + jnp.max(jnp.abs(ddx(op.ext_nodes, op.dx)))
        nl =jnp.max(op.c_Cinv) * (dPdx * ky_max + Pr * ky_max / op.dx) * 2.0
    else:
        nl = 0.0
    if op.exb_x is not None:                      # the sheared flow advects at |γ_E x| k_y
        nl = nl + jnp.max(jnp.abs(op.exb_x)) * ky_max
    if op.coll_dense:
        nl = nl + jnp.max(jnp.abs(op.nu)) * op.rad_coll
    if op.coll_flr:                               # the FLR increment's radius bound (classical diffusion ν b_max …)
        nl = nl + jnp.max(jnp.abs(op.nu)) * op.rad_flr
    if op.hkz > 0:                                # numerics.hyper_kz: its largest rate ν (L−1)^p max|k_z|
        nl = nl + op.hkz * jnp.max(op.hkz_ell) * jnp.max(op.kz_abs)
    return stream + drift + nl


_FD4 = 1.3722   # max |symbol| of the 4th-order centred first-derivative stencil, in units of 1/Δ


def cfl_rates_spectral(G, phi, Phi, op: VlasovOperators, L: int, M: int, Pn=None):
    """Upper bound on the spectral radius of the right-hand side as the sum of the per-term radii,
    each from the actual spectral radius of its velocity-space matrix and the 4th-order stencil
    symbol.  Differences from :func:`cfl_rates`: the drift carries ``T̂_i(x)`` and ρ(2v̂²+μ̂) (not
    2L+2M), the mirror term is counted, streaming uses the largest Hermite root (not √(2L)), and the
    E×B rate uses ``max|k_y Φ|`` (the zonal potential does not advect in x) evaluated on the node
    values that are actually advected.  Measured on saturated Cyclone states (2026-09-18): bound /
    Arnoldi radius = 1.4–1.5 at (16, 8) and (32, 16); ``cfl_rates`` gives 1.02 and 0.39–0.67."""
    stream = op.rad_v * jnp.max(jnp.abs(op.c_stream)) * _FD4 / op.dz
    mirror = op.rad_mirror * jnp.max(jnp.abs(op.c_mirror))
    drift_ky = 0.0 if op.split_drift_ky else drift_abs(op, "y") * op.kymax     # advanced exactly when split
    kx_drift = op.kx_bound if op.drift_fourier else _FD4 / op.dx          # the Fourier ∂_x reaches its largest retained k_x
    drift = op.rad_energy * jnp.max(op.Ti * (drift_ky + drift_abs(op, "x") * kx_drift))
    if op.c_Ky_co is not None:      # the Coriolis drift: ρ(V) |c_co| (k_y + k_x) √T̂ (not split)
        drift = drift + op.rad_v * jnp.max(jnp.sqrt(op.Ti) * (jnp.abs(op.c_Ky_co) * op.kymax + jnp.abs(op.c_Kx_co) * kx_drift))
    nl = 0.0
    if op.nonlinear:
        P = Phi if Pn is None else Pn
        w = jnp.where(jnp.arange(P.shape[-2]) == 0, 1.0, 2.0)[:, None]
        dP = fourier_dx(P, op.kx_sym) if op.bracket_fourier else ddx(P, op.dx)
        vy = jnp.max(jnp.sum(w * jnp.abs(dP), axis=-2))                     # |∂_xΦ| ≤ Σ_ky |·| in real space
        if op.ext_nodes is not None:                  # the imposed zonal flow adds to v_y: |a + b| ≤ |a| + |b|
            vy = vy + jnp.max(jnp.abs(ddx(op.ext_nodes, op.dx)))
        vx =jnp.max(jnp.sum(w * jnp.abs(op.ky * P), axis=-2))
        kx_nl = op.kx_bound if op.bracket_fourier else _FD4 / op.dx
        nl = jnp.max(op.c_Cinv) * (vy * op.kymax + vx * kx_nl)
    if op.exb_x is not None:                      # the sheared flow advects at |γ_E x| k_y
        nl = nl + jnp.max(jnp.abs(op.exb_x)) * op.kymax
    if op.coll_dense:
        nl = nl + jnp.max(jnp.abs(op.nu)) * op.rad_coll
    if op.coll_flr:                               # the FLR increment's radius bound (classical diffusion ν b_max …)
        nl = nl + jnp.max(jnp.abs(op.nu)) * op.rad_flr
    if op.hkz > 0:                                # numerics.hyper_kz: its largest rate ν (L−1)^p max|k_z|
        nl = nl + op.hkz * jnp.max(op.hkz_ell) * jnp.max(op.kz_abs)
    return stream + mirror + drift + nl


CFL_ESTIMATES = {"legacy": cfl_rates, "spectral": cfl_rates_spectral}
