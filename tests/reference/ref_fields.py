"""Gyroaverage operators and the gyrokinetic Poisson (quasi-neutrality) solve (§2.2, §4.3 K4).

Real-space-in-``x`` operators are precomputed from the Fourier symbol of each
Laguerre gyro-coefficient ``𝒥_m(b)`` and of the polarisation ``1 − Γ_0(b)``
with ``b = ρ̂²(x,z) k_⊥² / 2`` (``ρ̂ = v_t/Ω``, ``v_t = sqrt(2T/m)``), ``k_⊥² = g^{xx} k_x² + 2 g^{xy} k_x k_y + g^{yy} k_y²``
(Kohn–Nirenberg quantisation, symmetrised), and applied as dense
``(N_x × N_x)`` matrices per ``(k_y, z)``.  This is the “real-space gyroaverage
with precomputed weights” route of Risk 3.  For non-axisymmetric geometry the
``y``-averaged metric is used inside these operators (exact for tokamaks; the
position-dependent 2-D kernel is the Risk-3 follow-up).

Quasi-neutrality with adiabatic electrons, normalised (single ion species, Z=1):

    (1/T̂_i)(1 − Γ_0) φ + (1/T̂_e)(φ − ⟨φ⟩_FS δ_{k_y,0}) = Σ_m 𝒥_m G_{0m}

The ``k_y = 0`` flux-surface-average coupling across ``z`` is handled by a
Woodbury (rank-``N_x``) correction; the ``k_y = 0, k_x = 0`` gauge mode is
projected out.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from velcro import basis
from tests.reference.ref_operators import kx_grid


@dataclass
class FieldSolver:
    gyro: jnp.ndarray      # (M, Nky, Nz, Nx, Nx)  real-space 𝒥_m operators
    Ainv: jnp.ndarray      # (Nky, Nz, Nx, Nx)     inverse of the polarisation operator (k_y ≠ 0 rows; k_y=0 row = A_z⁻¹)
    W: jnp.ndarray         # (Nx, Nx)  Woodbury matrix for the k_y = 0 flux-surface coupling
    fsw: jnp.ndarray       # (Nx, Nz)  flux-surface-average weights
    inv_Te: jnp.ndarray    # (Nx,)
    M: int

    def gyroaverage(self, phi: jnp.ndarray) -> jnp.ndarray:
        """``Φ_m(x, k_y, z) = Σ_x' 𝒥_m[x, x'] φ(x', k_y, z)`` → shape ``(M, Nx, Nky, Nz)``."""
        return jnp.einsum("mkzxw,wkz->mxkz", self.gyro, phi)

    def gyro_density(self, G0: jnp.ndarray) -> jnp.ndarray:
        """``Σ_m 𝒥_m G_{0m}`` with ``G0`` of shape ``(M, Nx, Nky, Nz)`` (adjoint = same symmetric matrix)."""
        return jnp.einsum("mkzxw,mwkz->xkz", self.gyro, G0)

    def solve(self, G0: jnp.ndarray) -> jnp.ndarray:
        """Solve quasi-neutrality for ``φ(x, k_y, z)`` given the ``ℓ = 0`` moments."""
        rhs = self.gyro_density(G0)                          # (Nx, Nky, Nz)
        # k_y ≠ 0 (and the A_z^{-1} part for k_y = 0)
        phi = jnp.einsum("kzxw,wkz->xkz", self.Ainv, rhs)
        # k_y = 0: Woodbury correction for the flux-surface-average term
        rhs0 = rhs[:, 0, :]
        rhs0 = rhs0 - jnp.sum(rhs0 * self.fsw) / jnp.sum(self.fsw)    # remove the gauge/volume mode
        p0 = jnp.einsum("zxw,wz->xz", self.Ainv[0], rhs0)             # A_z⁻¹ rhs
        avg = jnp.sum(p0 * self.fsw, axis=1)                          # ⟨A⁻¹ rhs⟩_FS
        phibar = self.W @ avg                                         # ⟨φ⟩_FS
        corr = jnp.einsum("zxw,w->xz", self.Ainv[0], self.inv_Te * phibar)
        phi0 = p0 + corr
        phi0 = phi0 - jnp.mean(phi0)
        return phi.at[:, 0, :].set(phi0)


def build_field_solver(geom, ky: np.ndarray, Ti: np.ndarray, Te: np.ndarray, M: int,
                       Lx: float, dtype=np.complex128) -> FieldSolver:
    """Precompute gyro-operators and the polarisation inverse.

    Parameters
    ----------
    geom : Geometry
    ky : (Nky,) retained binormal wavenumbers
    Ti, Te : (Nx,) normalised ion / electron temperature profiles
    M : number of Laguerre moments
    """
    Nx, Nz = len(geom.x), len(geom.z)
    Nky = len(ky)
    kx = kx_grid(Nx, Lx)
    # y-averaged metric and field (exact for axisymmetric geometry)
    gxx, gxy, gyy, B = (geom.y_averaged(a)[:, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    rho2 = 0.5 * Ti[:, None] / B**2                           # ρ̂_i²(x, z)/2:  b = k⊥² ρ_i²/2 for v_t = sqrt(2T/m)
    x = geom.x
    # phase matrix E[x, x', kx] = exp(i kx (x - x')) / Nx
    E = np.exp(1j * kx[None, None, :] * (x[:, None, None] - x[None, :, None])) / Nx
    b = (rho2[:, :, None, None] * (gxx[:, :, None, None] * kx[None, None, :, None] ** 2
                                   + 2 * gxy[:, :, None, None] * kx[None, None, :, None] * ky[None, None, None, :]
                                   + gyy[:, :, None, None] * ky[None, None, None, :] ** 2))  # (Nx, Nz, Nkx, Nky)
    b = np.maximum(b, 0.0)
    Jm = basis.gyro_coefficients(b, M)                        # (M, Nx, Nz, Nkx, Nky)
    G0b = basis.gamma0(b)                                      # (Nx, Nz, Nkx, Nky)

    def quantise(symbol):  # symbol (Nx, Nz, Nkx, Nky) evaluated at output point x -> op (Nky, Nz, Nx, Nx)
        op = np.einsum("xwq,xzqk->kzxw", E, symbol)
        return 0.5 * (op + np.conj(np.swapaxes(op, -1, -2)))  # symmetrise (Hermitian)

    gyro = np.stack([quantise(Jm[m]) for m in range(M)], axis=0)
    pol = quantise(1.0 - G0b)
    inv_Ti, inv_Te = 1.0 / Ti, 1.0 / Te
    I = np.eye(Nx)
    A = inv_Ti[None, None, :, None] * pol + inv_Te[None, None, :, None] * I[None, None]
    Ainv = np.linalg.inv(A)
    # Woodbury for k_y = 0:  (I − K) ⟨φ⟩ = ⟨A⁻¹ rhs⟩ ,  K = Σ_z w_z A_z⁻¹ diag(1/T_e)
    fsw = geom.fs_weights()                                    # (Nx, Nz)
    K = np.einsum("xz,zxw->xw", fsw, Ainv[0]) * inv_Te[None, :]
    ones = np.ones((Nx, 1))
    W = np.linalg.inv(I - K + ones @ ones.T / Nx)             # regularised on the gauge mode
    return FieldSolver(
        gyro=jnp.asarray(gyro, dtype=dtype), Ainv=jnp.asarray(Ainv, dtype=dtype),
        W=jnp.asarray(W.real), fsw=jnp.asarray(fsw), inv_Te=jnp.asarray(inv_Te), M=M,
    )
