"""Hermite–Laguerre velocity-space basis (§3.1 of the design document).

Conventions (Mandell, Dorland & Landreman 2018):

* ``v̂ = v∥ / v_t`` with ``v_t = sqrt(2 T / m)``; ``μ̂ = μ B / T``.
* ``F0 = n0 / (π^{3/2} v_t^3) exp(-v̂² - μ̂)``.
* ``F1 = F0 Σ_{ℓm} ψ_ℓ(v̂) L_m(μ̂) G_{ℓm}`` where ``ψ_ℓ = H_ℓ / sqrt(2^ℓ ℓ!)``
  are orthonormal under the weight ``exp(-v̂²)/sqrt(π)`` and ``L_m`` are
  Laguerre polynomials, orthonormal under ``exp(-μ̂)``.
* ``G_{ℓm} = (1/n0) ∫ ψ_ℓ L_m F1 d³v``.

All velocity-space operators used by the solver are expressed as small dense
matrices acting on the ``(ℓ, m)`` index pair.  They are built once from exact
three-term recurrences, and their correctness is checked against Gauss
quadrature in ``tests/test_basis.py``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import eval_hermite, eval_laguerre, gammaln, ive


# --------------------------------------------------------------------------
# Basis functions (numpy, used in tests / diagnostics / initial conditions)
# --------------------------------------------------------------------------
def hermite_norm(l: int) -> float:
    """Normalisation ``1/sqrt(2^ℓ ℓ!)`` making ``ψ_ℓ`` orthonormal."""
    return float(np.exp(-0.5 * (l * np.log(2.0) + gammaln(l + 1))))


def psi(l: int, v: np.ndarray) -> np.ndarray:
    """Orthonormal Hermite function ``ψ_ℓ(v̂)`` (without the Gaussian weight)."""
    return hermite_norm(l) * eval_hermite(l, v)


def laguerre(m: int, mu: np.ndarray) -> np.ndarray:
    """Laguerre polynomial ``L_m(μ̂)``."""
    return eval_laguerre(m, mu)


def hermite_quadrature(n: int):
    """Gauss–Hermite nodes/weights for ``∫ f(v) e^{-v²}/sqrt(π) dv``."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w / np.sqrt(np.pi)


def laguerre_quadrature(n: int):
    """Gauss–Laguerre nodes/weights for ``∫ f(μ) e^{-μ} dμ``."""
    return np.polynomial.laguerre.laggauss(n)


# --------------------------------------------------------------------------
# Recurrence matrices.  A matrix ``A`` represents an operator mapping the
# coefficient vector ``c_{ℓ}`` (or ``c_m``) of a distribution ``F0 Σ c ψ L`` to the
# coefficients of the transformed distribution: ``(A c)_ℓ``.
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def hermite_v_matrix(L: int) -> np.ndarray:
    """``V``: multiplication by ``v̂``.  ``v̂ ψ_ℓ = √((ℓ+1)/2) ψ_{ℓ+1} + √(ℓ/2) ψ_{ℓ-1}``.

    Truncation at ``L`` drops the coupling to ``ψ_L`` (closure by truncation).
    """
    V = np.zeros((L, L))
    for l in range(L):
        if l + 1 < L:
            V[l + 1, l] = np.sqrt((l + 1) / 2.0)
        if l - 1 >= 0:
            V[l - 1, l] = np.sqrt(l / 2.0)
    return V


@lru_cache(maxsize=None)
def hermite_dv_matrix(L: int) -> np.ndarray:
    """``Dv``: ``∂/∂v̂`` acting on the weighted basis ``e^{-v̂²} ψ_ℓ``.

    ``∂_v̂ (e^{-v̂²} ψ_ℓ) = -√(2(ℓ+1)) e^{-v̂²} ψ_{ℓ+1}``.
    """
    D = np.zeros((L, L))
    for l in range(L - 1):
        D[l + 1, l] = -np.sqrt(2.0 * (l + 1))
    return D


@lru_cache(maxsize=None)
def laguerre_mu_matrix(M: int) -> np.ndarray:
    """``Mu``: multiplication by ``μ̂``.  ``μ̂ L_m = (2m+1) L_m - (m+1) L_{m+1} - m L_{m-1}``."""
    A = np.zeros((M, M))
    for m in range(M):
        A[m, m] = 2 * m + 1
        if m + 1 < M:
            A[m + 1, m] = -(m + 1)
        if m - 1 >= 0:
            A[m - 1, m] = -m
    return A


@lru_cache(maxsize=None)
def laguerre_dmu_matrix(M: int) -> np.ndarray:
    """``P``: ``∂/∂μ̂`` acting on the weighted basis ``e^{-μ̂} L_m``.

    ``∂_μ̂ (e^{-μ̂} L_m) = -e^{-μ̂} Σ_{k≤m} L_k``.
    """
    P = np.zeros((M, M))
    for m in range(M):
        P[: m + 1, m] = -1.0
    return P


@lru_cache(maxsize=None)
def mirror_matrix(L: int, M: int) -> np.ndarray:
    """Mirror-force / ``μ̂``-basis-variation operator per unit ``v_t ∂_z ln B``.

    Acting on ``g = F0 Σ c ψ_ℓ L_m`` at fixed ``(v∥, μ)``, the parallel dynamics
    ``-v∥ b·∇ g + (μ/m)(b·∇B) ∂g/∂v∥`` produce, besides ``-v_t v̂ ∂_z c``, a term
    proportional to ``∂_z ln B`` because ``μ̂ = μ B(z)/T`` varies along the field
    line.  In operator form (Kronecker ordering ``(ℓ, m)`` → ``ℓ*M + m``):

        Mirror = -(V ⊗ (Mu P)) + ½ (Dv ⊗ Mu)

    so that ``∂_t c = v_t (∂_z ln B) Mirror c + ...``.  Verified against direct
    quadrature in ``tests/test_basis.py``.
    """
    V, Dv = hermite_v_matrix(L), hermite_dv_matrix(L)
    Mu, P = laguerre_mu_matrix(M), laguerre_dmu_matrix(M)
    return -np.kron(V, Mu @ P) + 0.5 * np.kron(Dv, Mu)


@lru_cache(maxsize=None)
def energy_matrix(L: int, M: int) -> np.ndarray:
    """``2 v̂² + μ̂`` (the magnetic-drift velocity dependence), Kronecker ``(ℓ,m)``."""
    V = hermite_v_matrix(L)
    Mu = laguerre_mu_matrix(M)
    return 2.0 * np.kron(V @ V, np.eye(M)) + np.kron(np.eye(L), Mu)


# --------------------------------------------------------------------------
# Gyroaverage in Laguerre space
# --------------------------------------------------------------------------
def gyro_coefficients(b: np.ndarray, M: int) -> np.ndarray:
    """Laguerre expansion of ``J_0(sqrt(2 b μ̂))``: ``𝒥_m(b) = (b/2)^m e^{-b/2} / m!``.

    Returns an array of shape ``(M,) + b.shape``.  Computed in log space for
    stability at large ``b``.
    """
    b = np.asarray(b, dtype=np.float64)
    m = np.arange(M, dtype=np.float64).reshape((M,) + (1,) * b.ndim)
    logb2 = np.log(np.maximum(b, 1e-300) / 2.0)
    out = np.exp(m * logb2 - b / 2.0 - gammaln(m + 1))
    return np.where((m > 0) & (b <= 0), 0.0, out)


def gamma0(b: np.ndarray) -> np.ndarray:
    """``Γ_0(b) = e^{-b} I_0(b) = Σ_m 𝒥_m(b)²`` (the polarisation factor)."""
    return ive(0, np.asarray(b, dtype=np.float64))


# --------------------------------------------------------------------------
# Projection helpers (initial conditions and diagnostics)
# --------------------------------------------------------------------------
def project(f_over_F0, L: int, M: int, nv: int = 64, nmu: int = 64) -> np.ndarray:
    """Project ``f/F0`` (callable of ``(v̂, μ̂)``) onto ``G_{ℓm}`` by quadrature."""
    v, wv = hermite_quadrature(nv)
    mu, wmu = laguerre_quadrature(nmu)
    VV, MM = np.meshgrid(v, mu, indexing="ij")
    vals = f_over_F0(VV, MM)
    G = np.zeros((L, M))
    for l in range(L):
        pl = psi(l, v)
        for m in range(M):
            G[l, m] = np.einsum("i,j,ij->", wv * pl, wmu * laguerre(m, mu), vals)
    return G


def reconstruct(G: np.ndarray, v: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Evaluate ``f/F0 = Σ G_{ℓm} ψ_ℓ(v̂) L_m(μ̂)`` on a ``(v, μ)`` grid."""
    L, M = G.shape
    out = np.zeros((len(v), len(mu)))
    for l in range(L):
        pl = psi(l, v)[:, None]
        for m in range(M):
            out += G[l, m] * pl * laguerre(m, mu)[None, :]
    return out
