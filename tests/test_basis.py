import numpy as np
import pytest

from velcro import basis


def test_hermite_orthonormality():
    v, w = basis.hermite_quadrature(48)
    L = 12
    Psi = np.array([basis.psi(l, v) for l in range(L)])
    gram = (Psi * w) @ Psi.T
    assert np.allclose(gram, np.eye(L), atol=1e-12)


def test_laguerre_orthonormality():
    mu, w = basis.laguerre_quadrature(48)
    M = 12
    Lg = np.array([basis.laguerre(m, mu) for m in range(M)])
    gram = (Lg * w) @ Lg.T
    assert np.allclose(gram, np.eye(M), atol=1e-10)


def test_v_and_mu_recurrences():
    v, w = basis.hermite_quadrature(48)
    L = 10
    Psi = np.array([basis.psi(l, v) for l in range(L)])
    V_quad = (Psi * w * v) @ Psi.T  # <ψ_l | v | ψ_l'>
    V = basis.hermite_v_matrix(L)
    # interior entries (truncation only affects last row/col coupling to L)
    assert np.allclose(V_quad, V, atol=1e-12)

    mu, wm = basis.laguerre_quadrature(48)
    M = 10
    Lg = np.array([basis.laguerre(m, mu) for m in range(M)])
    Mu_quad = (Lg * wm * mu) @ Lg.T
    assert np.allclose(Mu_quad, basis.laguerre_mu_matrix(M), atol=1e-9)


def test_gyro_coefficients_sum_to_gamma0():
    b = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 10.0])
    J = basis.gyro_coefficients(b, 80)
    assert np.allclose((J**2).sum(0), basis.gamma0(b), rtol=1e-10)
    # J0 reconstruction on quadrature nodes: J0(sqrt(2 b mu)) = sum_m J_m L_m(mu)
    from scipy.special import j0
    mu, _ = basis.laguerre_quadrature(20)
    Lg = np.array([basis.laguerre(m, mu) for m in range(80)])
    for i, bi in enumerate(b[:-1]):
        assert np.allclose(J[:, i] @ Lg, j0(np.sqrt(2 * bi * mu)), atol=1e-8)


def test_mirror_operator_against_quadrature():
    """Compare the (ℓ,m)-space mirror matrix with the direct differential operator.

    For g = F0 Σ c ψ_ℓ L_m evaluated at fixed (v∥, μ) with μ̂ = μ B(z)/T, the
    z-dependence through B produces  ∂_z ln B · [ -v̂ μ̂ ∂_μ̂ (e^{-μ̂} L_m) ψ_ℓ
    + ½ μ̂ ∂_v̂ (e^{-v̂²} ψ_ℓ) L_m ] × (weights).  We evaluate that expression
    analytically on a quadrature grid and project.
    """
    L, M = 6, 5
    rng = np.random.default_rng(0)
    c = rng.normal(size=(L, M))
    v, wv = basis.hermite_quadrature(60)
    mu, wm = basis.laguerre_quadrature(60)
    VV, MM = np.meshgrid(v, mu, indexing="ij")

    # derivative of weighted basis functions
    def dpsi_w(l, x):  # d/dx [e^{-x^2} psi_l] / e^{-x^2}
        return -np.sqrt(2.0 * (l + 1)) * basis.psi(l + 1, x)

    def dlag_w(m, x):  # d/dx [e^{-x} L_m] / e^{-x}
        return -sum(basis.laguerre(k, x) for k in range(m + 1))

    rhs = np.zeros_like(VV)
    for l in range(L):
        for m in range(M):
            rhs += c[l, m] * (
                -VV * MM * dlag_w(m, MM) * basis.psi(l, VV)
                + 0.5 * MM * dpsi_w(l, VV) * basis.laguerre(m, MM)
            )
    proj = np.zeros((L, M))
    for l in range(L):
        for m in range(M):
            proj[l, m] = np.einsum("i,j,ij->", wv * basis.psi(l, v), wm * basis.laguerre(m, mu), rhs)
    op = basis.mirror_matrix(L, M) @ c.reshape(-1)
    # Truncation: the matrix result is exact for rows whose sources live inside the basis;
    # the quadrature includes couplings to ψ_L and L_M dropped by truncation.  Compare on
    # rows ℓ < L-1 (Hermite couples ±1) — Laguerre coupling from the rhs is downward only.
    assert np.allclose(op.reshape(L, M)[: L - 1], proj[: L - 1], atol=1e-9)


def test_mirror_density_row_is_magnetic_compression():
    """Continuity: ∂_t n = -B ∇∥(n u∥ / B), so the density row of the mirror
    operator must equal +(∂_z ln B) × (the parallel-flux row of V)."""
    L, M = 8, 6
    mir = basis.mirror_matrix(L, M).reshape(L, M, L, M)
    V = basis.hermite_v_matrix(L)
    expected = np.zeros((L, M))
    expected[:, 0] = V[0]
    assert np.allclose(mir[0, 0], expected)
    # the mirror force itself (Dv ⊗ Mu part) conserves particles
    Dv = basis.hermite_dv_matrix(L)
    assert np.allclose(Dv[0], 0.0)


def test_project_reconstruct_roundtrip():
    L, M = 8, 6
    f = lambda v, mu: (1 + 0.3 * v + 0.2 * v**2 * mu - 0.1 * mu**2)
    G = basis.project(f, L, M)
    v = np.linspace(-2, 2, 7)
    mu = np.linspace(0, 4, 5)
    VV, MM = np.meshgrid(v, mu, indexing="ij")
    assert np.allclose(basis.reconstruct(G, v, mu), f(VV, MM), atol=1e-10)
