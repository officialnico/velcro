"""Gyroaveraged potential at the Gauss–Laguerre nodes of the E×B nonlinearity (``numerics.node_gyro``).

Independent truth: the Laguerre series of the gyroaverage sums to a Bessel function,
``Σ_{m≥0} 𝒥_m(b) L_m(μ) = J0(√(2bμ))``, which is bounded by 1.  The series *truncated at M* is what the
``"laguerre"`` option evaluates at the nodes; it diverges at the outer nodes for ``M ≳ 16`` (found
2026-09-18: |Φ| = 3632 instead of 12.7 at μ = 51.7 on a saturated (32, 16) state), and that spurious
E×B velocity set the explicit stability limit.  ``"exact"`` applies ``J0(√(2bμ_j))`` directly.
"""
import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import j0

from velcro import basis, geometry
from velcro.config import VelcroConfig
from velcro.fields import build_field_solver
from velcro.operators import ky_grid
from velcro.simulation import Simulation
from velcro.vlasov import rhs


def _slab_solver(M, Nx=32, Lx=20.0, Ny=12, Nz=4):
    g = geometry.slab(Nx, Lx, Ny, Ly=2 * np.pi * 5, Nz=Nz)
    ky = ky_grid(Ny, g.Ly)
    return g, ky, build_field_solver(g, ky, np.ones(Nx), np.ones(Nx), M, Lx, node_gyro=True)


# ---------------------------------------------------------------- independent truth
def test_laguerre_series_of_the_gyroaverage_sums_to_J0():
    """The identity the fix rests on, checked with Velcro's own 𝒥_m against scipy's J0."""
    b = np.array([0.05, 0.5, 2.0, 6.0])
    for mu in (0.1, 1.0, 5.0, 20.0):
        J = basis.gyro_coefficients(b, 120)                                  # (120, 4): converged series
        series = sum(J[m] * basis.laguerre(m, np.array(mu)) for m in range(120))
        assert np.allclose(series, j0(np.sqrt(2 * b * mu)), atol=1e-9)


@pytest.mark.parametrize("M", [8, 16, 24])
def test_node_operator_matches_J0_symbol_and_is_bounded(M):
    """In a uniform slab the node operator must reproduce J0(√(2bμ_j)) e^{ikx x}; |Φ(μ_j)| ≤ |φ|."""
    g, ky, fs = _slab_solver(M)
    mu, _ = basis.laguerre_quadrature(M)
    for n in (0, 2, 7, 15):
        kx = 2 * np.pi * n / 20.0
        phi = np.exp(1j * kx * g.x)[:, None, None] * np.ones((1, len(ky), len(g.z)))
        Pn = np.asarray(fs.gyroaverage_nodes(jnp.asarray(phi)))               # (Nmu, Nx, Nky, Nz)
        b = 0.5 * (kx ** 2 + ky ** 2)
        expect = j0(np.sqrt(2 * b[None, :] * mu[:, None]))[:, None, :, None] * phi[None]
        assert np.allclose(Pn, expect, atol=1e-10)
        assert np.abs(Pn).max() <= 1.0 + 1e-10


# ---------------------------------------------------------------- property / convergence
def _broadband_phi(g, ky, seed=0, slope=0.0):
    """Random φ with |φ_k| ∝ (1 + k_x²)^(−slope/2); slope = 0 is white in k_x (b up to (π/Δx)²/2 ≈ 12)."""
    rng = np.random.default_rng(seed)
    shape = (len(g.x), len(ky), len(g.z))
    phi = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    kx = 2 * np.pi * np.fft.fftfreq(len(g.x), d=(g.x[1] - g.x[0]))
    return np.fft.ifft(np.fft.fft(phi, axis=0) * ((1 + kx ** 2) ** (-slope / 2))[:, None, None], axis=0)


def _node_values(M, slope=0.0):
    g, ky, fs = _slab_solver(M)
    phi = jnp.asarray(_broadband_phi(g, ky, slope=slope))
    mu, w = basis.laguerre_quadrature(M)
    Lmat = np.array([basis.laguerre(m, mu) for m in range(M)]).T               # (Nmu, M)
    trunc = np.einsum("jm,m...->j...", Lmat, np.asarray(fs.gyroaverage(phi)))
    exact = np.asarray(fs.gyroaverage_nodes(phi))
    return trunc, exact, float(jnp.abs(phi).max()), w


def test_truncated_sum_agrees_with_exact_where_the_quadrature_weight_is():
    """At M = 8 the two agree in the Gauss-weighted norm to the series truncation error: < 1 % for a
    turbulence-like decaying k_x spectrum (results obtained at M ≤ 8 with the truncated sum stand),
    ≈ 5 % for white noise, where b reaches 12 and eight Laguerre terms cannot represent J0."""
    def werr(slope):
        trunc, exact, _, w = _node_values(8, slope)
        w = w[:, None, None, None]
        return np.sqrt(np.sum(w * np.abs(trunc - exact) ** 2) / np.sum(w * np.abs(exact) ** 2))
    assert werr(3.0) < 1e-2
    assert werr(0.0) < 1e-1


def test_truncated_sum_diverges_at_outer_nodes_for_M16_exact_stays_bounded():
    trunc, exact, phimax, _ = _node_values(16)
    assert np.abs(exact).max() <= 1.5 * phimax                                # a gyroaverage cannot amplify (KN-symmetrised: small slack)
    assert np.abs(trunc[-1]).max() > 20 * phimax                              # the artefact, outermost node
    assert np.abs(trunc[:8]).max() <= 1.5 * phimax                            # inner nodes are fine


def _slab_sim(node_gyro, M=6):
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=32, Lx=20.0, Ny=16, Nz=4, L=3, M=M),
        geometry=dict(kind="slab", rho_star=1e-3),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=True,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01, node_gyro=node_gyro),
        init=dict(kind="noise", amplitude=1.0, seed=3),
        run=dict(t_end=0.1))))


def test_exact_nodes_conserve_free_energy_to_roundoff():
    """The skew bracket conserves Σ_j w_j |G(μ_j)|² for any advecting potential, so the exact node
    values keep the round-off conservation of the E×B term (white-noise state, grid-scale in x)."""
    sim = _slab_sim("exact")
    G = sim.G
    phi, Phi = sim._solve_fields(G)
    Pn = sim.fields.gyroaverage_nodes(phi)
    NL = np.asarray(rhs(G, phi, Phi, sim.op, Pn=Pn)) - np.asarray(rhs(G, phi, Phi, dataclasses.replace(sim.op, nonlinear=False)))
    w = np.where(np.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, :, None]
    Gn = np.asarray(G)
    assert np.abs(NL).max() > 0
    inj = abs(np.sum(np.real(np.conj(Gn) * NL) * w)) / (np.sqrt(np.sum(np.abs(Gn) ** 2 * w)) * np.sqrt(np.sum(np.abs(NL) ** 2 * w)))
    assert inj < 1e-12


def test_simulation_uses_the_exact_nodes_when_configured():
    """The flag reaches the right-hand side: relative to the E×B term, the two options agree for a
    smooth low-k state (b small, series converged) and differ for a grid-scale state."""
    a, b = _slab_sim("laguerre"), _slab_sim("exact")

    def rel_diff(G):
        phi, Phi = a._solve_fields(G)
        nl = np.asarray(a._rhs(G)) - np.asarray(rhs(G, phi, Phi, dataclasses.replace(a.op, nonlinear=False)))
        return np.linalg.norm(np.asarray(a._rhs(G)) - np.asarray(b._rhs(G))) / np.linalg.norm(nl)

    G = np.asarray(a.G)
    Gk = np.fft.fft(G, axis=2); Gk[:, :, np.abs(np.fft.fftfreq(G.shape[2])) > 0.07] = 0
    smooth, rough = rel_diff(jnp.asarray(np.fft.ifft(Gk, axis=2))), rel_diff(a.G)
    assert smooth < 1e-3
    assert rough > 10 * smooth and rough > 1e-3
