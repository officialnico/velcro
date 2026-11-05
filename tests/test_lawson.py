"""Integrating-factor (Lawson) RK4 for the binormal magnetic drift (``numerics.integrator = "lawson"``).

Pieces and their tests (one against an independent answer, one property each):
  node transforms      — eigenvalues are the Gauss nodes (numpy's hermgauss / laggauss); orthogonality
  drift propagator     — equals scipy's expm of the dense drift matrix; group property and unitarity
  right-hand-side split— f_rk4 = f_lawson + A G with A built independently from the dense energy matrix; A anti-Hermitian
  CFL bound            — still bounds the measured radius of the split operator; is smaller than the unsplit bound
  time stepper         — exact for a pure drift problem at any dt; 4th-order convergence; agrees with RK4;
                         stable where RK4 is not; nonlinear trajectory and free-energy balance match RK4
"""
import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from velcro import basis
from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.vlasov import apply_drift_propagator, drift_phase, rhs

L, M = 6, 3


def make(integrator, nonlinear=False, L=L, M=M, Nx=8, Nz=8, dt=None, **num):
    numerics = dict(hyper_x=0.05, hyper_z=0.05, hyper_v=0.5, krook_rate=1.0, krook_width=0.12, integrator=integrator,
                    cfl_estimate="spectral", dt=dt)
    numerics.update(num)
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=60.0, Ny=10, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=1 / 300, r0=0.18, q0=1.4, shat=0.8, n0=5),
        physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=0.02, nonlinear=nonlinear),
        numerics=numerics, init=dict(kind="noise", amplitude=1e-2 if nonlinear else 1.0, seed=1), run=dict(t_end=0.1))))


@pytest.fixture(scope="module")
def pair():
    return make("rk4"), make("lawson")


def rand_like(G, seed):
    r = np.random.default_rng(seed)
    return jnp.asarray(r.normal(size=G.shape) + 1j * r.normal(size=G.shape))


def dense_drift_action(sim, G):
    """A G = −(2v̂²+μ̂) i k_y K̂_y T̂ G, from the dense energy matrix (not from the node transforms)."""
    op = sim.op
    E = basis.energy_matrix(L, M).reshape(L, M, L, M)
    spatial = 1j * np.asarray(op.ky) * np.asarray(op.c_Ky) * np.asarray(op.Ti) * np.asarray(G)
    return -np.einsum("lmpq,pq...->lm...", E, spatial)


# ------------------------------------------------------------------ node transforms
def test_node_eigenvalues_are_the_gauss_nodes(pair):
    op = pair[1].op
    v, _ = np.polynomial.hermite.hermgauss(L)
    mu, _ = np.polynomial.laguerre.laggauss(M)
    assert np.allclose(np.sort(np.asarray(op.lam_v)), np.sort(2 * v ** 2), atol=1e-12)
    assert np.allclose(np.sort(np.asarray(op.lam_mu)), np.sort(mu), atol=1e-12)


def test_node_transforms_are_orthogonal_and_diagonalise_the_energy(pair):
    op = pair[1].op
    Uv, Um = np.asarray(op.U_v), np.asarray(op.U_mu)
    assert np.allclose(Uv.T @ Uv, np.eye(L), atol=1e-13) and np.allclose(Um.T @ Um, np.eye(M), atol=1e-13)
    U = np.kron(Uv, Um)
    lam = (np.asarray(op.lam_v)[:, None] + np.asarray(op.lam_mu)[None, :]).reshape(-1)
    assert np.allclose(U @ np.diag(lam) @ U.T, basis.energy_matrix(L, M), atol=1e-12)


# ------------------------------------------------------------------ propagator
def test_propagator_equals_expm_of_the_dense_drift(pair):
    sim = pair[1]; op = sim.op
    G = rand_like(sim.G, 0); h = 0.37
    out = np.asarray(apply_drift_propagator(G, drift_phase(h, op), op))
    E = basis.energy_matrix(L, M)
    for (ix, ik, iz) in [(0, 1, 0), (3, 2, 5), (7, 3, 2)]:
        w = float(np.asarray(op.ky)[0, ik, 0] * np.asarray(op.c_Ky)[ix, 0, iz] * np.asarray(op.Ti)[ix, 0, 0])
        expect = expm(-1j * h * w * E) @ np.asarray(G)[:, :, ix, ik, iz].reshape(-1)
        assert np.allclose(out[:, :, ix, ik, iz].reshape(-1), expect, atol=1e-11)


def test_propagator_is_a_unitary_group(pair):
    sim = pair[1]; op = sim.op
    G = rand_like(sim.G, 1)
    P = lambda h: drift_phase(h, op)
    two = apply_drift_propagator(apply_drift_propagator(G, P(0.2), op), P(0.5), op)
    assert np.allclose(two, apply_drift_propagator(G, P(0.7), op), atol=1e-11)
    assert np.allclose(apply_drift_propagator(apply_drift_propagator(G, P(0.4), op), P(-0.4), op), G, atol=1e-11)
    assert np.isclose(jnp.linalg.norm(apply_drift_propagator(G, P(3.0), op)), jnp.linalg.norm(G), rtol=1e-12)
    assert np.allclose(apply_drift_propagator(G, P(0.0), op), G, atol=1e-13)


# ------------------------------------------------------------------ right-hand-side split
@pytest.mark.parametrize("nonlinear", [False, True])
def test_split_rhs_plus_dense_drift_is_the_full_rhs(nonlinear):
    a, b = make("rk4", nonlinear), make("lawson", nonlinear)
    G = rand_like(a.G, 2) * (1e-2 if nonlinear else 1.0)
    full, split = np.asarray(a._rhs(G)), np.asarray(b._rhs(G))
    assert np.allclose(full, split + dense_drift_action(a, G), rtol=1e-11, atol=1e-11 * np.abs(full).max())
    assert np.abs(full - split).max() > 1e-3 * np.abs(full).max()          # the split removed something


def test_removed_drift_is_anti_hermitian(pair):
    G = rand_like(pair[0].G, 3)
    AG = dense_drift_action(pair[0], G)
    w = np.where(np.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, :, None]
    assert abs(np.sum(np.real(np.conj(np.asarray(G)) * AG) * w)) < 1e-10 * np.sum(np.abs(np.asarray(G)) * np.abs(AG) * w)


# ------------------------------------------------------------------ CFL bound
def _radius(sim, iters=400):
    v = sim.G / jnp.linalg.norm(sim.G); lam = 0.0
    for _ in range(iters):
        y = sim._rhs(v); lam = float(jnp.linalg.norm(y)); v = y / lam
    return lam


def test_cfl_bound_of_the_split_operator(pair):
    a, b = pair
    ra, rb = float(a._rate(a.G)), float(b._rate(b.G))
    lam_b = _radius(b)
    assert lam_b <= rb <= 2.5 * lam_b, f"bound {rb:.1f} vs measured radius of the split operator {lam_b:.1f}"
    assert rb < 0.75 * ra and lam_b < 0.75 * _radius(a)                    # the split removes a stiff term


# ------------------------------------------------------------------ time stepper
def _advance(sim, G, dt, n):
    return np.asarray(sim._nsteps(G, dt, n))


def test_pure_drift_problem_is_integrated_exactly_at_any_dt():
    """With every other term off, Lawson must return e^{hA}G to round-off for dt far above the RK4 limit."""
    sim = make("lawson", hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0)
    op = sim.op
    z = lambda k: jnp.zeros_like(getattr(op, k))
    sim.op = dataclasses.replace(op, c_stream=z("c_stream"), c_mirror=z("c_mirror"), c_Kx=z("c_Kx"), c_Cinv=z("c_Cinv"),
                                 nu=z("nu"), source_rate=0.0)
    sim.fields = dataclasses.replace(sim.fields, gyro=0 * sim.fields.gyro)      # Φ = 0: no field-coupled drift
    sim._compile()
    G = rand_like(sim.G, 4); dt = 5.0
    lam_max = float((op.lam_v.max() + op.lam_mu.max()) * jnp.max(jnp.abs(op.ky * op.c_Ky * op.Ti)))
    assert dt * lam_max > 10 * 2.8                                              # RK4 would be violently unstable
    out = _advance(sim, G, dt, 3)
    assert np.allclose(out, apply_drift_propagator(G, drift_phase(3 * dt, op), op), atol=1e-10)


def test_lawson_is_fourth_order_and_agrees_with_rk4(pair):
    a, b = pair
    G = rand_like(a.G, 5); T = 0.04
    ref = _advance(a, G, T / 256, 256)
    err = [np.linalg.norm(_advance(b, G, T / n, n) - ref) / np.linalg.norm(ref) for n in (2, 4, 8)]
    assert err[2] < 1e-6
    assert 12 < err[0] / err[1] < 20 and 12 < err[1] / err[2] < 20             # 4th order: 16x per halving


def test_lawson_is_accurate_where_rk4_is_unstable():
    """At dt = 0.6 × 2.8/ρ(split operator) — dt ρ(full) ≈ 3.9, outside RK4's stability region — Lawson follows a
    fine-step reference over the same physical time (the ITG grows physically, so compare, don't bound), and RK4
    at that dt diverges.  Measured 2026-09-19 at (16, 8): Lawson is stable to ≈ 0.7 of that limit, not 1.0 (the
    split terms do not commute with the propagator), so the usable gain is ≈ 1.7x in dt, not the 2.3x radius ratio."""
    a, b = make("rk4", L=16, M=8), make("lawson", L=16, M=8)
    lam_a, lam_b = _radius(a), _radius(b)
    dt = 0.6 * 2.8 / lam_b
    assert dt * lam_a > 3.5
    G = rand_like(a.G, 6)
    ref = _advance(a, G, dt / 16, 200 * 16)
    law = _advance(b, G, dt, 200)
    assert np.linalg.norm(law - ref) < 2e-2 * np.linalg.norm(ref)
    rk = _advance(a, G, dt, 200)
    assert (not np.isfinite(rk).all()) or np.linalg.norm(rk) > 1e6 * np.linalg.norm(ref)


def test_nonlinear_trajectory_and_free_energy_match_rk4():
    a, b = make("rk4", True, dt=2e-3), make("lawson", True, dt=2e-3)
    G = a.G * 30.0                                                              # amplitude where the E×B term matters
    Ga, Gb = _advance(a, G, 2e-3, 100), _advance(b, G, 2e-3, 100)
    assert np.linalg.norm(Ga - Gb) < 1e-6 * np.linalg.norm(Ga)
    Wa, Wb = float(a._diagnostics(jnp.asarray(Ga))["W"]), float(b._diagnostics(jnp.asarray(Gb))["W"])
    Qa, Qb = float(a._diagnostics(jnp.asarray(Ga))["Q"]), float(b._diagnostics(jnp.asarray(Gb))["Q"])
    assert np.isclose(Wa, Wb, rtol=1e-6) and np.isclose(Qa, Qb, rtol=1e-5, atol=1e-12)


def test_lawson_refuses_non_axisymmetric_geometry_and_is_off_by_default():
    assert VelcroConfig().numerics.integrator == "rk4"
    assert make("rk4").op.split_drift_ky is False and make("lawson").op.split_drift_ky is True



# ------------------------------------------------------------------ GX's Hermite closure (adiabatic path)
def _closure_energy():
    V1 = basis.hermite_v_matrix(L + 1)
    A = (V1 @ V1)[:L, :L]
    return A, 2.0 * np.kron(A, np.eye(M)) + np.kron(np.eye(L), basis.laguerre_mu_matrix(M))


def test_gx_hermite_closure_adiabatic_changes_only_the_last_hermite_row():
    """numerics.gx_hermite_closure on the adiabatic path: 2v̂² from the L×L block of the (L+1)-mode product. Only the
    ℓ = L−1 diagonal of 2v̂² + μ̂ changes (2L−1 instead of L−1), in the band and in the node transform."""
    base, sim = make("lawson"), make("lawson", gx_hermite_closure=True)
    A, E = _closure_energy()
    D = E - basis.energy_matrix(L, M)
    assert np.allclose(np.diag(D)[(L - 1) * M:], L) and np.count_nonzero(np.abs(D) > 1e-12) == M
    Uv, lam = np.asarray(sim.op.U_v), np.asarray(sim.op.lam_v)
    assert np.allclose(Uv @ np.diag(lam) @ Uv.T, 2 * A, atol=1e-12)
    assert base.op.energy_off == sim.op.energy_off
    for off, c0, c1 in zip(base.op.energy_off, base.op.energy_band, sim.op.energy_band):
        d = np.asarray(c1) - np.asarray(c0)
        assert np.allclose(d, 0) if tuple(off) != (0, 0) else (np.allclose(d[:-1], 0) and np.allclose(d[-1], L))


def test_gx_hermite_closure_adiabatic_propagator_is_expm_and_the_split_holds():
    """With the closure the Lawson propagator is still expm of the (new) dense drift, and f_rk4 = f_lawson + A G with A
    from the new energy matrix."""
    A, E = _closure_energy()
    sim = make("lawson", gx_hermite_closure=True); op = sim.op
    G = rand_like(sim.G, 4); h = 0.37
    out = np.asarray(apply_drift_propagator(G, drift_phase(h, op), op))
    for (ix, ik, iz) in [(0, 1, 0), (3, 2, 5), (7, 3, 2)]:
        w = float(np.asarray(op.ky)[0, ik, 0] * np.asarray(op.c_Ky)[ix, 0, iz] * np.asarray(op.Ti)[ix, 0, 0])
        expect = expm(-1j * h * w * E) @ np.asarray(G)[:, :, ix, ik, iz].reshape(-1)
        assert np.allclose(out[:, :, ix, ik, iz].reshape(-1), expect, atol=1e-11)
    rk = make("rk4", gx_hermite_closure=True)
    full, split = np.asarray(rk._rhs(G)), np.asarray(sim._rhs(G))
    spatial = 1j * np.asarray(op.ky) * np.asarray(op.c_Ky) * np.asarray(op.Ti) * np.asarray(G)
    drift = -np.einsum("lmpq,pq...->lm...", E.reshape(L, M, L, M), spatial)
    assert np.allclose(full, split + drift, rtol=1e-11, atol=1e-11 * np.abs(full).max())
    ref = _advance(rk, G, 1e-3 / 64, 256)
    assert np.linalg.norm(_advance(sim, G, 1e-3 / 4, 16) - ref) < 1e-8 * np.linalg.norm(ref)
