"""Low-rank (matrix-free) field solver against the dense one: exact at rank 1 in a slab, to tolerance with profiles."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from velcro import geometry
from velcro.fields import build_field_solver
from velcro.fields_lowrank import build_lowrank_field_solver
from velcro.operators import ky_grid


def _pair(kind, Nx=24, Lx=30.0, Ny=12, Nz=6, M=4, profile=False, node_gyro=True, tol=1e-10):
    if kind == "slab":
        g = geometry.slab(Nx, Lx, Ny, Ly=2 * np.pi * 5, Nz=Nz)
    else:
        g = geometry.s_alpha(Nx, Lx, Ny, Nz, rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=2)
    ky = ky_grid(Ny, g.Ly)
    x = np.asarray(g.x)
    Ti = 1.0 + (0.3 * (x - x.mean()) / (x.max() - x.min()) if profile else 0.0) * np.ones(Nx)
    Te = 1.5 * np.ones(Nx)
    dense = build_field_solver(g, ky, Ti, Te, M, Lx, node_gyro=node_gyro)
    low = build_lowrank_field_solver(g, ky, Ti, Te, M, Lx, node_gyro=node_gyro, tol=tol, cg_tol=1e-12, cg_maxiter=200)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(Nx, len(ky), Nz)) + 1j * rng.normal(size=(Nx, len(ky), Nz))
    phi[:, 0] = phi[:, 0].real
    G0 = rng.normal(size=(M, Nx, len(ky), Nz)) + 1j * rng.normal(size=(M, Nx, len(ky), Nz))
    G0[:, :, 0] = G0[:, :, 0].real
    return dense, low, jnp.asarray(phi), jnp.asarray(G0)


def test_slab_is_rank_one_and_exact():
    """No x dependence: every symbol factorises at rank 1 and the operators agree to round-off."""
    dense, low, phi, G0 = _pair("slab")
    assert low.rank == 1
    for name in ("gyroaverage", "gyroaverage_nodes", "gyro_density", "solve"):
        arg = G0 if name in ("gyro_density", "solve") else phi
        a, b = np.asarray(getattr(dense, name)(arg)), np.asarray(getattr(low, name)(arg))
        assert np.allclose(a, b, atol=1e-10 * np.abs(a).max()), name


def test_global_tokamak_with_profile_matches_dense_to_tolerance():
    """Metric and T_i vary in x: the rank is finite but far below N_x, and the truncation error is set by the cut."""
    dense, low, phi, G0 = _pair("s_alpha", profile=True)
    assert 1 < low.rank < 24 // 2
    for name in ("gyroaverage", "gyroaverage_nodes", "gyro_density", "solve"):
        arg = G0 if name in ("gyro_density", "solve") else phi
        a, b = np.asarray(getattr(dense, name)(arg)), np.asarray(getattr(low, name)(arg))
        assert np.allclose(a, b, atol=1e-7 * np.abs(a).max()), name


def test_looser_cut_lowers_the_rank_and_degrades_gracefully():
    dense, tight, phi, G0 = _pair("s_alpha", profile=True, tol=1e-10)
    _, loose, _, _ = _pair("s_alpha", profile=True, tol=1e-3)
    assert loose.rank < tight.rank
    a, b = np.asarray(dense.solve(G0)), np.asarray(loose.solve(G0))
    err = np.abs(a - b).max() / np.abs(a).max()
    assert 1e-7 < err < 1e-2


def test_polarisation_operator_is_hermitian_positive_definite():
    """The CG operator [Pol + T_i/T_e] on random vectors: ⟨u, A v⟩ = ⟨A u, v⟩ and ⟨u, A u⟩ > 0."""
    _, low, phi, _ = _pair("s_alpha", profile=True)
    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)).transpose(1, 2, 0)
    v = jnp.asarray(rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)).transpose(1, 2, 0)
    Au, Av = low.polarisation(u), low.polarisation(v)
    lhs, rhs = np.sum(np.conj(np.asarray(u)) * np.asarray(Av)), np.sum(np.conj(np.asarray(Au)) * np.asarray(v))
    assert np.isclose(lhs, rhs, rtol=1e-10)
    assert np.sum(np.conj(np.asarray(u)) * np.asarray(Au)).real > 0


def test_memory_scales_linearly_in_Nx():
    """The whole solver holds O(N_x) numbers per operator (the k_y = 0 row is the only N_x² object)."""
    _, low, _, _ = _pair("s_alpha", profile=True)
    Nky, Nz, M, Nx, R = low.gyro_U.shape
    assert low.gyro_U.size + low.gyro_V.size == Nky * Nz * M * 2 * Nx * R
    assert low.Ainv0.shape == (Nz, Nx, Nx)


def test_simulation_switch_reproduces_the_dense_step():
    """A global nonlinear s–α case stepped with both solvers: the right-hand sides agree to the CG tolerance."""
    from velcro.config import VelcroConfig
    from velcro.simulation import Simulation
    base = dict(grid=dict(Nx=24, Lx=30.0, Ny=12, Nz=6, L=6, M=3),
                geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=2),
                physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), nu=0.05, nonlinear=True, Ti_profile=True),
                numerics=dict(dt=0.01, field_solver="dense"),
                init=dict(kind="noise", amplitude=1e-2, seed=3), run=dict(t_end=0.1, precision="fp64"))
    dense = Simulation(VelcroConfig.model_validate(base))
    base["numerics"] = dict(dt=0.01, field_solver="lowrank", lowrank_tol=1e-10, cg_tol=1e-10, cg_maxiter=200)
    low = Simulation(VelcroConfig.model_validate(base))
    assert low.fields.rank > 1
    G = dense.G
    a, b = np.asarray(dense._rhs(G)), np.asarray(low._rhs(G))
    assert np.allclose(a, b, atol=1e-7 * np.abs(a).max())
    Ga, Gb = np.asarray(dense._nsteps(G, 0.01, 3)), np.asarray(low._nsteps(G, 0.01, 3))
    assert np.allclose(Ga, Gb, atol=1e-7 * np.abs(Ga).max())
