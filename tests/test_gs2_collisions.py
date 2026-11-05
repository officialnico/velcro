"""GS2's model collision operator in the Hermite–Laguerre basis (velcro/gs2_collisions.py, collision_model = "gs2").

Truths used (none produced by the code under test):
* GS2's own formulas (collisions.fpp init_vnew, transcribed here independently): hee = e^{−E}/√(πE) + (1 − 1/2E) erf √E,
  hsg = erf v/2v² − e^{−v²}/(√π v), vnew_D = hee/E^{3/2}, vnew_s = 4 hsg/√E, vnew_E = E(vnew_s(2 − 0.5/E) − 2 vnew_D);
* the energy-loss identity ν_E = −(1/v²F₀) d(ν_∥v⁵F₀)/dv (finite differences), which is what makes GS2's energy restoring
  exact;
* the like-species test-particle Landau operator (velcro/landau.py's independent Rosenbluth-potential route) = GS2's
  Lorentz + energy diffusion without the restoring terms;
* the e–i Lorentz operator's (1, 0) element −4/(3√π) (closed form) and landau.lorentz_matrix;
* pitch-angle scattering's eigenfunctions v̂ⁿPₙ(ξ) with eigenvalue −n(n+1)/2 at constant ν (const_v): v̂∥ (n = 1) and
  2v̂∥² − μ̂ (n = 2);
* conservation (particles; like-species parallel momentum and energy) and the H-theorem (symmetric, negative
  semi-definite), every GS2 collision_model; the e–i block conserves particles and energy and removes momentum;
* quadrature convergence at (48, 16) (a finer, wider grid changes nothing).
"""
import numpy as np
import pytest
from scipy.special import erf

from velcro import gs2_collisions as gc
from velcro import landau as la
from velcro.collisions import collision_matrix

SQRT_PI = np.sqrt(np.pi)


def _e(L, M, l, m):
    return np.eye(L * M)[l * M + m]


# ------------------------------------------------------------------ GS2's frequencies
def test_frequencies_are_gs2s_formulas():
    E = np.linspace(0.02, 30.0, 200)
    v = np.sqrt(E)
    hee = np.exp(-E) / np.sqrt(np.pi * E) + (1.0 - 0.5 / E) * erf(np.sqrt(E))
    hsg = 0.5 * erf(v) / v ** 2 - np.exp(-v ** 2) / (SQRT_PI * v)
    nuD, nus = hee / E ** 1.5, hsg / np.sqrt(E) * 4.0
    nuE = E * (nus * (2.0 - 0.5 / E) - 2.0 * nuD)
    f = gc.frequencies(v)
    assert np.allclose(f["nuD"], nuD, rtol=1e-12) and np.allclose(f["nus"], nus, rtol=1e-12)
    assert np.allclose(f["nuE"], nuE, rtol=1e-11, atol=1e-13)
    assert np.allclose(f["nuD_ei"], 1 / v ** 3)
    # the series of G joins the closed form at x = 0.1, and is right below it
    x = np.array([0.0999999, 0.1000001, 0.05, 0.01])
    closed = (erf(x) - 2 * x * np.exp(-x ** 2) / SQRT_PI) / (2 * x ** 2)
    assert np.allclose(gc.chandrasekhar_G(x), closed, rtol=1e-9)
    # const_v: every frequency at v̂ = 1
    fc = gc.frequencies(v, const_v=True)
    f1 = gc.frequencies(np.ones(1))
    assert all(np.allclose(fc[k], f1[k][0]) for k in f1)


def test_energy_restoring_frequency_is_the_energy_loss_weight():
    x = np.linspace(0.2, 6.0, 60)
    F = lambda y: (2 * gc.chandrasekhar_G(y) / y ** 3) * y ** 5 * np.exp(-y ** 2)      # ν_∥ v⁵ F₀ (up to π^{-3/2})
    h = 1e-5
    w = -(F(x + h) - F(x - h)) / (2 * h) / (x ** 2 * np.exp(-x ** 2))
    assert np.allclose(gc.frequencies(x)["nuE"], w, rtol=1e-7, atol=1e-9)


# ------------------------------------------------------------------ against independent operators
@pytest.mark.parametrize("L,M", [(6, 3), (8, 4)])
def test_lorentz_plus_energy_diffusion_is_the_landau_test_part(L, M):
    p = gc._pieces(L, M, False, *gc._grid_size(L, M))
    mine = p["lorentz"] + p["ediffuse"]
    ref = la.test_particle_matrix(L, M, alpha=1.0, mass_ratio=1.0)
    assert np.abs(mine - ref).max() <= 1e-9 * np.abs(ref).max()


def test_ei_block_is_the_lorentz_operator():
    L, M = 8, 4
    T = gc.ei_matrix(L, M)
    assert np.isclose(T[M, M], -4 / (3 * SQRT_PI), rtol=1e-12)              # the (1, 0) element, closed form
    ref = la.lorentz_matrix(L, M)
    assert np.abs(T - ref).max() <= 1e-10 * np.abs(ref).max()
    assert not gc.ei_matrix(L, M, "ediffuse").any()


def test_const_v_pitch_angle_eigenfunctions():
    L, M = 8, 4
    p = gc._pieces(L, M, True, *gc._grid_size(L, M))
    nuD1 = float(gc.frequencies(np.ones(1))["nuD"][0])
    A = p["lorentz"]
    u1 = _e(L, M, 1, 0)                                       # v̂∥ = ψ₁/√2: n = 1, eigenvalue −ν_D
    assert np.allclose(A @ u1, -nuD1 * u1, atol=1e-12)
    u2 = np.sqrt(2) * _e(L, M, 2, 0) + _e(L, M, 0, 1)          # 2v̂∥² − μ̂ = √2 ψ₂ + L₁: n = 2, eigenvalue −3ν_D
    assert np.allclose(A @ u2, -3 * nuD1 * u2, atol=1e-12)
    # and the like-species restoring term gives the momentum back: (lorentz + R_L) v̂∥ = 0
    C = gc.like_matrix(L, M, "lorentz", const_v=True)
    assert np.allclose(C @ u1, 0, atol=1e-12)


# ------------------------------------------------------------------ conservation and the H-theorem
@pytest.mark.parametrize("terms,const_v", [("default", False), ("lorentz", False), ("ediffuse", False), ("lorentz", True)])
@pytest.mark.parametrize("L,M", [(8, 4), (16, 8)])
def test_like_species_conserves_and_dissipates(terms, const_v, L, M):
    C = gc.like_matrix(L, M, terms, const_v)
    sc = np.abs(C).max()
    assert np.abs(C - C.T).max() <= 1e-12 * sc                                   # self-adjoint
    ev = np.linalg.eigvalsh(0.5 * (C + C.T))
    assert ev.max() <= 1e-11 * sc                                                # H-theorem
    for row in la.moment_functionals(L, M):                                      # n, u∥, E
        assert np.abs(row @ C).max() <= 1e-11 * sc
    nnull = int(np.sum(np.abs(ev) <= 1e-10 * sc))
    # exactly n, u∥, E — except pitch-angle scattering alone, which also spares every isotropic function (a polynomial in
    # v̂∥² + μ̂); energy diffusion damps any v̂-dependence of h/F₀ at fixed ξ, and no polynomial is a function of ξ alone
    assert nnull > 3 if terms == "lorentz" else nnull == 3


def test_ei_block_conserves_particles_and_energy_and_takes_momentum():
    L, M = 16, 8
    T = gc.ei_matrix(L, M)
    dens, mom, en = la.moment_functionals(L, M)
    sc = np.abs(T).max()
    assert np.abs(dens @ T).max() <= 1e-12 * sc and np.abs(en @ T).max() <= 1e-11 * sc
    assert (mom @ T @ _e(L, M, 1, 0)) < 0                                        # friction on the ions at rest
    assert np.linalg.eigvalsh(0.5 * (T + T.T)).max() <= 1e-12 * sc


def test_quadrature_is_converged_at_48_16():
    L, M = 48, 16
    nv, vmax = gc._grid_size(L, M)
    A = gc.like_matrix(L, M)
    B = gc.like_matrix(L, M, nv=int(1.5 * nv), vmax=vmax + 3.0)
    assert np.abs(A - B).max() <= 1e-9 * np.abs(A).max()
    T, U = gc.ei_matrix(L, M), gc.ei_matrix(L, M, nv=int(1.5 * nv), vmax=vmax + 3.0)
    assert np.abs(T - U).max() <= 1e-9 * np.abs(T).max()


# ------------------------------------------------------------------ the solver carries it
def _cfg(**phys):
    from velcro.config import VelcroConfig
    geo = dict(kind="s_alpha", rho_star=1e-5, q0=1.4, shat=0.8, n0=1700)
    ph = dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), nonlinear=False, Ti_profile=False, n_profile=False,
              nu=0.01, collision_model="gs2", electrons="kinetic", beta=0.0,
              electron=dict(mass=2.7e-4, nu=0.05), ei_collisions=True)
    ph.update(phys)
    num = dict(hyper_x=0.0, hyper_z=0.0, hyper_v=0.0, krook_rate=0.0, krook_width=0.0, integrator="rk4", dt=1e-4,
               parallel_derivative="fourier", field_solver="fft", radial_drift="fourier", bracket_dx="fourier")
    return VelcroConfig.model_validate(dict(
        grid=dict(Nx=8, Lx=40.0, Ny=7, Nz=16, L=6, M=3), geometry=geo, physics=ph, numerics=num,
        init=dict(kind="noise", amplitude=1.0, seed=1), run=dict(t_end=1e-3, precision="fp64")))


@pytest.mark.parametrize("terms", ["default", "lorentz", "ediffuse"])
def test_simulation_carries_the_gs2_operator(terms):
    from velcro.simulation import Simulation
    sim = Simulation(_cfg(gs2_collision_terms=terms))
    L, M = 6, 3
    b = sim.op.base
    assert b.coll_dense
    assert np.allclose(np.asarray(b.coll).reshape(L * M, L * M), gc.like_matrix(L, M, terms), atol=1e-14)
    assert np.allclose(np.asarray(sim.op.landau_T[1]).reshape(L * M, L * M), gc.ei_matrix(L, M, terms), atol=1e-14)
    assert np.allclose(np.asarray(sim.op.landau_nu), [0.0, 0.05])
    assert not np.asarray(sim.op.landau_T[0]).any() and not np.asarray(sim.op.landau_F).any()
    sim.run()                                                                    # a few steps under the CFL bound
    assert np.isfinite(np.asarray(sim.G)).all()


def test_config_refusals():
    with pytest.raises(ValueError, match="collision_flr"):
        _cfg(collision_flr=True)
    with pytest.raises(ValueError, match="gs2_const_v"):
        _cfg(gs2_const_v=True)
    with pytest.raises(ValueError, match="need collision_model = 'gs2'"):
        _cfg(collision_model="dougherty", gs2_collision_terms="lorentz")
    from velcro.simulation import Simulation
    with pytest.raises(ValueError, match="β > 0"):
        Simulation(_cfg(beta=1e-3))


def test_collision_matrix_routes_gs2_and_leaves_the_others():
    L, M = 6, 3
    assert np.allclose(collision_matrix("gs2", L, M, "lorentz"), gc.like_matrix(L, M, "lorentz"))
    assert np.allclose(collision_matrix("landau", L, M), la.like_species_matrix(L, M))
