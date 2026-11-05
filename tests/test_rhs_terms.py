"""Each linear term of the Vlasov right-hand side, isolated, in s-α geometry: one test against an
independent answer (velocity-space coefficients from Gauss quadrature of the continuous operator,
not from the recurrence matrices the code uses) and one property test (what the term must conserve).
Also the flux diagnostic and the explicit-stability estimate."""
import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from velcro import basis
from velcro.config import VelcroConfig
from velcro.operators import ddx, ddz
from velcro.simulation import Simulation
from velcro.vlasov import rhs, cfl_rates

L, M = 6, 3


@pytest.fixture(scope="module")
def sim():
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=16, Lx=40.0, Ny=10, Nz=8, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
        physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01,
                      parallel_form="advective"),   # term-by-term split; the skew form is tested against it below
        init=dict(kind="noise", amplitude=1.0, seed=5), run=dict(t_end=0.1))))


def only(op, *keep):
    """The operators with every linear coefficient zeroed except those named in ``keep``."""
    z = {k: jnp.zeros_like(getattr(op, k)) for k in ("c_stream", "c_mirror", "c_Kx", "c_Ky", "c_Cinv") if k not in keep}
    return dataclasses.replace(op, **z)


def rand(shape, seed):
    r = np.random.default_rng(seed)
    return jnp.asarray(r.normal(size=shape) + 1j * r.normal(size=shape))


def coeffs(f):
    """⟨ψ_ℓ L_m | f⟩ by Gauss quadrature of the continuous function f(v̂, μ̂)."""
    return basis.project(f, L, M)


def weights(G):
    return np.where(np.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, :, None]


# ------------------------------------------------------------------ streaming and E∥
def test_streaming_of_a_density_perturbation_is_minus_v_dz(sim):
    """−v∥ ∇∥ acting on g(x,y,z)·F0:  v̂·1 = ψ_1/√2, so only ℓ = 1 is driven, by −c ∂_z g / √2;
    the potential enters through H = G + Φ/T the same way (E∥)."""
    op = only(sim.op, "c_stream")
    shp = sim.G.shape
    g, P = rand(shp[2:], 1), rand((M,) + shp[2:], 2)
    G = jnp.zeros(shp, dtype=sim.dtype).at[0, 0].set(g)
    c1 = coeffs(lambda v, mu: v)[1, 0]
    assert abs(c1 - 1 / np.sqrt(2)) < 1e-12
    out = np.asarray(rhs(G, None, jnp.zeros_like(P), op))
    expect = -np.asarray(op.c_stream) * c1 * np.asarray(ddz(g, op.dz, op.phase))
    assert np.allclose(out[1, 0], expect, atol=1e-12)
    rest = out.copy(); rest[1, 0] = 0
    assert np.abs(rest).max() < 1e-12
    outP = np.asarray(rhs(jnp.zeros(shp, dtype=sim.dtype), None, P, op))
    expectP = -np.asarray(op.c_stream) * c1 * np.asarray(ddz(P * op.inv_Ti, op.dz, op.phase))
    assert np.allclose(outP[1], expectP, atol=1e-12)


def _parallel_injection(Nz):
    """Relative rate of change of the Jacobian-weighted free energy Σ_z J Σ_ℓm |G|² under streaming +
    mirror force, for a state smooth in z (k_y = 0, where the twist-and-shift phase is 1)."""
    s = Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=8, Lx=40.0, Ny=10, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01,
                      parallel_form="advective"),   # term-by-term split; the skew form is tested against it below
        init=dict(kind="noise", amplitude=1.0, seed=5), run=dict(t_end=0.1))))
    z = s.geom.z
    a = np.random.default_rng(0).normal(size=(L, M, 8))
    G = np.zeros(s.G.shape, complex)
    G[:, :, :, 0, :] = a[..., None] * (1 + 0.5 * np.cos(z) + 0.3 * np.sin(2 * z))
    G = jnp.asarray(G)
    zero = jnp.zeros((M,) + G.shape[2:], dtype=G.dtype)
    fsw = np.asarray(s.op.fsw)[None, None, :, None, :]

    def inj(*keep):
        out = np.asarray(rhs(G, None, zero, only(s.op, *keep)))
        return np.sum(fsw * np.real(np.conj(np.asarray(G)) * out)) / np.sum(fsw * np.abs(np.asarray(G)) * np.abs(out))
    return inj("c_stream"), inj("c_stream", "c_mirror")


def test_streaming_plus_mirror_conserves_jacobian_weighted_free_energy():
    """b·∇z varies along the field line, so streaming alone does not conserve the free energy; the
    mirror force supplies the compensating term (phase-space volume ∝ B).  Together they conserve
    Σ_z J Σ|G|² to the truncation order of the z stencil (4th: 16x per doubling of N_z)."""
    (s16, both16), (s32, both32) = _parallel_injection(16), _parallel_injection(32)
    assert abs(s32) > 1e-3                        # streaming alone: O(ε) and not converging away
    assert abs(both32) < 1e-4
    assert abs(both32) < abs(both16) / 8


def _par_sim(Nz, form):
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=8, Lx=40.0, Ny=10, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01, parallel_form=form),
        init=dict(kind="noise", amplitude=1.0, seed=5), run=dict(t_end=0.1))))


def test_skew_parallel_form_is_the_same_operator():
    """For a state smooth in z the skew form equals the advective one up to the truncation error of the stencil
    (4th order: the difference falls 16x per doubling of N_z); the advective form is tested against quadrature above."""
    diff = []
    for Nz in (16, 32):
        sa, ss = _par_sim(Nz, "advective"), _par_sim(Nz, "skew")
        z = sa.geom.z; a = np.random.default_rng(0).normal(size=(L, M, 8))
        G = np.zeros(sa.G.shape, complex); G[:, :, :, 0, :] = a[..., None] * (1 + 0.5 * np.cos(z) + 0.3 * np.sin(2 * z))
        G = jnp.asarray(G)
        P = np.zeros((M,) + G.shape[2:], complex); P[:, :, 0, :] = np.random.default_rng(3).normal(size=(M, 8, 1)) * (1 + 0.2 * np.cos(z) + 0.1 * np.sin(3 * z))
        P = jnp.asarray(P)
        oa = np.asarray(rhs(G, None, P, only(sa.op, "c_stream", "c_mirror")))
        os_ = np.asarray(rhs(G, None, P, only(ss.op, "c_stream", "c_mirror")))
        diff.append(np.abs(oa - os_).max() / np.abs(oa).max())
    assert diff[1] < 2e-4 and diff[1] < diff[0] / 8, diff


def test_skew_parallel_form_conserves_free_energy_for_any_state():
    """Σ_z J Σ_ℓm |G|² is conserved to round-off by streaming + mirror in the skew form for white noise in z, at k_y = 0 and
    with the twist-and-shift phase (k_y ≠ 0); the advective form injects at O(1) of the term's size for such a state."""
    ss, sa = _par_sim(16, "skew"), _par_sim(16, "advective")
    G = rand(ss.G.shape, 8); zero = jnp.zeros((M,) + G.shape[2:], dtype=G.dtype)
    fsw = np.asarray(ss.op.fsw)[None, None, :, None, :]

    def inj(s):
        out = np.asarray(rhs(G, None, zero, only(s.op, "c_stream", "c_mirror")))
        return np.sum(fsw * np.real(np.conj(np.asarray(G)) * out), axis=(0, 1, 2, 4)) / np.sum(fsw * np.abs(np.asarray(G)) * np.abs(out), axis=(0, 1, 2, 4))
    assert np.abs(inj(ss)).max() < 1e-13                     # every k_y separately
    assert np.abs(inj(sa)).max() > 1e-3


# ------------------------------------------------------------------ magnetic drift
def test_drift_of_a_maxwellian_perturbation(sim):
    """−½B⁻¹(2v̂²+μ̂)[K_x ∂_x + i k_y K_y] acting on g·F0: velocity coefficients from quadrature of 2v̂²+μ̂."""
    op = only(sim.op, "c_Kx", "c_Ky")
    shp = sim.G.shape
    g = rand(shp[2:], 4)
    G = jnp.zeros(shp, dtype=sim.dtype).at[0, 0].set(g)
    c = coeffs(lambda v, mu: 2 * v ** 2 + mu)
    assert np.isclose(c[0, 0], 2.0) and np.isclose(c[2, 0], np.sqrt(2.0)) and np.isclose(abs(c[0, 1]), 1.0)
    spatial = np.asarray(op.c_Kx) * np.asarray(ddx(g, op.dx)) + np.asarray(op.c_Ky) * 1j * np.asarray(op.ky) * np.asarray(g)
    out = np.asarray(rhs(G, None, jnp.zeros((M,) + shp[2:], dtype=sim.dtype), op))
    assert np.allclose(out, -c[:, :, None, None, None] * spatial[None, None], atol=1e-11)


def test_binormal_drift_conserves_free_energy(sim):
    """(2v̂²+μ̂) is symmetric and i k_y K_y(x,z) is anti-Hermitian, so the k_y part of the drift on G
    cannot change Σ|G|²."""
    op = only(sim.op, "c_Ky")
    G = rand(sim.G.shape, 6)
    out = np.asarray(rhs(G, None, jnp.zeros((M,) + G.shape[2:], dtype=G.dtype), op))
    inj = np.sum(np.real(np.conj(np.asarray(G)) * out) * weights(G))
    assert abs(inj) < 1e-10 * np.sum(np.abs(np.asarray(G)) * np.abs(out) * weights(G))


# ------------------------------------------------------------------ gradient drive
@pytest.mark.parametrize("q", [0, 1])
def test_drive_coefficients_from_quadrature(sim, q):
    """−½C⁻¹ i k_y [ω_n + ω_T(v̂²+μ̂−3/2) − ω_B μ̂] Φ_q L_q(μ̂): coefficients by quadrature."""
    op = only(sim.op, "c_Cinv")
    shp = sim.G.shape
    p = rand(shp[2:], 7 + q)
    P = jnp.zeros((M,) + shp[2:], dtype=sim.dtype).at[q].set(p)
    Lq = lambda mu: basis.laguerre(q, mu)
    cn, cT, cB = (coeffs(f) for f in (lambda v, mu: Lq(mu), lambda v, mu: (v ** 2 + mu - 1.5) * Lq(mu), lambda v, mu: mu * Lq(mu)))
    field = -np.asarray(op.c_Cinv) * 1j * np.asarray(op.ky) * np.asarray(p)
    expect = (np.asarray(op.omega_n) * cn[:, :, None, None, None] + np.asarray(op.omega_T) * cT[:, :, None, None, None]
              - np.asarray(op.c_omegaB) * cB[:, :, None, None, None]) * field[None, None]
    out = np.asarray(rhs(jnp.zeros(shp, dtype=sim.dtype), None, P, op))
    assert np.allclose(out, expect, atol=1e-11)


def test_drive_without_the_gradB_part_is_the_diamagnetic_drive_only():
    """``physics.drive_gradB = False``: −½C⁻¹ i k_y [ω_n + ω_T(v̂²+μ̂−3/2)] Φ, coefficients by quadrature; and the
    difference to the default is exactly +½C⁻¹ i k_y ω_B μ̂ Φ."""
    cfg = dict(grid=dict(Nx=16, Lx=40.0, Ny=10, Nz=8, L=L, M=M),
               geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
               physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=0.0, nonlinear=False,
                            Ti_profile=False, n_profile=False, drive_gradB=False),
               numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01),
               init=dict(kind="noise", amplitude=1.0, seed=5), run=dict(t_end=0.1))
    s0 = Simulation(VelcroConfig.model_validate(cfg))
    cfg["physics"]["drive_gradB"] = True
    s1 = Simulation(VelcroConfig.model_validate(cfg))
    shp = s0.G.shape; p = rand(shp[2:], 21)
    P = jnp.zeros((M,) + shp[2:], dtype=s0.dtype).at[0].set(p)
    G0 = jnp.zeros(shp, dtype=s0.dtype)
    cn, cT, cB = (coeffs(f) for f in (lambda v, mu: 1.0 + 0 * v * mu, lambda v, mu: v ** 2 + mu - 1.5, lambda v, mu: mu))
    field = (-np.asarray(s0.op.c_Cinv) * 1j * np.asarray(s0.op.ky) * np.asarray(p))[None, None]
    out0 = np.asarray(rhs(G0, None, P, only(s0.op, "c_Cinv")))
    assert np.allclose(out0, (2.2 * cn + 6.9 * cT)[:, :, None, None, None] * field, atol=1e-11)
    out1 = np.asarray(rhs(G0, None, P, only(s1.op, "c_Cinv")))
    wB = np.asarray(s1.geom.omega_B)                                          # (Nx, 1, Nz)
    assert np.allclose(out1 - out0, -cB[:, :, None, None, None] * wB[None, None] * field, atol=1e-11)


def test_drive_vanishes_for_zonal_modes_and_flat_profiles(sim):
    op = only(sim.op, "c_Cinv")
    P = rand((M,) + sim.G.shape[2:], 9)
    G0 = jnp.zeros(sim.G.shape, dtype=sim.dtype)
    assert np.abs(np.asarray(rhs(G0, None, P, op))[:, :, :, 0]).max() == 0
    flat = dataclasses.replace(op, omega_n=0 * op.omega_n, omega_T=0 * op.omega_T, c_omegaB=0 * op.c_omegaB)
    assert np.abs(np.asarray(rhs(G0, None, P, flat))).max() == 0


# ------------------------------------------------------------------ Krook buffers
def _krook_sim(zonal):
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=16, Lx=40.0, Ny=10, Nz=8, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.7, krook_width=0.25, krook_zonal=zonal, dt=0.01),
        init=dict(kind="noise", amplitude=1.0, seed=5), run=dict(t_end=0.1))))


@pytest.mark.parametrize("zonal", [True, False])
def test_krook_buffer_profile_and_zonal_switch(zonal):
    """−ν_K(x) G with ν_K = rate·cos²(π d/2w) within w = width·L_x of either edge (written out here independently);
    with ``krook_zonal = False`` the k_y = 0 modes are untouched and the others unchanged."""
    s = _krook_sim(zonal)
    op = only(s.op)                                   # every linear coefficient off: only the Krook term is left
    G = rand(s.G.shape, 12)
    out = np.asarray(rhs(G, None, jnp.zeros((M,) + G.shape[2:], dtype=G.dtype), op))
    x = np.asarray(s.geom.x); d = np.minimum(x - x.min(), x.max() - x); w = 0.25 * 40.0
    nuK = np.where(d < w, 0.7 * np.cos(0.5 * np.pi * d / w) ** 2, 0.0)
    expect = -nuK[None, None, :, None, None] * np.asarray(G)
    if not zonal:
        expect[:, :, :, 0, :] = 0
    assert np.allclose(out, expect, atol=1e-13)
    assert nuK.max() == 0.7 and (nuK == 0).sum() >= 6


def test_krook_never_increases_free_energy():
    s = _krook_sim(False); G = rand(s.G.shape, 13)
    out = np.asarray(rhs(G, None, jnp.zeros((M,) + G.shape[2:], dtype=G.dtype), only(s.op)))
    assert np.all(np.real(np.conj(np.asarray(G)) * out) <= 0)


# ------------------------------------------------------------------ heat-flux diagnostic
def test_heat_flux_against_direct_quadrature(sim):
    """Q = ⟨Σ_ky w Re[v_x* ∫(v̂²+μ̂) g J0 …]⟩ recomputed in numpy with quadrature energy coefficients."""
    G = rand(sim.G.shape, 10)
    phi, Phi = sim._solve_fields(G)
    Phi, Gn = np.asarray(Phi), np.asarray(G)
    vx = -np.asarray(sim.op.c_Cinv) * 1j * np.asarray(sim.op.ky) * Phi                       # (M, Nx, Nky, Nz)
    E = np.zeros((M, L, M))                                                                   # ⟨ψ_0 L_m|(v̂²+μ̂)|ψ_p L_q⟩
    for p_ in range(L):
        for q_ in range(M):
            E[:, p_, q_] = coeffs(lambda v, mu: (v ** 2 + mu) * basis.psi(p_, v) * basis.laguerre(q_, mu))[0]
    EG = np.einsum("mpq,pq...->m...", E, Gn)
    w = np.where(np.arange(Gn.shape[3]) == 0, 1.0, 2.0)[None, None, :, None]
    Qxz = np.sum(np.real(np.conj(vx) * EG) * w, axis=(0, 2))
    fsw = np.asarray(sim.op.fsw)
    assert np.isclose(float(sim._diagnostics(G)["Q"]), np.sum(Qxz * fsw) / fsw.shape[0], rtol=1e-10)


def test_heat_flux_is_quadratic_and_zero_without_binormal_structure(sim):
    G = rand(sim.G.shape, 11)
    Q1, Q3 = float(sim._diagnostics(G)["Q"]), float(sim._diagnostics(3.0 * G)["Q"])
    assert np.isclose(Q3, 9.0 * Q1, rtol=1e-10)
    zonal = jnp.zeros_like(G).at[:, :, :, 0].set(G[:, :, :, 0])
    assert abs(float(sim._diagnostics(zonal)["Q"])) < 1e-14 * abs(Q1)


# ------------------------------------------------------------------ explicit stability estimate
def _radius_and_rate(l, m, estimate, Ti_profile):
    s = Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=16, Lx=60.0, Ny=10, Nz=8, L=l, M=m),
        geometry=dict(kind="s_alpha", rho_star=1 / 300, r0=0.18, q0=1.4, shat=0.8, n0=5),
        physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=0.02, nonlinear=False,
                     Ti_profile=Ti_profile),
        numerics=dict(hyper_x=0.05, hyper_z=0.05, hyper_v=0.5, krook_rate=1.0, krook_width=0.12, cfl_estimate=estimate),
        init=dict(kind="noise", amplitude=1.0, seed=1), run=dict(t_end=0.1))))
    v = s.G / jnp.linalg.norm(s.G)
    lam = 0.0
    for _ in range(400):                       # power iteration; the operator is complex-linear when nonlinear = False
        y = s._rhs(v)
        lam = float(jnp.linalg.norm(y))
        v = y / lam
    return lam, float(s._rate(s.G))


@pytest.mark.parametrize("LM", [(6, 3), (12, 5), (24, 8)])
def test_spectral_cfl_estimate_bounds_the_measured_radius(LM):
    """The per-term bound is never below the measured spectral radius of the linear operator and
    stays within 2x of it, with a global T_i profile (T_i up to 2 at the inner edge)."""
    lam, rate = _radius_and_rate(*LM, "spectral", True)
    assert lam <= rate <= 2.0 * lam, f"bound {rate:.1f} vs measured radius {lam:.1f}"


def test_legacy_cfl_estimate_underestimates_with_a_temperature_profile():
    """Documents why 'spectral' exists: the legacy rates ignore T_i(x) in the drift and the mirror term."""
    lam, rate = _radius_and_rate(24, 8, "legacy", True)
    assert rate < 0.8 * lam
