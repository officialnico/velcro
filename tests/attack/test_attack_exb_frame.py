"""Adversarial checks of the shearing frame of ``physics.exb_shear`` (tier 2, KE_VALIDATION E8).

The frame stores G = e^{−iθ_ky x} p with p x-periodic.  The remap replaces (p, θ) by (e^{−inΔk_x x} p, θ − nΔk_x) row by
row: the SAME physical field.  Any correct frame right-hand side F must therefore commute with it,

    F(e^{−inΔk_x x} p, θ − nΔk_x) = e^{−inΔk_x x} F(p, θ)        (n an integer per k_y row),

for every per-row n — not only for the n that keep θ ∝ k_y.  The remap of ``_compile_exb_frame`` moves each row on its
own (|θ_ky| > Δk_x/2), so after the first remap θ is no longer ∝ k_y; the bracket's docstring and
test_frame_fourier_bracket_and_drift_are_the_plain_fourier_operator_on_the_sheared_field rely on θ ∝ k_y.
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.vlasov import exb_remap

FOURIER_X = dict(bracket_dx="fourier", radial_drift="fourier")


def _slab(nonlinear=True, numerics=None, Nx=32, Ny=10, gE=0.7):
    physics = dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=nonlinear,
                   Ti_profile=False, n_profile=False, exb_shear=dict(gamma_E=gE, frame="shearing"))
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=20.0, Ny=Ny, Nz=4, L=3, M=2),
        geometry=dict(kind="slab", rho_star=1e-3),
        physics=physics,
        numerics=dict(hyper_x=0.0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01, field_solver="fft",
                      node_gyro="exact", **(numerics or {})),
        init=dict(kind="noise", amplitude=1.0, seed=3),
        run=dict(t_end=0.1, precision="fp64"))))


TUBE = dict(kind="s_alpha", rho_star=1e-5, r0=0.18, q0=1.4, shat=0.8, n0=1700)
LX_TUBE = 47.2689


def _tube(nonlinear=True, numerics=None, Nx=64, Ny=10, gE=0.3):
    physics = dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), nu=0.01, nonlinear=nonlinear, Ti_profile=False,
                   n_profile=False, exb_shear=dict(gamma_E=gE, frame="shearing"))
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=LX_TUBE, Ny=Ny, Nz=8, L=4, M=2),
        geometry=TUBE, physics=physics,
        numerics=dict(hyper_x=0.0, hyper_z=0.05, hyper_v=0.1, krook_rate=0.0, krook_width=0.0,
                      integrator="rk4", node_gyro="exact", dt=0.002, field_solver="fft", **(numerics or {})),
        init=dict(kind="noise", amplitude=1e-3, seed=5),
        run=dict(t_end=0.01, precision="fp64"))))


def _noise(sim, seed, scale=1.0, kmax=None, zflat=False):
    r = np.random.default_rng(seed)
    G = scale * (r.normal(size=sim.G.shape) + 1j * r.normal(size=sim.G.shape))
    G[:, :, :, 0, :] = G[:, :, :, 0, :].real
    if zflat:
        G[...] = G[..., :1]
    if kmax is not None:
        Gh = np.fft.fft(G, axis=2)
        m = np.abs(np.fft.fftfreq(G.shape[2], d=1.0 / G.shape[2]))
        Gh[:, :, m > kmax] = 0.0
        G = np.fft.ifft(Gh, axis=2)
        G[:, :, :, 0, :] = G[:, :, :, 0, :].real
    return jnp.asarray(G, dtype=sim.dtype)


def _gauge(sim, p, s):
    """(p, θ ∝ k_y) and its row-by-row remapped twin (p′, θ′), n_j = round(j s): the same physical field."""
    ky = np.asarray(sim.ky)
    j = np.rint(ky / ky[1]).astype(int)
    dkx = 2 * np.pi / sim.cfg.grid.Lx
    x = np.asarray(sim.geom.x)
    theta = s * j * dkx
    n = np.rint(j * s).astype(int)
    ph = np.exp(-1j * n[None, :] * dkx * x[:, None])[None, None, :, :, None]          # e^{−i n_j Δk_x x}
    p2 = jnp.asarray(np.asarray(p) * ph)
    th2 = theta - n * dkx
    return (jnp.asarray(theta, dtype=sim.rdtype), p2, jnp.asarray(th2, dtype=sim.rdtype), ph, n)


@pytest.mark.parametrize("s", [0.3, 0.45, 0.8])
def test_frame_rhs_commutes_with_the_remap_slab_nonlinear(s):
    """Slab, z-flat, no gradients/drifts/dissipation: the right-hand side is the E×B bracket alone, evaluated with the
    Fourier ∂_x (exact).  Low-k_x state (|m| ≤ 4 of 32): no aliasing either way."""
    sim = _slab(nonlinear=True, numerics=FOURIER_X)
    p = _noise(sim, 1, scale=10.0, kmax=4, zflat=True)
    theta, p2, th2, ph, n = _gauge(sim, p, s)
    assert len(set(n)) > 1
    F = np.asarray(sim.rhs_fn(p, theta))
    F2 = np.asarray(sim.rhs_fn(p2, th2))
    want = F * ph
    err = np.abs(F2 - want).max() / np.abs(want).max()
    print(f"s={s}: n_j={n.tolist()}  gauge error of the nonlinear frame rhs: {err:.3e}")
    assert err < 1e-10, err


def test_frame_rhs_commutes_with_the_remap_slab_linear_control():
    """The control: the same check with the nonlinearity off (only the E∥/streaming of a z-structured state and the
    field solve act) — the linear frame is gauge covariant."""
    sim = _slab(nonlinear=False, numerics=FOURIER_X)
    p = _noise(sim, 2, scale=1.0, kmax=4)
    theta, p2, th2, ph, n = _gauge(sim, p, 0.3)
    F = np.asarray(sim.rhs_fn(p, theta))
    F2 = np.asarray(sim.rhs_fn(p2, th2))
    err = np.abs(F2 - F * ph).max() / np.abs(F).max()
    print(f"linear slab gauge error {err:.3e}")
    assert err < 1e-11, err


@pytest.mark.parametrize("nonlinear", [False, True])
def test_frame_rhs_commutes_with_the_remap_tube(nonlinear):
    """s-α tube with every term on and the Fourier x-operators (the E8-A6 configuration, without the 2/3 mask)."""
    sim = _tube(nonlinear=nonlinear, numerics=FOURIER_X)
    p = _noise(sim, 3, scale=30.0, kmax=6)
    theta, p2, th2, ph, n = _gauge(sim, p, 0.35)
    F = np.asarray(sim.rhs_fn(p, theta))
    F2 = np.asarray(sim.rhs_fn(p2, th2))
    err = np.abs(F2 - F * ph).max() / np.abs(F).max()
    print(f"tube nonlinear={nonlinear}: n_j={n.tolist()} gauge error {err:.3e}")
    assert err < 1e-10, err


def test_size_of_the_nonlinear_gauge_error_relative_to_the_bracket():
    """Report only: the gauge error against the size of the nonlinear term itself (slab, Fourier operators)."""
    sim = _slab(nonlinear=True, numerics=FOURIER_X)
    lin = _slab(nonlinear=False, numerics=FOURIER_X)
    p = _noise(sim, 4, scale=10.0, kmax=4, zflat=True)
    for s in (0.2, 0.3, 0.45):
        theta, p2, th2, ph, n = _gauge(sim, p, s)
        NL = np.asarray(sim.rhs_fn(p, theta)) - np.asarray(lin.rhs_fn(p, theta))
        NL2 = np.asarray(sim.rhs_fn(p2, th2)) - np.asarray(lin.rhs_fn(p2, th2))
        diff = NL2 - NL * ph
        per_ky = [float(np.abs(diff[:, :, :, k]).max() / max(np.abs(NL[:, :, :, k]).max(), 1e-300)) for k in range(NL.shape[3])]
        print(f"s={s} n={n.tolist()}: |ΔNL|/|NL| per k_y row: " + ", ".join(f"{v:.2e}" for v in per_ky))


# ------------------------------------------------------------------ remap on odd N_x
@pytest.mark.parametrize("Nx", [32, 33])
def test_exb_remap_drops_only_the_wrapping_mode(Nx):
    """exb_remap shifts the x spectrum of a row by one mode and must drop exactly the entry that would wrap from +k_max to
    −k_max (or back).  For even N_x that is the Nyquist neighbour the code zeroes; for odd N_x the indices differ."""
    r = np.random.default_rng(0)
    shape = (2, 1, Nx, 3, 2)
    G = r.normal(size=shape) + 1j * r.normal(size=shape)
    dkx = 2 * np.pi / 20.0
    x = -10.0 + 20.0 * np.arange(Nx) / Nx
    theta = np.array([0.0, 0.6 * dkx, -0.6 * dkx])
    G2, th2 = exb_remap(jnp.asarray(G), jnp.asarray(theta), dkx, float(x[0]))
    G2 = np.asarray(G2)
    # exact: p′ = e^{∓iΔk_x x} p — the coefficient of frequency f comes from f ± 1 — with the entry whose source lies
    # outside the band (the one that would wrap around) dropped
    freqs = np.rint(np.fft.fftfreq(Nx, d=1.0 / Nx)).astype(int)
    kmax, kmin = freqs.max(), freqs.min()
    for row, sgn in ((1, 1), (2, -1)):
        Gh = np.fft.fft(G[:, :, :, row, :], axis=2)
        Wh_exact = np.zeros_like(Gh)
        for i, f in enumerate(freqs):
            g = f + sgn
            if kmin <= g <= kmax:
                Wh_exact[:, :, i] = Gh[:, :, int(np.where(freqs == g)[0][0])] * np.exp(-1j * sgn * dkx * x[0])
        got = np.fft.fft(G2[:, :, :, row, :], axis=2)
        err = np.abs(got - Wh_exact).max() / np.abs(Wh_exact).max()
        print(f"Nx={Nx} row {row}: remap error vs exact shift-with-drop {err:.3e}")
        assert err < 1e-12, (Nx, row, err)


# ------------------------------------------------------------------ the float64 J0 of the FFT solver
def test_fft_solver_j0_float64_at_large_argument():
    """FFTFieldSolver._j0 (float64: jax bessel_jn, n_iter 60) against scipy for the arguments the node gyroaverage reaches:
    a = √(2 b μ_j) with b up to ~50 and μ_j up to ~50 (M = 16) → a up to ~70."""
    from scipy.special import j0
    from velcro.fields import _j0
    x = np.linspace(0.0, 120.0, 2401)
    got = np.asarray(_j0(jnp.asarray(x)))
    err = np.abs(got - j0(x))
    for lo, hi in ((0, 25), (25, 40), (40, 55), (55, 70), (70, 120)):
        m = (x >= lo) & (x < hi)
        print(f"_j0 float64 |error| on [{lo}, {hi}): {err[m].max():.3e}")
    assert err[x < 70].max() < 1e-10, err[x < 70].max()
