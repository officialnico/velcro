"""End to end: the shearing frame evolved through several remaps equals the frame evolved with no remap at all.

Both runs integrate the same physical equations; the remap only re-labels the representation (e^{−inΔk_x x} p,
θ − nΔk_x), dropping nothing while the state has no content near the band edge.  Without remaps θ_ky = k_y γ_E t stays
∝ k_y (the regime in which even the old bracket was exact), so the no-remap run is the reference.  The slab (no x
dependence of any coefficient, so no band-edge content is generated linearly) with z structure (streaming, E∥, the
field solve) and the Fourier x-operators.  A weak nonlinearity (amplitude 0.1, |k_x| ≤ 2Δk_x, t = 0.6 at γ_E = 7: the
rows are remapped 1, 3 and 4 times — not additive) keeps the cascade four doublings away from the band edge, so the
remaps drop < 1e-8 and the two physical fields must agree far below the size of the old bracket's error (which was
~0.3 of the nonlinear change of the state, i.e. ~1e-3 here).
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from test_attack_exb_frame import FOURIER_X, _noise, _slab


def _physical(sim, p, theta):
    x = np.asarray(sim.geom.x)
    return np.asarray(p) * np.exp(-1j * np.asarray(theta)[None, None, None, :, None] * x[None, None, :, None, None])


@pytest.mark.parametrize("amp,tol", [(1e-6, 1e-11), (0.1, 1e-6)])
def test_frame_through_remaps_equals_the_frame_without_remaps(amp, tol):
    sim = _slab(nonlinear=True, numerics=FOURIER_X, Nx=64, gE=7.0)
    p0 = _noise(sim, 7, scale=amp, kmax=2)
    dt, n = 0.005, 120
    rate = np.asarray(sim.exb_rate)
    dkx = sim.exb_dkx
    pA, thA = sim._nsteps(p0, dt, n, sim.exb_theta)                       # the code's own RK4 + remap after every step
    nrem = np.rint(rate * n * dt / dkx).astype(int)
    print(f"remaps per k_y row: {nrem.tolist()}")
    J = len(nrem)
    assert nrem.max() >= 3 and any(nrem[a] + nrem[b] != nrem[a + b] for a in range(1, J) for b in range(a, J - a))
    f = sim.rhs_fn

    @jax.jit
    def rk4(p, th):                                                      # the same RK4 with θ continued, never remapped
        k1 = f(p, th)
        k2 = f(p + 0.5 * dt * k1, th + 0.5 * dt * rate)
        k3 = f(p + 0.5 * dt * k2, th + 0.5 * dt * rate)
        k4 = f(p + dt * k3, th + dt * rate)
        return p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), th + dt * rate
    pB, thB = p0, jnp.asarray(np.zeros_like(rate))
    for _ in range(n):
        pB, thB = rk4(pB, thB)
    GA, GB = _physical(sim, pA, thA), _physical(sim, pB, thB)
    lin = _slab(nonlinear=False, numerics=FOURIER_X, Nx=64, gE=7.0)
    pL, thL = lin._nsteps(p0, dt, n, lin.exb_theta)
    nl_change = np.abs(GA - _physical(lin, pL, thL)).max() / np.abs(GA).max()          # what the bracket did
    err = np.abs(GA - GB).max() / np.abs(GB).max()
    spec = np.abs(np.fft.fft(np.asarray(pA), axis=2))
    edge = spec[:, :, 24:41].max() / spec.max()
    print(f"amp {amp}: |G_remap − G_noremap| / |G| = {err:.2e}; the nonlinearity changed the state by {nl_change:.2e}; "
          f"band-edge content of the remapped state {edge:.1e}")
    assert err < tol, err
