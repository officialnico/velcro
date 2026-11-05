"""Smoke and conservation tests of the nonlinear E×B term."""
import numpy as np
import pytest

from velcro.config import VelcroConfig
from velcro.simulation import Simulation


def test_nonlinear_run_is_stable_and_finite():
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=16, Lx=40.0, Ny=10, Nz=8, L=6, M=3),
        geometry=dict(kind="s_alpha", rho_star=1 / 400, r0=0.18, q0=1.4, shat=0.8, n0=4),
        physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=0.05, nonlinear=True),
        numerics=dict(hyper_x=0.05, hyper_z=0.05, hyper_v=0.5, krook_rate=1.0, krook_width=0.15, cfl=0.4),
        init=dict(kind="noise", amplitude=1e-2, seed=2),
        run=dict(t_end=3.0, diag_every=1.0),
    ))
    sim = Simulation(cfg)
    d = sim.run(verbose=False)
    assert np.all(np.isfinite(d.phi2))
    assert len(d.t) >= 3


def test_exb_advection_conserves_free_energy_in_slab():
    """With gradients, drifts and dissipation off, the E×B nonlinearity alone conserves
    Σ|G|² (∫ g² d³v) to round-off with the skew-symmetric bracket, for a smooth *and* for a
    white-noise (grid-scale) state; the advective form only does so for smooth fields."""
    import jax.numpy as jnp
    from velcro.vlasov import rhs
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=32, Lx=20.0, Ny=16, Nz=4, L=3, M=3),
        geometry=dict(kind="slab", rho_star=1e-3),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=True,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01),
        init=dict(kind="noise", amplitude=1.0, seed=3),
        run=dict(t_end=0.1),
    ))
    sim = Simulation(cfg)
    # smooth initial condition: keep only low k_x
    G = np.asarray(sim.G)
    Gk = np.fft.fft(G, axis=2)
    kx = np.fft.fftfreq(G.shape[2])
    Gk[:, :, np.abs(kx) > 0.15] = 0
    G = jnp.asarray(np.fft.ifft(Gk, axis=2))
    phi, Phi = sim._solve_fields(G)
    import dataclasses
    op_lin = dataclasses.replace(sim.op, nonlinear=False)
    NL = np.asarray(rhs(G, phi, Phi, sim.op)) - np.asarray(rhs(G, phi, Phi, op_lin))   # bracket term alone
    w = np.where(np.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, :, None]
    def injection(G, op):
        phi, Phi = sim._solve_fields(G)
        NL = np.asarray(rhs(G, phi, Phi, op)) - np.asarray(rhs(G, phi, Phi, dataclasses.replace(op, nonlinear=False)))
        dW = np.sum(np.real(np.conj(np.asarray(G)) * NL) * w)
        return abs(dW) / (np.sqrt(np.sum(np.abs(np.asarray(G)) ** 2 * w)) * np.sqrt(np.sum(np.abs(NL) ** 2 * w)))
    op_adv = dataclasses.replace(sim.op, bracket_form="advective")
    assert injection(G, sim.op) < 1e-12
    assert injection(G, op_adv) < 1e-4
    G_noise = sim.G                                   # unfiltered white noise: grid-scale in x
    assert injection(G_noise, sim.op) < 1e-12
    assert injection(G_noise, op_adv) > 1e-3
