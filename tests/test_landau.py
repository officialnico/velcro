"""Phase 1 validation (§5.1): ion Landau damping of the ion-acoustic wave in a slab.

The least-damped eigenvalue of the code's exact linear operator (one k_y block,
hyper-collisions closing the Hermite hierarchy) must approach the analytic root
of 1 + (1 − Γ0 + 1/τ)/Γ0 + ζ Z(ζ) = 0.
"""
import numpy as np
import pytest

from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.analysis import landau_root_with_flr, linear_matrix


def _sim(L):
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=4, Lx=40.0, Ny=7, Nz=8, L=L, M=1),
        geometry=dict(kind="slab", rho_star=1e-3, Lz_over_Lref=1.0),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=5.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0.0, hyper_v=10.0, krook_rate=0.0, krook_width=0.0, dt=0.005),
        init=dict(kind="landau", amplitude=1e-3, ky_index=1, kx_index=0),
        run=dict(t_end=1.0, diag_every=0.05),
    ))
    return Simulation(cfg)


def test_landau_damping_eigenvalue():
    sim = _sim(48)
    A = linear_matrix(sim, 1)
    w_an = landau_root_with_flr(5.0, b=0.5 * float(sim.ky[1] ** 2), kpar=2 * np.pi)
    om = 1j * np.linalg.eigvals(A)
    nearest = om[np.argmin(np.abs(om - w_an))]
    assert abs(nearest - w_an) / abs(w_an) < 0.01, (nearest, w_an)


def test_landau_damping_time_domain():
    """Time-stepped decay of the probe matches the analytic root to a few percent
    (Prony fit on the two ±ω branches excited by a density initial condition)."""
    from velcro.analysis import prony
    sim = _sim(48)
    d = sim.run(t_end=1.5, verbose=False)
    t, pr = np.array(d.t), np.array(d.probe)
    sel = t >= 0.3
    om, a = prony(t[sel], pr[sel], order=2)
    w_an = landau_root_with_flr(5.0, b=0.5 * float(sim.ky[1] ** 2), kpar=2 * np.pi)
    w = om[np.argmin(np.abs(np.abs(om.real) - w_an.real))]
    assert abs(abs(w.real) - w_an.real) / w_an.real < 0.03
    assert abs(w.imag - w_an.imag) / abs(w_an.imag) < 0.15
