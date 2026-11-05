"""Rosenbluth–Hinton zonal-flow residual test (§5.1).

A k_y = 0 potential with radial wavenumber k_x in a circular tokamak relaxes,
after GAM oscillations damp, to the residual  φ(∞)/φ(0) = 1 / (1 + 1.6 q² / sqrt(ε)).
Run:  python scripts/rosenbluth_hinton.py
"""
import sys, time
import numpy as np
from velcro.config import VelcroConfig
from velcro.simulation import Simulation


def run(q0=1.4, eps=0.18, L=16, M=8, Nx=16, Nz=32, t_end=60.0, rho_star=1 / 1000, kx_index=1, Lx=40.0, verbose=False):
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=Lx, Ny=7, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=rho_star, r0=eps, q0=q0, shat=0.0, n0=1),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=False,
                     Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0.0, hyper_z=0.02, hyper_v=0.5, krook_rate=0.0, krook_width=0.0, cfl=0.4, dt_max=0.05),
        init=dict(kind="zonal", amplitude=1e-3, kx_index=kx_index, z_profile="flat"),
        run=dict(t_end=t_end, diag_every=0.5)))
    sim = Simulation(cfg)
    d = sim.run(verbose=verbose)
    zon = np.array(d.phi_zonal)                     # (nt, Nx)
    kx = 2 * np.pi * kx_index / Lx
    amp = zon @ np.cos(kx * sim.geom.x) / np.sum(np.cos(kx * sim.geom.x) ** 2)   # projection on cos(kx x)
    t = np.array(d.t)
    rh = 1.0 / (1.0 + 1.6 * q0**2 / np.sqrt(eps))
    return t, amp / amp[0], rh, sim


if __name__ == "__main__":
    t0 = time.time()
    t, ratio, rh, sim = run(verbose=("-v" in sys.argv))
    late = ratio[t > 0.7 * t[-1]].mean()
    print(f"Rosenbluth–Hinton: simulated residual {late:.3f}, theory {rh:.3f}  (q={1.4}, eps={0.18}, {time.time()-t0:.0f}s)")
    print("t:", np.round(t[::10], 1)); print("phi/phi0:", np.round(ratio[::10], 3))
