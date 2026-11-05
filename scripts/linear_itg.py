"""Linear ITG growth rate in global s-α geometry (Cyclone Base Case parameters)."""
import sys, time
import numpy as np
from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.analysis import prony

def run(n0, L=12, M=4, Nx=48, Nz=16, t_end=40.0, verbose=False, Lx=100.0, rho_star=1/1000, hyper_v=0.5):
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=Lx, Ny=7, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=rho_star, r0=0.18, q0=1.4, shat=0.8, n0=n0),
        physics=dict(omega_n=dict(kind="constant", value=2.2), omega_T=dict(kind="constant", value=6.9),
                     tau=1.0, nu=0.0, nonlinear=False, Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0.02, hyper_z=0.02, hyper_v=hyper_v, krook_rate=1.0, krook_width=0.1, cfl=0.4, dt_max=0.05),
        init=dict(kind="noise", amplitude=1e-4, seed=1),
        run=dict(t_end=t_end, diag_every=0.5),
    ))
    sim = Simulation(cfg)
    ky = sim.ky[1]
    t0 = time.time()
    d = sim.run(verbose=verbose)
    t = np.array(d.t); p = np.array(d.phi2); pr = np.array(d.probe)
    sel = t >= 0.6 * t[-1]
    # growth of the k_y = ky[1] mode itself: Σ|φ|² over all k_y reports whichever of k_y and 2k_y grows faster
    gamma = 0.5 * np.polyfit(t[sel], np.log(np.stack(d.phi_ky)[sel, 1]), 1)[0]
    om, a = prony(t[sel], pr[sel], order=1)
    return ky, gamma, om[0], time.time() - t0, sim

if __name__ == "__main__":
    # with rho_star = 1/1000: C_y = 128.6 rho, so k_y rho_i = n0 / 128.6
    for n0 in (int(a) for a in (sys.argv[1:] or ["26", "51", "77", "103"])):
        ky, gamma, om, dt, sim = run(n0)
        print(f"n0={n0:3d}  ky rho_i = {ky:.3f}   gamma R/v_t = {gamma:.4f}   omega (prony) = {om:.4f}   [{dt:.0f}s, {sim.step_count} steps]", flush=True)
