"""M0 velocity-space economy pre-test (§5.7, §7 of the design document).

Scans the minimum converged (L, M) for the linear ITG growth rate as a function
of collisionality, per collision model, and records the Hermite / Laguerre
free-energy spectra of the converged eigenmode.  Uses Velcro's own exact
linear operator (jax.jacfwd) in a narrow s-α box at small ρ* (flux-tube-like),
so it runs on a laptop CPU in minutes.

Usage:  python scripts/m0_velocity_economy.py [--nu 0 0.01 0.05 0.2] [--models dougherty lenard-bernstein]
Output: m0_economy.npz and a printed table.
"""
import argparse, time
import numpy as np
from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.analysis import linear_matrix


def growth_rate(L, M, nu, model, n0=64, Nx=8, Nz=12, rho_star=1 / 1000, Lx=24.0, return_vec=False):
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=Lx, Ny=7, Nz=Nz, L=L, M=M),
        geometry=dict(kind="s_alpha", rho_star=rho_star, r0=0.18, q0=1.4, shat=0.8, n0=n0),
        physics=dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), tau=1.0, nu=nu, collision_model=model,
                     nonlinear=False, Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0.0, hyper_z=0.02, hyper_v=0.2, krook_rate=0.0, krook_width=0.0, dt=0.01),
        init=dict(kind="noise"), run=dict(t_end=1.0)))
    sim = Simulation(cfg)
    A = linear_matrix(sim, 1)
    lam, vec = np.linalg.eig(A)
    i = np.argmax(lam.real)
    om = 1j * lam[i]
    if return_vec:
        v = vec[:, i].reshape(L, M, Nx, Nz)
        herm = np.sum(np.abs(v) ** 2, axis=(1, 2, 3)); lag = np.sum(np.abs(v) ** 2, axis=(0, 2, 3))
        return om, herm / herm.sum(), lag / lag.sum(), float(sim.ky[1])
    return om, float(sim.ky[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.2])
    ap.add_argument("--models", nargs="+", default=["dougherty"])
    ap.add_argument("--Ls", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24])
    ap.add_argument("--Ms", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--out", default="m0_economy.npz")
    a = ap.parse_args()
    results = {}
    print(f"{'model':18s} {'nu':>6s} {'ky':>6s} {'L*':>4s} {'M*':>4s} {'gamma':>8s} {'omega_r':>8s}  {'gamma_ref(L,M max)':>18s}  time")
    for model in a.models:
        for nu in a.nu:
            t0 = time.time()
            Lmax, Mmax = a.Ls[-1], a.Ms[-1]
            om_ref, herm, lag, ky = growth_rate(Lmax, Mmax, nu, model, return_vec=True)
            table = {}
            Lstar = Mstar = None
            # converge L at M = Mmax, then M at L = L*
            for L in a.Ls:
                om, _ = growth_rate(L, Mmax, nu, model)
                table[(L, Mmax)] = om
                if abs(om.imag - om_ref.imag) <= a.tol * abs(om_ref.imag) and Lstar is None:
                    Lstar = L
            Lstar = Lstar or Lmax
            for M in a.Ms:
                om, _ = growth_rate(Lstar, M, nu, model)
                table[(Lstar, M)] = om
                if abs(om.imag - om_ref.imag) <= a.tol * abs(om_ref.imag) and Mstar is None:
                    Mstar = M
            Mstar = Mstar or Mmax
            om_star = table[(Lstar, Mstar)]
            print(f"{model:18s} {nu:6.3f} {ky:6.3f} {Lstar:4d} {Mstar:4d} {om_star.imag:8.4f} {om_star.real:8.4f}  "
                  f"{om_ref.imag:8.4f}{'':10s}  {time.time()-t0:5.0f}s", flush=True)
            results[f"{model}_nu{nu}"] = dict(Lstar=Lstar, Mstar=Mstar, omega=om_star, omega_ref=om_ref, hermite=herm,
                                              laguerre=lag, ky=ky, table={str(k): v for k, v in table.items()})
    np.savez(a.out, **{k: np.array(str(v)) for k, v in results.items()},
             **{f"{k}_hermite": v["hermite"] for k, v in results.items()},
             **{f"{k}_laguerre": v["laguerre"] for k, v in results.items()})
    # Hermite-spectrum slope diagnostic (§5.7): fit |G_ℓ|² ∝ ℓ^{-p} on ℓ ≥ 2
    for k, v in results.items():
        h = v["hermite"]; l = np.arange(len(h))
        sel = (l >= 2) & (h > 0)
        p = -np.polyfit(np.log(l[sel]), np.log(h[sel]), 1)[0]
        print(f"{k:28s} Hermite spectrum slope p = {p:.2f}  (m^-1/2 phase-mixing ↔ p=0.5; steeper = compact)")


if __name__ == "__main__":
    main()
