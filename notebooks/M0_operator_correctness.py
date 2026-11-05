# %% [markdown]
# # Velcro M0: operator correctness and the velocity-economy pre-test
#
# This notebook is the M0 deliverable of the design document (§7.1): it walks from the
# primitive operators to the first validation cases, running the checks inline.
# Every cell is executable from a fresh checkout (`uv pip install -e ".[dev]"`).
#
# 1. Hermite–Laguerre basis: orthonormality, recurrences vs Gauss quadrature, mirror operator
# 2. Gyroaverage coefficients 𝒥_m(b) and Γ₀
# 3. Finite-difference convergence orders, twist-and-shift ghosts, FFT round trip, bracket
# 4. Field solve: manufactured solution
# 5. Geometry: analytic s–α vs numerically constructed Fourier equilibrium
# 6. Landau damping: exact linear operator eigenvalue vs analytic root
# 7. M0 velocity-economy pre-test: minimum (L, M) vs collisionality

# %%
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from velcro import basis, geometry, operators as ops
np.set_printoptions(precision=4, linewidth=120)

# %% [markdown]
# ## 1. Basis
# `ψ_ℓ` (orthonormal Hermite functions) and `L_m` (Laguerre polynomials).  The recurrence
# matrices used by the solver are compared against quadrature.

# %%
v, wv = basis.hermite_quadrature(48); mu, wm = basis.laguerre_quadrature(48)
L, M = 10, 8
Psi = np.array([basis.psi(l, v) for l in range(L)]); Lag = np.array([basis.laguerre(m, mu) for m in range(M)])
print("Hermite Gram error :", np.abs((Psi * wv) @ Psi.T - np.eye(L)).max())
print("Laguerre Gram error:", np.abs((Lag * wm) @ Lag.T - np.eye(M)).max())
print("v̂ recurrence error :", np.abs((Psi * wv * v) @ Psi.T - basis.hermite_v_matrix(L)).max())
print("μ̂ recurrence error :", np.abs((Lag * wm * mu) @ Lag.T - basis.laguerre_mu_matrix(M)).max())

# %% [markdown]
# The mirror operator (parallel dynamics at fixed μ with `μ̂ = μB(z)/T`) is
# `Mirror = −(V ⊗ Mu·P) + ½ (Dv ⊗ Mu)`; its density row reproduces the magnetic
# compression `−B ∇∥(n u∥/B)` exactly.

# %%
mir = basis.mirror_matrix(6, 4).reshape(6, 4, 6, 4)
print("density row of Mirror (ℓ'=1 column) :", mir[0, 0, :, 0])
print("V row 0                              :", basis.hermite_v_matrix(6)[0])

# %% [markdown]
# ## 2. Gyroaverage in Laguerre space
# `J₀(√(2bμ̂)) = Σ_m 𝒥_m(b) L_m(μ̂)` with `𝒥_m = (b/2)^m e^{-b/2}/m!`, and `Σ_m 𝒥_m² = Γ₀(b)`.

# %%
b = np.array([0.0, 0.5, 1.0, 4.0])
J = basis.gyro_coefficients(b, 60)
print("Σ 𝒥_m²      :", (J**2).sum(0)); print("Γ₀(b)       :", basis.gamma0(b))

# %% [markdown]
# ## 3. Spatial operators

# %%
for Nx in (16, 32, 64):
    x = np.arange(Nx) * 2 * np.pi / Nx
    f = jnp.asarray(np.sin(2 * x), dtype=complex)[:, None, None]
    err = float(jnp.max(jnp.abs(ops.ddx(f, 2 * np.pi / Nx)[:, 0, 0] - 2 * np.cos(2 * x))))
    print(f"ddx  Nx={Nx:3d}  max error {err:.3e}")
Ny = 24; nky = ops.n_ky(Ny)
f = np.random.default_rng(0).normal(size=(2, nky, 2)) + 0j; f[:, 0] = f[:, 0].real
print("FFT round trip error:", float(jnp.max(jnp.abs(ops.to_spec_y(ops.to_real_y(jnp.asarray(f), Ny), nky) - f))))

# %% [markdown]
# ## 4. Field solve (slab, manufactured solution)
# The quasi-neutrality operator is Hermitian positive definite per `(k_y, z)`; the
# solver reproduces a prescribed φ from the density it implies (see `tests/test_fields.py`).

# %%
from velcro.fields import build_field_solver
g = geometry.slab(32, 20.0, 12, Ly=2 * np.pi * 5, Nz=4); ky = ops.ky_grid(12, g.Ly)
fs = build_field_solver(g, ky, np.ones(32), np.ones(32), 4, 20.0)
A = np.linalg.inv(np.asarray(fs.Ainv))
print("Hermitian error:", max(np.abs(A[k, 0] - A[k, 0].conj().T).max() for k in range(len(ky))))
print("min eigenvalue :", min(np.linalg.eigvalsh(A[k, 0]).min() for k in range(len(ky))))

# %% [markdown]
# ## 5. Geometry cross-check
# The analytic global s–α tokamak against the general Fourier-surface equilibrium
# evaluated for concentric circular surfaces (agreement to O(r/R)).

# %%
r0 = 0.02; qs = lambda s: 1.4 * (1 + 0.8 * (np.sqrt(s) / 0.5 - 1))
ga = geometry.s_alpha(8, 4, 1, 16, rho_star=1 / 2000, r0=r0, q0=1.4, shat=0.8)
gf = geometry.circular_tokamak_fourier(1.0, r0 / 0.5, 1.0, qs).build(8, 4, 1, 16, s0=0.25, n0=1, rho_star=1 / 2000)
for k in ("B", "gpar", "gxy", "gyy", "Kx", "Ky", "Cinv"):
    a, bb = np.asarray(getattr(ga, k)), np.asarray(getattr(gf, k))
    print(f"{k:5s} rel. difference at box centre: {np.abs(a[4] - bb[4]).max() / np.abs(a[4]).max():.3e}")

# %% [markdown]
# ## 6. Landau damping (Phase 1 validation, §5.1)
# The exact linear operator of the code (via `jax.jacfwd`) has a least-damped eigenvalue
# converging to the analytic ion-acoustic root as the Hermite resolution grows.

# %%
from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.analysis import landau_root_with_flr, linear_matrix
for L in (16, 32, 48):
    cfg = VelcroConfig.model_validate(dict(
        grid=dict(Nx=4, Lx=40.0, Ny=7, Nz=8, L=L, M=1), geometry=dict(kind="slab", rho_star=1e-3),
        physics=dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=5.0, nonlinear=False, Ti_profile=False, n_profile=False),
        numerics=dict(hyper_x=0, hyper_z=0, hyper_v=10.0, krook_rate=0, krook_width=0, dt=0.005),
        init=dict(kind="landau", ky_index=1), run=dict(t_end=1.0)))
    sim = Simulation(cfg)
    om = 1j * np.linalg.eigvals(linear_matrix(sim, 1))
    w_an = landau_root_with_flr(5.0, b=0.5 * float(sim.ky[1] ** 2), kpar=2 * np.pi)
    print(f"L={L:2d}: nearest eigen-ω = {om[np.argmin(np.abs(om - w_an))]:.4f}   analytic {w_an:.4f}")

# %% [markdown]
# ## 7. M0 velocity-economy pre-test (§5.7, §7)
# `scripts/m0_velocity_economy.py` scans the minimum converged (L, M) for the linear ITG
# growth rate against collisionality per collision model and records the Hermite
# free-energy spectrum slope.  Run it and load the result here.

# %%
import os
if os.path.exists("m0_economy.npz"):
    d = np.load("m0_economy.npz")
    for k in d.files:
        if k.endswith("_hermite"):
            h = d[k]; l = np.arange(len(h)); sel = (l >= 2) & (h > 0)
            print(k, "slope", -np.polyfit(np.log(l[sel]), np.log(h[sel]), 1)[0])
else:
    print("run: python scripts/m0_velocity_economy.py  (writes m0_economy.npz)")
