# GKX's kinetic-electron example, in Velcro

GKX's kinetic-electron Cyclone case as native Velcro configs:
- the example's own box: y0 28.2, so k_y ρ_i 0.0355 … 0.75, L_x 176.25 ρ_i, jtwist 5;
- grid 64 × 64 × 24;
- kinetic ions and electrons, m_e/m_i 2.7 × 10⁻⁴, β 10⁻⁴ with A∥.

Velcro's (Hermite, Laguerre) = (16, 4) is GKX's (Nm, Nl) = (16, 4).

## Install

Tested with jax 0.11.1 on CUDA 12:

```
pip install "jax[cuda12]==0.11.1"
pip install -e .
```

## Run

```
python -m velcro run examples/gkx_ke/velcro_fast.json --output fast.npz
```

| config | time step | use |
|---|---|---|
| `velcro_fast.json` | fixed, 1/79 R/v_t (0.0249 a/v_ti) | the fastest run of this case |
| `velcro_lowmem.json` | the same fixed step, block-tridiagonal implicit solver | much less GPU memory, a little slower |
| `velcro_adaptive.json` | adaptive (CFL 0.9) | the general default |

- The fixed step is set for this case's box. For other cases, start from the adaptive config.
- Velcro's time unit is R/v_t, and t_GX = 1.9642 t_Velcro. The configs run to t = 204 (t ≈ 400 a/v_ti); `--t-end` changes it.
- The output file holds the diagnostics per unit of time. The heat fluxes are `Qi_es`, `Qi_em`, `Qe_es`, `Qe_em`, … in Velcro's gyro-Bohm units; on this case GX's/GKX's Q is 0.3666 × Velcro's.

## Compare with GKX

`gkx_cyclone.toml` is GKX's deck of this case, and `velcro_cyclone.toml` is Velcro's. The Velcro file is
`velcro_fast.json` written as TOML.

```
gkx examples/gkx_ke/gkx_cyclone.toml
python -m velcro run examples/gkx_ke/velcro_cyclone.toml --output velcro_cyclone.npz
```

| | GKX (`gkx_cyclone.toml`) | Velcro (`velcro_cyclone.toml`) |
|---|---|---|
| physics, box, grid, moments | the same | the same |
| time step | `imex-ars3`, fixed dt 0.035 a/v_ti | IMEX ARK3, fixed 1/79 R/v_t (0.0249 a/v_ti) |
| run length | t_max = 400 a/v_ti | t_end = 204 R/v_t (400.7 a/v_ti) |
| noise seed | 22 | 0: compare time averages, not trajectories |
| units | a/v_ti, ρ_i, GX's gyro-Bohm flux | R/v_t, ρ_ref = √2 ρ_i; t_GX = 1.9642 t, Q_GX = 0.3666 Q |

`python examples/compare_speed.py --pair ke` runs GKX for 3 minutes of wall time, then Velcro for 3 minutes, on the
same GPU. It prints each code's start-up time, its wall seconds per a/v_ti once stepping, how far it got, and at the
end both codes' heat fluxes over the window they both reached, with a chart (`compare_speed.png`): progress against
wall time and the heat fluxes against time. `--pair adiabatic` does the same for
`examples/gkx_adiabatic`, and `--budget` changes the 3 minutes.

To compare the two runs:
- the growth rate in the linear phase;
- ⟨Q_i⟩ and ⟨Q_e⟩ over the same saturated window, for example t ≥ 150 a/v_ti;
- wall seconds per a/v_ti after the first (compile) interval, on the same GPU, one code at a time.

## A collisional electromagnetic variant

`gkx_cyclone_coll_em.toml` is the same case with collisions and δB∥ added:
- conserving Dougherty collisions (GKX `lenard_bernstein`; ν_ii 0.01, ν_ee 0.6086 v_ti/a, like-species only);
- A∥ and δB∥ at β 10⁻³.

E×B flow shear is not part of it: GKX 2.8.1 has no flow-shear option. Velcro's twin is
`velcro_cyclone_coll_em.toml` (IMEX, adaptive step).

This variant has not been run end to end yet: on a 16 GB GPU, GKX stopped while setting up its implicit step for it.
