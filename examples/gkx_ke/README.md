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
