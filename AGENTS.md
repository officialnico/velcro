# AGENTS.md

For an AI coding agent that installs Velcro, runs it, reads its outputs and compares them with GKX. Each statement
names the file it comes from; where this file and the code disagree, the code is right.

## What Velcro is

Velcro 0.2.0 is a continuum δf gyrokinetic turbulence code in Python/JAX; the same source runs on CPU and on NVIDIA or
AMD GPUs through XLA (`README.md`). It solves on field-aligned `(x, k_y, z)` grids with `x` in real space and a
Hermite–Laguerre moment velocity space. Electrons are adiabatic (the default; the box may then be radially global) or
kinetic (`physics.electrons = "kinetic"`, real mass ratio, flux tube only: `check_config` in `velcro/kinetic.py`), with
A∥ when `physics.beta > 0` and optional δB∥ (`physics.bpar`). Integrators (`numerics.integrator`): `"rk4"`, `"lawson"`
(integrating-factor RK4) and, for kinetic electrons only, `"imex"` (the electrons' parallel streaming implicit).
Geometries (`geometry.kind`): `slab`, `s_alpha`, `miller`, `vmec`, `fourier`, and `gx_file` (GX's own geometry
arrays). A run uses one GPU: multi-GPU is under "Not in this release" in `README.md`.

## Install

From the repository root, with Python ≥ 3.11 (`pyproject.toml`; `tests/fast_tests.json` was measured on Python 3.12):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "jax[cuda12]==0.11.1"   # the JAX tested on CUDA 12 (examples/gkx_ke/README.md); AMD GPUs: jax[rocm]
pip install -e .                    # pip install -e ".[dev]" also installs pytest, pytest-xdist, matplotlib
python -m velcro info               # velcro 0.2.0  jax 0.11.1  devices [...]: the list must show a GPU
```

## Run

```bash
python -m velcro run CONFIG.toml|CONFIG.json [--t-end T] [--output out.npz] [--quiet]
```

- A `.toml` file is read as TOML, any other suffix as JSON (`VelcroConfig.from_file` in `velcro/config.py`).
- `--t-end` replaces `run.t_end`, `--output` replaces `run.output` (`velcro/__main__.py`).
- The output file is written once, when the run ends, and only if `run.output` or `--output` is set. A run whose |φ|²
  turns non-finite raises `FloatingPointError: simulation diverged` and writes nothing (`Simulation.run`).
- Every `run.diag_every` of simulated time it prints `t={t:9.3f} dt={dt:.3e} |φ|²={phi2:.4e} Q={Q:.4e}  [{elapsed:6.1f}s]`
  (nothing with `--quiet`): the simulated time and the current step in L_ref/v_t, the `phi2` and `Q` diagnostics (`Q` is
  the ions' total heat flux), and the wall seconds since the time loop started. The first interval includes compilation.
- A line starting `WARNING: imex:` means the IMEX factors did not fit the device: the run uses block-Thomas instead of
  dense factors, or Lawson RK4 instead of the IMEX (`numerics.imex_fallback = "lawson"`, the default; `"error"` refuses
  to run instead) (`velcro/imex.py`, `Simulation._imex_prepare`).

From Python (`velcro/simulation.py`):

```python
from velcro import VelcroConfig, Simulation
sim = Simulation(VelcroConfig.from_file("examples/gkx_ke/velcro_fast.json"))
print(sim.solver_info())   # the integrator that runs and, with the IMEX, the factorisation in use
sim.run(t_end=204.0); sim.save("fast.npz")   # run(..., verbose=False) drops the printed lines
```

## The output file

`Simulation.save` writes one `np.savez` file. n = the number of diagnostic records (one at t = 0, then one per
`run.diag_every`); S = 1 species with adiabatic electrons, 2 with kinetic ones (ions first).

| key | shape | content |
|---|---|---|
| `G` | (S·L, M, Nx, Nky, Nz) | final state (complex), species stacked on axis 0 |
| `t` | () | time of `G` |
| `x`, `z`, `ky` | (Nx,), (Nz,), (Nky,) | x in ρ_ref, z on [−π, π), k_y in 1/ρ_ref |
| `config` | () | the validated config as a JSON string, every default filled in; `numerics.integrator` is what ran |
| `diag_t`, `dt` | (n,), (n,) | time of each record; the time step in use there (one value per record, not per step) |
| `phi2` | (n,) | Σ\|φ\|² over the grid, k_y > 0 counted twice (a sum, so it depends on the grid) |
| `W` | (n,) | Σ\|G\|², weighted by n̂T̂ per species with kinetic electrons |
| `Q`, `Gamma` | (n,) | ion heat and particle flux, box averages; with kinetic electrons `Q = Qi_es + Qi_em + Qi_bpar` |
| `hermite`, `laguerre` | (n, S·L), (n, M) | Σ\|G\|² per Hermite row and per Laguerre index |
| `phi_ky` | (n, Nky) | Σ over x and z of \|φ\|² per k_y |
| `phi_zonal` | (n, Nx) | flux-surface-averaged k_y = 0 potential |
| `probe` | (n,) | complex φ at x index Nx//2, k_y index 1, z index Nz//2 |

Kinetic electrons add (n,) series (`KineticDiagnostics`, `kinetic_fluxes` in `velcro/kinetic.py`): `Qi_es`, `Qi_em`,
`Qi_bpar`, `Qe_es`, `Qe_em`, `Qe_bpar` (each species' heat flux from φ, from A∥ flutter and from δB∥; `*_bpar` is 0
without `physics.bpar`), `Qe = Qe_es + Qe_em + Qe_bpar`, `Gamma_e`, and `apar2` = Σ|A∥|². The E×B shearing frame adds
`exb_theta`. The file holds no wall-clock time. Read the config back with `json.loads(str(d["config"]))`.

Mean heat fluxes over a window (Velcro time units):

```python
import numpy as np
d = np.load("fast.npz")
t, T0, T1 = d["diag_t"], 100.0, 204.0                     # set T0, T1 to your saturated window
w = (t >= T0) & (t <= T1)
for name, key in (("<Q_i>", "Q"), ("<Q_e>", "Qe")):
    print(name, d[key][w].mean(), "Velcro;", 0.3666 * d[key][w].mean(), "GX/GKX units (GKX example only)")
```

## Units, and conversion to GX/GKX

Velcro (`README.md`, "Normalisation"; the docstring of `velcro/gx_geometry.py`): v_t = √(2T/m) = √2 v_ti,
ρ_ref = v_t/Ω = √2 ρ_i, time in L_ref/v_t, gradients ω_n = L_ref/L_n and ω_T = L_ref/L_T, ρ* = ρ_ref/L_ref an input
(`geometry.rho_star`), fluxes in Velcro's gyro-Bohm units. L_ref is the major radius R (always in `s_alpha`; by default
in `miller`, `vmec` and `gx_file`, where `geometry.Lref_over_a` changes it). GX and GKX use a, ρ_i and v_ti. With R = L_ref:

| quantity | GX/GKX from Velcro | GKX example, R/a = 2.7778 | source |
|---|---|---|---|
| time | t_GX = (R/a)/√2 · t | 1.9642 t | `velcro/gx_geometry.py`, `velcro/kinetic.py` |
| rates γ, ω, ν | rate_GX = √2 (a/R) · rate | rate/1.9642 | the same, and `velcro/config.py` |
| heat flux | Q_GX = 2√2 (a/R)² Q / ⟨\|∇ρ\|⟩ | 0.3666 Q | `velcro/gx_geometry.py` |
| wavenumbers | k ρ_i = k ρ_ref/√2 (Velcro's `ky` is k_y ρ_ref) | | `README.md` |
| gradients | a/L_T = (a/R) ω_T, a/L_n = (a/R) ω_n | | `README.md` |

⟨|∇ρ|⟩ is GX's flux normaliser: 1 in s-α, `sim.geom.extra["grho_avg"]` in the `miller`, `vmec` and `gx_file`
geometries. `physics.beta` is GX's β, and `physics.electron.mass` is m_e/m_i (`velcro/config.py`).

The box: Velcro's modes are k_y = 2πn/L_y, n = 0 … (Ny−1)//3 (`ky_grid` in `velcro/operators.py`). GKX's y0 is
`geometry.Ly = 2π y0/√2` (in ρ_ref) in the `miller`, `vmec` and `gx_file` geometries. In `s_alpha`,
L_y = 2π r0/(q0 ρ* n0) with an integer toroidal mode number `geometry.n0` (`velcro/geometry.py`), so k_y ρ_i equals
GKX's only up to that integer: the example's n0 = 645 (ρ* = 10⁻⁵) gives y0 = 28.19 for GKX's 28.2. Set
`grid.Lx = jtwist/(|S| k_y,min)`, with S = ŝ in s-α as in the example (jtwist 5); the `miller`, `vmec` and `gx_file`
geometries refuse an L_x for which S k_y,min L_x is not an integer (`velcro/simulation.py`).

## GKX's kinetic-electron example (`examples/gkx_ke/`)

GKX's kinetic-electron Cyclone case as native Velcro configs (`examples/gkx_ke/README.md`): GKX's box (y0 28.2,
L_x 176.25 ρ_i, jtwist 5), grid 64 × 64 × 24, (L, M) = (16, 4), kinetic ions and electrons, m_e/m_i 2.7 × 10⁻⁴,
β 10⁻⁴ with A∥. All three use `geometry.kind = "s_alpha"` with `gpar = "constant"` (GX's s-α model),
`numerics.integrator = "imex"`, `run.precision = "fp32"`, `run.t_end = 204` (≈ 400 a/v_ti), `run.diag_every = 1`
and `run.output = null`, so always pass `--output`. They differ only in the two fields of the table below.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m velcro run examples/gkx_ke/velcro_fast.json     --output fast.npz     | tee fast.log
python -m velcro run examples/gkx_ke/velcro_lowmem.json   --output lowmem.npz   | tee lowmem.log
python -m velcro run examples/gkx_ke/velcro_adaptive.json --output adaptive.npz | tee adaptive.log
```

| config | `numerics.dt` | `numerics.imex_solver` | use |
|---|---|---|---|
| `velcro_fast.json` | 1/79 R/v_t (0.0249 a/v_ti), fixed | `"auto"`: dense while one factor set fits `imex_dense_gib` (4 GiB), else block-Thomas | the fastest of the three on this case |
| `velcro_lowmem.json` | the same fixed step | `"thomas"`: block-Thomas, L times less factor memory than dense | when the dense factors do not fit |
| `velcro_adaptive.json` | `null`: adaptive step from an estimate of the fastest explicit rate, `numerics.cfl` 0.9 | `"auto"` | the general default |

The fixed step is set for this case's box only, and a fixed `numerics.dt` skips the step control (`Simulation.run`).
Start every other case from `velcro_adaptive.json`.

Known issue in 0.2.0: `python -m velcro run` re-estimates the adaptive step only at a chunk start where the step count is
a multiple of `numerics.steps_per_dt_update` (`Simulation.run`). Chunks end at every diagnostic, so the step can stay at
its first estimate for many time units, including through the turbulence's growth. If an adaptive run diverges there,
rerun with a fixed `numerics.dt` below the first `dt` the adaptive run printed.

## Matching a GKX run

1. Same case: convert GKX's inputs with the tables above (gradients × R/a, collision frequencies × (R/a)/√2, β and
   m_e/m_i unchanged) and use the same box (y0, jtwist).
2. Same grid: the same Nx × Ny × Nz; `grid.L` (Hermite moments, v∥) is GKX's Nm and `grid.M` (Laguerre moments, μ) is
   GKX's Nl (`velcro/config.py`; the example: (16, 4) = (Nm, Nl)). Velcro keeps (Ny−1)//3 + 1 k_y modes and, with
   `numerics.x_dealias`, |k_x| ≤ ((Nx−1)//3) Δk_x (`velcro/config.py`).
3. Same precision: `run.precision`, `"fp64"` (the default) or `"fp32"`; fp32 on consumer GPUs (`README.md`). Velcro
   sets `jax_default_matmul_precision = "highest"`, so its fp32 contractions are not TF32 (`velcro/simulation.py`);
   report GKX's setting with it.
4. One code at a time on the GPU, the same GPU for both, `XLA_PYTHON_CLIENT_PREALLOCATE=false` for both.
5. The same physical time span and averaging window (here t_GX = 1.9642 t_Velcro).
6. Speed: wall seconds per unit of simulated time after the first printed interval (compilation and the first IMEX
   factors), from the printed elapsed times, so do not pass `--quiet`. Divide by (R/a)/√2 for seconds per a/v_ti
   (script below).
7. A run that printed `WARNING: imex:` did not run the configured solver: say so next to its time, or rerun it with
   more free device memory. Report the config used and its `dt`.

```bash
python - fast.log <<'EOF'
import re, sys
rows = re.findall(r"^t=\s*(\S+) .*\[\s*(\S+)s\]$", open(sys.argv[1], encoding="utf-8").read(), re.M)
(t0, w0), (t1, w1) = [(float(a), float(b)) for a, b in (rows[0], rows[-1])]
s = (w1 - w0) / (t1 - t0)
print(f"{s:.4g} wall s per R/v_t, {s / 1.9642:.4g} per a/v_ti")   # 1.9642 = (R/a)/sqrt(2) of the GKX example
EOF
```

## Other cases

- This release ships no GX/GKX deck translator. Write a Velcro config by hand (TOML or JSON): every field and its
  meaning is in `velcro/config.py` (sections `grid`, `geometry`, `physics`, `numerics`, `init`, `run`).
- Start from the nearest example: `examples/cyclone_nonlinear.toml` (adiabatic electrons, global s-α, nonlinear),
  `examples/landau_slab.toml` (slab Landau damping, `nonlinear = false`), `examples/gkx_ke/*.json` (kinetic electrons,
  flux tube, every field written out). Start adaptive (`numerics.dt = null`); keep `grid.M ≤ 24` (`README.md`).
- Kinetic electrons and the `miller`, `vmec` and `gx_file` geometries need `physics.Ti_profile = false` and
  `physics.n_profile = false` (both default to true; `velcro/kinetic.py`, `velcro/simulation.py`). The GKX example also
  sets `numerics.krook_rate = 0` and `numerics.krook_width = 0` (an x-periodic box).
- The example's `numerics` block picks the options `velcro/config.py` describes as GX's: `parallel_bc = "gx_chains"`,
  `parallel_derivative = "fourier"`, `x_dealias`, `bracket_dx` and `radial_drift` = `"fourier"`, `hyper_kz`,
  `gx_hyper`, `gx_damp_ends`. `gx_hyper`, `gx_hyperz` and `gx_damp_ends` are rates in v_t/R: GX's rate × (R/a)/√2.
- Geometry: `s_alpha`; `miller` (GX's parameter names, needs `Rmaj` and `Ly`); `vmec` (a wout `file` and `Ly`); or
  `gx_file`, GX's own geometry output (`RUN.out.nc`, `RUN.eik.out` or `RUN.eik.nc`) with `Ly` set and `grid.Nz` equal to
  its ntheta (`velcro/gx_geometry.py`).
- Unknown keys are ignored without an error (pydantic's default; `velcro/config.py` does not forbid extras). Print what
  was read: `python -c 'from velcro.config import VelcroConfig as C; print(C.from_file("my.json").model_dump_json(indent=1))'`.
- To run a GKX or GX deck as it stands, ask the Velcro authors for the matching config.

## Tests

```bash
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -m fast   # the quick CPU subset
python -m pytest tests                       # the full suite
```

Run them from the repository root: `conftest.py` marks as `fast` exactly the tests listed, with their measured CPU
durations, in `tests/fast_tests.json`. The `slow` marker in `pyproject.toml` mentions `--runslow`, but no such option
exists: slow tests run in the full suite (`-m "not slow"` skips them). Tests that need `gkx` skip without it.

## Licence

All rights reserved; see `LICENSE`. Cite Velcro with `CITATION.cff`.
