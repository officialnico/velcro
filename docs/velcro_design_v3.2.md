# Velcro: An Open-Source GPU-Native Global Gyrokinetic Turbulence Code for Stellarators

### Architecture, Numerical Methods, and Validation Plan

*Technical design document, v3.2*

*v3.2 integrates the Risk 1 revisions and the validated velocity-space convergence analysis. A summary of changes from v3.1 is appended as a revision note. Items flagged "[cite to confirm]" need the author to complete one bibliographic entry.*

-----

## Abstract

We present the design and implementation plan for a new open-source global gyrokinetic turbulence code targeting stellarator geometry, aimed at filling a specific and currently empty cell in the stellarator simulation ecosystem: an openly licensed, GPU-native, continuum (Eulerian) global code. Global nonlinear gyrokinetic turbulence simulation in non-axisymmetric magnetic equilibria is today served by GENE-3D [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694), a consortium-access continuum code developed at IPP Garching that is not redistributable and not commercially licensable, and by EUTERPE [Kornilov et al. 2004; Kleiber et al. 2024], a particle-in-cell code that is CPU-only and not distributed as open source. The most advanced open, GPU-native alternatives — GX [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631), stella [Barnes et al. 2019](https://doi.org/10.1016/j.jcp.2019.01.025), GS2 [Dorland et al. 2000](https://doi.org/10.1103/PhysRevLett.85.5579) — are all deliberately restricted to the flux-tube (radially local) regime. Our approach adopts the Laguerre–Hermite pseudo-spectral velocity-space representation of Mandell, Dorland & Landreman [2018] — shown to retain its economy under strongly non-Maxwellian edge and scrape-off-layer conditions [Frei et al. 2024; Frei et al. 2025] — and applies it to global radial coordinates on field-aligned grids derived from the GVEC ideal-MHD equilibrium solver [Hindenlang et al. 2025](https://doi.org/10.1088/1361-6587/adba11). The velocity scheme is proven for stellarator ITG in the flux-tube limit and for the non-Maxwellian tokamak edge; *economy in the low-collisionality, trapped-particle-driven stellarator core remains the one quantitative open question* (§6), testable in flux-tube before any code is written (§7) and graceful in failure. The implementation is in Python with JAX, following the GX precedent [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631) for compiler-managed GPU portability and free differentiability of the solver. We outline the governing equations, discretization scheme, parallelization strategy, validation pathway against published GENE-3D benchmarks and W7-X experimental data, and the project plan to first usable code.

-----

## Table of Contents

1. [Introduction and motivation](#1-introduction-and-motivation)
1. [Governing equations](#2-governing-equations)
1. [Discretization](#3-discretization)
1. [GPU and parallelization strategy](#4-gpu-and-parallelization-strategy)
1. [Validation](#5-validation)
1. [Technical risks](#6-technical-risks)
1. [Roadmap and operating mode](#7-roadmap-and-operating-mode)
1. [Conclusion](#8-conclusion)
1. [References](#references)

-----

## 1. Introduction and motivation

The stellarator concept is the leading alternative to the tokamak for magnetic confinement fusion. Modern optimized stellarators such as Wendelstein 7-X (W7-X) have reduced neoclassical (collisional) transport to the point that turbulent transport is now the binding constraint on energy confinement [Helander 2014; Beidler et al. 2021]. Predicting and minimizing turbulent transport in candidate reactor configurations is therefore the central computational task of the stellarator design problem. The quantitative basis for this prediction is the gyrokinetic Vlasov–Maxwell system, a five-dimensional reduction of the full kinetic plasma equations in which fast gyromotion has been averaged out [Brizard & Hahm 2007; Abel et al. 2013].

### 1.1 The software landscape

Production gyrokinetic codes fall into three classes by geometric scope:

- **Flux-tube (radially local).** The simulation domain is a thin twisted domain following a single field line on a chosen flux surface. Examples: GS2, stella, CGYRO, GX, and the standard distribution of GENE. This is the workhorse regime for design iteration.
- **Full flux surface.** One closed flux surface is simulated globally; the radial direction remains local. Examples: the global-on-surface variant of GENE due to Merz et al.
- **Global (“full-torus”).** The full plasma volume including radial profile variation is simulated. Captures turbulence spreading, profile stiffness, large-scale zonal flows, and finite-ρ* effects, where ρ* = ρ_i / a is the ratio of ion gyroradius to minor radius. For tokamaks: XGC, ORB5, GENE-global, GYSELA. For stellarators: GENE-3D [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694), EUTERPE [Kornilov et al. 2004](https://doi.org/10.1063/1.1737420), and the stellarator branch of XGC.

For stellarators specifically, the open, fast, modifiable end of the global landscape is empty. The two established global stellarator codes occupy the opposite corner:

- **GENE-3D** [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694) is the production continuum (Eulerian) global code and the de facto reference. It is access-restricted through the GENE consortium at IPP Garching, is not redistributable, and its consortium licensing does not extend to commercial use. Outside parties can often obtain *results* through collaboration, but cannot freely run, inspect, fork, or deploy the code.
- **EUTERPE** [Kornilov et al. 2004; Kleiber et al. 2024] is a global stellarator code, but it is particle-in-cell (PIC) rather than continuum, is CPU-parallel only with no GPU-native path, and is not distributed as open source. Its PIC sampling noise is least favourable in precisely the low-transport, near-marginal regimes where optimized stellarators operate.

GX [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631) is the closest thing to the tool we want — GPU-native, fast, well-validated, open under a permissive licence — but is *flux-tube by design*, a deliberate choice its authors motivate by the reactor-relevant small-ρ* limit. **No open-source, GPU-native, continuum global gyrokinetic stellarator code exists.** That specific cell — open and forkable, GPU-native and fast on small hardware, continuum, global — is the gap Velcro targets.

This gap is not primarily an academic-access problem; the academic stellarator community can and does obtain GENE-3D results through the consortium. It matters for three constituencies the consortium model does not serve:

1. **The private stellarator sector** (Type One Energy, Proxima Fusion, Thea Energy, Renaissance Fusion) cannot rely on consortium-licensed tooling for commercial reactor design, and in several cases is a direct competitor to the institutions that hold the code. For these efforts the absence of an openly licensed global stellarator code is a concrete blocker, not a preference.
1. **Independent verification.** Confinement predictions that inform device-scale investment decisions currently rest, at the global level, on a code that outside parties cannot inspect or reproduce from source. Flux-tube validity at reactor scale is itself an empirical question answerable only by comparison against global simulation, and that comparison should be independently checkable [McMillan et al. 2010; Görler et al. 2011].
1. **Modification and coupling.** Adding a diagnostic, a closure, a new geometry, or coupling to a transport or optimization loop requires the right to change and redistribute the source — which consortium access typically does not grant, and which open licensing does.

### 1.2 Scope, niche, and what this code is not

We describe the technical architecture of a new code, *Velcro*, which combines (a) the global radial scope of GENE-3D, (b) the Laguerre–Hermite velocity discretization of GX, and (c) GPU-native execution following the patterns established by the GENE GPU port and GENE-X.

We commit to a specific niche: **fast iteration for stellarator design and physics interpretation at the single-node to small-cluster scale.** A typical target run is a global W7-X turbulence calculation completing overnight on an 8-GPU node, suitable for inclusion in an optimization loop or for direct comparison against a discharge. Velcro is *not* designed to be a capability-class code at the 10³+ GPU scale that GENE-3D occupies on Marconi-Fusion or LUMI; supporting that regime is a possible M5+ extension but is not the design point. The asymmetric play is to do what GENE-3D does for problem sizes one to two orders of magnitude smaller, an order of magnitude faster per problem, and with full open-source reproducibility. Section 4 discusses the multi-node scaling case explicitly.

Because the velocity-space cost is collisionality-gated (§3.1, §6), we make the target operating regime an explicit design parameter rather than leaving it implicit: the core collisionality band ν* ∈ [0.01, 0.1] characteristic of optimized-stellarator discharges. The velocity-resolution budget (L, M) is to be sized against this band, and the performance claim (§4.6) is conditioned on it. This makes the speed claim falsifiable against a stated regime rather than asserted in the abstract.

Section 2 sets out the governing equations. Section 3 describes the discretization scheme. Section 4 details the GPU and parallelization strategy. Section 5 sets out the validation plan, including how to verify against a closed-source reference code without source access. Section 6 addresses technical risks. Section 7 describes the roadmap and the per-milestone operating mode.

-----

## 2. Governing equations

We restrict the initial scope of the code to the electrostatic, δf gyrokinetic system in 3D non-axisymmetric MHD equilibrium geometry, with adiabatic electrons. Electromagnetic effects, kinetic electrons, and multiple ion species are deferred to subsequent code generations (§7). The reduced scope mirrors the initial release of GENE-3D [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694) and corresponds to the regime in which most ion-temperature-gradient (ITG) and trapped-electron-mode (TEM) physics of interest for stellarator confinement can be captured. (Note that the trapped-electron-mode convergence question of §6 requires kinetic electrons and is therefore tested with GX rather than with Velcro's adiabatic-electron M1 release; see §7.)

### 2.1 Gyrokinetic Vlasov equation

The gyrocenter distribution function F_σ(**X**, v∥, μ, t) for species σ is split into a Maxwellian equilibrium and a fluctuating piece, F_σ = F_{0σ} + F_{1σ}, with F_{1σ}/F_{0σ} ~ δ ≪ 1 in the gyrokinetic ordering [Frieman & Chen 1982](https://doi.org/10.1063/1.863762). The fluctuation obeys

$$
\frac{\partial F_{1\sigma}}{\partial t}
- \mathbf{v}_\parallel \cdot \nabla F_{1\sigma}
- (\mathbf{v}_{\nabla B} + \mathbf{v}_c) \cdot \nabla F_{1\sigma}
- \mathbf{v}_{E1} \cdot \left[ \nabla F_{0\sigma} + \mu \nabla B_0 \frac{F_{0\sigma}}{T_{0\sigma}} \right]
- \dot{v}_\parallel \frac{\partial F_{1\sigma}}{\partial v_\parallel}
= 0
\quad (1)
$$

where **v**∥ = v∥ **b̂**₀, **v**_{∇B} and **v**_c are the grad-B and curvature drifts, and **v**_{E1} = (c/B₀²) **B**₀ × ∇φ̄₁ is the fluctuating E × B drift containing the only nonlinearity retained. The overbar denotes the gyroaverage operator. Definitions follow Maurer et al. [2020] and need not be repeated here. (Corrected 2026-09-21: the $\mu\nabla B_0 F_0/T_0$ term — Eq. (13) of Maurer et al. — cancels the $\nabla B$ part of $\nabla F_0$ taken at fixed $(v_\parallel, \mu)$, so the drive contains the density and temperature gradients only; earlier versions of this document and of the code omitted it, see `physics.drive_gradB`.)

### 2.2 Gyrokinetic Poisson equation

The electrostatic potential φ₁ is determined self-consistently from the quasi-neutrality condition with adiabatic electrons,

$$
\sum_\sigma \frac{q_\sigma^2}{m_\sigma}
\int \frac{F_{0\sigma}}{T_{0\sigma}}
\left[ \phi_1(\mathbf{x}) - \langle\bar{\phi}_1\rangle(\mathbf{x}) \right] dv_\parallel\, d\mu
- \frac{e n_{0e}}{T_{0e}} \left( \phi_1 - \langle \phi_1 \rangle_{\text{FS}} \right)
= \sum_{\sigma \neq e} \frac{q_\sigma}{m_\sigma}
\int \langle F_{1\sigma} \rangle(\mathbf{x})\, dv_\parallel\, d\mu
\quad (2)
$$

where ⟨·⟩_FS denotes a flux-surface average and the double-gyroaverage operator ⟨·̄⟩ is defined in the standard way [Maurer et al. 2020; Abel et al. 2013].

### 2.3 Field-aligned coordinates

The computational coordinate system is the standard field-aligned set (x, y, z), where x = a ρ_tor is the radial coordinate based on toroidal flux, y = σ_{B_p} C_y (q θ* − φ) is the binormal field-line label, and z = σ_{B_p} θ* is the parallel coordinate aligned with the PEST poloidal angle. The MHD equilibrium magnetic field has the contravariant representation

$$
\mathbf{B}_0 = C(x)\, \nabla x \times \nabla y,
\qquad
C(x) = \frac{x B_{\text{axis}}}{|q(x)| C_y}
\quad (3)
$$

Metric coefficients g^{ij} = ∇u^i · ∇u^j and the Jacobian J⁻¹ = (∇x × ∇y) · ∇z are supplied by the equilibrium solver. We use GVEC [Hindenlang et al. 2025](https://doi.org/10.1088/1361-6587/adba11), an open-source variational equilibrium code which represents the flux-surface geometry by Fourier modes [cos(mθ − nφ), sin(mθ − nφ)] in poloidal and toroidal angles and by spline finite elements of arbitrary polynomial degree (typically k ∈ [3, 7]) in the radial direction. GVEC is already in production use as the equilibrium feeder for GENE-3D [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694) and exposes Python bindings suitable for direct integration.

-----

## 3. Discretization

### 3.1 Velocity-space representation: Laguerre–Hermite spectral basis

The central numerical departure of our code from GENE-3D is the velocity-space representation. GENE-3D uses a fixed grid in (v∥, μ) with finite differences. We instead project F_{1σ} onto a Hermite–Laguerre orthogonal basis,

$$
F_{1\sigma}(\mathbf{x}, v_\parallel, \mu, t)
= F_{0\sigma}(\mathbf{x}, v_\parallel, \mu)
\sum_{\ell=0}^{L-1} \sum_{m=0}^{M-1}
\mathcal{N}_{\ell m}\, H_\ell(\hat{v}_\parallel)\, L_m(\hat{\mu})\,
G_{\ell m \sigma}(\mathbf{x}, t)
\quad (4)
$$

where H_ℓ are physicists’ Hermite polynomials in the normalized parallel velocity v̂∥ = v∥ / v_{th,σ}(**x**₀), L_m are Laguerre polynomials in the normalized magnetic moment μ̂ = μ B₀ / T_{0σ}(**x**), 𝒩_{ℓm} is a normalization constant, and G_{ℓmσ} are the spectral coefficients that the code actually evolves [Mandell et al. 2018; Mandell et al. 2024].

The Hermite–Laguerre basis has three properties that motivate this choice:

1. **Interpolation between fluid and kinetic regimes.** The truncated expansion at low (L, M) reduces to the Beer–Hammett gyrofluid hierarchy [Beer & Hammett 1996](https://doi.org/10.1063/1.871538) with Landau-fluid closures, while at (L, M) → ∞ recovers the fully kinetic limit. A single parameter pair tunes computational cost against physical fidelity. Mandell et al. [2024] report convergence to grid-equivalent accuracy at (L, M) ≈ (24, 12) for *flux-tube* nonlinear ITG.
1. **Spectral accuracy in velocity.** Polynomial moments of F_{1σ} are computed exactly to machine precision rather than by quadrature, eliminating velocity-grid aliasing.
1. **Efficient — and load-bearing — collision operator.** The Hermite–Laguerre representation diagonalizes the linearized Landau–Boltzmann operator at leading order, allowing an exact implicit treatment of collisions at low cost. This is not merely a convenience. As the mechanism discussion below makes explicit, collisions are *the* process that arrests the velocity-space cascade and keeps the moment count economical, so the **fidelity of the collision model directly controls the speed advantage in the core**. The model used (Landau, Sugama, Lorentz, or Dougherty) is therefore a first-class validation target, not an implementation detail, and convergence economy must be reported per collision model (§5.7).

The most consequential recent evidence for the basis is the work of Frei et al. [2024], who implemented a Hermite–Laguerre velocity discretization inside GENE-X (a grid-based full-f edge code) and validated it against the existing grid implementation on the TCV-X21 reference case — a strongly non-Maxwellian edge and scrape-off-layer scenario dominated by trapped electron mode turbulence, with adverse conditions for any basis premised on near-Maxwellian distributions. The spectral implementation reproduced the outboard midplane profiles (density, temperature, radial electric field) in excellent agreement with the grid version while requiring significantly lower velocity-space resolution, with a measured speedup of approximately 50× for the TCV-X21 case in that full-f setting. The transferable result for our purposes is not the speedup figure — which is partly a full-f phenomenon, full-f having to resolve the entire distribution, and which will not carry directly to a δf code — but the demonstration that the Hermite–Laguerre basis retains its economy under strongly non-Maxwellian, trapped-electron-mode-dominated conditions, i.e. exactly where a near-Maxwellian basis was expected to fail. A subsequent peer-reviewed study [Frei et al. 2025] confirms the agreement extends to fluctuation spectra, turbulent fluxes, phase shifts, and power crossing the separatrix, and that it is robust under increasing spectral resolution.

Together these establish **representability** — that a near-Maxwellian basis does not simply break down when the distribution is strongly non-Maxwellian. They do not, however, settle the distinct question of **economy** in the low-collisionality core. The edge is collisional; since collisions are the mechanism that keeps the spectrum compact, the edge result lands in the *favorable* regime and does not transfer directly to the collisionless stellarator core. This distinction — representability, settled; core economy, open — is the precise content of Risk 1 (§6).

This has now been shown for δf specifically, not only full-f: δf gyromoment simulations reduce complexity relative to continuum δf gyrokinetics [Mencke et al. 2026], and a δf nonlinear study reaches grid-equivalent accuracy on the Cyclone Base Case with fewer velocity-space degrees of freedom than continuum GENE [Hoffmann et al. 2023b]. Since Velcro is δf and Cyclone is its M2 target, these bear more directly on the design than the full-f edge speedups.

A mechanism underlies this, and naming it sharpens Risk 1. The economy is conditional: it requires the distribution to stay free of arbitrarily fine velocity structure. Collisionless phase mixing drives structure to ever-finer scales without bound — the spectral analogue of filamentation — and the moments needed grow without limit; collisions arrest the cascade and keep the spectrum compact. The linear, collisional balance is quantitative: the collisional cutoff in Hermite index scales as m_c ~ (k∥ v_th / ν)^{2/3}, with the inertial free-energy spectrum decaying as m^{−1/2} [Zocco & Schekochihin 2011; Kanekar et al. 2015; Schekochihin et al. 2016]. This is why the basis survives the edge, which is strongly non-Maxwellian but also collisional. The dependence is documented directly: gyromoment convergence improves as collisions are added [Hoffmann et al. 2023a, 2023b], and in a linear device fine velocity structure proliferates only once collisionality is artificially reduced [Mencke et al. 2026]. The real convergence question is therefore not whether the basis converges in stellarator geometry, but whether stellarator turbulence filaments faster than the ambient collisionality can damp it.

That, however, is only the **Hermite (v∥) half** of the story. The stellarator-specific risk is at least as much in **Laguerre (μ)**: trapped-passing structure and bounce/drift resonances in a 3D |B| well inject velocity-space structure in μ, and nonlinear phase mixing couples Laguerre moments even in the slab limit [Mandell et al. 2018]. The convergence question thus has two axes — Hermite content set by phase-mixing versus collisionality, and Laguerre content set by trapped-particle and magnetic-drift structure — and the stellarator core stresses both. Crucially, **the two axes are not symmetric in their mitigation.** The Hermite cascade is suppressed in strong turbulence by nonlinear ExB advection (stochastic echo / anti-phase-mixing), which steepens the free-energy spectrum from m^{−1/2} toward m^{−3/2} [Schekochihin et al. 2016]; no analogous mechanism acts on the Laguerre direction. The μ axis is therefore the binding, least-mitigated half of the convergence risk, and any mitigation must target it specifically rather than scaling (L, M) uniformly (§6, Risk 1).

We adopt the specific basis normalization and recursion relations given in [Mandell et al. 2018; Mandell et al. 2024]. Implementation is straightforward because the GX project has published reference code for the velocity-space machinery, and Frei et al. [2024] document the additional bookkeeping required for non-equilibrium normalizations.

### 3.2 Spatial discretization

The three spatial directions are discretized as follows:

- **Radial (x): fourth-order centered finite differences.** Grid is non-uniform; we use the same node spacing as the underlying GVEC radial spline grid for compatibility.
- **Binormal (y): Fourier pseudospectral.** Periodic by construction; we evolve a finite set of Fourier modes k_y and use the 2/3 rule for de-aliasing in the nonlinear product.
- **Parallel (z): fourth-order centered finite differences.** Twist-and-shift boundary conditions at z = ±π as in the standard field-aligned construction [Beer et al. 1995; Maurer et al. 2020].

A fourth-order hyperdiffusion operator with second-order stencils is included in each spatial and velocity direction to damp grid-scale modes [Pueschel et al. 2010](https://doi.org/10.1016/j.cpc.2010.04.010).

### 3.3 Time integration

Time advance uses an explicit fourth-order Runge–Kutta scheme. The gyrokinetic field equation (2) is solved at each RK4 substage to obtain the updated φ₁ before computing the right-hand side of (1). Time step is set adaptively by a CFL condition in the binormal Fourier direction and the parallel finite-difference direction.

### 3.4 Nonlinear term: pseudospectral evaluation

The E × B nonlinearity in (1) takes the form {φ̄₁, F_{1σ}}_{xy}, the Poisson bracket in the perpendicular plane. Evaluated naively in real space this is O(N_x N_y) work per grid point. We use the standard Orszag pseudospectral approach [Orszag 1971](https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2): compute ∂_x φ̄₁ and ∂_y F_{1σ} in spectral space (binormal) by multiplication, inverse FFT to real space, multiply pointwise, forward FFT, and apply de-aliasing. Cost per evaluation is O(N_x N_y log N_y) per parallel and velocity node, dominated by the FFTs themselves. On GPU we use cuFFT or rocFFT.

-----

## 4. GPU and parallelization strategy

### 4.1 Implementation language

Velcro is implemented in Python with JAX, following the GX precedent [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631) rather than the Fortran/C++ pattern of GENE-X [Trilaksono et al. 2025](https://doi.org/10.1145/3731599.3768148). This is a consequential choice and warrants explicit justification because the alternative is defensible.

Arguments for the Fortran/C++ path, in honesty: explicit control of memory layout and reduction order at the kernel level; a more mature multi-GPU communication story at the 10³+ GPU scale; first-class AMD GPU support through HIP without depending on a third-party stack’s vendor priorities; direct kernel-level reuse of the GENE-X codebase, which implements many of the operators we need on a closely related problem class.

Arguments for JAX, which we ultimately choose:

1. **Codebase size and legibility.** A directly comparable C++ project at gyselalib++‘s current scope is ~84,000 lines across 426 files [Bourne et al. 2025](https://doi.org/10.21105/joss.08582); the same physics in JAX is expected to land in the 10,000–25,000-line range. The compression is not magic — it reflects pushing the templated coordinate machinery, named-dimension array wrappers, FFTs, tridiagonal solves, autodiff, and device portability into maintained substrates (`jax.numpy`, XLA) rather than writing and owning them. The honest flip side is that this is the same dependency surface we flag as Risk 5: the code we don’t write is also code we don’t control. The benefit we do claim is concrete: a codebase one or two people can hold in their heads in full, where a question like “is this operator consistent with that one” or “what would adding electromagnetic effects touch” can be answered against the whole source rather than a slice. Large consortium codes lose this property structurally, regardless of team skill.
1. **Compiler-managed device portability.** JAX traces Python into XLA’s hardware-independent IR, compiled to CPU, NVIDIA GPU, AMD GPU, or TPU without source changes. The gap against hand-written CUDA is real but on the order of 1.5–2× per kernel and shrinking; the wins we expect from spectral velocity compression and algorithmic choices dominate it, and the gap is carried explicitly in the §4.6 projection.
1. **Differentiability.** `jax.grad` over the whole solver is free. This is the single strongest argument for the choice: sensitivity of heat flux to magnetic geometry for stellarator optimization, parameter inference against experimental data, and gradient-checked conservation laws for verification all come without a separate adjoint code — which for the Fortran/C++ stack would be a multi-person-year project in its own right.
1. **Free improvements from a large user base.** XLA and JAX are developed by Google and a broad ML community; performance improvements, new backends, and bug fixes arrive without our effort. The caveat is the loss of the escape hatch — an XLA regression means waiting for a patch rather than writing our own kernel. We accept this trade and treat its scaling-specific form as Risk 5.

The honest dependency this creates is on JAX’s continued investment in multi-GPU primitives (`shard_map`, `pjit`, distributed compilation) which are less mature than MPI+Kokkos at the 10³+ GPU scale. This is treated explicitly as Risk 5 in §6.

### 4.2 Memory layout and target architecture

The primary data structure is the spectral coefficient array G_{ℓmσ}(x_i, k_y, z_k, t) of shape (N_x, N_{k_y}, N_z, L, M, N_σ), stored in single precision (FP32) as a JAX array. Total memory per snapshot for a representative W7-X global simulation (N_x, N_{k_y}, N_z, L, M) = (256, 128, 64, 24, 12) with one ion species is approximately ~2.5 GB — fits on a single H100 (80 GB) with room for multiple working buffers and the RK4 stages.

For higher resolution or kinetic-electron runs the simulation is distributed across multiple GPUs using JAX’s `shard_map` and explicit device meshes. Domain decomposition is performed in the z (parallel) and x (radial) directions, following the GENE-3D pattern [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694). The binormal k_y direction stays local on each GPU to avoid transposes for the linear physics; transposes are only required for the nonlinear pseudospectral product.

### 4.3 Kernel design

The entire solver is implemented as JIT-compiled JAX functions. Equinox is used for the State pytree and pydantic for input validation. The principal kernels are:

- **K1. Spatial stencil application.** Fourth-order finite differences in x and z, applied across all (L, M, k_y, σ) indices via `jax.numpy` array operations. Memory-bandwidth-bound; XLA’s automatic fusion of pointwise operations should reach a substantial fraction of HBM bandwidth peak, comparable to the ~90% of bandwidth peak reported for hand-written CUDA stencils on V100 by [Germaschewski et al. 2021](https://doi.org/10.1063/5.0046327) with a 1.5–2× gap acceptable.
- **K2. Hermite–Laguerre recursion.** Three-term recurrences along the ℓ and m indices, expressed as `jax.lax.scan` and JIT-compiled. Compute-bound but small.
- **K3. Nonlinear product.** `jax.numpy.fft.rfftn` / `irfftn` calls plus a pointwise multiplication. JAX dispatches to cuFFT on NVIDIA and rocFFT on AMD without source changes. Cost dominated by the 2D FFT pair per (z, ℓ, m) slice.
- **K4. Field solve.** The gyrokinetic Poisson equation (2) reduces, after Fourier transform in y, to a set of one-dimensional elliptic problems in x per (k_y, z), with a tridiagonal-plus-correction structure from the polarization integral. Solved by `jax.lax.linalg.tridiagonal_solve` plus an explicit Sherman–Morrison correction.

### 4.4 Communication, scaling, and the design point

Multi-GPU communication is handled by JAX’s collective operations (`jax.lax.psum`, `jax.lax.all_to_all`), which dispatch to NCCL on NVIDIA and RCCL on AMD. The dominant communication cost is the all-to-all transpose required for the pseudospectral nonlinear product when the radial direction is distributed. We mitigate by:

- Choosing decomposition to keep k_y local always, so the transpose is only over x;
- Using JAX’s compiler-managed overlap of communication with parallel-direction stencil application;
- Topology-aware device meshes so transposes use NVLink within a node before crossing InfiniBand.

The GENE GPU port [Germaschewski et al. 2021](https://doi.org/10.1063/5.0046327) reports that inter-node communication becomes the binding constraint above ~10³ GPUs; the GENE-X port [Trilaksono et al. 2025](https://doi.org/10.1145/3731599.3768148) confirms the same pattern. JAX’s distribution primitives at this scale are less mature than MPI+Kokkos. This is the design point question we must take seriously rather than wave away. Velcro’s stated niche (§1.2) is the regime below this wall: single node up to one or two cabinets, where intra-node NVLink and short-hop InfiniBand keep the transpose cost manageable, and where JAX’s current scaling story is well-validated by the ML community. We do not claim a capability-class scaling story competing with GENE-3D on Marconi-Fusion or LUMI. Pushing past 10³ GPUs is a possible M5+ extension and is treated as Risk 5 in §6 rather than as a baseline target.

### 4.5 Mixed precision

Stencil application and the nonlinear product run in FP32. Time-step accumulators, the field solve, and reduction operations for global diagnostics run in FP64. JAX handles mixed precision through explicit dtype control on arrays. Maurer et al. [2020] performed a controlled-precision study showing that turbulence statistics (heat flux time averages, frequency spectra) are preserved under reduced precision for the Vlasov advance, with deviations limited to instantaneous field-level details. Expected speedup from FP32 stencils is 1.5–2× on top of the GPU baseline.

### 4.6 Expected performance

Combining the JAX/GPU factor (we conservatively assume parity with the ~15× CPU-to-GPU speedup reported in [Germaschewski et al. 2021](https://doi.org/10.1063/5.0046327) less a 1.5–2× XLA-vs-hand-written-CUDA gap, netting ~8–10× over CPU), the FP32 mixed-precision factor (1.5×), and the Laguerre–Hermite resolution savings vs grid-based velocity representation (~4× at equivalent ITG fidelity in the δf flux-tube setting per [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631); larger savings are reported in the full-f edge setting [Frei et al. 2024](https://arxiv.org/abs/2411.09232) but do not carry directly to δf and are not banked here), we project useful global stellarator turbulence simulations on the order of **hours on a single 8-GPU node** for problem sizes that require multi-day runs on hundreds of CPU nodes with GENE-3D today. This is the central performance claim. The XLA gap is included in the projection rather than ignored, and the speedup stack rests on the δf ~4× base rather than the full-f figure.

The δf ~4× velocity-compression factor is, however, the **regime-sensitive term** in this stack, and the analysis behind Risk 1 lets us say where it breaks. The speed factor relative to a velocity grid is (grid DOF)/(L·M). The banked ~4× corresponds to the smooth-ITG operating point — roughly (L, M) = (24, 5) ≈ 120 spectral coefficients against a 32 × 16 = 512-point velocity grid (≈4.3×). Grid parity (1×) is reached at L·M = 512, i.e. by (L, M) = (24, 21) or (48, 11). Moderate growth in *either* moment axis therefore erases the speed advantage, and the low-collisionality, trapped-particle-driven core plausibly pushes both — the Laguerre axis without the nonlinear-echo rescue that protects the Hermite axis (§3.1, §6). The GPU and FP32 factors are regime-independent; the velocity-compression factor is the one to verify against the worst-case regime before banking it, which is the purpose of the M0 convergence pre-test (§7).

-----

## 5. Validation

A new gyrokinetic code is not scientifically usable until it has been validated against established results. Our validation strategy proceeds in overlapping phases, each independent and each providing falsification opportunities.

### 5.1 Phase 1: Analytic limits

- **Landau damping.** Linearized 1D1V electrostatic problem in a uniform background. Damping rates must match the analytic Landau formula [Landau 1946](https://doi.org/10.1016/B978-0-08-010586-4.50066-3) to within numerical tolerance.
- **Rosenbluth–Hinton zonal-flow residual.** The long-time residual of an initial zonal-flow perturbation is given analytically [Rosenbluth & Hinton 1998](https://doi.org/10.1103/PhysRevLett.80.724); we reproduce it in the appropriate limit.
- **Linear ITG dispersion in slab geometry.** Standard toroidal-ITG dispersion relations with adiabatic electrons.

### 5.2 Phase 2: Tokamak (axisymmetric) benchmarks

Setting the number of toroidal field periods to one and feeding a circular-tokamak equilibrium reduces our code to a global tokamak gyrokinetic solver. We benchmark against the Cyclone Base Case [Dimits et al. 2000](https://doi.org/10.1063/1.873896), the canonical inter-code reference. Both linear growth rates and nonlinear saturated heat flux must agree with the published Cyclone results across the full set of participating codes.

### 5.3 Phase 3: GENE-3D test-case reproduction

The Maurer et al. [2020] paper, Section 5, publishes specific stellarator test cases including W7-X high-mirror configuration with documented input profiles, resolution settings, and reported linear growth rates and nonlinear heat fluxes. We reproduce these cases verbatim. Matching the reported numbers within the reported error bars is the strongest code-to-code test of correctness obtainable without source access to the reference code. We note its limit: it tests agreement with GENE-3D’s *published configuration*, so a bug exercised by neither code’s published cases would not surface here. Phase 4 addresses this by widening the set of codes and configurations.

### 5.4 Phase 4: Community intercomparison

The EU fusion modeling community runs periodic stellarator turbulence intercomparison exercises [Bañón Navarro et al. 2023; Wilms et al. 2024] in which GENE-3D, GX, EUTERPE, and stella produce numbers for the same configurations using their respective approximations. Velcro is run on the same configurations and the numbers are published alongside. Sitting inside the existing inter-code spread is strong evidence of physical correctness, and because the participating codes use independent numerical schemes, it is a more searching test than agreement with any single reference.

### 5.5 Phase 5: W7-X experimental validation

The most stringent test is direct comparison with experimental observations from W7-X campaigns: ion temperature profiles, energy confinement times, fluctuation spectra from Doppler reflectometry, heat flux deposition patterns. Published datasets from the OP1.2 and OP2 campaigns [Beidler et al. 2021; Bañón Navarro et al. 2023] provide the reference. No gyrokinetic code achieves absolute agreement with experiment on the first attempt, and we do not expect to. Two distinct things are being checked here, and we keep them separate: first, whether Velcro’s *residual* against the data matches the residual GENE-3D reports against the same data — agreement on residuals is evidence the two codes are doing the same physics, which is code-to-code validation, not experimental validation; and second, the open physics question of how far either code’s residual can be closed by the effects deferred to later milestones (kinetic electrons, electromagnetic fluctuations). The first is a correctness check on Velcro; the second is a research result that an independent open code is well-placed to pursue precisely because it can be freely modified and rerun.

### 5.6 Working around source-access constraints

The standard objection is that we cannot truly validate against a closed-source reference. This objection misses how code intercomparison actually works in computational physics. Reference codes are validated by their published outputs, not by their source. Nobody reads VMEC’s source to check stellarator equilibria; they check that VMEC’s flux surfaces match measured ones from real devices. The same principle applies here: GENE-3D is validated by what it predicts, and Velcro is validated against what it predicts in turn — with Phase 4 ensuring the comparison is not hostage to a single reference.

Where source-level disagreement is suspected, the standard remedy is to engage a collaborator at an institution with GENE-3D access to run parallel cases and share *output data* (not source). This is permitted under all gyrokinetic-code licensing agreements we are aware of and is normal practice in the field.

### 5.7 Velocity-space convergence economy (standing)

Because the speed claim is collisionality-gated and the Laguerre axis is unmitigated (§3.1, §4.6, §6), velocity-space convergence economy is promoted from a one-off check to a standing validation deliverable rather than left implicit. Once the code exists, a self-convergence study maps the minimum converged (L, M) — for both linear growth rate and nonlinear heat flux — across the target collisionality band (§1.2), **reported per collision model** (Landau / Sugama / Lorentz / Dougherty), since the collision model controls the economy. As a robustness diagnostic the nonlinear Hermite free-energy spectrum is recorded: a slope steepening from m^{−1/2} toward m^{−3/2} confirms that nonlinear echo is suppressing the parallel-velocity cascade [Schekochihin et al. 2016], converting the “Hermite axis is rescued in turbulence” expectation from assumption to measurement. This phase begins as the pre-code M0 scan (§7) and continues as a maintained artifact thereafter.

-----

## 6. Technical risks

We address the principal risks honestly:

**Risk 1: Hermite–Laguerre convergence economy at finite ρ* and in global stellarator geometry.** The convergence properties of the basis are demonstrated for flux-tube ITG and TEM [Mandell et al. 2018; Mandell et al. 2024] and, more relevantly, for global full-f edge and scrape-off-layer turbulence on the TCV-X21 case, where the basis retained its economy under strongly non-Maxwellian conditions and agreement with grid results proved robust under increasing spectral resolution [Frei et al. 2024; Frei et al. 2025]. That last result retires the *existential* form of the risk — the worry that a near-Maxwellian basis would simply fail off-Maxwellian is now contradicted by direct evidence. What remains is narrower and non-existential, and it has two distinct axes that are not symmetric.

*The Hermite (v∥) axis* is adverse in the linear limit but expected to self-heal in turbulence. The linear collisional cutoff scales as m_c ~ (k∥ v_th / ν)^{2/3} (a scaling we have reproduced numerically against the published exponent to ~1.5%), which for the W7-X core (ν* = 0.05–0.01) gives a Hermite count of L ≈ 45–135 — roughly 2–5× the tokamak-ITG value of L ≈ 24, before any turbulence reduction. The mitigation is the nonlinear echo / anti-phase-mixing effect, which steepens the spectrum from m^{−1/2} toward m^{−3/2} and so concentrates the free energy at low moments [Schekochihin et al. 2016], plausibly returning the effective L toward the ITG value. This rescue is an expectation, not a result, for stellarator near-marginal turbulence; it is converted to a measurement by the spectrum diagnostic of §5.7.

*The Laguerre (μ) axis* is the binding, least-mitigated half. Trapped-passing structure and magnetic-drift resonances broaden the gyromoment spectrum in μ, and the gyromoment literature documents that the required moment count rises as collisionality falls, as trapped-particle drive turns on, and as magnetic drifts dominate [Frei et al., flux-tube GM; Morren et al. 2024 — for core TEM the truncation order needed was “considerable”]. All three conditions are maximal in the optimized-stellarator core, and no echo-type mechanism reduces the μ cascade. The performance consequence is concrete (§4.6): the banked ~4× corresponds to (L, M) ≈ (24, 5) ≈ 120 against a 512-point grid, and grid parity is reached at L·M = 512 — by (24, 21) or (48, 11) — so even moderate growth in μ erases the speed advantage.

The remaining untested cell is therefore narrow and specific: *nonlinear, low-collisionality, TEM-or-mixed, global* stellarator gyromoment economy. Everything around it is mapped — stellarator ITG flux-tube (favorable, GX), tokamak edge full-f (favorable, Frei), core TEM linear (adverse, Frei/Morren). What is unpublished is the nonlinear global low-ν case Velcro targets.

*Mitigation.* The failure mode is graceful: higher (L, M) costs speed, not correctness, so the exposure is to the performance claim (§4.6), not to code validity. The basis is parametrically tunable, so (L, M) can be raised adaptively. Because the μ axis is unrescued, the mitigation is named specifically rather than left as “raise (L, M)”: adaptive Laguerre refinement, a trapped-passing-aware closure, or a hybrid representation in which μ uses an enriched or grid representation while v∥ stays Hermite-spectral. Committing to one of these is premature before the M0 number, but the lever is identified. The whole question is directly testable at M0 (flux-tube, in GX) and M2/M4, with a measurable collisionality crossover.

**Risk 2: Field-solve scaling in global geometry.** Maurer et al. [2020] note that the gyrokinetic Poisson solve becomes communication-heavy at high resolution because the radial direction cannot be decoupled. Our chosen decomposition (radial distributed, binormal local) inherits this. Mitigation is the same as in GENE-3D: solve in batches across k_y to amortize communication latency.

**Risk 3: Gyroaverage operator in stellarator geometry.** The gyroaverage in field-aligned coordinates is non-trivial when the flux-surface geometry varies along the field line, as it does in stellarators. GENE-3D uses a real-space gyroaverage with precomputed weights; GX uses a spectral gyroaverage that exploits the flux-tube geometry. Neither is directly transferable to our setting. We will implement the real-space approach first and assess whether a spectral variant is feasible. Related work on portable gyroaverage operators in the gyselalib++ project [Bourne et al. 2025](https://doi.org/10.21105/joss.08582) is relevant to the implementation pattern even though the underlying numerical scheme differs.

**Risk 4: Validation collaboration access.** Phase 4 (community intercomparison) depends on the willingness of existing groups to share output data and accept a new entrant into intercomparison exercises. This is normally available to academic collaborators in good faith but is not contractually guaranteed. Mitigation: phases 1–3 and 5 are self-contained and provide adequate validation independently.

**Risk 5: Scaling beyond the design point.** Velcro’s stated niche is the regime below ~10³ GPUs (§1.2, §4.4), where intra-node and short-hop communication dominate, transposes can be kept tractable, and JAX’s `shard_map`/`pjit` distribution primitives are well-validated by ML practice. Pushing past this scale — required if Velcro is to compete with GENE-3D on its native capability-class runs rather than complement it at smaller scale — requires solving two compounding problems: the intrinsic inter-node all-to-all bottleneck that has bound every grid-based continuum gyrokinetic code at this scale, and the relative immaturity of JAX’s distribution primitives at the 10³+ GPU scale compared to MPI+Kokkos. We do not promise a solution as part of M0–M5. If the niche shifts and Velcro is asked to scale beyond its design point, the architecture would need rework at the communication layer, and we would be depending on the JAX team’s continued investment in large-scale distributed compilation. This is the most consequential risk to the long-term competitive positioning of the code, distinct from its short-term correctness.

-----

## 7. Roadmap and operating mode

|Milestone|Scope                                                                                                 |Target month|
|---------|------------------------------------------------------------------------------------------------------|------------|
|M0       |Repository, build system, GVEC integration, primitive operators with unit tests; **velocity-economy pre-test (flux-tube, GX/GYACOMO) — see below**|0–3         |
|M1       |Linear Vlasov advance, field solve, FP64 CPU baseline, Landau / Rosenbluth–Hinton / linear ITG passing|3–6         |
|M2       |Nonlinear advance, Cyclone Base Case match                                                            |6–9         |
|M3       |GPU port, FP32 mixed precision                                                                        |9–12        |
|M4       |GENE-3D W7-X test-case reproduction                                                                   |12–15       |
|M5       |Multi-GPU scaling within design point, full release                                                   |15–18       |
|M6       |Kinetic electrons                                                                                     |18–24       |
|M7       |Electromagnetic fluctuations                                                                          |24–30       |

*Indicative milestones. M0–M5 constitute the minimum viable release.*

> **Status note (2026-10-02).** This document is the original plan and is kept as written. Since then:
> - M6 (kinetic electrons) and part of M7 (A∥, and δB∥ validated linearly) were done ahead of M4–M5, in the flux-tube
>   configuration. They are validated against GX: linear, Rosenbluth–Hinton / GAM, β scan, nonlinear ITG and TEM fluxes.
> - The README summarises the gates.
> - Still open from this roadmap: M4 (GENE-3D W7-X), M5 (multi-GPU), nonlinear electromagnetic runs at higher β, and
>   kinetic electrons in 3D (VMEC) geometry.

**The Risk 1 pre-test is pulled forward to M0.** The decisive number for the velocity-space economy is obtainable in flux-tube before any Velcro code is written, and it splits into two tiers:

- **(a) Mechanism scan, CPU, no GPU required.** The collisionality→moment-economy mechanism is largely geometry-independent and can be tested in GYACOMO (open, Fortran/MPI, the same Hermite–Laguerre basis, with kinetic electrons and advanced collision operators) on a Cyclone-Base-Case-with-kinetic-electrons TEM case scanned over collisionality. This is the cheapest path to the dominant effect. Note that GYACOMO is tokamak/Z-pinch/Miller only (no VMEC, no global), so it cannot supply the stellarator-geometry increment.
- **(b) Geometry-specific scan, GPU.** A (L, M) convergence scan for a W7-X TEM case at realistic core collisionality, run in GX (open, GPU-native, already does stellarator flux-tube). This adds the stellarator-geometry increment on top of (a).

Both scans output the nonlinear Hermite free-energy spectrum (§5.7) so the Hermite-axis echo rescue is measured rather than assumed. If the required (L, M) stays economical, the narrow remaining form of Risk 1 is largely pre-retired; if it climbs toward grid parity, the §4.6 speed claim is revised before the GPU and multi-GPU investment of M3–M5 — the cheapest possible point to learn it. M2 remains the first *in-Velcro* test of Risk 1; the M0 pre-test front-runs it in flux-tube using tools that already exist.

### 7.1 Operating mode

Each milestone is delivered not as a code drop but as a runnable, narrated notebook that walks through the new functionality from primitive operators to the validation case the milestone is named after, with the test suite passing inline. M0 is the operator-correctness notebook (orthonormality, finite-difference convergence orders, FFT round-trip, field-solve verification) plus the velocity-economy pre-test result. M1 adds the linear physics tests. M2 ends with a notebook that reproduces a Cyclone heat-flux curve from a fresh repository checkout. The notebook is documentation, validation evidence, and onboarding material in one artifact, and it is the medium in which design decisions are recorded.

This is a deliberate operating choice. Conventional gyrokinetic codes accumulate code at a rate set by their team size and their funding cycle, and their documentation is a separate artifact that lags the source by months or years. The notebook-per-milestone discipline keeps the project’s surface area small enough that one or two people can hold the whole of it — code, validation evidence, and design rationale — at once, and keeps documentation pinned to runnable code rather than drifting from it. It also produces an artifact at each milestone that an outside reader can audit in an afternoon. We treat this small, auditable surface area as a structural advantage to be defended, not an artifact of being early.

-----

## 8. Conclusion

The open-source global gyrokinetic stellarator code described here fills an identifiable structural gap in the fusion simulation ecosystem: an openly licensed, GPU-native, continuum global code, where the established global stellarator codes are either consortium-restricted and continuum (GENE-3D) or open-question-licensed, CPU-only and particle-in-cell (EUTERPE), and the open GPU-native codes are flux-tube by design (GX). The technical bet is well-bounded: the velocity-space scheme has been proven in flux-tube geometry [Mandell et al. 2018, 2024] and, decisively *for representability*, in the non-Maxwellian edge regime where the basis was previously thought to be weak [Frei et al. 2024, 2025]; the JAX/GPU implementation pattern has been proven for gyrokinetics by GX [Mandell et al. 2024](https://doi.org/10.1017/S0022377824000631) and the GPU performance envelope is well-characterized by parallel Fortran/C++ efforts [Germaschewski et al. 2021; Trilaksono et al. 2025]; and the global geometric framework has been proven in closed-source codes [Maurer et al. 2020](https://doi.org/10.1016/j.jcp.2020.109694). The novelty lies in combining all three, which has not been done.

The validation pathway against a closed reference code is standard practice in computational physics and does not require source access. The principal remaining technical risk is the convergence *economy* of the Hermite–Laguerre basis at finite ρ* in the low-collisionality stellarator core — a narrower and non-existential question after Frei et al., with two asymmetric axes: a Hermite (v∥) cost that is adverse linearly but plausibly rescued by nonlinear echo, and a Laguerre (μ) cost from trapped-particle structure that has no such rescue and is therefore the binding term. The question is empirically testable in flux-tube as soon as the M0 milestone — before any Velcro code is written — and its failure mode degrades the speed advantage gracefully rather than breaking the code. The principal remaining strategic risk is scaling beyond Velcro’s stated design point of ~10³ GPUs, which is acknowledged rather than promised.

-----

## References

- Abel, I. G., Plunk, G. G., Wang, E., Barnes, M., Cowley, S. C., Dorland, W., & Schekochihin, A. A. (2013). *Multiscale gyrokinetics for rotating tokamak plasmas: fluctuations, transport and energy flows.* Rep. Prog. Phys. 76, 116201. <https://doi.org/10.1088/0034-4885/76/11/116201>
- Bañón Navarro, A., Di Siena, A., Velasco, J. L., Wilms, F., Merlo, G., Görler, T., García-Regaña, J. M., Sanchez, E., & Jenko, F. (2023). *First-principles based plasma profile predictions for optimized stellarators.* Nucl. Fusion 63, 054003. <https://doi.org/10.1088/1741-4326/acc3af>
- Barnes, M., Parra, F. I., & Landreman, M. (2019). *stella: An operator-split, implicit-explicit δf-gyrokinetic code for general magnetic field configurations.* J. Comput. Phys. 391, 365. <https://doi.org/10.1016/j.jcp.2019.01.025>
- Beer, M. A., Cowley, S. C., & Hammett, G. W. (1995). *Field-aligned coordinates for nonlinear simulations of tokamak turbulence.* Phys. Plasmas 2, 2687. <https://doi.org/10.1063/1.871232>
- Beer, M. A., & Hammett, G. W. (1996). *Toroidal gyrofluid equations for simulations of tokamak turbulence.* Phys. Plasmas 3, 4046. <https://doi.org/10.1063/1.871538>
- Beidler, C. D., Smith, H. M., Alonso, A., et al. (2021). *Demonstration of reduced neoclassical energy transport in Wendelstein 7-X.* Nature 596, 221. <https://doi.org/10.1038/s41586-021-03687-w>
- Bourne, E., Grandgirard, V., Asahi, Y., Bigot, J., Donnel, P., et al. (2025). *Gyselalib++: A Portable C++ Library for Semi-Lagrangian Kinetic and Gyrokinetic Simulations.* J. Open Source Softw. 10(113), 8582. <https://doi.org/10.21105/joss.08582>
- Brizard, A. J., & Hahm, T. S. (2007). *Foundations of nonlinear gyrokinetic theory.* Rev. Mod. Phys. 79, 421. <https://doi.org/10.1103/RevModPhys.79.421>
- Dimits, A. M., Bateman, G., Beer, M. A., et al. (2000). *Comparisons and physics basis of tokamak transport models and turbulence simulations.* Phys. Plasmas 7, 969. <https://doi.org/10.1063/1.873896>
- Dorland, W., Jenko, F., Kotschenreuther, M., & Rogers, B. N. (2000). *Electron temperature gradient turbulence.* Phys. Rev. Lett. 85, 5579. <https://doi.org/10.1103/PhysRevLett.85.5579>
- Frei, B. J., Ulbl, P., Trilaksono, J., & Jenko, F. (2024). *Spectrally Accelerated Edge and Scrape-Off Layer Gyrokinetic Turbulence Simulations.* arXiv:2411.09232. <https://arxiv.org/abs/2411.09232>
- Frei, B. J., Ulbl, P., Pitzal, C., Zholobenko, W., & Jenko, F. (2025). *Turbulence and transport in spectrally accelerated full-f gyrokinetic simulations.* Nucl. Fusion 65, 116026. <https://doi.org/10.1088/1741-4326/ae114a>
- Frieman, E. A., & Chen, L. (1982). *Nonlinear gyrokinetic equations for low-frequency electromagnetic waves in general plasma equilibria.* Phys. Fluids 25, 502. <https://doi.org/10.1063/1.863762>
- Germaschewski, K., Allen, B., Dannert, T., et al. (2021). *Toward exascale whole-device modeling of fusion devices: Porting the GENE gyrokinetic microturbulence code to GPU.* Phys. Plasmas 28, 062501. <https://doi.org/10.1063/5.0046327>
- Görler, T., Lapillonne, X., Brunner, S., Dannert, T., Jenko, F., Merz, F., & Told, D. (2011). *The global version of the gyrokinetic turbulence code GENE.* J. Comput. Phys. 230, 7053. <https://doi.org/10.1016/j.jcp.2011.05.034>
- Helander, P. (2014). *Theory of plasma confinement in non-axisymmetric magnetic fields.* Rep. Prog. Phys. 77, 087001. <https://doi.org/10.1088/0034-4885/77/8/087001>
- Hindenlang, F., Plunk, G. G., & Maj, O. (2025). *GVEC: a galerkin variational equilibrium code for 3D MHD equilibria.* Plasma Phys. Control. Fusion 67, 045002. <https://doi.org/10.1088/1361-6587/adba11>
- Hoffmann, A. C. D., Frei, B. J., & Ricci, P. (2023a). *Gyrokinetic simulations of plasma turbulence in a Z-pinch using a moment-based approach and advanced collision operators.* J. Plasma Phys. 89(2), 905890214. <https://doi.org/10.1017/S0022377823000284>
- Hoffmann, A. C. D., Frei, B. J., & Ricci, P. (2023b). *Gyrokinetic moment-based simulations of the Dimits shift.* J. Plasma Phys. 89(6), 905890611. <https://doi.org/10.1017/S0022377823001320>  *(This is also the reference for the GYACOMO code used in the M0 mechanism pre-test, §7.)*
- Kanekar, A., Schekochihin, A. A., Dorland, W., & Loureiro, N. F. (2015). *Fluctuation–dissipation relations for a plasma-kinetic Langevin equation.* J. Plasma Phys. 81(1), 305810104. <https://doi.org/10.1017/S0022377814000622>
- Kleiber, R., Borchardt, M., Hatzky, R., Könies, A., Leyh, H., Mishchenko, A., Riemann, J., Slaby, C., García-Regaña, J. M., Sánchez, E., & Cole, M. (2024). *EUTERPE: A global gyrokinetic code for stellarator geometry.* Comput. Phys. Commun. 295, 109013. <https://doi.org/10.1016/j.cpc.2023.109013>
- Kornilov, V., Kleiber, R., Hatzky, R., Villard, L., & Jost, G. (2004). *Gyrokinetic global three-dimensional simulations of linear ion-temperature-gradient modes in Wendelstein 7-X.* Phys. Plasmas 11, 3196. <https://doi.org/10.1063/1.1737420>
- Landau, L. D. (1946). *On the vibrations of the electronic plasma.* J. Phys. U.S.S.R. 10, 25. <https://doi.org/10.1016/B978-0-08-010586-4.50066-3>
- Mandell, N. R., Dorland, W., & Landreman, M. (2018). *Laguerre–Hermite pseudo-spectral velocity formulation of gyrokinetics.* J. Plasma Phys. 84, 905840108. <https://doi.org/10.1017/S0022377818000041>
- Mandell, N. R., Dorland, W., Abel, I., Gaur, R., Kim, P., Martin, M., & Qian, T. (2024). *GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design.* J. Plasma Phys. 90, 905900402. <https://doi.org/10.1017/S0022377824000631>
- Maurer, M., Bañón Navarro, A., Dannert, T., Restelli, M., Hindenlang, F., Görler, T., Told, D., Jarema, D., Merlo, G., & Jenko, F. (2020). *GENE-3D: A global gyrokinetic turbulence code for stellarators.* J. Comput. Phys. 420, 109694. <https://doi.org/10.1016/j.jcp.2020.109694>
- McMillan, B. F., Lapillonne, X., Brunner, S., Villard, L., Jolliet, S., Bottino, A., Görler, T., & Jenko, F. (2010). *System size effects on gyrokinetic turbulence.* Phys. Rev. Lett. 105, 155001. <https://doi.org/10.1103/PhysRevLett.105.155001>
- Mencke, J. E., Ricci, P., & Da Silva, L. (2026). *Comprehensive full-f drift-kinetic and δf gyrokinetic simulations of a linear plasma device based on the gyro-moment approach.* Preprint, arXiv:2603.13123. <https://arxiv.org/abs/2603.13123>
- Morren, et al. (2024). *Gyrokinetic moment-based study of trapped-electron-mode convergence.* Phys. Plasmas (2024). *[full author list / volume / article number to confirm]*
- Orszag, S. A. (1971). *On the elimination of aliasing in finite-difference schemes by filtering high-wavenumber components.* J. Atmos. Sci. 28, 1074. <https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2>
- Pueschel, M. J., Dannert, T., & Jenko, F. (2010). *On the role of numerical dissipation in gyrokinetic Vlasov simulations of plasma microturbulence.* Comput. Phys. Commun. 181, 1428. <https://doi.org/10.1016/j.cpc.2010.04.010>
- Rosenbluth, M. N., & Hinton, F. L. (1998). *Poloidal flow driven by ion-temperature-gradient turbulence in tokamaks.* Phys. Rev. Lett. 80, 724. <https://doi.org/10.1103/PhysRevLett.80.724>
- Schekochihin, A. A., Parker, J. T., Highcock, E. G., Dellar, P. J., Dorland, W., & Hammett, G. W. (2016). *Phase mixing versus nonlinear advection in drift-kinetic plasma turbulence.* J. Plasma Phys. 82(2), 905820212. <https://doi.org/10.1017/S0022377816000374>
- Trilaksono, J., Ulbl, P., Williams, J. J., Pfeiler, C.-M., Finkbeiner, M., Dannert, T., Laure, E., Markidis, S., & Jenko, F. (2025). *OpenACC and OpenMP-Accelerated Fortran/C++ Gyrokinetic Fusion Code GENE-X for Heterogeneous Architectures.* Proceedings of the Platform for Advanced Scientific Computing Conference (PASC ’25), ACM. <https://doi.org/10.1145/3731599.3768148>
- Wilms, F., Bañón Navarro, A., Windisch, T., Bozhenkov, S., Warmer, F., Fuchert, G., Ford, O., Zhang, D., Stange, T., & Jenko, F. (2024). *Global gyrokinetic analysis of Wendelstein 7-X discharge: unveiling the importance of trapped-electron-mode and electron-temperature-gradient turbulence.* Nucl. Fusion 64, 096040.
- Zocco, A., & Schekochihin, A. A. (2011). *Reduced fluid-kinetic equations for low-frequency dynamics, magnetic reconnection, and electron heating in low-beta plasmas.* Phys. Plasmas 18, 102309. <https://doi.org/10.1063/1.3628639>

-----

## Revision note (v3.1 → v3.2)

Integrated changes, by section:

- **Abstract / §8** — softened the Frei edge result from "decisive for the convergence question" to "decisive for representability"; stated core economy as the one quantitative open question.
- **§1.2** — added target core collisionality band ν* ∈ [0.01, 0.1] as an explicit design parameter against which (L, M) and the speed claim are sized.
- **§3.1** — recast the collision-operator property from "negligible cost" to load-bearing (its model fidelity controls core economy); split the Frei reassurance into representability (settled) vs core economy (open); added the Laguerre/μ axis and the Hermite/Laguerre rescue asymmetry; added the m_c ~ (k∥v_th/ν)^{2/3} and m^{−1/2} scalings with citations.
- **§4.6** — flagged the ~4× as the regime-sensitive term and quantified grid parity at L·M = 512 (e.g. (24,21), (48,11)).
- **§5.7 (new)** — velocity-space convergence economy as a standing validation phase, reported per collision model, with the Hermite-spectrum echo diagnostic.
- **§6 Risk 1** — rewritten into two asymmetric axes (Hermite adverse-but-rescued; Laguerre binding-and-unmitigated), with validated quantitative bounds, the narrow remaining untested cell, a named μ-direction mitigation lever, and graceful failure retained.
- **§7** — pulled the Risk 1 pre-test forward to M0 as a two-tier scan (GYACOMO CPU mechanism + GX W7-X geometry increment) with the spectrum diagnostic; updated the roadmap table and §7.1.
- **References** — added Schekochihin et al. 2016, Zocco & Schekochihin 2011, Kanekar et al. 2015, Morren et al. 2024; noted Hoffmann et al. 2023b as the GYACOMO reference.
- **Equations (1)–(4) and §3.4** — repaired LaTeX subscript corruption from the v3.1 source so the math renders; no physics changed.

**To confirm:** the Morren et al. 2024 entry needs its full author list, volume, and article number completed.
