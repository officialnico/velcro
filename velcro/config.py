"""Input validation (pydantic) for a Velcro run."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class GridConfig(BaseModel):
    Nx: int = Field(64, ge=4, description="radial points (uniform, periodic FD with Krook buffers)")
    Lx: float = Field(100.0, gt=0, description="radial box length in ρ_ref")
    Ny: int = Field(48, ge=6, description="real-space binormal points; Nky = (Ny-1)//3 + 1 modes kept")
    Nz: int = Field(32, ge=4, description="parallel points on z ∈ [-π, π)")
    L: int = Field(16, ge=2, description="Hermite moments (v∥)")
    M: int = Field(6, ge=1, description="Laguerre moments (μ)")
    x_offset: float = Field(0.0, ge=0.0, lt=1.0, description="radial grid shifted by this fraction of Δx (s-α and gx_file only).  "
                            "With q₀n₀ an integer, x = 0 is a rational surface of every k_y, and so are further grid points "
                            "(θ_B = 2π ŝ k_y x): there the field line closes, the k_z = 0 mode has k∥ = 0, and kinetic "
                            "electrons in that cell do not stream — a whole cell (≫ the physical electron layer) of "
                            "non-adiabatic electrons, which drives a spurious fast electron-direction mode.  With 0.5 a grid point "
                            "lies on a rational surface of k_y index n only if 2^k | n, 2^k the power of 2 in "
                            "2N_x/gcd(2N_x, j_twist) (32 for N_x 96, j_twist 10)")


class GeometryConfig(BaseModel):
    kind: Literal["slab", "s_alpha", "fourier", "gx_file", "miller", "vmec"] = "s_alpha"
    rho_star: float = Field(1.0 / 180.0, gt=0, description="ρ_ref / L_ref")
    # slab
    Lz_over_Lref: float = 1.0
    # s-alpha
    r0: float = Field(0.18, description="reference minor radius / R0")
    q0: float = 1.4
    shat: float = 0.8
    alpha_mhd: float = 0.0
    gpar: Literal["velcro", "constant"] = Field(
        "velcro", description="s-α parallel gradient b·∇θ: 'velcro' = 1/(qR₀(1 + r cos θ)) (the default, validated with "
                              "adiabatic electrons); 'constant' = 1/(qR₀), GX's s-α model (only b·∇θ and the Jacobian change)")
    n0: int = Field(1, ge=1, description="minimum toroidal mode number: L_y = 2π C_y / n0")
    q_poly: Optional[list[float]] = Field(None, description="polynomial q(r/a) = Σ c_k (r/a)^k (c_0 first) replacing "
                                          "q0 (1 + ŝ (r − r0)/r0); q0 is then q(r0/a) and ŝ is ignored; needs a_over_R0")
    a_over_R0: float = Field(0.3616, gt=0, description="inverse aspect ratio ε = a/R0, used with q_poly (CBC: 0.3616)")
    # fourier equilibrium (VMEC/GVEC wout file); gx_file: GX's geometry (RUN.out.nc, RUN.eik.out or RUN.eik.nc)
    file: Optional[str] = None
    s0: float = 0.25
    L_ref: Optional[float] = None
    # gx_file (velcro/gx_geometry.py): GX's flux-tube arrays, x-independent; ŝ, q and the twist come from the file
    Ly: Optional[float] = Field(None, gt=0, description="gx_file: binormal box in ρ_ref, 2π/k_y,min (GX: 2π y0/√2)")
    Lref_over_a: Optional[float] = Field(None, gt=0, description="gx_file: Velcro's L_ref in GX's a (default the file's "
                                         "rmaj); the same R/a as the rate and gradient conversions of the deck")
    # miller (velcro/miller.py): the native Miller local equilibrium in GX's parameter names (a = 1); flux tube, needs Ly
    rhoc: float = Field(0.5, gt=0, description="miller: r/a of the surface")
    Rmaj: Optional[float] = Field(None, gt=0, description="miller: R₀/a of the surface (L_ref = Rmaj·a unless Lref_over_a)")
    R_geo: Optional[float] = Field(None, gt=0, description="miller: F/(a B_ref), the toroidal-field radius (default Rmaj)")
    shift: float = Field(0.0, description="miller: |dR₀/dρ| (R₀ decreases outward, GX's convention)")
    akappa: float = Field(1.0, gt=0, description="miller: elongation κ")
    akappri: float = Field(0.0, description="miller: dκ/dρ")
    tri: float = Field(0.0, gt=-1.0, lt=1.0, description="miller: triangularity δ")
    tripri: float = Field(0.0, description="miller: dδ/dρ")
    betaprim: float = Field(0.0, description="miller: GX's betaprim (dβ/dρ, ≤ 0 for a pressure falling outward): the "
                                             "local equilibrium's pressure gradient (curvature drift ≠ ∇B drift, and F')")
    # vmec (velcro/vmec_tube.py): GX's Boozer-coordinate flux tube of a VMEC wout (file), at s0 = normalised toroidal
    # flux, field line alpha, npol poloidal turns of the Boozer angle, cut as GX's pyvmec module; flux tube, needs Ly
    alpha: float = Field(0.0, description="vmec: the field-line label α = θ_B − ι φ_B of the tube")
    npol: int = Field(1, ge=1, description="vmec: poloidal turns of the Boozer angle before the cut")
    vmec_cut: Literal["none", "fix aspect", "gds21", "gbdrift0"] = Field(
        "none", description="vmec: GX's flux-tube cut: 'fix aspect' ends the tube where 2ŝ gds21/gds22 · x0/y0 is an "
                            "integer (jtwist), 'gds21' / 'gbdrift0' where that array vanishes, 'none' keeps ±npol π")
    vmec_x0_over_y0: float = Field(1.0, gt=0, description="vmec: x0/y0 of the 'fix aspect' cut (GX: x0 defaults to y0)")
    vmec_jtwist: Optional[int] = Field(None, description="vmec: the jtwist the 'fix aspect' cut aims for (default: any "
                                       "integer in ±30, GX's search)")
    vmec_which_crossing: int = Field(-1, description="vmec: which of the admissible cuts (GX's which_crossing, −1 = the last)")
    mboz: Optional[int] = Field(None, ge=1, description="vmec: Boozer poloidal modes (default 2 mpol of the wout, GX's)")
    nboz: Optional[int] = Field(None, ge=0, description="vmec: Boozer toroidal modes (default 2 ntor of the wout, GX's)")


class ProfileSpec(BaseModel):
    """A normalised gradient profile ``ω(x) = -L_ref d ln f / dx``.

    ``constant``: ω = value everywhere.  ``flattop``: value in the interior,
    smoothly (tanh, scale ``width`` in ρ_ref) reduced to zero within
    ``edge`` ρ_ref of each radial boundary (so the buffers see flat profiles).
    """
    kind: Literal["constant", "flattop", "peaked", "plateau"] = "constant"
    value: float = 0.0
    width: float = 4.0
    edge: float = 10.0
    # ``peaked``: ω = value · sech²(x / width) (Görler et al. 2011 Eq. 53, the linear global benchmark profile).
    # ``plateau``: ω = value · [1 − sech²((x − x_l)/width) − sech²((x − x_r)/width)] with x_l = x_min + edge,
    #              x_r = x_max − edge (Görler et al. 2011 Eq. 58, the ITM nonlinear benchmark profile).


class ExtZonalSpec(BaseModel):
    """A static external zonal potential ``φ_ext(x) = amp cos(2π kx_index x / L_x + phase)`` (k_y = 0, uniform in z).

    It enters **only** the E×B nonlinearity: its gyroaverage at the μ nodes is added to the k_y = 0 row of the advecting
    potential, so it advects every k_y ≠ 0 component of G in y at v_y = ½C⁻¹ ∂_x J0 φ_ext.  Nothing else sees it: not the
    field solve, the drives, the magnetic drift, E∥ or the flux diagnostics.  An imposed sheared E×B flow for
    intervention tests (the validation notes, test D)."""
    amp: float = 0.0
    kx_index: int = Field(1, ge=1, description="radial wavenumber in units of 2π/L_x")
    phase: float = 0.0


class ExBShearSpec(BaseModel):
    """An equilibrium E×B flow sheared in x, ``v_y = γ_E x`` (``gamma_E`` in v_t/L_ref, x in ρ_ref; GX's ``g_exb`` is
    −kxfac·γ_E/rate_factor), as a proper term of the equations (velcro/vlasov.py, simulation.py; design and the
    difference from the imposed static flow ``physics.ext_zonal`` in the validation notes):

    * ``frame = "local"``: the exact local term −i k_y γ_E x G on every k_y row (every operator sees the advected
      perturbation).  Exact in the buffered (Krook) box; in a periodic box the periodic image is not the sheared one;
    * ``frame = "shearing"``: the periodic flux tube in the shearing frame y′ = y − γ_E t x, G = e^{−iθ(t)x} p with
      θ_ky = k_y γ_E t: p stays x-periodic (the twist-and-shift ghosts, the Krook/source terms and the Lawson k_y drift
      are unchanged), every k_x becomes k_x − θ_ky(t) in the field solve, the gyroaverages and the radial drift, and when
      |θ| passes Δk_x/2 the spectrum of p is shifted by one k_x (exact, GX's remap with the phase kept continuous) with the
      mode leaving the band dropped.  The bracket is NOT taken on p: it is invariant under the shear only while θ ∝ k_y,
      which the row-by-row remap breaks, so it is evaluated on the physical fields e^{−iθx}p with ∂_x − iθ and mapped
      back (operators.frame_bracket, GX's ExBshear_phase algorithm).  Needs ``numerics.field_solver = "fft"`` (the
      flux-tube solver evaluates its symbols at k_x − θ every step), the skew bracket and no GQL cut-off;
    * ``"auto"``: shearing when the box has no Krook buffers, local otherwise.
    ``u_prime``: the parallel-velocity-gradient drive −½C⁻¹ i k_y (2v̂ ω_u) Φ with ω_u = −L_ref d(u∥/v_t)/dx (toroidal
    rotation: ω_u = (q/ε) γ_E), a separate switch; 0 = off.  With kinetic electrons (the validation notes) every
    species gets the drive 2v̂ ω_u,s with ω_u,s = ω_u/v̂t_s (u∥ is shared, v̂ is in each species' own thermal units)."""
    gamma_E: float = 0.0
    frame: Literal["auto", "local", "shearing"] = "auto"
    u_prime: float = 0.0
    u_prime_over_B: bool = Field(False, description="the PVG drive's coefficient is u_prime/B(z) (B in B_ref): purely toroidal "
                                 "rotation, u∥ = (I/B)Ω_φ with I = R B_φ constant on the surface, so ω_u ∝ I/B along the "
                                 "field line (GS2's itor_over_B); u_prime is then ω_u at B = B_ref.  Default False: constant")
    gx_qn_grid: bool = Field(False, description="shearing frame only: the quasineutrality denominator "
                             "(1 − Γ₀(b))/T̂_i + 1/T̂_e at the frame's grid k_x (θ = 0), the numerator and every gyroaverage at "
                             "k_x − θ — GX at 3865a53 computes qneutFacPhi once (Solver_GK constructor) and never after "
                             "geo_shift moves kperp2 with kxstar.  Default False: both at k_x − θ")
    gx_hyper_grid: bool = Field(False, description="shearing frame only: numerics.hyper_x on the frame representation "
                                "(the grid k_x, as GX's hyperdiffusion, which uses kx[idx]); default False: on the physical "
                                "field at k_x − θ")
    gx_step_split: bool = Field(False, description="shearing frame only: GX's time splitting (ts_rk3.cu advance at "
                                "3865a53: flow_shear_shift before the stages) — at the start of every step θ advances by "
                                "k_y γ_E dt and the spectrum is remapped, then every RK stage uses that θ.  Default False: "
                                "each stage carries its own θ(t + c dt) and the remap follows the step")


class SpeciesSpec(BaseModel):
    """A kinetic species other than the reference ions (``physics.electrons = "kinetic"``).  Charge in units of e,
    mass, temperature and density relative to the reference ions (flux tube: constant profiles).  ``omega_n`` /
    ``omega_T`` default to the ions' (quasineutral background); ``nu`` is the like-species collision frequency in
    v_tref/L_ref (GX's ``vnewk`` in v_ti/a times 1.9642 for the CBC s-α decks)."""
    name: str = "electron"
    Z: float = -1.0
    mass: float = Field(2.7e-4, gt=0, description="m_s/m_i (GX's KE decks: 2.7e-4)")
    temp: float = Field(1.0, gt=0, description="T_s/T_i")
    dens: float = Field(1.0, gt=0, description="n_s/n_i")
    omega_n: Optional[ProfileSpec] = None
    omega_T: Optional[ProfileSpec] = None
    nu: float = Field(0.0, ge=0)


class PhysicsConfig(BaseModel):
    omega_n: ProfileSpec = ProfileSpec(value=2.2)
    omega_T: ProfileSpec = ProfileSpec(value=6.9)
    tau: float = Field(1.0, gt=0, description="T_e / T_i at the reference surface")
    Ti_profile: bool = Field(True, description="let T_i(x) follow ∫ω_T (global); else T_i ≡ 1")
    n_profile: bool = Field(True, description="let n(x) follow ∫ω_n; else n ≡ 1")
    collision_model: Literal["dougherty", "lenard-bernstein", "hyper", "landau", "gs2"] = Field(
        "dougherty", description="'landau': the linearised Landau operator at k⊥ = 0 (velcro/landau.py), like-species for "
                                 "every species (applied to H on both paths by default: physics.collisions_on) and, with "
                                 "ei_collisions, the electrons' e–i *test* block only (a stationary ion Maxwellian at "
                                 "ν̂_ei = Z_i ν_e: GX's β = 0 model; no drag toward the ion flow, so refused at β > 0; the "
                                 "ions feel no e–i collisions); nu is then ν̂_aa (Helander & Sigmar), (16, 8) at most.  "
                                 "'gs2': GS2's model operator at k⊥ = 0 (velcro/gs2_collisions.py: Lorentz with ν_D(v) and "
                                 "energy diffusion with ν_∥(v), each with GS2's momentum / energy restoring; with "
                                 "ei_collisions the electrons' pitch-angle scattering off the ions at Z_i ν_e/v̂³, GS2's zeff "
                                 "term, at β = 0 only), nu = GS2's vnewk (ν̂_ss), any (L, M); see gs2_collision_terms")
    gs2_collision_terms: Literal["default", "lorentz", "ediffuse"] = Field(
        "default", description="collision_model = 'gs2': GS2's collision_model — 'default' (Lorentz + energy diffusion), "
                               "'lorentz' (pitch-angle scattering only, e–i included) or 'ediffuse' (energy diffusion only)")
    gs2_const_v: bool = Field(False, description="collision_model = 'gs2' with gs2_collision_terms = 'lorentz': GS2's "
                              "const_v (every frequency at v̂ = 1)")
    nu: float = Field(0.0, ge=0, description="reference collision frequency in v_t/L_ref")
    collisions_on: Literal["g", "h"] = Field(
        "h", description="adiabatic electrons only: what the collision operator acts on.  'h' (the default since "
                         "2026-10-03): the non-adiabatic H = G + Φ/T̂_i, the gyrokinetic operator's "
                         "argument (GX and Velcro's kinetic path), for which the H-theorem holds; 'g': G itself, the default "
                         "of every adiabatic run before 2026-10-03 (recorded results with ν > 0 are records of it) — with "
                         "'g' collisions can raise the free energy at finite k⊥ (tier-2 review; the validation notes); "
                         "the ITG growth rate moves by 1–2 % of the collisional shift between the two")
    collision_flr: bool = Field(False, description="the gyrokinetic finite-Larmor-radius terms of the collision operator "
                                "(velcro/collisions_flr.py; derivation and validation in the validation notes), both paths, every "
                                "species: 'dougherty' / 'lenard-bernstein' the exact gyroaverage (GX's operator: −ν b and the J₀/J₁-"
                                "weighted u∥, u⊥, T restoring); 'landau' the exact test-particle classical diffusion −ν b K and the "
                                "field part's gyro-harmonics n = 0 and ±1, exact on sector bases of M + 2 and M + 1 Laguerre "
                                "functions (|n| ≥ 2 enters at O(b²) and is omitted; M ≤ 10).  b → 0 is the k⊥ = 0 "
                                "operator.  x-periodic flux tubes only (no Krook buffers, no T/n profiles, not the shearing frame): "
                                "the terms are applied in k_x with the symbols at the reference surface.  Adiabatic electrons need "
                                "collisions_on = 'h'.  False (the default): the k⊥ = 0 operators, every run so far")
    nonlinear: bool = True
    gql_cutoff: Optional[int] = Field(None, ge=0, description="generalised quasilinear E×B nonlinearity: k_y indices ≤ cutoff are 'low'; "
                                      "eddy–eddy (high·high → high) interactions are dropped.  0 = quasilinear (zonal modes + linear "
                                      "drift waves in the instantaneous zonal fields); None = full nonlinearity")
    drive_gradB: bool = Field(False, description="True restores the −ω_B μ̂ Φ part of v_E·∇F0 (∇F0 at fixed μ) that the code carried "
                              "until 2026-09-21.  In the gyrokinetic equation that piece is cancelled by the v_E·μ∇B part of the "
                              "parallel acceleration acting on F0, which the right-hand side does not contain; with it the linear "
                              "growth rate is 12–15 % below GX, the threshold 4.3 instead of 3.9–4.0 and χ_i 1.65 instead of 2.33 "
                              "(GX 2.24).")
    ext_zonal: Optional[ExtZonalSpec] = Field(None, description="static external zonal E×B potential that only advects "
                                              "(ExtZonalSpec); None or amp = 0 = off")
    coriolis_mach: float = Field(0.0, description="GS2's 'mach' (Ω_φ a/v_t,ref of this tube): the Coriolis drift of a toroidally "
                                 "rotating plasma, GS2's wcoriolis (dist_fn.fpp) — mach·v̂∥·(cdrift k_y + cdrift0 k_y θ₀)/2 with the "
                                 "species factor (T/Z)/v̂t_s, added to the magnetic drift on H; no centrifugal terms (GS2 has none).  "
                                 "Native Miller geometry only.  0 = off")
    exb_shear: Optional[ExBShearSpec] = Field(None, description="equilibrium E×B flow shear v_y = γ_E x as a proper term "
                                              "(ExBShearSpec); None or gamma_E = u_prime = 0 = off")
    electrons: Literal["adiabatic", "kinetic"] = Field(
        "adiabatic", description="'kinetic': electrons are a second kinetic species (``physics.electron``) with their own "
                                 "Hermite–Laguerre moments; quasineutrality with both species kinetic (no adiabatic response, no "
                                 "flux-surface-average treatment of the zonal row).  See velcro/kinetic.py")
    electron: SpeciesSpec = SpeciesSpec()
    beta: float = Field(0.0, ge=0, description="β_ref = 8π n_ref T_ref/B_ref² (GX's beta); > 0 adds A∥ (Ampère's law, "
                                               "g formulation) — kinetic electrons only")
    flutter: bool = Field(True, description="with β > 0: the magnetic-flutter part −v∥{J0A∥, g} of the nonlinearity")
    bpar: bool = Field(False, description="with β > 0: the compressional perturbation δB∥ (GX's fbpar; perpendicular pressure "
                                          "balance coupled to quasineutrality, the μ̂·2J1(a)/a term of ⟨χ⟩)")
    ei_collisions: bool = Field(False, description="kinetic electrons only: electron–ion collisions as GX's `ei_colls = true` "
                                "(velcro/kinetic.py): the electrons' Dougherty operator with ν_e + ν_ei (ν_ei = Z_i ν_e) on the "
                                "diagonal and in the energy restoring, and momentum restoring toward the ion flow at ν_ei (the "
                                "ion flow is taken as 0 when β = 0, as in GX).  Needs collision_model = 'dougherty', or "
                                "'landau' at β = 0 (the e–i Landau test block, see collision_model)")


class NumericsConfig(BaseModel):
    bracket: Literal["skew", "advective"] = Field(
        "skew", description="E×B bracket discretisation: 'skew' conserves free energy to round-off "
                            "for any field; 'advective' is the original Orszag form (4th-order residual)")
    node_gyro: Literal["laguerre", "exact"] = Field(
        "exact", description="gyroaveraged potential at the μ nodes of the nonlinearity: 'laguerre' = "
                                "truncated Σ_m 𝒥_m L_m(μ_j) (diverges at the outer nodes for M ≳ 16); 'exact' = J0(√(2bμ_j)) φ")
    parallel_form: Literal["advective", "skew"] = Field(
        "skew", description="discretisation of streaming + mirror force.  'advective': −c(z) V ∂_z H + c ∂_z ln B · Mirror H with "
                                 "centred differences, which conserves the Jacobian-weighted free energy only to truncation order and "
                                 "lets a grid-scale parallel mode grow at 0.1–0.2 v_t/R unless hyper_z ≳ 0.05.  'skew': the same operator "
                                 "written as −V·(1/2J)[w ∂_z + ∂_z w] H + c ∂_z ln B (Mirror − ½V) H with w = J c: both parts are exactly "
                                 "skew in the J-weighted inner product (axisymmetric geometry; otherwise falls back to 'advective')")
    hyper_x: float = Field(0.05, ge=0, description="coefficient of −Δx⁴∂⁴_x (grid-scale)")
    hyper_z: float = Field(0.05, ge=0)
    hyper_ky: float = Field(0.0, ge=0, description="coefficient of −(k_y/k_y,max)⁴")
    hyper_v: float = Field(0.5, ge=0, description="hyper-collisions ν_h[(ℓ/(L−1))ᵖ + (m/(M−1))ᵖ], p = hyper_v_power]")
    hyper_v_power: int = Field(4, ge=1, description="power p of the hyper-collision operator; higher p confines it to the "
                                                    "truncation (GAM Landau damping without eroding the RH residual)")
    hyper_kz: float = Field(0.0, ge=0, description="GX's |k∥|-scaled Hermite hypercollisions (its `hypercollisions = true`, "
                                                   "`nu_hyper_m`): −ν ℓ^p |k_z| G_ℓ for ℓ ≥ 3, with ν = hyper_kz (p+½)/(L−1)^(p+½)·"
                                                   "2.3·v_ts⟨|b·∇z|⟩ per species; kinetic electrons, and adiabatic "
                                                   "electrons with Velcro's own parallel boundary ("
                                                   "for the ions, v̂t = 1).  0 = off")
    hyper_kz_p: int = Field(6, ge=1, description="order p of hyper_kz (GX's p_hyper_m)")
    parallel_derivative: Literal["fd4", "fourier", "fourier_capped"] = Field(
        "fd4", description="∂_z of streaming and E∥, kinetic electrons only (the adiabatic code keeps 'fd4').  'fd4': the "
                           "4th-order centred stencil; its symbol (8 sin θ − sin 2θ)/6Δz vanishes at the grid Nyquist θ = π, "
                           "so near a rational surface (θ = k_y τ(x) ≈ 0 mod 2π) grid-scale modes along z do not stream and "
                           "their electrons act as k∥ = 0 electrons.  'fourier': i k_z on the Fourier modes of the "
                           "Bloch-periodic field line, k_z = n − θ/2π, as GX (time step ∝ Δz/π).  'fourier_capped': the same "
                           "with |k_z| capped at FD4's largest symbol 1.3722/Δz — exact for |k_z Δz| ≤ 1.3722, no zero at "
                           "the Nyquist, and FD4's time step")
    kpar_floor: float = Field(0.0, ge=0.0, description="with a Fourier parallel_derivative: the electrons' |k_z| ≥ "
                              "kpar_floor·ŝ k_y Δx on the k_y ≠ 0 rows (k_z per radian of z; exact zeros stay 0, so use it with "
                              "grid.x_offset).  A grid point near a rational surface stands for a cell of width Δx, across "
                              "which k_z varies by ŝk_yΔx; the electrons' non-adiabatic layer (≈ ωqR/(ŝ k_y v_te), ≈ 0.05 ρ_i "
                              "for the GX decks) is far thinner, so a whole cell of non-streaming electrons would stand in for "
                              "it.  0.5 floors |k_z| at its value half a cell from the rational surface.  The ions' layer is "
                              "≈ √(m_i/m_e) wider and resolved: no floor.  0 = off")
    parallel_bc: Literal["velcro", "gx_chains"] = Field(
        "velcro", description="kinetic electrons only.  'velcro': every x point's field line with its twist phase (the real-x "
                              "flux tube; grid points on rational surfaces close).  'gx_chains': GX's linked boundary — the "
                              "retained k_x (x_dealias) linked by the twist into chains m, m+s, … (s = ŝ k_y L_x), each chain "
                              "closed periodically as GX's linked FFT does, and ∂_z taken spectrally along the chain with "
                              "parallel_derivative's symbol (needs x_dealias and a Fourier parallel_derivative)")
    gx_damp_ends: float = Field(0.0, ge=0.0, description="with parallel_bc = 'gx_chains': GX's absorbing layers at the two ends "
                                "of every k_y > 0 chain (grad_parallel_linked applyBCs → dampEnds_linked): −ν(ξ) A h, h the "
                                "non-adiabatic part G + (Z/T)χ, ν = 1 − 2ξ²/(1+ξ⁴) over gx_damp_width of the chain at each end "
                                "(ξ = distance to the end / width), A this rate in v_t/R.  GX's A is damp_ends_amp/dt = 0.1/dt_GX "
                                "(v_ti/a) = 1.9642·0.1/dt_GX here.  0 = off")
    gx_damp_width: float = Field(0.125, gt=0.0, lt=0.5, description="GX's damp_ends_widthfrac")
    gx_damp_amp: float = Field(0.0, ge=0.0, description="GX's damp_ends_amp, per step: the end-damping rate becomes "
                               "gx_damp_amp/dt₀ with dt₀ Velcro's CFL step at t = 0 (GX's rule amp/dt with Velcro's own step, "
                               "so a tenth of h per step at amp 0.1, as in GX).  Overrides gx_damp_ends when > 0")
    gx_hyper: float = Field(0.0, ge=0.0, description="kinetic electrons only: GX's k-space hyperviscosity (linear.cu "
                            "hyperdiff), −gx_hyper ((k_x² + k_y²)/(k_x,max² + k_y,max²))^gx_hyper_p on every moment of G, "
                            "k_x,max the largest retained k_x ((N_x−1)//3 Δk_x) and k_y,max the largest k_y; the rate in v_t/R "
                            "(GX's D_hyper × 1.9642 for the GX decks).  0 = off")
    gx_hyper_p: int = Field(2, ge=1, description="GX's p_hyper")
    gx_hyperz: float = Field(0.0, ge=0.0, description="kinetic electrons with parallel_bc 'gx_chains' only: GX's parallel "
                             "hyperdiffusion (its `hyperz = true`, grad_parallel_linked.cu hyperz / device_funcs.cu "
                             "hyperkzLinked_kernel), −gx_hyperz |k_z/k_z,max|^gx_hyperz_p on every moment of G along each linked "
                             "chain, k_z,max = N_z/2 (the Nyquist of one 2π segment); the rate in v_t/R (GX's nu_hyper_z × "
                             "(R/a)/√2).  0 = off (GX's default; high-β runs need parallel dissipation)")
    gx_hyperz_p: int = Field(6, ge=1, description="GX's p_hyper_z")
    gx_hyper_ref_ny: int = Field(0, ge=0, description="kinetic electrons only: normalise gx_hyper's k⊥ to the k_y,max of a "
                                 "grid with this N_y instead of the run's own (k_x,max unchanged), so a finer k_y grid keeps "
                                 "the same dissipation at every retained k (the new rows are damped more, ∝ (k⊥/k⊥,ref)^2p).  "
                                 "Must be ≤ the run's N_y and share its k_y spacing.  0 = the run's own grid (GX's rule)")
    gx_hermite_closure: bool = Field(False, description="the v̂² of the magnetic-drift factor 2v̂² + μ̂ "
                                     "as GX's stencil has it, the L×L block of the (L+1)-mode product (V V)[:L,:L], instead of "
                                     "the product of the truncated V (which drops the ψ_L state on the last Hermite row: 2v̂² there "
                                     "is 2L−1 in GX and L−1 otherwise); in the drift band and in the Lawson drift propagator")
    zonal_nyquist_project: bool = Field(False, description="kinetic electrons only: project the k_z Nyquist mode ((−1)^j along "
                                        "z) out of the k_y = 0 row of the state and the right-hand side.  With a Fourier ∂_z that "
                                        "mode has no streaming on the (real) zonal row, and nothing else damps its ℓ ≤ 2 moments, "
                                        "so a grid-scale zigzag builds up there nonlinearly; GX streams it at +N_z/2 (not "
                                        "real-preserving), which phase-mixes it away.  The projection removes the unresolved mode")
    gx_nyquist: bool = Field(False, description="with parallel_bc = 'gx_chains': the Nyquist mode of every k_y > 0 chain "
                             "streams at +N_z/2 (capped as parallel_derivative), as GX's init_kzLinked sets it, instead of 0.  "
                             "The chains of k_y > 0 are complex, so the symbol stays real and ∂_z anti-Hermitian; the k_y = 0 "
                             "chains keep 0 (their reality)")
    x_dealias: bool = Field(False, description="2/3-rule projection along x of the right-hand side and the initial state, "
                            "keeping |k_x| ≤ ((N_x−1)//3) Δk_x — GX's retained set for the same N_x (63 of 96).  Kinetic "
                            "electrons, or adiabatic electrons in an x-periodic box (no Krook buffers; in the shearing frame of "
                            "physics.exb_shear the remap drops the mode leaving the band, as GX's g_shift).  The twist-and-shift "
                            "links these modes into GX's chains; structure below ~3 cells (e.g. one grid point on a rational "
                            "surface) cannot form")
    bracket_dx: Literal["fd4", "fourier"] = Field(
        "fd4", description="∂_x inside the skew E×B (and flutter) bracket.  'fourier': i k_x by FFT along x (Nyquist 0), as "
                           "GX's pseudo-spectral bracket (with x_dealias the product is then truncated to GX's retained k_x); "
                           "the skew form still conserves the free energy.  Kinetic electrons, or adiabatic electrons in an "
                           "x-periodic box (the skew bracket without a GQL cut-off)")
    radial_drift: Literal["fd4", "fourier"] = Field(
        "fd4", description="∂_x of the linear radial magnetic drift (flux tube, periodic in x; kinetic electrons, or "
                           "adiabatic electrons without Krook buffers).  "
                           "'fd4': the 4th-order stencil, accurate to k_xΔx ≈ 1 — on GX's grid about half of GX's k_x range, "
                           "which cuts the ŝθ sin θ part of the drift in the ballooning tails of the higher-k_y modes.  "
                           "'fourier': i k_x by FFT along x (the Nyquist mode 0), as GX.  The E×B bracket keeps FD4")
    krook_rate: float = Field(0.5, ge=0, description="Krook damping rate in the radial buffers")
    krook_width: float = Field(0.1, ge=0, lt=0.5, description="buffer width as a fraction of L_x, each side")
    krook_zonal: bool = Field(True, description="False: the radial Krook buffers leave k_y = 0 alone (zonal modes do not "
                                                "see the twist phase, so they are legitimately x-periodic, as in a flux tube)")
    source_rate: float = Field(0.0, ge=0, description="Krook relaxation of the k_y=0 flux-surface-averaged profile moments")
    integrator: Literal["rk4", "lawson", "imex"] = Field(
        "rk4", description="'lawson' = integrating-factor RK4: the binormal magnetic drift of G, "
                           "−(2v̂²+μ̂) i k_y K_y T_i G (the stiffest linear term), is advanced exactly as a phase "
                           "rotation at the velocity nodes; everything else stays explicit. Axisymmetric geometry only; "
                           "uses the 'spectral' CFL bound without the k_y drift.  'imex' (kinetic electrons only, "
                           "velcro/imex.py): an additive Runge–Kutta scheme (imex_scheme) with "
                           "the electrons' parallel streaming, mirror force and hypercollisions and both species' "
                           "absorbing chain ends implicit, field response included (exact per linked chain: dense chain "
                           "inverses + a Woodbury capacitance per chain), everything else explicit; with imex_lawson the "
                           "binormal drift of G stays exact (Lawson); dt from the explicit part's spectral bound")
    imex_scheme: Literal["ark3", "ars232", "ars443"] = Field(
        "ark3", description="integrator 'imex': 'ark3' = ARK3(2)4L[2]SA (Kennedy & Carpenter 2003; 3rd order, L-stable, "
                            "3 implicit stages, explicit stability 2.48 on the imaginary axis); 'ars232' / 'ars443' = "
                            "Ascher–Ruuth–Spiteri (2nd / 3rd order, L-stable; explicit stability 1.73 / 1.57)")
    imex_lawson: bool = Field(True, description="integrator 'imex': advance the binormal drift of G exactly (the Lawson "
                              "integrating factor of 'lawson', applied to the additive Runge–Kutta stages)")
    imex_cache: int = Field(2, ge=1, description="integrator 'imex': step sizes whose factors (chain inverses or "
                            "block-Thomas factors, imex_solver) are kept on the device (dense: Σ_chain-kinds (L·M·N·N_z)² "
                            "complex numbers, 2.8 GB at (12,6), 8.8 GB at (16,8) on GX's 96 × 36 × 24 grid in fp32)")
    imex_solver: Literal["auto", "dense", "thomas"] = Field(
        "auto", description="integrator 'imex': how each chain group's implicit electron operator K = I − hγ f_I is "
                            "factorised.  'dense' = its inverse K⁻¹, (L·M·N·N_z)² complex numbers per group (the path "
                            "validated deck by deck); 'thomas' = an exact block-Thomas factorisation along the Hermite "
                            "index ℓ (streaming and mirror force couple ℓ ± 1 only, so K is block-tridiagonal in ℓ): L "
                            "blocks of (M·N·N_z)², L times less memory, and a stage reads ≈ (2L + 2)/(L² + 2L) of the "
                            "dense bytes; 'auto' = dense while one factor set fits imex_dense_gib, otherwise 'thomas' for "
                            "every group)")
    imex_factor_storage: Literal["full", "bf16"] = Field(
        "full", description="integrator 'imex', block-Thomas groups: how the factors P_ℓ are kept on the device.  'full' = "
                            "the run's complex dtype (the validated path); 'bf16' = their real and imaginary parts in "
                            "bfloat16, applied with accumulation in the run's real dtype: half the factor memory and half "
                            "the bytes each stage solve reads, at ≈ 8e-5 of the state per step on GX's JET box at (24,8) "
                            "against a step's own 5.4e-3.  A memory fallback only: XLA does not fuse the bf16 → fp32 "
                            "conversion into the product, so the factor products run ≈ 2.8× slower.  "
                            "Experimental: no run is validated with it")
    imex_dense_gib: float = Field(4.0, gt=0.0, description="integrator 'imex', imex_solver 'auto': the GiB one factor set "
                                  "may take before every chain group switches from dense inverses to block-Thomas.  4 picks "
                                  "the faster solver at both measured points: the validated "
                                  "(12,6) decks need 2.6 GiB and stay dense (bitwise their validated path); (16,8) on GX's "
                                  "96 × 36 × 24 grid needs 8.2 GiB and goes block-Thomas, faster there; GX's JET box at "
                                  "(12,6) needs 62.3 GiB dense, 5.2 block-Thomas")
    imex_fallback: Literal["lawson", "error"] = Field(
        "lawson", description="integrator 'imex': what a run does when the device cannot hold the implicit factors it "
                              "needs (velcro/imex.py ImexSystem.need_bytes: the factor set, the capacitance, and the larger "
                              "of a factor build's temporaries and the step's working set, against imex_device_gib or the "
                              "device's free memory) — 'auto' tries dense, then block-Thomas; when neither fits (or a "
                              "forced imex_solver does not), 'lawson' runs it as Lawson RK4 with a warning that states the "
                              "sizes (RK4 on a geometry Lawson cannot take), 'error' refuses the run")
    imex_device_gib: Optional[float] = Field(
        None, gt=0.0, description="integrator 'imex': the free device memory (GiB) the factor plan is checked against; "
                                  "None = the device's own (bytes_limit − bytes_in_use from memory_stats; a backend that "
                                  "reports none, e.g. the CPU, is not checked).  Set it for a share of a device, or to test")
    imex_dt_levels: float = Field(0.85, ge=0.0, lt=1.0, description="integrator 'imex': dt_from_cfl returns the largest "
                                  "step dt_max·r^k (r = this) below the bound, so a fluctuating bound (the turbulence's "
                                  "E×B and flutter speeds) switches among a few factored steps instead of refactoring at "
                                  "every estimate; 0 = the bound itself with imex_dt_stick")
    imex_dt_stick: float = Field(0.8, gt=0.0, le=1.0, description="integrator 'imex' with imex_dt_levels = 0: dt_from_cfl "
                                 "keeps the previous step while it is within [imex_dt_stick, 1] × the new bound")
    imex_deduce_fi: bool = Field(True, description="integrator 'imex': take f_I at an implicit stage from the stage equation, "
                                 "(G_i − r_i)/(hγ), instead of evaluating it again (one f_I evaluation per step instead of "
                                 "one per stage; the two differ by the stage solve's round-off over hγ)")
    imex_nl_bound: Literal["sum", "max"] = Field(
        "sum", description="integrator 'imex': the E×B / flutter speeds in the explicit part's bound — 'sum' as "
                           "kinetic.cfl_rate_kinetic (Σ over k_y of |∂_x Φ| and |k_y Φ| at the μ nodes, an upper bound of "
                           "their real-space values), 'max' the real-space maxima (the bracket's grid and ∂_x)")
    imex_rate: Literal["bound", "jacobian"] = Field(
        "bound", description="integrator 'imex': the rate dt_from_cfl divides the explicit table's stability interval by. "
                             "'bound' = the explicit part's per-term spectral bound (ImexSystem.explicit_rate, summed as "
                             "kinetic.cfl_rate_kinetic; ≈ 2.7× above the real radius at the saturated q 1.4 state); "
                             "'jacobian' = an Arnoldi estimate of the explicit part's own "
                             "spectral radius at the state (imex.JacobianRate), refreshed at every dt_from_cfl, never "
                             "below imex_rate_floor × the bound")
    imex_rate_krylov: int = Field(20, ge=4, description="integrator 'imex', imex_rate 'jacobian': Arnoldi steps per "
                                  "estimate (one jvp of the explicit part each; m + 1 state-sized vectors on the device)")
    imex_rate_floor: float = Field(0.32, gt=0.0, le=1.0, description="integrator 'imex', imex_rate 'jacobian': the rate is "
                                   "at least this × the per-term bound, so the step is never longer than 1/this × the "
                                   "bound's (0.32 at cfl 0.9: CFL 2.8 of the bound, the step validated on the q 1.4 deck; "
                                   "CFL 4 diverged there)")
    cfl: float = Field(0.5, gt=0)
    cfl_estimate: Literal["legacy", "spectral"] = Field(
        "legacy", description="'legacy' = the original rough rates (underestimates the spectral radius for "
                              "L ≳ 32 and with T_i profiles); 'spectral' = per-term bound from the actual "
                              "velocity-matrix radii (never below the measured radius; 1.4–1.5x above it)")
    dt: Optional[float] = Field(None, description="fixed time step (overrides CFL)")
    dt_max: float = 0.1
    field_solver: Literal["dense", "lowrank", "fft"] = Field(
        "dense", description="'dense': N_x×N_x gyro/polarisation matrices per (k_y, z) (fast, memory ∝ N_x²); 'lowrank': "
                             "SVD-factorised symbols applied with FFTs and a preconditioned-CG quasi-neutrality solve "
                             "(memory ∝ N_x; exact at rank 1 without profiles) — for N_x ≳ 192 and global runs; 'fft' (kinetic "
                             "electrons only): the flux-tube limit, symbols at the reference surface applied as FFT multipliers")
    lowrank_tol: float = Field(1e-8, gt=0, description="relative singular-value cut of each symbol table")
    cg_tol: float = Field(1e-6, gt=0, description="relative residual of the k_y ≠ 0 field solve")
    cg_maxiter: int = Field(50, ge=1)
    steps_per_dt_update: int = 20


class InitConfig(BaseModel):
    kind: Literal["noise", "mode", "zonal", "landau"] = "noise"
    amplitude: float = 1e-3
    seed: int = 0
    ky_index: int = 1
    kx_index: int = 0
    l: int = 0
    m: int = 0
    z_profile: Literal["flat", "ballooning"] = "ballooning"


class RunConfig(BaseModel):
    t_end: float = 50.0
    diag_every: float = Field(0.5, gt=0, description="time between diagnostics")
    output: Optional[str] = None
    precision: Literal["fp64", "fp32"] = "fp64"
    max_steps: int = 10_000_000


class VelcroConfig(BaseModel):
    grid: GridConfig = GridConfig()
    geometry: GeometryConfig = GeometryConfig()
    physics: PhysicsConfig = PhysicsConfig()
    numerics: NumericsConfig = NumericsConfig()
    init: InitConfig = InitConfig()
    run: RunConfig = RunConfig()

    @model_validator(mode="after")
    def _check(self):
        if self.geometry.kind == "fourier" and not self.geometry.file:
            raise ValueError("geometry.kind='fourier' requires geometry.file (VMEC/GVEC wout)")
        if self.geometry.kind == "gx_file" and (not self.geometry.file or self.geometry.Ly is None):
            raise ValueError("geometry.kind='gx_file' requires geometry.file (GX's out.nc / eik file) and geometry.Ly")
        if self.geometry.kind == "miller" and (self.geometry.Ly is None or self.geometry.Rmaj is None):
            raise ValueError("geometry.kind='miller' requires geometry.Rmaj (R₀/a) and geometry.Ly")
        if self.geometry.kind == "vmec" and (not self.geometry.file or self.geometry.Ly is None):
            raise ValueError("geometry.kind='vmec' requires geometry.file (a VMEC wout) and geometry.Ly")
        ph = self.physics
        if ph.collision_model == "gs2":
            if ph.collision_flr:
                raise ValueError("physics.collision_flr with collision_model = 'gs2' (the k⊥ = 0 operator only; "
                                 "velcro/gs2_collisions.py)")
            if ph.gs2_const_v and ph.gs2_collision_terms != "lorentz":
                raise ValueError("physics.gs2_const_v needs gs2_collision_terms = 'lorentz' (velcro/gs2_collisions.py)")
            if self.grid.L < 3 or self.grid.M < 2:
                raise ValueError("collision_model = 'gs2' needs grid.L ≥ 3 and grid.M ≥ 2")
        elif ph.gs2_const_v or ph.gs2_collision_terms != "default":
            raise ValueError("physics.gs2_collision_terms / gs2_const_v need collision_model = 'gs2'")
        return self

    @classmethod
    def from_file(cls, path: str) -> "VelcroConfig":
        import json, pathlib
        p = pathlib.Path(path)
        text = p.read_text()
        if p.suffix in (".toml",):
            import tomllib
            return cls.model_validate(tomllib.loads(text))
        return cls.model_validate(json.loads(text))
