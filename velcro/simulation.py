"""Run driver: setup, RK4 time advance with CFL control, diagnostics, output (§3.3, §7.1)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from velcro.config import VelcroConfig


def _setup_jax(precision: str):
    import jax
    jax.config.update("jax_enable_x64", precision == "fp64")
    # On GPUs jax lowers float32 contractions to TF32 (10-bit mantissa) by default.  The
    # velocity-space matrices and the DFT transforms need true single precision (the fp32 path
    # is checked against fp64 in tests/test_perf_equivalence.py); fp64 is unaffected.
    jax.config.update("jax_default_matmul_precision", "highest")


@dataclass
class Diagnostics:
    t: list = field(default_factory=list)
    phi2: list = field(default_factory=list)        # Σ|φ|²
    W: list = field(default_factory=list)           # Σ|G|² (free-energy proxy)
    Q: list = field(default_factory=list)           # volume-averaged ion heat flux
    Gamma: list = field(default_factory=list)       # particle flux
    hermite: list = field(default_factory=list)     # (L,) Σ|G_ℓ·|²
    laguerre: list = field(default_factory=list)    # (M,)
    phi_ky: list = field(default_factory=list)      # (Nky,) Σ_{x,z}|φ|²
    phi_zonal: list = field(default_factory=list)   # (Nx,) flux-surface averaged k_y=0 potential
    probe: list = field(default_factory=list)       # complex φ at (x=0, ky[probe_ky], z=0)
    dt: list = field(default_factory=list)

    def asdict(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


@dataclass
class KineticDiagnostics(Diagnostics):
    """:class:`Diagnostics` of a kinetic-electron run: ``Q`` / ``Gamma`` are the ions' (E×B + flutter), ``hermite`` has
    one entry per stacked (species, ℓ) row, plus the per-species split of the fluxes and Σ|A∥|²."""
    apar2: list = field(default_factory=list)       # Σ|A∥|²
    Qi_es: list = field(default_factory=list)       # ion heat flux, E×B part
    Qi_em: list = field(default_factory=list)       # ion heat flux, magnetic-flutter part
    Qe: list = field(default_factory=list)          # electron heat flux (E×B + flutter)
    Qe_es: list = field(default_factory=list)
    Qe_em: list = field(default_factory=list)
    Gamma_e: list = field(default_factory=list)     # electron particle flux
    Qi_bpar: list = field(default_factory=list)     # the compressional (δB∥) parts (0 without physics.bpar; with it the
    Qe_bpar: list = field(default_factory=list)     # *_es parts are the φ parts alone and Q = es + em + bpar)


class Simulation:
    """A Velcro simulation.  Build with a :class:`VelcroConfig`, then :meth:`run`."""

    def __init__(self, cfg: VelcroConfig):
        _setup_jax(cfg.run.precision)
        import jax
        import jax.numpy as jnp
        from velcro import geometry as geo
        from velcro.fields import build_field_solver
        from velcro.operators import ky_grid
        from velcro.vlasov import build_operators

        self.cfg = cfg
        g, gc = cfg.grid, cfg.geometry
        self.dtype = jnp.complex128 if cfg.run.precision == "fp64" else jnp.complex64
        self.rdtype = jnp.float64 if cfg.run.precision == "fp64" else jnp.float32
        # geometry
        if g.x_offset and gc.kind not in ("s_alpha", "gx_file", "miller", "vmec"):
            raise ValueError("grid.x_offset is implemented for the s-alpha, gx_file, miller and vmec geometries only")
        if gc.kind == "slab":
            self.geom = geo.slab(g.Nx, g.Lx, g.Ny, Ly=2 * np.pi * 10.0, Nz=g.Nz, rho_star=gc.rho_star,
                                 Lz_over_Lref=gc.Lz_over_Lref)
        elif gc.kind == "s_alpha":
            q_profile, q0 = None, gc.q0
            if gc.q_poly is not None:
                c = np.asarray(gc.q_poly, dtype=float)
                q_profile = lambda r: np.polyval(c[::-1], r / gc.a_over_R0)      # r in R0 units → r/a
                q0 = float(q_profile(gc.r0))
            self.geom = geo.s_alpha(g.Nx, g.Lx, g.Ny, g.Nz, rho_star=gc.rho_star, r0=gc.r0, q0=q0,
                                    shat=gc.shat, n0=gc.n0, alpha_mhd=gc.alpha_mhd, q_profile=q_profile,
                                    gpar_model=gc.gpar, x_offset=g.x_offset)
        elif gc.kind == "gx_file":
            from velcro.gx_geometry import from_gx_geometry
            if cfg.physics.drive_gradB:
                raise ValueError("physics.drive_gradB needs ∂_x ln B, which GX's flux-tube geometry does not carry")
            self.geom = from_gx_geometry(gc.file, g.Nx, g.Lx, g.Ny, g.Nz, Ly=gc.Ly, rho_star=gc.rho_star,
                                         Lref_over_a=gc.Lref_over_a, x_offset=g.x_offset)
        elif gc.kind == "miller":
            from velcro.miller import from_miller
            if cfg.physics.drive_gradB:
                raise ValueError("physics.drive_gradB needs ∂_x ln B, which the Miller flux tube does not carry")
            self.geom = from_miller(g.Nx, g.Lx, g.Ny, g.Nz, Ly=gc.Ly, rhoc=gc.rhoc, Rmaj=gc.Rmaj, q=gc.q0, shat=gc.shat,
                                    R_geo=gc.R_geo, shift=gc.shift, akappa=gc.akappa, akappri=gc.akappri, tri=gc.tri,
                                    tripri=gc.tripri, betaprim=gc.betaprim, rho_star=gc.rho_star,
                                    Lref_over_a=gc.Lref_over_a, x_offset=g.x_offset)
        elif gc.kind == "vmec":
            from velcro.vmec_tube import from_vmec_tube
            if cfg.physics.drive_gradB:
                raise ValueError("physics.drive_gradB needs ∂_x ln B, which the VMEC flux tube does not carry")
            self.geom = from_vmec_tube(gc.file, g.Nx, g.Lx, g.Ny, g.Nz, Ly=gc.Ly, s0=gc.s0, alpha=gc.alpha, npol=gc.npol,
                                       cut=gc.vmec_cut, x0_over_y0=gc.vmec_x0_over_y0, jtwist=gc.vmec_jtwist,
                                       which_crossing=gc.vmec_which_crossing, mboz=gc.mboz, nboz=gc.nboz,
                                       rho_star=gc.rho_star, Lref_over_a=gc.Lref_over_a, x_offset=g.x_offset)
        else:
            eq = geo.from_vmec_wout(gc.file, L_ref=gc.L_ref)
            self.geom = eq.build(g.Nx, g.Lx, g.Ny, g.Nz, s0=gc.s0, n0=gc.n0, rho_star=gc.rho_star)
        self.ky = ky_grid(g.Ny, self.geom.Ly)
        self.Nky = len(self.ky)
        # profiles
        if gc.kind in ("gx_file", "miller", "vmec"):    # the twist-and-shift must be an integer shift of the x-periodic box
            w = float(self.geom.extra["twist_shat"]) * (2 * np.pi / float(self.geom.Ly)) * g.Lx
            if abs(w - round(w)) > 1e-3 * max(1.0, abs(round(w))):
                raise ValueError(f"geometry.kind = '{gc.kind}': the twist S k_y,min L_x = {w:.5f} is not an integer, so the "
                                 "twisted field line is not x-periodic — set L_x = jtwist/(|S| k_y,min) (GX's x0 rule; "
                                 f"S = {float(self.geom.extra['twist_shat']):.6f}, a GX deck translation must do this)")
        if gc.kind in ("gx_file", "miller", "vmec") and (cfg.physics.Ti_profile or cfg.physics.n_profile):
            raise ValueError(f"geometry.kind = '{gc.kind}' is a flux tube (x-independent geometry): set physics.Ti_profile "
                             "and physics.n_profile to false — the global T_i(x)/n(x) profiles belong to the global s-α / "
                             "fourier geometries (with the default ρ* they span T_i 0.5–2.2 across the box); the kinetic "
                             "path refuses them for every geometry")
        self.prof = self._profiles()
        # field solver and Vlasov operators
        self.kinetic = cfg.physics.electrons == "kinetic"
        if not self.kinetic and (cfg.physics.beta > 0
                                 or cfg.physics.bpar or cfg.numerics.parallel_derivative != "fd4"
                                 or cfg.numerics.kpar_floor > 0
                                 or cfg.numerics.parallel_bc != "velcro"
                                 or cfg.numerics.gx_nyquist
                                 or cfg.numerics.gx_hyper > 0
                                 or cfg.numerics.zonal_nyquist_project or cfg.physics.ei_collisions):
            raise ValueError("physics.beta > 0, numerics.parallel_derivative ≠ 'fd4', "
                             "numerics.parallel_bc ≠ 'velcro' and numerics.gx_hyper > 0 need physics.electrons = 'kinetic'")
        if not self.kinetic and (cfg.numerics.radial_drift != "fd4" or cfg.numerics.bracket_dx != "fd4"
                                 or cfg.numerics.x_dealias) \
                and cfg.numerics.krook_rate > 0 and cfg.numerics.krook_width > 0:
            raise ValueError("numerics.radial_drift / bracket_dx = 'fourier' and numerics.x_dealias need an x-periodic box "
                             "(no Krook buffers) with adiabatic electrons")
        if self.kinetic:
            from velcro import kinetic as ke
            ke.check_config(cfg, self.geom)
            self.species = ke.species_list(cfg, self.prof, self.geom.x)
            build = ke.build_kinetic_fft_solver if cfg.numerics.field_solver == "fft" else ke.build_kinetic_field_solver
            self.fields = build(self.geom, self.ky, self.species, g.M, g.L, g.Lx, cfg.physics.beta,
                                dtype=self.dtype, node_gyro=cfg.numerics.node_gyro == "exact", bpar=cfg.physics.bpar)
        elif cfg.numerics.field_solver == "fft":            # the flux-tube limit (symbols at x = 0 as FFT multipliers)
            from velcro.fields import build_fft_field_solver
            self.fields = build_fft_field_solver(self.geom, self.ky, self.prof["Ti"], self.prof["Te"], g.M, g.Lx,
                                                 node_gyro=cfg.numerics.node_gyro == "exact")
        elif cfg.numerics.field_solver == "lowrank":
            from velcro.fields_lowrank import build_lowrank_field_solver
            self.fields = build_lowrank_field_solver(self.geom, self.ky, self.prof["Ti"], self.prof["Te"], g.M, g.Lx,
                                                     dtype=self.dtype, node_gyro=cfg.numerics.node_gyro == "exact",
                                                     tol=cfg.numerics.lowrank_tol, cg_tol=cfg.numerics.cg_tol,
                                                     cg_maxiter=cfg.numerics.cg_maxiter)
        else:
            self.fields = build_field_solver(self.geom, self.ky, self.prof["Ti"], self.prof["Te"], g.M, g.Lx,
                                             dtype=self.dtype, node_gyro=cfg.numerics.node_gyro == "exact")
        self.op = build_operators(self.geom, self.prof, cfg, self.ky, self.geom.twist_phase, self.dtype)
        if self.kinetic:
            self.op = ke.build_kinetic_operators(self.op, self.geom, cfg, self.species, self.ky, self.dtype,
                                                 fields=self.fields)
            if self.op.chain_damp is not None:
                self._set_gx_damping(cfg)
        # physics.exb_shear in the shearing frame: the frame phase θ_ky(t) = k_y γ_E t is state outside G
        self.exb_theta = None
        from velcro.vlasov import exb_frame_kind
        if exb_frame_kind(cfg) == "shearing" and cfg.physics.exb_shear.gamma_E != 0.0:
            if self.kinetic:          # kinetic electrons (the validation notes): the FFT solver's symbols at k_x − θ
                if cfg.numerics.field_solver != "fft":
                    raise ValueError("physics.exb_shear with kinetic electrons needs numerics.field_solver = 'fft'")
                self.fields.frame = ke.kinetic_frame_data(self.geom, self.ky, self.species, g.Lx, cfg.physics.beta, g.M,
                                                          qn_grid=bool(cfg.physics.exb_shear.gx_qn_grid))
            if cfg.physics.ext_zonal is not None and cfg.physics.ext_zonal.amp != 0.0:
                raise ValueError("physics.exb_shear and physics.ext_zonal together are not supported")
            self.exb_theta = jnp.zeros(self.Nky, dtype=self.rdtype)
            self.exb_rate = jnp.asarray(np.asarray(self.ky) * cfg.physics.exb_shear.gamma_E, dtype=self.rdtype)
            self.exb_dkx = float(2 * np.pi / g.Lx)
        self.phi_ext = None
        ez = cfg.physics.ext_zonal
        if ez is not None and ez.amp != 0.0:
            import dataclasses
            self.phi_ext, nodes = self._external_zonal(ez)
            self.op = dataclasses.replace(self.op, ext_nodes=nodes)
        self.imex = None                    # numerics.integrator = "imex": the implicit part (velcro/imex.py)
        self.imex_rate = None               # ... and with numerics.imex_rate = "jacobian" its adaptive rate
        self.imex_fallback = None           # ... and, when its factors did not fit the device, what ran instead
        if cfg.numerics.integrator == "imex" and not self.kinetic:
            raise ValueError("numerics.integrator = 'imex' needs physics.electrons = 'kinetic' (its implicit part is the "
                             "electrons' parallel dynamics)")
        if self.kinetic:
            self._compile_kinetic()
        else:
            self._compile()
        self.diag = KineticDiagnostics() if self.kinetic else Diagnostics()
        self.t = 0.0
        self.step_count = 0
        self.G = self.initial_condition()

    def _set_gx_damping(self, cfg):
        """GX's chain-end damping rate A: gx_damp_ends, or gx_damp_amp/dt₀ (GX's amp/dt rule with Velcro's CFL step at
        t = 0, from the linear rates).  The damping acts through h, so its eigenvalues reach A·op.damp_cfl; RK4 needs
        dt·A·damp_cfl below its real-axis limit 2.78 — refused otherwise, as that grid puts damped points at small k⊥."""
        import dataclasses
        import jax.numpy as jnp
        from velcro import kinetic as ke
        num = cfg.numerics
        G0 = jnp.zeros((len(self.species) * cfg.grid.L, cfg.grid.M, cfg.grid.Nx, self.Nky, cfg.grid.Nz), dtype=self.dtype)
        rate0 = float(ke.cfl_rate_kinetic(ke.kinetic_fields(G0, self.fields, self.op, num.node_gyro == "exact"), self.op,
                                          cfg.grid.L, cfg.grid.M))
        dt0 = min(num.cfl * 2.8 / max(rate0, 1e-12), num.dt_max)
        A = num.gx_damp_amp / dt0 if num.gx_damp_amp > 0 else num.gx_damp_ends
        if num.gx_damp_amp > 0 and dt0 * A * self.op.damp_cfl > 2.5:     # (an explicit rate caps dt instead: dt_from_cfl)
            raise ValueError(f"GX end damping too stiff for this grid: dt·A·(1 + f) = {dt0 * A * self.op.damp_cfl:.2f} > 2.5 "
                             f"(A {A:.1f}, bound factor {self.op.damp_cfl:.1f}: damped chain ends at small k⊥)")
        rdt = jnp.zeros((), self.dtype).real.dtype
        self.op = dataclasses.replace(self.op, chain_damp=tuple(jnp.asarray(d * A, dtype=rdt) for d in self.op.chain_damp),
                                      damp_cfl=float(A * self.op.damp_cfl))
        self.gx_damp_rate, self.gx_damp_dt0 = A, dt0

    # ------------------------------------------------------------------
    def _profiles(self) -> dict:
        cfg, x = self.cfg, self.geom.x
        Lx = cfg.grid.Lx

        def profile(spec):
            if spec.kind == "constant":
                return spec.value * np.ones_like(x)
            if spec.kind == "peaked":
                return spec.value / np.cosh(x / spec.width) ** 2
            if spec.kind == "plateau":     # edges measured from the box boundaries ±L_x/2 (the periodic grid stops one Δx short)
                xl, xr = -0.5 * Lx + spec.edge, 0.5 * Lx - spec.edge
                return spec.value * (1.0 - 1.0 / np.cosh((x - xl) / spec.width) ** 2 - 1.0 / np.cosh((x - xr) / spec.width) ** 2)
            xl, xr = x.min() + spec.edge, x.max() - spec.edge
            return spec.value * 0.5 * (np.tanh((x - xl) / spec.width) - np.tanh((x - xr) / spec.width))

        omega_n, omega_T = profile(cfg.physics.omega_n), profile(cfg.physics.omega_T)
        rs = self.geom.rho_star

        def integrate(omega):  # f(x) = exp(-ρ* ∫ ω dx), f = 1 at x = 0
            dx = x[1] - x[0]
            c = np.concatenate([[0.0], np.cumsum(0.5 * (omega[1:] + omega[:-1]) * dx)])
            c -= np.interp(0.0, x, c)
            return np.exp(-rs * c)

        Ti = integrate(omega_T) if cfg.physics.Ti_profile else np.ones_like(x)
        n = integrate(omega_n) if cfg.physics.n_profile else np.ones_like(x)
        Te = cfg.physics.tau * Ti           # same shape as T_i (τ fixed) — adiabatic electrons
        nu = cfg.physics.nu * n / Ti**1.5
        return dict(omega_n=omega_n, omega_T=omega_T, Ti=Ti, Te=Te, n=n, nu=nu)

    def _external_zonal(self, ez):
        """``physics.ext_zonal`` → (φ_ext as a field (Nx, Nky, Nz), non-zero on the k_y = 0 row only; its gyroaverage at
        the μ nodes, (Nmu, Nx, 1, Nz)), the node values taken as the nonlinearity takes them for φ (exact J0 or the
        truncated Laguerre sum)."""
        import jax.numpy as jnp
        x, Nz = self.geom.x, len(self.geom.z)
        k = 2 * np.pi * ez.kx_index / self.cfg.grid.Lx
        phi = np.zeros((len(x), self.Nky, Nz), dtype=np.complex128)
        phi[:, 0, :] = (ez.amp * np.cos(k * x + ez.phase))[:, None]
        phi = jnp.asarray(phi, dtype=self.dtype)
        if self.cfg.numerics.node_gyro == "exact":
            nodes = self.fields.gyroaverage_nodes(phi)
        else:
            nodes = jnp.einsum("jm,m...->j...", self.op.lag_to_nodes, self.fields.gyroaverage(phi))
        return phi, nodes[:, :, :1, :]

    # ------------------------------------------------------------------
    def _compile(self):
        import jax
        import jax.numpy as jnp
        from velcro.vlasov import rhs, CFL_ESTIMATES

        fs, op = self.fields, self.op
        L, M = self.cfg.grid.L, self.cfg.grid.M
        if self.exb_theta is not None:
            return self._compile_exb_frame()

        def solve_fields(G):
            phi = fs.solve(G[0])
            Phi = fs.gyroaverage(phi)
            return phi, Phi

        exact_nodes = self.cfg.numerics.node_gyro == "exact" and op.nonlinear

        def f(G):
            phi, Phi = solve_fields(G)
            return rhs(G, phi, Phi, op, Pn=fs.gyroaverage_nodes(phi) if exact_nodes else None)

        def rk4(G, dt):
            k1 = f(G)
            k2 = f(G + 0.5 * dt * k1)
            k3 = f(G + 0.5 * dt * k2)
            k4 = f(G + dt * k3)
            return G + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        lawson = self.cfg.numerics.integrator == "lawson"
        if lawson and not op.axisymmetric:
            raise ValueError("numerics.integrator = 'lawson' needs an axisymmetric geometry (K_y independent of y)")
        from velcro.vlasov import drift_phase, apply_drift_propagator

        def lawson_rk4(G, dt, P1):
            """Integrating-factor RK4 for ∂_t G = A G + N(G), A = binormal drift (exact), N = f with A removed:
            a = N(u); b = N(E½(u + ½h a)); c = N(E½u + ½h b); d = N(E₁u + h E½c);
            u⁺ = E₁u + h/6 (E₁a + 2 E½(b + c) + d),  E½ = e^{hA/2}, E₁ = e^{hA}.
            With E₁ = E½∘E½ the four E₁ terms fold into the E½ arguments: E₁u + h E½c = E½(E½u + h c) and
            u⁺ = E½(E½u + h/6 E½a + h/3 (b + c)) + h/6 d — four propagator applications per step instead of six."""
            E = lambda X: apply_drift_propagator(X, P1, op)      # E½
            a = f(G)
            Eu, Ea = E(G), E(a)
            b = f(Eu + 0.5 * dt * Ea)
            c = f(Eu + 0.5 * dt * b)
            d = f(E(Eu + dt * c))
            return E(Eu + (dt / 6.0) * Ea + (dt / 3.0) * (b + c)) + (dt / 6.0) * d

        def nsteps(G, dt, n):
            if lawson:
                P1 = drift_phase(0.5 * dt, op).astype(G.dtype)
                return jax.lax.fori_loop(0, n, lambda i, g: lawson_rk4(g, dt, P1), G)
            return jax.lax.fori_loop(0, n, lambda i, g: rk4(g, dt), G)

        self._solve_fields = jax.jit(solve_fields)
        self._rhs = jax.jit(f)
        self._nsteps = jax.jit(nsteps)  # n is traced: fori_loop lowers to a while_loop, no recompiles
        estimate = "spectral" if lawson else self.cfg.numerics.cfl_estimate
        rate_fn = CFL_ESTIMATES[estimate]

        def rate(G):
            phi, Phi = solve_fields(G)
            if estimate == "spectral":
                return rate_fn(G, phi, Phi, op, L, M, Pn=fs.gyroaverage_nodes(phi) if exact_nodes else None)
            return rate_fn(G, phi, Phi, op, L, M)

        self._rate = jax.jit(rate)

        energy_half = jnp.asarray((op.energy.reshape(L * M, L * M) / 2.0).reshape(L, M, L, M))  # v²+μ/2 … use (v²+μ)
        from velcro import basis
        Ehalf = (np.kron(basis.hermite_v_matrix(L) @ basis.hermite_v_matrix(L), np.eye(M))
                 + np.kron(np.eye(L), basis.laguerre_mu_matrix(M))).reshape(L, M, L, M)
        Ehalf = jnp.asarray(Ehalf)
        ky = op.ky
        fsw = op.fsw

        def fluxes(G, phi, Phi):
            # radial E×B velocity of the gyroaveraged potential: v_x = −½ C⁻¹ ∂_y Φ
            vx = -op.c_Cinv * (1j * ky) * Phi                       # (M, Nx, Nky, Nz)
            E_G = jnp.einsum("lmpq,pq...->lm...", Ehalf, G)[0]      # (M, ...) energy-weighted ℓ=0 row
            w = jnp.where(jnp.arange(vx.shape[2]) == 0, 1.0, 2.0)[None, None, :, None]
            Q = jnp.sum(jnp.real(jnp.conj(vx) * E_G) * w, axis=(0, 2))      # (Nx, Nz)
            Gam = jnp.sum(jnp.real(jnp.conj(vx) * G[0]) * w, axis=(0, 2))
            Qv = jnp.sum(Q * fsw) / fsw.shape[0]
            Gv = jnp.sum(Gam * fsw) / fsw.shape[0]
            return Qv, Gv

        def diagnostics(G):
            phi, Phi = solve_fields(G)
            Qv, Gv = fluxes(G, phi, Phi)
            w = jnp.where(jnp.arange(phi.shape[1]) == 0, 1.0, 2.0)
            phi2 = jnp.sum(jnp.abs(phi) ** 2 * w[None, :, None])
            W = jnp.sum(jnp.abs(G) ** 2 * w[None, None, None, :, None])
            herm = jnp.sum(jnp.abs(G) ** 2 * w[None, None, None, :, None], axis=(1, 2, 3, 4))
            lag = jnp.sum(jnp.abs(G) ** 2 * w[None, None, None, :, None], axis=(0, 2, 3, 4))
            phi_ky = jnp.sum(jnp.abs(phi) ** 2, axis=(0, 2))
            zonal = jnp.real(jnp.sum(phi[:, 0, :] * fsw, axis=1))
            probe = phi[phi.shape[0] // 2, min(1, phi.shape[1] - 1), phi.shape[2] // 2]
            return dict(phi2=phi2, W=W, Q=Qv, Gamma=Gv, hermite=herm, laguerre=lag, phi_ky=phi_ky, phi_zonal=zonal, probe=probe)

        self._diagnostics = jax.jit(diagnostics)

    def _compile_exb_frame(self):
        """``_compile`` in the shearing frame of ``physics.exb_shear``: every field solve, right-hand side, rate and
        diagnostic takes the frame phase θ (Nky,); ``_nsteps(G, dt, n, θ) → (G, θ)`` advances θ by k_y γ_E dt per step
        (the RK stages at their own θ) and remaps the spectrum when |θ| passes Δk_x/2 (velcro.vlasov.exb_remap).  The
        wrappers ``_solve_fields``, ``_rhs``, ``_rate`` and ``_diagnostics`` read ``self.exb_theta``."""
        import jax
        import jax.numpy as jnp
        from velcro.vlasov import rhs, cfl_rates_spectral, drift_phase, apply_drift_propagator, exb_remap

        fs, op = self.fields, self.op
        L, M = self.cfg.grid.L, self.cfg.grid.M
        exact_nodes = self.cfg.numerics.node_gyro == "exact" and op.nonlinear
        rate_k, dkx = self.exb_rate, self.exb_dkx
        x0 = float(np.asarray(self.geom.x)[0])
        ex = self.cfg.physics.exb_shear
        qn_grid, step_split = bool(ex.gx_qn_grid), bool(ex.gx_step_split)      # GX's choices, opt-in

        def solve_fields(G, theta):
            phi = fs.solve(G[0], theta, theta_den=jnp.zeros_like(theta)) if qn_grid else fs.solve(G[0], theta)
            return phi, fs.gyroaverage(phi, theta)

        def f(G, theta):
            phi, Phi = solve_fields(G, theta)
            Pn = fs.gyroaverage_nodes(phi, theta) if exact_nodes else None
            return rhs(G, phi, Phi, op, Pn=Pn, theta=theta.reshape((1, -1, 1)))

        def rk4(G, dt, th, rate=rate_k):
            k1 = f(G, th)
            k2 = f(G + 0.5 * dt * k1, th + 0.5 * dt * rate)
            k3 = f(G + 0.5 * dt * k2, th + 0.5 * dt * rate)
            k4 = f(G + dt * k3, th + dt * rate)
            return G + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        lawson = self.cfg.numerics.integrator == "lawson"
        if lawson and not op.axisymmetric:
            raise ValueError("numerics.integrator = 'lawson' needs an axisymmetric geometry (K_y independent of y)")

        def lawson_rk4(G, dt, P1, th, rate=rate_k):
            E = lambda X: apply_drift_propagator(X, P1, op)
            a = f(G, th)
            Eu, Ea = E(G), E(a)
            b = f(Eu + 0.5 * dt * Ea, th + 0.5 * dt * rate)
            c = f(Eu + 0.5 * dt * b, th + 0.5 * dt * rate)
            d = f(E(Eu + dt * c), th + dt * rate)
            return E(Eu + (dt / 6.0) * Ea + (dt / 3.0) * (b + c)) + (dt / 6.0) * d

        def nsteps(G, dt, n, theta):
            P1 = drift_phase(0.5 * dt, op).astype(G.dtype) if lawson else None
            mask = op.x_mask if op.x_dealias else None

            def body(i, carry):
                g, th = carry
                g = lawson_rk4(g, dt, P1, th) if lawson else rk4(g, dt, th)
                return exb_remap(g, th + dt * rate_k, dkx, x0, mask)

            def body_split(i, carry):           # gx_step_split: shift first, then every stage at the new θ
                g, th = exb_remap(carry[0], carry[1] + dt * rate_k, dkx, x0, mask)
                z = jnp.zeros_like(rate_k)
                return (lawson_rk4(g, dt, P1, th, z) if lawson else rk4(g, dt, th, z)), th
            return jax.lax.fori_loop(0, n, body_split if step_split else body, (G, theta))

        sf = jax.jit(solve_fields)
        fj = jax.jit(f)
        self._sf_theta = sf
        self._solve_fields = lambda G: sf(G, self.exb_theta)
        self._rhs = lambda G: fj(G, self.exb_theta)
        self._nsteps = jax.jit(nsteps)
        self.rhs_fn = f

        def rate(G, theta):
            phi, Phi = solve_fields(G, theta)
            Pn = fs.gyroaverage_nodes(phi, theta) if exact_nodes else None
            return cfl_rates_spectral(G, phi, Phi, op, L, M, Pn=Pn)
        rj = jax.jit(rate)
        self._rate = lambda G: rj(G, self.exb_theta)
        from velcro import basis
        Ehalf = jnp.asarray((np.kron(basis.hermite_v_matrix(L) @ basis.hermite_v_matrix(L), np.eye(M))
                             + np.kron(np.eye(L), basis.laguerre_mu_matrix(M))).reshape(L, M, L, M))
        ky, fsw = op.ky, op.fsw

        def diagnostics(G, theta):
            phi, Phi = solve_fields(G, theta)
            vx = -op.c_Cinv * (1j * ky) * Phi
            E_G = jnp.einsum("lmpq,pq...->lm...", Ehalf, G)[0]
            w = jnp.where(jnp.arange(vx.shape[2]) == 0, 1.0, 2.0)[None, None, :, None]
            Q = jnp.sum(jnp.real(jnp.conj(vx) * E_G) * w, axis=(0, 2))
            Gam = jnp.sum(jnp.real(jnp.conj(vx) * G[0]) * w, axis=(0, 2))
            Qv, Gv = jnp.sum(Q * fsw) / fsw.shape[0], jnp.sum(Gam * fsw) / fsw.shape[0]
            w1 = jnp.where(jnp.arange(phi.shape[1]) == 0, 1.0, 2.0)
            phi2 = jnp.sum(jnp.abs(phi) ** 2 * w1[None, :, None])
            W = jnp.sum(jnp.abs(G) ** 2 * w1[None, None, None, :, None])
            herm = jnp.sum(jnp.abs(G) ** 2 * w1[None, None, None, :, None], axis=(1, 2, 3, 4))
            lag = jnp.sum(jnp.abs(G) ** 2 * w1[None, None, None, :, None], axis=(0, 2, 3, 4))
            phi_ky = jnp.sum(jnp.abs(phi) ** 2, axis=(0, 2))
            zonal = jnp.real(jnp.sum(phi[:, 0, :] * fsw, axis=1))
            probe = phi[phi.shape[0] // 2, min(1, phi.shape[1] - 1), phi.shape[2] // 2]
            return dict(phi2=phi2, W=W, Q=Qv, Gamma=Gv, hermite=herm, laguerre=lag, phi_ky=phi_ky, phi_zonal=zonal, probe=probe)
        dj = jax.jit(diagnostics)
        self._diagnostics = lambda G: dj(G, self.exb_theta)

    def _compile_kinetic(self):
        """``_compile`` for kinetic electrons (velcro/kinetic.py): the same RK4 / Lawson RK4 steppers, with the binormal
        drift of every species in the Lawson factor, and per-species diagnostics.  ``rhs_fn`` is the (unjitted)
        right-hand side, ``_fields`` returns (φ, A∥, χ, node values), ``_solve_fields`` (φ, the ions' Φ)."""
        import jax
        import jax.numpy as jnp
        from velcro import basis
        from velcro import kinetic as ke

        if self.cfg.numerics.integrator == "imex":     # before any step is built: may turn the run into a Lawson one
            self._imex_prepare(frame=self.exb_theta is not None)
        fs, op = self.fields, self.op
        L, M = self.cfg.grid.L, self.cfg.grid.M
        exact = self.cfg.numerics.node_gyro == "exact"
        lawson = self.cfg.numerics.integrator == "lawson"
        if self.exb_theta is not None:
            return self._compile_kinetic_frame()

        def fields(G):
            return ke.kinetic_fields(G, fs, op, exact)

        def f(G):
            return ke.rhs_kinetic(G, fields(G), op)

        def rk4(G, dt):
            k1 = f(G)
            k2 = f(G + 0.5 * dt * k1)
            k3 = f(G + 0.5 * dt * k2)
            k4 = f(G + dt * k3)
            return G + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def lawson_rk4(G, dt, P1):        # Simulation._compile's lawson_rk4 with the per-species propagator
            E = lambda X: op.propagate(X, P1)
            a = f(G)
            Eu, Ea = E(G), E(a)
            b = f(Eu + 0.5 * dt * Ea)
            c = f(Eu + 0.5 * dt * b)
            d = f(E(Eu + dt * c))
            return E(Eu + (dt / 6.0) * Ea + (dt / 3.0) * (b + c)) + (dt / 6.0) * d

        def nsteps(G, dt, n):
            if lawson:
                P1 = op.drift_phase(0.5 * dt).astype(G.dtype)
                return jax.lax.fori_loop(0, n, lambda i, g: lawson_rk4(g, dt, P1), G)
            return jax.lax.fori_loop(0, n, lambda i, g: rk4(g, dt), G)

        def solve_fields(G):
            phi, _ = fs.solve(G)
            return phi, fs.gyroaverage(phi)[0]

        self.rhs_fn = f
        self._fields = jax.jit(fields)
        self._solve_fields = jax.jit(solve_fields)
        self._rhs = jax.jit(f)
        self._nsteps = jax.jit(nsteps)
        self._rate = jax.jit(lambda G: ke.cfl_rate_kinetic(fields(G), op, L, M))
        if self.cfg.numerics.integrator == "imex":
            self._setup_imex(fields, lambda G, F: ke.rhs_kinetic(G, F, op))
        V = basis.hermite_v_matrix(L)
        Ehalf = jnp.asarray((np.kron(V @ V, np.eye(M)) + np.kron(np.eye(L), basis.laguerre_mu_matrix(M))).reshape(L, M, L, M))
        fsw = op.fsw

        def diagnostics(G):
            F = fields(G)
            phi, apar = F[0], F[1]
            Q, Gam = ke.kinetic_fluxes(G, F, op, Ehalf, fs=fs if self.cfg.physics.bpar else None)
            w = jnp.where(jnp.arange(phi.shape[1]) == 0, 1.0, 2.0)
            wG = w[None, None, None, :, None]
            phi2 = jnp.sum(jnp.abs(phi) ** 2 * w[None, :, None])
            apar2 = jnp.zeros((), phi2.dtype) if apar is None else jnp.sum(jnp.abs(apar) ** 2 * w[None, :, None])
            nT = jnp.repeat(op.dens_temp, L)[:, None, None, None, None]
            W = jnp.sum(nT * jnp.abs(G) ** 2 * wG)
            herm = jnp.sum(jnp.abs(G) ** 2 * wG, axis=(1, 2, 3, 4))
            lag = jnp.sum(jnp.abs(G) ** 2 * wG, axis=(0, 2, 3, 4))
            phi_ky = jnp.sum(jnp.abs(phi) ** 2, axis=(0, 2))
            zonal = jnp.real(jnp.sum(phi[:, 0, :] * fsw, axis=1))
            probe = phi[phi.shape[0] // 2, min(1, phi.shape[1] - 1), phi.shape[2] // 2]
            return dict(phi2=phi2, apar2=apar2, W=W, Q=Q[0, 0] + Q[0, 1] + Q[0, 2], Qi_es=Q[0, 0], Qi_em=Q[0, 1],
                        Qe=Q[1, 0] + Q[1, 1] + Q[1, 2], Qe_es=Q[1, 0], Qe_em=Q[1, 1], Gamma=Gam[0, 0] + Gam[0, 1] + Gam[0, 2],
                        Gamma_e=Gam[1, 0] + Gam[1, 1] + Gam[1, 2], Qi_bpar=Q[0, 2], Qe_bpar=Q[1, 2],
                        hermite=herm, laguerre=lag, phi_ky=phi_ky, phi_zonal=zonal, probe=probe)

        self._diagnostics = jax.jit(diagnostics)

    def _kinetic_diagnostics(self, F, G):
        """The kinetic diagnostics dict of fields ``F`` (from ``kinetic_fields``) and the state G (traced)."""
        import jax.numpy as jnp
        from velcro import basis
        from velcro import kinetic as ke
        op, fs = self.op, self.fields
        L, M = self.cfg.grid.L, self.cfg.grid.M
        V = basis.hermite_v_matrix(L)
        Ehalf = jnp.asarray((np.kron(V @ V, np.eye(M)) + np.kron(np.eye(L), basis.laguerre_mu_matrix(M))).reshape(L, M, L, M))
        phi, apar = F[0], F[1]
        Q, Gam = ke.kinetic_fluxes(G, F, op, Ehalf, fs=fs if self.cfg.physics.bpar else None)
        w = jnp.where(jnp.arange(phi.shape[1]) == 0, 1.0, 2.0)
        wG = w[None, None, None, :, None]
        phi2 = jnp.sum(jnp.abs(phi) ** 2 * w[None, :, None])
        apar2 = jnp.zeros((), phi2.dtype) if apar is None else jnp.sum(jnp.abs(apar) ** 2 * w[None, :, None])
        nT = jnp.repeat(op.dens_temp, L)[:, None, None, None, None]
        fsw = op.fsw
        return dict(phi2=phi2, apar2=apar2, W=jnp.sum(nT * jnp.abs(G) ** 2 * wG), Q=Q[0, 0] + Q[0, 1] + Q[0, 2],
                    Qi_es=Q[0, 0], Qi_em=Q[0, 1], Qe=Q[1, 0] + Q[1, 1] + Q[1, 2], Qe_es=Q[1, 0], Qe_em=Q[1, 1],
                    Gamma=Gam[0, 0] + Gam[0, 1] + Gam[0, 2], Gamma_e=Gam[1, 0] + Gam[1, 1] + Gam[1, 2], Qi_bpar=Q[0, 2],
                    Qe_bpar=Q[1, 2], hermite=jnp.sum(jnp.abs(G) ** 2 * wG, axis=(1, 2, 3, 4)),
                    laguerre=jnp.sum(jnp.abs(G) ** 2 * wG, axis=(0, 2, 3, 4)), phi_ky=jnp.sum(jnp.abs(phi) ** 2, axis=(0, 2)),
                    phi_zonal=jnp.real(jnp.sum(phi[:, 0, :] * fsw, axis=1)),
                    probe=phi[phi.shape[0] // 2, min(1, phi.shape[1] - 1), phi.shape[2] // 2])

    def _compile_kinetic_frame(self):
        """``_compile_kinetic`` in the shearing frame of ``physics.exb_shear`` (the validation notes), the kinetic version
        of :meth:`_compile_exb_frame`: the fields, right-hand side, rate and diagnostics take the frame phase θ (N_ky,),
        shared by the species; ``_nsteps(G, dt, n, θ) → (G, θ)`` advances θ by k_y γ_E dt per step (every RK / Lawson-RK
        stage at its own θ, or GX's split with ``gx_step_split``) and remaps both species' spectra when |θ| passes Δk_x/2
        (velcro.vlasov.exb_remap on the stacked state; the x_dealias mask drops the mode leaving the band).  The
        wrappers ``_fields``, ``_solve_fields``, ``_rhs``, ``_rate`` and ``_diagnostics`` read ``self.exb_theta``."""
        import jax
        import jax.numpy as jnp
        from velcro import kinetic as ke
        from velcro.vlasov import exb_remap

        fs, op = self.fields, self.op
        L, M = self.cfg.grid.L, self.cfg.grid.M
        exact = self.cfg.numerics.node_gyro == "exact"
        lawson = self.cfg.numerics.integrator == "lawson"
        rate_k, dkx = self.exb_rate, self.exb_dkx
        x0 = float(np.asarray(self.geom.x)[0])
        step_split = bool(self.cfg.physics.exb_shear.gx_step_split)

        def fields(G, theta):
            return ke.kinetic_fields(G, fs, op, exact, theta=theta)

        def f(G, theta):
            return ke.rhs_kinetic(G, fields(G, theta), op, theta=theta)

        def rk4(G, dt, th, rate=rate_k):
            k1 = f(G, th)
            k2 = f(G + 0.5 * dt * k1, th + 0.5 * dt * rate)
            k3 = f(G + 0.5 * dt * k2, th + 0.5 * dt * rate)
            k4 = f(G + dt * k3, th + dt * rate)
            return G + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def lawson_rk4(G, dt, P1, th, rate=rate_k):
            E = lambda X: op.propagate(X, P1)
            a = f(G, th)
            Eu, Ea = E(G), E(a)
            b = f(Eu + 0.5 * dt * Ea, th + 0.5 * dt * rate)
            c = f(Eu + 0.5 * dt * b, th + 0.5 * dt * rate)
            d = f(E(Eu + dt * c), th + dt * rate)
            return E(Eu + (dt / 6.0) * Ea + (dt / 3.0) * (b + c)) + (dt / 6.0) * d

        def nsteps(G, dt, n, theta):
            P1 = op.drift_phase(0.5 * dt).astype(G.dtype) if lawson else None
            mask = op.x_mask if op.x_dealias else None

            def body(i, carry):
                g, th = carry
                g = lawson_rk4(g, dt, P1, th) if lawson else rk4(g, dt, th)
                return exb_remap(g, th + dt * rate_k, dkx, x0, mask)

            def body_split(i, carry):           # gx_step_split: shift first, then every stage at the new θ
                g, th = exb_remap(carry[0], carry[1] + dt * rate_k, dkx, x0, mask)
                z = jnp.zeros_like(rate_k)
                return (lawson_rk4(g, dt, P1, th, z) if lawson else rk4(g, dt, th, z)), th
            return jax.lax.fori_loop(0, n, body_split if step_split else body, (G, theta))

        def solve_fields(G, theta):
            F = fields(G, theta)
            return F[0], F[2][0, 0]                      # φ and the ions' ℓ = 0 row of χ (= 𝒥_m(b_i*) φ without δB∥)

        fj, sf, rj = jax.jit(fields), jax.jit(solve_fields), jax.jit(f)
        self.rhs_fn = f
        self._fields_theta = fj
        self._sf_theta = sf
        self._fields = lambda G: fj(G, self.exb_theta)
        self._solve_fields = lambda G: sf(G, self.exb_theta)
        self._rhs = lambda G: rj(G, self.exb_theta)
        self._nsteps = jax.jit(nsteps)
        ratej = jax.jit(lambda G, theta: ke.cfl_rate_kinetic(fields(G, theta), op, L, M))
        self._rate = lambda G: ratej(G, self.exb_theta)
        dj = jax.jit(lambda G, theta: self._kinetic_diagnostics(fields(G, theta), G))
        self._diagnostics = lambda G: dj(G, self.exb_theta)
        if self.cfg.numerics.integrator == "imex":
            self._setup_imex(fields, lambda G, F, theta: ke.rhs_kinetic(G, F, op, theta=theta), frame=True)

    def _imex_prepare(self, frame: bool):
        """``numerics.integrator = "imex"``: the implicit part and its factor plan, before any step is built.  When the
        device cannot hold the factors (``ImexSystem.plan``: dense, then block-Thomas), ``numerics.imex_fallback =
        "lawson"`` turns the run into a Lawson RK4 run (RK4 on a geometry Lawson cannot take) with a warning stating the
        sizes, and ``"error"`` refuses it (MemoryError)."""
        import dataclasses
        from velcro import imex as im
        num = self.cfg.numerics
        imp = im.ImexSystem(self, im.tableau(num.imex_scheme), frame=frame)
        if imp.kind is not None:
            self._imex_pending = imp                    # _setup_imex takes it
            return
        msg = imp.plan_report()
        if num.imex_fallback == "error":
            raise MemoryError(f"{msg} (numerics.imex_fallback = 'error')")
        to = "lawson" if self.op.axisymmetric else "rk4"
        print(f"WARNING: {msg}; running numerics.integrator = '{to}' instead (numerics.imex_fallback = 'lawson'; "
              f"'error' refuses)", flush=True)
        self.imex_fallback = dict(integrator=to, free_bytes=imp.plan["free"], report=msg,
                                  need_bytes={k: n["total"] for k, n in imp.plan["tried"]})
        cfg = self.cfg.model_copy(deep=True)
        cfg.numerics.integrator = to
        self.cfg = cfg
        if bool(self.op.base.split_drift_ky) != (to == "lawson"):     # imex_lawson off: Lawson splits the drift out
            self.op = dataclasses.replace(self.op, base=dataclasses.replace(self.op.base, split_drift_ky=to == "lawson"))

    def _setup_imex(self, fields, f_full, frame: bool = False):
        """``numerics.integrator = "imex"`` (velcro/imex.py): the implicit part, its step, ``_nsteps`` (the factors of a
        step size cached outside the compiled loop, so a jvp of the step reuses them) and the explicit part's rate.
        ``fields(G[, θ])`` / ``f_full(G, F[, θ])`` are the compile's kinetic_fields and right-hand side."""
        import jax
        import jax.numpy as jnp
        from velcro import imex as im
        num = self.cfg.numerics
        if num.imex_lawson and not self.op.axisymmetric:
            raise ValueError("numerics.imex_lawson needs an axisymmetric geometry (K_y independent of y)")
        self.imex = getattr(self, "_imex_pending", None) or im.ImexSystem(self, im.tableau(num.imex_scheme), frame=frame)
        self._imex_pending = None
        step = im.make_step(self, self.imex, f_full, fields, frame=frame)
        self._imex_step = step
        self._imex_last_dt = None

        def note(dt):                               # the step last taken: dt_from_cfl keeps it while it is still stable
            try:
                self._imex_last_dt = float(dt)
            except Exception:                       # traced (a step inside another jit): nothing to remember
                pass
        self._imex_note = note
        self.imex_rate = None                       # numerics.imex_rate = "jacobian": the adaptive rate (imex.JacobianRate)
        if not frame:
            inner = jax.jit(lambda G, dt, n, fac: jax.lax.fori_loop(0, n, lambda i, g: step(g, dt, fac), G))

            def nsteps(G, dt, n):
                fac = self.imex.factors(dt)
                note(dt)
                return inner(G, dt, n, fac)
            self._nsteps = nsteps
            bound = jax.jit(lambda G: self.imex.explicit_rate(fields(G)))
            self._rate = bound
            if num.imex_rate == "jacobian":
                self.imex_rate = im.JacobianRate(self, self.imex, f_full, fields, lambda G, th: bound(G))
                self._rate = self.imex_rate
            return
        from velcro.vlasov import exb_remap
        rate_k, dkx = self.exb_rate, self.exb_dkx
        x0 = float(np.asarray(self.geom.x)[0])
        mask = self.op.x_mask if self.op.x_dealias else None
        split = bool(self.cfg.physics.exb_shear.gx_step_split)

        def body(carry, dt, fac):
            g, th = carry
            if split:                               # gx_step_split: shift first, then every stage at the new θ
                g, th = exb_remap(g, th + dt * rate_k, dkx, x0, mask)
                return step(g, dt, fac, th, jnp.zeros_like(rate_k)), th
            return exb_remap(step(g, dt, fac, th, rate_k), th + dt * rate_k, dkx, x0, mask)
        inner = jax.jit(lambda G, dt, n, theta, fac: jax.lax.fori_loop(0, n, lambda i, c: body(c, dt, fac), (G, theta)))

        def nsteps(G, dt, n, theta):
            fac = self.imex.factors(dt, frame=True)
            note(dt)
            return inner(G, dt, n, theta, fac)
        self._nsteps = nsteps
        ratej = jax.jit(lambda G, theta: self.imex.explicit_rate(fields(G, theta)))
        self._rate = lambda G: ratej(G, self.exb_theta)
        if num.imex_rate == "jacobian":
            self.imex_rate = im.JacobianRate(self, self.imex, f_full, fields, ratej, frame=True)
            self._rate = lambda G: self.imex_rate(G, self.exb_theta)

    # ------------------------------------------------------------------
    def initial_condition(self):
        import jax.numpy as jnp
        if getattr(self, "kinetic", False):
            return self._initial_condition_kinetic()
        G0 = self._initial_ions()
        if self.cfg.numerics.x_dealias:             # the 2/3 rule's band, as the kinetic path (nothing beyond it is evolved)
            from velcro.operators import x_dealias_mask
            G0 = np.fft.ifft(np.fft.fft(G0, axis=-3) * x_dealias_mask(G0.shape[-3])[:, None, None], axis=-3)
        return jnp.asarray(G0, dtype=self.dtype)

    def _initial_ions(self) -> np.ndarray:
        """The ions' (or the only species') initial G, (L, M, Nx, Nky, Nz) complex128."""
        cfg = self.cfg
        L, M = cfg.grid.L, cfg.grid.M
        Nx, Nz = len(self.geom.x), len(self.geom.z)
        shape = (L, M, Nx, self.Nky, Nz)
        rng = np.random.default_rng(cfg.init.seed)
        G = np.zeros(shape, dtype=np.complex128)
        x, z = self.geom.x, self.geom.z
        Lx = cfg.grid.Lx
        zprof = (1 + np.cos(z)) / 2 if cfg.init.z_profile == "ballooning" else np.ones_like(z)
        # x envelope that vanishes in the buffers
        wb = cfg.numerics.krook_width * Lx
        env = np.ones_like(x)
        if wb > 0:
            d = np.minimum(x - x.min(), x.max() - x)
            env = 0.5 * (1 - np.cos(np.pi * np.clip(d / wb, 0, 1)))
        a = cfg.init.amplitude
        ic = cfg.init
        if ic.kind == "noise":
            noise = rng.normal(size=shape[2:]) + 1j * rng.normal(size=shape[2:])
            G[0, 0] = a * noise * env[:, None, None] * zprof[None, None, :]
            if L > 1:
                G[1, 0] = 0.3 * a * (rng.normal(size=shape[2:]) + 1j * rng.normal(size=shape[2:])) * env[:, None, None] * zprof
        elif ic.kind in ("mode", "landau"):
            kx = 2 * np.pi * ic.kx_index / Lx
            G[ic.l, ic.m, :, ic.ky_index, :] = a * np.exp(1j * kx * x)[:, None] * env[:, None] * zprof[None, :]
            if ic.kind == "landau":  # single parallel mode kz=1 on ky mode, sine in z
                G[ic.l, ic.m, :, ic.ky_index, :] = a * env[:, None] * np.exp(1j * (kx * x[:, None] + z[None, :]))
        elif ic.kind == "zonal":  # Rosenbluth–Hinton: k_y = 0 density perturbation with radial wavenumber
            kx = 2 * np.pi * ic.kx_index / Lx
            G[0, 0, :, 0, :] = a * np.cos(kx * x)[:, None] * env[:, None]
        return G

    def _initial_condition_kinetic(self):
        """Ions as :meth:`_initial_ions` (the same draws as an adiabatic run of the same seed); ``noise`` puts an
        independent density noise on each other species, every other kind the ions' moments (GX initialises every species
        alike: Gate 3's zonal test starts with the same gyrocentre density in both).  The x mean of the k_y = 0 row (not
        evolved) is removed."""
        import jax.numpy as jnp
        from velcro.kinetic import no_gauge_state
        cfg, ic = self.cfg, self.cfg.init
        Gi = self._initial_ions()
        blocks = [Gi]
        x, z, Lx = self.geom.x, self.geom.z, cfg.grid.Lx
        zprof = (1 + np.cos(z)) / 2 if ic.z_profile == "ballooning" else np.ones_like(z)
        wb = cfg.numerics.krook_width * Lx
        env = np.ones_like(x)
        if wb > 0:
            d = np.minimum(x - x.min(), x.max() - x)
            env = 0.5 * (1 - np.cos(np.pi * np.clip(d / wb, 0, 1)))
        for s in range(1, len(self.species)):
            Ge = np.zeros_like(Gi) if ic.kind == "noise" else Gi.copy()
            if ic.kind == "noise":
                rng = np.random.default_rng([ic.seed, s])
                shp = Gi.shape[2:]
                Ge[0, 0] = ic.amplitude * (rng.normal(size=shp) + 1j * rng.normal(size=shp)) * env[:, None, None] * zprof
            blocks.append(Ge)
        G0 = np.concatenate(blocks, axis=0)
        if cfg.numerics.x_dealias:
            from velcro.kinetic import x_dealias_mask
            G0 = np.fft.ifft(np.fft.fft(G0, axis=-3) * x_dealias_mask(G0.shape[-3])[:, None, None], axis=-3)
        if cfg.numerics.zonal_nyquist_project and G0.shape[-1] % 2 == 0:
            s = (-1.0) ** np.arange(G0.shape[-1])
            G0[..., 0, :] -= np.mean(G0[..., 0, :] * s, axis=-1, keepdims=True) * s
        return no_gauge_state(jnp.asarray(G0, dtype=self.dtype))

    # ------------------------------------------------------------------
    def dt_from_cfl(self, G) -> float:
        rate = float(self._rate(G))
        imex = getattr(self, "imex", None)
        # integrator "imex": the explicit table's stability interval, the explicit part's rate (velcro/imex.py)
        dt = self.cfg.numerics.cfl * (2.8 if imex is None else imex.tab.erk_bound) / max(rate, 1e-12)
        dt = min(dt, self.cfg.numerics.dt_max)
        if getattr(self.op, "damp_cfl", 0.0) > 0 and imex is None:   # GX's end damping (through h): its real eigenvalues
            dt = min(dt, 2.5 / self.op.damp_cfl)                       # ≤ A(1 + f) (implicit with "imex")
        if getattr(self, "kinetic", False) and self.exb_theta is not None:
            # the frame's remap moves θ by one Δk_x per step at most: the fastest row may not advance by more than that or
            # θ walks off the band silently (on the adiabatic path the caller caps the step)
            rate = float(np.max(np.abs(np.asarray(self.exb_rate))))
            if rate > 0:
                dt = min(dt, 0.9 * self.exb_dkx / rate)
        if imex is not None:
            r = self.cfg.numerics.imex_dt_levels
            if r > 0:                   # the largest level dt_max·r^k at or below the bound: a few factored step sizes
                k = max(0, int(np.ceil(np.log(dt / self.cfg.numerics.dt_max) / np.log(r) - 1e-9)))
                dt = self.cfg.numerics.dt_max * r ** k
            else:                       # keep the factored step while it stays within [imex_dt_stick, 1] × the bound
                last = self._imex_last_dt
                if last is not None and self.cfg.numerics.imex_dt_stick * dt <= last <= dt * (1 + 1e-12):
                    dt = last
        return float(dt)

    def record(self):
        d = self._diagnostics(self.G)
        self.diag.t.append(self.t)
        for k, v in d.items():
            getattr(self.diag, k).append(np.asarray(v))
        self.diag.dt.append(self._dt)

    def run(self, t_end: Optional[float] = None, verbose: bool = True):
        cfg = self.cfg
        t_end = cfg.run.t_end if t_end is None else t_end
        self._dt = cfg.numerics.dt or self.dt_from_cfl(self.G)
        self.record()
        t0 = time.time()
        next_diag = self.t + cfg.run.diag_every
        while self.t < t_end - 1e-12 and self.step_count < cfg.run.max_steps:
            if cfg.numerics.dt is None and self.step_count % cfg.numerics.steps_per_dt_update == 0:
                self._dt = self.dt_from_cfl(self.G)
            # take steps up to the next diagnostic time
            n = max(1, int(np.floor((min(next_diag, t_end) - self.t) / self._dt)))
            n = min(n, cfg.numerics.steps_per_dt_update)
            if self.exb_theta is not None:
                self.G, self.exb_theta = self._nsteps(self.G, self._dt, n, self.exb_theta)
            else:
                self.G = self._nsteps(self.G, self._dt, n)
            self.t += n * self._dt
            self.step_count += n
            if self.t >= next_diag - 1e-9:
                self.record()
                next_diag += cfg.run.diag_every
                if verbose:
                    print(f"t={self.t:9.3f} dt={self._dt:.3e} |φ|²={float(self.diag.phi2[-1]):.4e} "
                          f"Q={float(self.diag.Q[-1]):.4e}  [{time.time()-t0:6.1f}s]", flush=True)
            if not np.isfinite(float(self.diag.phi2[-1])):
                raise FloatingPointError("simulation diverged")
        if cfg.run.output:
            self.save(cfg.run.output)
        return self.diag

    def solve_fields_at(self, G, theta=None):
        """(φ, Φ) of G in the shearing-frame phase ``theta`` (Nky,) — ignored without the frame.  For tools that jit
        their own functions of G: ``_solve_fields`` reads ``self.exb_theta`` when it is called, so inside a foreign
        jit it would freeze θ at its value at trace time; pass θ as an argument of the jitted function instead."""
        if self.exb_theta is None:
            return self._solve_fields(G)
        if theta is None:
            raise ValueError("the E×B shearing frame: pass the frame phase theta (sim.exb_theta at call time)")
        return self._sf_theta(G, theta)

    def save(self, path: str):
        extra = {} if self.exb_theta is None else dict(exb_theta=np.asarray(self.exb_theta))
        diag = {("diag_t" if k == "t" else k): v for k, v in self.diag.asdict().items()}   # t is the state's time
        np.savez(path, G=np.asarray(self.G), t=self.t, x=self.geom.x, z=self.geom.z, ky=self.ky,
                 config=self.cfg.model_dump_json(), **extra, **diag)

    def load_state(self, state):
        """Continue from a saved state (a path or a loaded npz/dict with 'G'): G and, in the shearing frame, the phase θ_ky
        it is the representation in (``save`` writes both).  A state without θ is a physical
        field (θ = 0, e.g. an unsheared spin-up).  A frame state with θ ≠ 0 holds e^{+iθx} times the physical field, so
        it cannot continue without the frame — refused rather than silently shifted in k_x."""
        import jax.numpy as jnp
        st = np.load(state) if isinstance(state, (str, bytes)) or hasattr(state, "__fspath__") else state
        keys = st.files if hasattr(st, "files") else st.keys()
        G = np.asarray(st["G"])
        if G.shape != tuple(self.G.shape):
            raise ValueError(f"state G has shape {G.shape}, this run's is {tuple(self.G.shape)}")
        theta = np.asarray(st["exb_theta"], dtype=float) if "exb_theta" in keys else None
        if self.exb_theta is None:
            if theta is not None and np.any(theta != 0.0):
                raise ValueError("the state is in the E×B shearing frame (exb_theta ≠ 0): continue it with the same "
                                 "physics.exb_shear in the shearing frame — without the frame its G is not the physical "
                                 "field")
        else:
            if theta is not None and theta.shape != tuple(self.exb_theta.shape):
                raise ValueError(f"state exb_theta has shape {theta.shape}, this run's is {tuple(self.exb_theta.shape)}")
            self.exb_theta = jnp.zeros_like(self.exb_theta) if theta is None else jnp.asarray(theta,
                                                                                               dtype=self.rdtype)
        self.G = jnp.asarray(G, dtype=self.dtype)

    # ------------------------------------------------------------------
    def solver_info(self) -> dict:
        """The time integrator this run actually uses, for its outputs: ``integrator`` after any out-of-memory fallback
        (``numerics.imex_fallback``; then ``fallback_from`` 'imex' and the sizes), and with the IMEX the factorisation
        ``numerics.imex_solver = "auto"`` chose (``imex_solver`` 'dense' / 'thomas', 'mixed' when a tool set groups by
        hand), its per-group ``imex_kinds`` and the step rule.  :meth:`pinned_numerics` turns it into the overrides that
        reproduce it."""
        num = self.cfg.numerics
        out = dict(integrator=str(num.integrator))
        imp = getattr(self, "imex", None)
        if imp is not None and imp.kind is not None:
            kinds = sorted(set(imp.kind))
            out.update(imex_solver=kinds[0] if len(kinds) == 1 else "mixed", imex_kinds=list(imp.kind),
                       imex_scheme=str(num.imex_scheme), imex_rate=str(num.imex_rate),
                       imex_factor_gib=float(imp.kinv_bytes) / 2 ** 30)
            if getattr(imp, "half_factors", False):     # only when set, so the validated runs' records are unchanged
                out["imex_factor_storage"] = "bf16"
        if getattr(self, "imex_fallback", None):
            fb = self.imex_fallback
            out.update(fallback_from="imex", fallback_free_bytes=fb.get("free_bytes"), fallback_need_bytes=fb.get("need_bytes"))
        return out

    def pinned_numerics(self) -> dict:
        """``numerics`` overrides that make a run from this one's state take the same path — the integrator that ran and,
        with the IMEX, the factorisation 'auto' chose — so a continuation on a device with less free
        memory cannot silently change it: with the IMEX pinned, ``imex_fallback`` 'error' refuses instead of falling back."""
        info = self.solver_info()
        pin = dict(integrator=info["integrator"])
        if info["integrator"] == "imex" and info.get("imex_solver") in ("dense", "thomas"):
            pin.update(imex_solver=info["imex_solver"], imex_fallback="error")
        return pin

    # ------------------------------------------------------------------
    def growth_rate(self, tmin_frac: float = 0.5):
        """Fit γ from the second half of the |φ|² time series (linear runs)."""
        t = np.asarray(self.diag.t); p = np.asarray(self.diag.phi2)
        sel = t >= tmin_frac * t[-1]
        slope = np.polyfit(t[sel], np.log(p[sel]), 1)[0]
        return 0.5 * slope

    def frequency(self, tmin_frac: float = 0.5, ky_index: int = 1):
        """Real frequency from the phase of φ at the outboard midplane (linear runs)."""
        raise NotImplementedError("use probe series from run_linear_probe")
