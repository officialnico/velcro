"""Implicit–explicit (IMEX) Runge–Kutta time stepping for kinetic electrons (``numerics.integrator = "imex"``).

In brief:

* **Partition.**  The implicit part f_I is the stiff linear electron dynamics along the field line — the electrons' parallel
  streaming and mirror force (on h = G + (Z/T)χ, so with E∥ and the whole field response: quasineutrality, Ampère and
  δB∥), their GX hypercollisions, and GX's absorbing chain ends of both species (on h, so with the field response).  The
  explicit part is the rest of the run's own right-hand side, f_E = f − f_I: the E×B and flutter nonlinearity, the
  drifts, the drive, collisions, hyperviscosity, the ions' parallel dynamics, and the tiny remainder of the parallel
  operator's radial variation (f_I takes the reference surface's coefficients, as the flux-tube FFT field solver does).
  With ``numerics.imex_lawson`` (default) the binormal drift of G stays exact, as in the Lawson RK4 integrator.
* **Scheme.**  ARK3(2)4L[2]SA of Kennedy & Carpenter (2003): third order, an L-stable, stiffly accurate ESDIRK implicit
  table (γ = 0.4359, three implicit stages, one matrix per step size) paired with a four-stage explicit table whose
  stability interval is 2.48 on the imaginary axis and 3.66 on the real one (RK4: 2.83 / 2.79).  With the Lawson factor
  E(τ) = e^{τL} of the binormal drift L the stages are G_i = E(c_i h)G_n + h Σ_j E((c_i − c_j)h)(a^E_ij f_E + a^I_ij f_I)(G_j)
  (the ARK applied to the integrating-factor variable), so each implicit stage solves (I − hγ f_I) G_i = r_i.
  ``numerics.imex_scheme`` also offers ARS(2,3,2) and ARS(4,4,3) (Ascher, Ruuth & Spiteri 1997).
* **Linear algebra.**  f_I is local in (k_x, k_y) along GX's linked chains (or along each untwisted field line), so each
  chain is an independent system.  Without the field response the electrons' chain operator depends only on the chain's
  length, its k_y class (Nyquist entry, end damping) — not on the chain's k_x — so one dense inverse
  K_g⁻¹ = (I − hγ Π A₀)⁻¹ per group of chains of the same kind serves all of them (assembled and inverted on the GPU,
  cached per step size).  The field response is rank F ≤ 3 per point (φ, A∥, δB∥/B) and enters by the
  Woodbury identity: with A_I = A₀(1 + CQ) (Q: moments → fields, C: fields → (Z/T)χ),
      X = y + (K⁻¹C − C) w,   y = K⁻¹R,   (I + Σ_s Q_s(C_s − K_s⁻¹C_s)) w = Q y,
  where the capacitance matrix (F·N·N_z square per chain, a few hundred) is LU-factored per chain.  The ions' implicit
  operator is their end damping alone (diagonal).  In the shearing frame of physics.exb_shear the field symbols move with
  θ, so the capacitance is rebuilt at each stage from the cached K⁻¹ (K itself does not depend on θ).
* **Block-Thomas (``numerics.imex_solver``).**  K is block-tridiagonal in the Hermite index ℓ: streaming (V) and the
  mirror force couple ℓ ± 1 only; hypercollisions, end damping and the zonal projection are diagonal in ℓ.  Its blocks
  are (M·nz)² (dense in ζ: the chain derivative is spectral).  Eliminating from ℓ = L − 1 down,
  S_{L−1} = D_{L−1}, S_ℓ = D_ℓ − B_ℓ S_{ℓ+1}⁻¹ A_{ℓ+1} (A_ℓ, B_ℓ the blocks below and above the diagonal), and keeping
  P_ℓ = S_ℓ⁻¹ costs L·(M·nz)² instead of (L·M·nz)².  K = I − hγA with the skew streaming and mirror force antisymmetric
  in the J-weighted product and the damping and hypercollisions dissipative, so no pivoting across blocks is needed (the
  fp64 factors equal the dense inverse to 10⁻¹³ at production size).  The stage
  solve sweeps down (ρ_ℓ = r_ℓ − B_ℓ P_{ℓ+1} ρ_{ℓ+1}) and back up (x_ℓ = P_ℓ(ρ_ℓ − A_ℓ x_{ℓ−1})); the field correction
  C w lives in ℓ ≤ 1, so it changes only ρ_0, ρ_1 and costs one more upward sweep: 2L + 2 block products per stage
  against the dense path's L² + 2L.  The capacitance needs K⁻¹'s ℓ ≤ 1 block, which is P_0, P_1 and two products.
  ``imex_solver = "auto"`` keeps the dense inverses while a factor set fits ``imex_dense_gib``, else every group is
  factored by block-Thomas.  Before any step is built, :meth:`ImexSystem.need_bytes` (factors, capacitance, build
  temporaries or the step's working set) is checked against the device's free memory (``imex_device_gib`` overrides it):
  dense, then block-Thomas; when neither fits the Simulation runs Lawson RK4 with a warning, or refuses
  (``numerics.imex_fallback``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

SQRT2 = float(np.sqrt(2.0))


# ------------------------------------------------------------------------------------------------ tables
@dataclass(frozen=True)
class Tableau:
    """An additive Runge–Kutta pair with a shared c: explicit A_E (strictly lower) with weights bE, implicit A_I (lower
    triangular, zero first row — ESDIRK — and the same γ on the rest of the diagonal) with weights bI; ``erk_bound`` is
    the explicit table's stability interval on the imaginary axis."""
    name: str
    AE: np.ndarray
    AI: np.ndarray
    bE: np.ndarray
    bI: np.ndarray
    c: np.ndarray
    gamma: float
    order: int
    erk_bound: float

    @property
    def s(self) -> int:
        return len(self.c)

    @property
    def stiffly_accurate(self) -> bool:
        """The implicit table's last row is bI and c_s = 1: the step's implicit terms are the last stage's."""
        return bool(np.allclose(self.AI[-1], self.bI, rtol=0, atol=1e-15) and abs(self.c[-1] - 1.0) < 1e-15)

    @property
    def gsa(self) -> bool:
        """Globally stiffly accurate: both tables' last rows are their weights, so the step is the last stage."""
        return self.stiffly_accurate and bool(np.allclose(self.AE[-1], self.bE, rtol=0, atol=1e-15))


def _ark324() -> Tableau:
    Fr = Fraction
    g = Fr(1767732205903, 4055673282236)
    AE = [[0, 0, 0, 0],
          [Fr(1767732205903, 2027836641118), 0, 0, 0],
          [Fr(5535828885825, 10492691773637), Fr(788022342437, 10882634858940), 0, 0],
          [Fr(6485989280629, 16251701735622), Fr(-4246266847089, 9704473918619), Fr(10755448449292, 10357097424841), 0]]
    b = [Fr(1471266399579, 7840856788654), Fr(-4482444167858, 7529755066697), Fr(11266239266428, 11593286722821), g]
    AI = [[0, 0, 0, 0],
          [Fr(1767732205903, 4055673282236), g, 0, 0],
          [Fr(2746238789719, 10658868560708), Fr(-640167445237, 6845629431997), g, 0],
          b]
    c = [0, Fr(1767732205903, 2027836641118), Fr(3, 5), 1]
    f = lambda a: np.array([[float(x) for x in r] for r in a]) if isinstance(a[0], list) else np.array([float(x) for x in a])
    return Tableau("ARK3(2)4L[2]SA", f(AE), f(AI), f(b), f(b), f(c), float(g), 3, 2.4841)


def _ars232() -> Tableau:
    g = 1.0 - 1.0 / np.sqrt(2.0)
    d = -2.0 * np.sqrt(2.0) / 3.0
    AE = np.array([[0, 0, 0], [g, 0, 0], [d, 1 - d, 0]])
    AI = np.array([[0, 0, 0], [0, g, 0], [0, 1 - g, g]])
    b = np.array([0, 1 - g, g])
    return Tableau("ARS(2,3,2)", AE, AI, b, b.copy(), np.array([0, g, 1.0]), float(g), 2, 1.7320)


def _ars443() -> Tableau:
    AE = np.array([[0, 0, 0, 0, 0], [1 / 2, 0, 0, 0, 0], [11 / 18, 1 / 18, 0, 0, 0], [5 / 6, -5 / 6, 1 / 2, 0, 0],
                   [1 / 4, 7 / 4, 3 / 4, -7 / 4, 0]])
    AI = np.array([[0, 0, 0, 0, 0], [0, 1 / 2, 0, 0, 0], [0, 1 / 6, 1 / 2, 0, 0], [0, -1 / 2, 1 / 2, 1 / 2, 0],
                   [0, 3 / 2, -3 / 2, 1 / 2, 1 / 2]])
    return Tableau("ARS(4,4,3)", AE, AI, AE[-1].copy(), AI[-1].copy(), AE.sum(axis=1), 0.5, 3, 1.5698)


TABLEAUX = {"ark3": _ark324, "ars232": _ars232, "ars443": _ars443}


def tableau(name: str) -> Tableau:
    if name not in TABLEAUX:
        raise ValueError(f"numerics.imex_scheme {name!r} not in {tuple(TABLEAUX)}")
    return TABLEAUX[name]()


def stability_function(tab: Tableau, part: str = "I"):
    """R(z) of one table: 1 + z bᵀ(I − zA)⁻¹1 (a callable of a complex array)."""
    A, b = (tab.AI, tab.bI) if part == "I" else (tab.AE, tab.bE)
    s = tab.s

    def R(z):
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        out = np.empty(z.shape, dtype=complex)
        for i, zz in enumerate(z.ravel()):
            out.ravel()[i] = 1.0 + zz * b @ np.linalg.solve(np.eye(s) - zz * A, np.ones(s))
        return out
    return R


# ------------------------------------------------------------------------------------------------ chain layout
@dataclass
class Group:
    """Chains of one kind: the same length N and the same parallel operator (symbol, |k_z|, end-damping profile, zonal
    projection), so one dense electron matrix per step size serves all of them."""
    N: int
    nz: int                 # N · N_z: the chain's extended coordinate ζ = p N_z + iz (member p)
    C: int                  # number of chains
    members: np.ndarray     # (C·N,) flat indices ikx·N_ky + n, chain after chain, member after member
    sym: np.ndarray         # (S, nz) real: the parallel derivative is i·sym on the chain's Fourier modes (FFT order)
    kabs: np.ndarray        # (nz,) |k_z| of the hypercollisions
    damp: np.ndarray        # (nz,) end-damping rate ν(ζ)·A (0: none)
    zonal: bool             # k_y = 0 chains: the zonal Nyquist projection (op.zonal_nyq) applies
    gauge: int              # position (chain index) of the k_x = k_y = 0 mode in this group, or −1


def _fd4_symbol(Nz: int, dz: float) -> np.ndarray:
    k = np.fft.fftfreq(Nz, d=1.0 / Nz) * dz                                          # κ = n Δz, FFT order
    return (8 * np.sin(k) - np.sin(2 * k)) / (6 * dz)


def build_layout(sim):
    """The chain groups of ``sim`` (velcro/kinetic.py's GX chains, or every (k_x, k_y) line of an untwisted tube) and
    the gather data: (groups, ret, back, chain_w)."""
    op = sim.op
    b = op.base
    S = op.S
    Nx, Nky, Nz = sim.cfg.grid.Nx, len(np.asarray(sim.ky)), len(np.asarray(sim.geom.z))
    groups = []
    if op.chains:
        chain_w = np.asarray(op.chain_w)
        for ci, (N, chs) in enumerate(op.chains):
            symc = np.asarray(op.chain_sym[ci])
            symc = (symc / 1j).real
            absc = np.asarray(op.chain_abs[ci]) if op.chain_abs is not None else np.zeros(N * Nz)
            dmpc = np.asarray(op.chain_damp[ci]) if op.chain_damp is not None else np.zeros((len(chs), N * Nz))
            keys = {}
            for k, ch in enumerate(chs):
                sym = symc[:, k] if symc.ndim == 3 else symc                       # (S, nz)
                zonal = ch[0] % Nky == 0
                key = (zonal, sym.tobytes(), dmpc[k].tobytes())
                keys.setdefault(key, []).append(k)
            for key, ks in keys.items():
                k0 = ks[0]
                sym = symc[:, k0] if symc.ndim == 3 else symc
                mem = np.asarray([i for k in ks for i in chs[k]], dtype=np.int64)
                gauge = next((j for j, k in enumerate(ks) if 0 in chs[k]), -1)
                groups.append(Group(N=N, nz=N * Nz, C=len(ks), members=mem, sym=np.array(sym, dtype=float),
                                    kabs=np.array(absc, dtype=float), damp=np.array(dmpc[k0], dtype=float),
                                    zonal=bool(key[0]), gauge=gauge))
    else:
        phase = np.asarray(b.phase)
        if not np.allclose(phase, 1.0, atol=1e-12):
            raise ValueError("numerics.integrator = 'imex' needs GX's linked chains (numerics.parallel_bc = 'gx_chains') or "
                             "an untwisted field line (a slab): with Velcro's own twisted field line the parallel operator "
                             "couples the radial points and the chain systems do not separate")
        chain_w = np.ones(Nx, dtype=complex)
        retained = np.flatnonzero(np.asarray(op.x_mask).reshape(-1) > 0) if op.x_dealias else np.arange(Nx)
        if op.par_fourier:
            dzs = np.asarray(op.dz_sym)                                                # (S, Nx, Nky, Nz)
            if not np.allclose(dzs, dzs[:, :1], atol=0):
                raise ValueError("imex: the parallel symbol varies along x (a twisted field line)")
            sym_ky = dzs[:, 0]                                                         # (S, Nky, Nz)
        else:
            s0 = _fd4_symbol(Nz, float(b.dz))
            sym_ky = np.broadcast_to(s0, (S, Nky, Nz)).copy()
        kz_abs = np.asarray(op.kz_abs) if op.kz_abs is not None else np.zeros((Nx, Nky, Nz))
        keys = {}
        for n in range(Nky):
            for ikx in retained:
                key = (n == 0, sym_ky[:, n].tobytes(), kz_abs[ikx, n].tobytes())
                keys.setdefault(key, []).append(int(ikx) * Nky + n)
        for key, mem in keys.items():
            f0 = mem[0]
            ikx, n = f0 // Nky, f0 % Nky
            groups.append(Group(N=1, nz=Nz, C=len(mem), members=np.asarray(mem, dtype=np.int64),
                                sym=np.array(sym_ky[:, n], dtype=float), kabs=np.array(kz_abs[ikx, n], dtype=float),
                                damp=np.zeros(Nz), zonal=bool(key[0]), gauge=mem.index(0) if 0 in mem else -1))
    ret = np.concatenate([g.members for g in groups])
    back = np.full(Nx * Nky, len(ret), dtype=np.int64)
    back[ret] = np.arange(len(ret))
    if len(np.unique(ret)) != len(ret):
        raise ValueError("imex: a mode is in two chains")
    return groups, ret, back, chain_w


def _real_space_operator(sym: np.ndarray) -> np.ndarray:
    """The dense n × n matrix of 'multiply the Fourier modes (FFT order) by sym' on an n-point periodic line."""
    n = len(sym)
    return np.fft.ifft(np.fft.fft(np.eye(n), axis=0) * np.asarray(sym)[:, None], axis=0)


# ------------------------------------------------------------------------------------------------ device memory
def device_free_bytes(num):
    """Bytes the IMEX may still use on the default device: ``numerics.imex_device_gib`` when set, else memory_stats'
    bytes_limit − bytes_in_use; None when the backend reports no limit (the CPU): no check."""
    if getattr(num, "imex_device_gib", None) is not None:
        return int(float(num.imex_device_gib) * 2 ** 30)
    try:
        import jax
        s = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001
        return None
    lim = s.get("bytes_limit", 0)
    return int(lim - s.get("bytes_in_use", 0)) if lim else None


def _gib(b) -> str:
    return "unknown" if b is None else f"{b / 2 ** 30:.2f} GiB"


# ------------------------------------------------------------------------------------------------ the implicit part
class ImexSystem:
    """The implicit part of ``numerics.integrator = "imex"`` for a kinetic-electron Simulation: its operator f_I, the
    stage solve (I − hγ f_I) X = R, the factors per step size, and the explicit part's CFL bound."""

    def __init__(self, sim, tab: Tableau, frame: bool = False):
        import jax.numpy as jnp
        from velcro import kinetic as ke
        t0 = time.time()
        self.sim, self.tab = sim, tab
        op, fs, cfg = sim.op, sim.fields, sim.cfg
        b = op.base
        if not isinstance(fs, ke.KineticFFTFieldSolver):
            raise ValueError("numerics.integrator = 'imex' needs numerics.field_solver = 'fft' (the flux-tube field solver: "
                             "its symbols are local in (k_x, k_y), so every chain is an independent system)")
        if b.gql_cutoff >= 0:
            raise ValueError("numerics.integrator = 'imex' with physics.gql_cutoff is not implemented")
        self.dtype = sim.dtype
        self.rdtype = jnp.zeros((), sim.dtype).real.dtype
        self.S, self.L, self.M = op.S, op.L, cfg.grid.M
        self.Nx, self.Nky, self.Nz = cfg.grid.Nx, len(np.asarray(sim.ky)), len(np.asarray(sim.geom.z))
        masses = [sp.mass for sp in sim.species]
        self.par = tuple(s for s, m in enumerate(masses) if m == min(masses))       # implicit parallel operator
        if len(self.par) != 1:
            raise ValueError("imex: exactly one lightest species (the electrons) is treated implicitly")
        self.e = self.par[0]
        self.skew = bool(b.parallel_skew)
        self.groups, ret, back, chain_w = build_layout(sim)
        self.ret, self.back = np.asarray(ret, dtype=np.int32), np.asarray(back, dtype=np.int32)
        self.chain_w = jnp.asarray(chain_w, dtype=self.dtype)
        self.x_dealias = bool(op.x_dealias)
        self.x_mask = op.x_mask
        self.zonal_nyq = bool(op.zonal_nyq)
        self.bpar = bool(fs.bpar)
        self.em = fs.Ainv is not None
        self.F = 1 + int(self.em) + int(self.bpar)
        i0 = self.Nx // 2
        r = lambda a, dt=None: jnp.asarray(np.asarray(a), dtype=dt or self.rdtype)
        # reference-surface profiles (x index N_x/2, x = 0), tiled along each chain
        e = self.e
        self.prof_c = np.asarray(op.c_stream)[e, i0, 0, :].astype(float)            # v̂t_e b·∇z
        self.prof_cm = np.asarray(op.c_mirror)[e, i0, 0, :].astype(float)           # v̂t_e b·∇z ∂_z ln B
        self.prof_j = np.asarray(b.jac)[i0, 0, :].astype(float)
        self.prof_ji = np.asarray(b.inv_jac)[i0, 0, :].astype(float)
        self.hkz_e = float(np.asarray(op.hkz)[e]) if op.use_hkz else 0.0
        ell = np.arange(self.L, dtype=float)
        self.ell_p = np.where(ell >= 3, ell ** float(cfg.numerics.hyper_kz_p), 0.0) if op.use_hkz else np.zeros(self.L)
        if op.use_hkz and not np.allclose(np.asarray(op.ell_p).reshape(-1), self.ell_p):
            raise ValueError("imex: unexpected hypercollision ℓ^p")
        mir = np.asarray(b.mirror_skew if self.skew else b.mirror).reshape(self.L * self.M, self.L * self.M)
        self.mir = mir.astype(float)
        self.mir_off = b.mirror_skew_off if self.skew else b.mirror_off
        self.mir_band = tuple(jnp.reshape(c, (self.L, self.M, 1, 1)) for c in
                              (b.mirror_skew_band if self.skew else b.mirror_band))
        Vm = np.asarray(b.V).reshape(self.L, self.M, self.L, self.M)[:, 0, :, 0].astype(float)
        self.Vm = Vm
        self.V_lo = jnp.reshape(b.V_lo, (self.L, 1, 1, 1))
        self.V_up = jnp.reshape(b.V_up, (self.L, 1, 1, 1))
        # the explicit bound's radial wavenumbers: the largest |k_x| the Fourier ∂_x reaches on the evolved (retained) modes,
        # else the FD4 stencil's 1.3722/Δx (kinetic.cfl_rate_kinetic takes π/Δx for the drift and the stencil's for the bracket)
        kxs = np.abs(np.asarray(op.kx_sym).reshape(-1)) if op.kx_sym is not None else None
        kmask = np.asarray(op.x_mask).reshape(-1) if op.x_dealias else np.ones(self.Nx)
        kx_ret = float(np.max(kxs * kmask)) if kxs is not None else ke._FD4 / float(b.dx)
        self.kx_drift = kx_ret if op.drift_fourier else ke._FD4 / float(b.dx)
        self.kx_nl = kx_ret if op.bracket_fourier else ke._FD4 / float(b.dx)
        self.nl_bound = str(cfg.numerics.imex_nl_bound)
        # ρ(2v̂² + μ̂) of the drift as applied: with numerics.gx_hermite_closure the last Hermite row's v̂² is GX's, whose
        # spectrum reaches 7 % higher at (12,6) than the plain product's (base.rad_energy, which cfl_rate_kinetic uses)
        lam_v = np.asarray(op.lam_v_k if op.lam_v_k is not None else b.lam_v)
        self.rad_energy = max(float(b.rad_energy), float(np.max(lam_v) + np.max(np.asarray(b.lam_mu))))
        # the explicit species' parallel streaming acts on h = G + (Z/T)χ: through the field it is that species' share of
        # the ω_H / kinetic-Alfvén branch, k∥ v̂t_s/√(2(β/(2m̂_s) + b_min)) at the smallest k⊥ (kinetic.build_kinetic_operators'
        # bound, there for the lightest species only).  The implicit electrons screen it when k∥v_te hγ ≫ 1, not at small
        # k∥ or small dt, so the bound keeps the unscreened factor
        Bmax = float(np.max(np.asarray(sim.geom.B)))
        kx_min = 2 * np.pi / cfg.grid.Lx
        ky_min = float(np.asarray(sim.ky)[1]) if self.Nky > 1 else np.inf
        gxx, gyy = float(np.min(np.asarray(sim.geom.gxx))), float(np.min(np.asarray(sim.geom.gyy)))
        b_min = 0.5 * min(kx_min ** 2 * gxx, ky_min ** 2 * gyy) / Bmax ** 2
        rad_v = float(b.rad_v)
        self.wave_expl = np.asarray([max(rad_v, 1.0 / (SQRT2 * np.sqrt(cfg.physics.beta / (2 * sp.mass) + b_min)))
                                     for sp in sim.species])
        # x-variation of the parallel coefficients: its share of the parallel operator stays in the explicit part
        cs, cm, jc = np.asarray(op.c_stream)[e, :, 0, :], np.asarray(op.c_mirror)[e, :, 0, :], np.asarray(b.jac)[:, 0, :]
        rel = lambda a, a0: float(np.max(np.abs(a - a0[None])) / max(np.max(np.abs(a0)), 1e-300))
        self.x_resid = max(rel(cs, self.prof_c), rel(cm, self.prof_cm) if np.any(self.prof_cm) else 0.0,
                           2.0 * rel(jc, self.prof_j))
        # device data per group
        self.gd = []
        for g in self.groups:
            tile = lambda p: np.tile(p, g.N)
            d = dict(sym=r(g.sym[e]), kabs=r(g.kabs), damp=r(g.damp), c=r(tile(self.prof_c)), cm=r(tile(self.prof_cm)),
                     j=r(tile(self.prof_j)), ji=r(tile(self.prof_ji)), jc=r(tile(self.prof_j * self.prof_c)))
            gm = np.ones((g.C, 1))
            if g.gauge >= 0:
                gm[g.gauge] = 0.0
            d["gauge"] = r(gm)
            self.gd.append(d)
        self.fmaps0 = tuple(self._field_maps(gi, self._symbols()) for gi in range(len(self.groups)))
        isz = jnp.zeros((), self.dtype).dtype.itemsize
        self.dense_bytes = [(self.L * self.M * g.nz) ** 2 * isz for g in self.groups]
        self.half_factors = str(getattr(cfg.numerics, "imex_factor_storage", "full")) == "bf16"
        psz = 4 if self.half_factors else isz                   # bf16: Re and Im of P_ℓ, 2 bytes each
        self.thomas_bytes = [self.L * (self.M * g.nz) ** 2 * psz for g in self.groups]
        self.mir4 = self.mir.reshape(self.L, self.M, self.L, self.M)
        self.hl = (self.hkz_e * self.ell_p) if self.hkz_e > 0 else np.zeros(self.L)
        self.frame = bool(frame)
        self.plan = self._plan(cfg.numerics)                    # the kinds the device can hold, or None (Simulation falls
        if self.plan["kinds"] is not None:                      # back to Lawson or refuses: numerics.imex_fallback)
            self.set_kinds(self.plan["kinds"])                  # kind per group; kinv_bytes = one factor set's bytes
        else:
            self.kind, self.kinv_bytes, self._tm = None, 0, {}
        self.setup_s = time.time() - t0
        self._cache = {}
        self._build = {}

    # -------------------------------------------------------------- does it fit the device (numerics.imex_fallback)
    def need_bytes(self, kinds) -> dict:
        """Device bytes a run needs with these factor kinds (one per group): the factor set (in the shearing frame with
        the Thomas groups' ℓ ≤ 1 blocks, kept for the per-stage capacitance), the capacitance LUs, and the larger of a
        factor build's temporaries (dense: the assembled K and its inverse, 2 n² for the largest group; Thomas: every
        group's ℓ ≤ 1 block, which the build returns for the capacitance, and 6 (M·nz)² for the largest group) and the
        step's working set: 32 state-sized arrays with the default adaptive rate (the peak above the factor set on GX's
        JET box, 9.14 GiB against 5.19), one more per Arnoldi vector beyond
        20."""
        isz = 2 * self.rdtype.itemsize
        num = self.sim.cfg.numerics
        g = self.groups
        th = [i for i, k in enumerate(kinds) if k == "thomas"]
        de = [i for i, k in enumerate(kinds) if k == "dense"]
        tblk = [(2 * self.M * gr.nz) ** 2 * isz for gr in g]
        factors = sum(self.dense_bytes[i] for i in de) + sum(self.thomas_bytes[i] for i in th)
        if self.frame:
            factors += sum(tblk[i] for i in th)
        cap = 0 if self.frame else sum(gr.C * ((self.F * gr.nz) ** 2 * isz + 4 * self.F * gr.nz) for gr in g)
        build = max([2 * self.dense_bytes[i] for i in de] +
                    ([sum(tblk[i] for i in th) + max(6 * (self.M * g[i].nz) ** 2 * isz for i in th)] if th else []))
        state = self.S * self.L * self.M * self.Nx * self.Nky * self.Nz * isz
        extra = max(0, int(num.imex_rate_krylov) - 20) if num.imex_rate == "jacobian" else 0
        work = (32 + extra) * state
        return dict(factors=int(factors), capacitance=int(cap), build=int(build), work=int(work),
                    total=int(factors + cap + max(build, work)))

    def _plan(self, num) -> dict:
        """The factor kinds the run uses: imex_solver's choice (:meth:`_choose_kinds`), then — when the device cannot hold
        it and it is dense — block-Thomas for every group; ``kinds`` None when nothing fits (no check where the device
        reports no limit).  ``free`` and ``tried`` [(kinds, need_bytes)] are kept for the report."""
        G = len(self.groups)
        first = self._choose_kinds(num)
        options = [first]
        if set(first) == {"dense"} and self.L >= 2 and self._block_tridiagonal():
            options.append(["thomas"] * G)
        free = device_free_bytes(num)
        tried = []
        for kinds in options:
            need = self.need_bytes(kinds)
            tried.append(("/".join(sorted(set(kinds))), need))
            if free is None or need["total"] <= free:
                if kinds is not first:
                    print(f"WARNING: imex: the dense chain inverses do not fit the device ({_gib(free)} free, "
                          f"{_gib(tried[0][1]['total'])} needed): block-Thomas for every group ({_gib(need['total'])})",
                          flush=True)
                return dict(kinds=kinds, free=free, tried=tried)
        return dict(kinds=None, free=free, tried=tried)

    def plan_report(self) -> str:
        """The sizes behind the plan, for the fallback's warning or error."""
        rows = [f"{k} {_gib(n['total'])} (factors {_gib(n['factors'])}, capacitance {_gib(n['capacitance'])}, build "
                f"{_gib(n['build'])}, step {_gib(n['work'])})" for k, n in self.plan["tried"]]
        return f"imex: the implicit factors do not fit the device ({_gib(self.plan['free'])} free): " + "; ".join(rows)

    # -------------------------------------------------------------- the factorisation per group (numerics.imex_solver)
    def _choose_kinds(self, num):
        """'dense' or 'thomas' per group: imex_solver forces one; 'auto' keeps the dense inverses while their set fits
        imex_dense_gib and otherwise factors every group by block-Thomas.  (Switching only the largest groups until the
        set fits was measured slower than switching all: the groups it keeps dense are
        then large too, and their dense solves read more than Thomas's sweeps.)"""
        mode = str(num.imex_solver)
        G = len(self.groups)
        if mode == "dense":
            return ["dense"] * G
        ok = self.L >= 2 and self._block_tridiagonal()
        if mode == "thomas":
            if not ok:
                raise ValueError("imex_solver 'thomas' needs L ≥ 2 and an implicit operator block-tridiagonal in ℓ")
            return ["thomas"] * G
        if not ok or sum(self.dense_bytes) <= float(num.imex_dense_gib) * 2 ** 30:
            return ["dense"] * G
        return ["thomas"] * G

    def set_kinds(self, kinds):
        """Factor group gi densely or by block-Thomas as ``kinds[gi]`` says (any mix; tests and tools — the config's
        imex_solver chooses at construction); drops the cached factors."""
        if len(kinds) != len(self.groups) or any(k not in ("dense", "thomas") for k in kinds):
            raise ValueError(f"set_kinds: one of 'dense' / 'thomas' per group ({len(self.groups)}), got {kinds}")
        if "thomas" in kinds and not (self.L >= 2 and self._block_tridiagonal()):
            raise ValueError("block-Thomas needs L ≥ 2 and an implicit operator block-tridiagonal in ℓ")
        self.kind = list(kinds)
        self.kinv_bytes = int(sum(db if k == "dense" else tb for k, db, tb in
                                  zip(self.kind, self.dense_bytes, self.thomas_bytes)))
        self._tm = {gi: self._thomas_mats(gi) for gi, k in enumerate(self.kind) if k == "thomas"}
        self._cache = {}

    def _block_tridiagonal(self) -> bool:
        """V and the mirror matrix couple ℓ to ℓ ± 1 only (zero ℓ-diagonal blocks), so K is block-tridiagonal in ℓ."""
        L = self.L
        far = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) != 1
        return bool(np.all(self.Vm[far] == 0) and np.all(self.mir4.transpose(0, 2, 1, 3)[far] == 0))

    def _thomas_mats(self, gi):
        """Group gi's ζ-space matrices of the block-Thomas factorisation (run dtype): T1 = Π S (streaming per unit V),
        T2 = Π diag(c_m) (the mirror force per unit mirror matrix; the vector c_m when Π = I), Dmp = Π diag(damp) and
        Kab = Π |k_z| (hypercollisions per unit ℓ^p), Π the zonal Nyquist projection on the output (I elsewhere)."""
        import jax.numpy as jnp
        g = self.groups[gi]
        nz = g.nz
        cdt = self.dtype
        D = _real_space_operator(1j * g.sym[self.e])
        tile = lambda p: np.tile(p, g.N)
        c, ji, jc = tile(self.prof_c), tile(self.prof_ji), tile(self.prof_j * self.prof_c)
        S4 = -0.5 * (c[:, None] * D + ji[:, None] * D * jc[None, :]) if self.skew else -c[:, None] * D
        cm = tile(self.prof_cm).astype(complex)
        proj = g.zonal and self.zonal_nyq
        u = 1.0 - 2.0 * (np.arange(nz) % 2)
        Pi = np.eye(nz) - np.outer(u, u) / nz if proj else None
        on = (lambda a: Pi @ a) if proj else (lambda a: a)
        Kab = on(_real_space_operator(g.kabs)) if self.hkz_e > 0 else np.zeros((nz, nz))
        return dict(T1=jnp.asarray(on(S4), cdt), T2=jnp.asarray(Pi * cm[None, :], cdt) if proj else None,
                    cm=jnp.asarray(cm, cdt), Dmp=jnp.asarray(on(np.diag(g.damp).astype(complex)), cdt),
                    Kab=jnp.asarray(Kab, cdt))

    def _off(self, t, l, lp, X):
        """The block of A coupling ℓ ← ℓ' = ℓ ± 1 applied to X (…, M, nz): V_ℓℓ' (T1 X) + Σ_n Mir[ℓ m, ℓ' n] (T2 X)_n
        (t: the group's :meth:`_thomas_mats`).  K's block is −hγ times it."""
        import jax.numpy as jnp
        out = None
        if self.Vm[l, lp] != 0:
            out = float(self.Vm[l, lp]) * jnp.einsum("zw,...mw->...mz", t["T1"], X)
        mb = self.mir4[l, :, lp, :]
        if np.any(mb):
            Y = X * t["cm"] if t["T2"] is None else jnp.einsum("zw,...mw->...mz", t["T2"], X)
            Z = self._mix_m(mb, Y)
            out = Z if out is None else out + Z
        return jnp.zeros_like(X) if out is None else out

    @staticmethod
    def _mix_m(mb, Y):
        """Σ_n mb[m, n] Y[…, n, z] for a banded (here tridiagonal) M × M matrix mb: one shifted copy of Y per nonzero
        diagonal, so it fuses with the elementwise work around it (no small matmul, no transpose)."""
        import jax.numpy as jnp
        M = mb.shape[0]
        out = None
        for d in range(-(M - 1), M):
            c = np.diagonal(mb, d)                                  # mb[m, m + d]
            if not np.any(c):
                continue
            coef = np.zeros(M)
            lo = max(0, -d)
            coef[lo:lo + len(c)] = c
            sl = Y[..., max(d, 0):M + min(d, 0), :]                 # Y[…, m + d, :] for the m that have one
            pad = [(0, 0)] * (Y.ndim - 2) + [(max(-d, 0), max(d, 0)), (0, 0)]
            term = jnp.asarray(coef[:, None], Y.real.dtype) * jnp.pad(sl, pad)
            out = term if out is None else out + term
        return jnp.zeros_like(Y) if out is None else out

    def _off_cols(self, t, l, lp, Z):
        """The same block applied to the columns of an (M·nz)² matrix Z: (block) @ Z."""
        import jax.numpy as jnp
        n = Z.shape[0]
        Zt = jnp.transpose(Z).reshape(n, self.M, n // self.M)           # rows of Zᵀ are the columns of Z
        return jnp.transpose(self._off(t, l, lp, Zt).reshape(n, n))

    def _off_right(self, t, l, lp, P):
        """P @ (the block coupling ℓ ← ℓ'), P (n, n) with n = M·nz."""
        import jax.numpy as jnp
        n = P.shape[0]
        P3 = P.reshape(n, self.M, n // self.M)                            # columns (m, z)
        out = None
        if self.Vm[l, lp] != 0:
            out = float(self.Vm[l, lp]) * jnp.einsum("imz,zw->imw", P3, t["T1"])
        mb = self.mir4[l, :, lp, :]
        if np.any(mb):
            Y = P3 * t["cm"] if t["T2"] is None else jnp.einsum("imz,zw->imw", P3, t["T2"])
            Z = jnp.einsum("mn,imw->inw", jnp.asarray(mb, P.dtype), Y)
            out = Z if out is None else out + Z
        return (jnp.zeros_like(P3) if out is None else out).reshape(n, n)

    def _thomas_factor(self, gi, gdt, t):
        """(P_0 … P_{L−1}), T: the block-Thomas factors of group gi at hγ = gdt (P_ℓ = S_ℓ⁻¹, eliminated from ℓ = L − 1
        down) and K⁻¹'s ℓ ≤ 1 block T (2·M·nz square, the capacitance's), from P_0, P_1 and the blocks between them."""
        import jax.numpy as jnp
        L, M = self.L, self.M
        nz = self.groups[gi].nz
        n = M * nz
        cdt = self.dtype
        eyeM = jnp.eye(M, dtype=cdt)

        def diag_block(l):                      # D_ℓ = I + hγ I_M ⊗ Π(diag(damp) + hkz ℓ^p |k_z|)
            d = t["Dmp"] + float(self.hl[l]) * t["Kab"] if self.hl[l] != 0 else t["Dmp"]
            return jnp.eye(n, dtype=cdt) + gdt * jnp.kron(eyeM, d)

        P = [None] * L
        P[L - 1] = jnp.linalg.inv(diag_block(L - 1))
        for l in range(L - 2, -1, -1):          # S_ℓ = D_ℓ − hγ² Off(ℓ, ℓ+1) P_{ℓ+1} Off(ℓ+1, ℓ)
            Z = self._off_right(t, l + 1, l, P[l + 1])
            P[l] = jnp.linalg.inv(diag_block(l) - (gdt * gdt) * self._off_cols(t, l, l + 1, Z))
        W = self._off_cols(t, 0, 1, P[1])                                # Off(0,1) P_1
        T01 = gdt * (P[0] @ W)                                           # −P_0 B_0 P_1
        T10 = gdt * (P[1] @ self._off_cols(t, 1, 0, P[0]))               # −P_1 A_1 P_0
        T = jnp.concatenate([jnp.concatenate([P[0], T01], axis=1),
                             jnp.concatenate([T10, P[1] + gdt * (T10 @ W)], axis=1)], axis=0)
        return tuple(P), T

    def _thomas_fn(self, gi):
        import jax
        key = ("thomas", gi)
        if key not in self._build:
            half = self.half_factors

            def build(gdt, t):
                P, T = self._thomas_factor(gi, gdt, t)          # T (the capacitance's block) from the full P_0, P_1
                if half:                                        # numerics.imex_factor_storage 'bf16': (Re P_ℓ, Im P_ℓ)
                    import jax.numpy as jnp
                    P = tuple((jnp.real(p).astype(jnp.bfloat16), jnp.imag(p).astype(jnp.bfloat16)) for p in P)
                return P, T
            self._build[key] = jax.jit(build)
        return self._build[key]

    @staticmethod
    def _pmul(Pl, x):
        """P_ℓ applied to the rows x (C, M, nz).  With imex_factor_storage 'bf16' Pl is (Re P_ℓ, Im P_ℓ): the rows' real
        and imaginary parts stacked (2C rows) meet each half once, accumulated in the rows' real dtype, so each part of
        P_ℓ is read once — half the bytes of the complex product."""
        import jax.numpy as jnp
        from jax import lax
        C = x.shape[0]
        xf = x.reshape(C, -1)
        if isinstance(Pl, tuple):
            pr, pi = Pl
            rdt = xf.real.dtype
            X2 = jnp.concatenate([jnp.real(xf), jnp.imag(xf)], axis=0)
            A = jnp.matmul(X2, pr.T, preferred_element_type=rdt)
            B = jnp.matmul(X2, pi.T, preferred_element_type=rdt)
            return lax.complex(A[:C] - B[C:], B[:C] + A[C:]).astype(x.dtype).reshape(x.shape)
        return jnp.matmul(xf, Pl.T).reshape(x.shape)

    def _thomas_down(self, t, Xe, P, gdt):
        """The downward sweep for the electrons' Xe (L, M, C, nz): y_ℓ = P_ℓ ρ_ℓ, ρ_ℓ = r_ℓ − B_ℓ y_{ℓ+1} (each (C, M, nz)),
        and K⁻¹Xe's ℓ ≤ 1 part (2, M, C, nz)."""
        import jax.numpy as jnp
        L = self.L
        R = jnp.transpose(Xe, (0, 2, 1, 3))                                          # (L, C, M, nz)
        ys = [None] * L
        ys[L - 1] = self._pmul(P[L - 1], R[L - 1])
        for l in range(L - 2, -1, -1):
            ys[l] = self._pmul(P[l], R[l] + gdt * self._off(t, l, l + 1, ys[l + 1]))
        x1 = ys[1] + gdt * self._pmul(P[1], self._off(t, 1, 0, ys[0]))
        return ys, jnp.transpose(jnp.stack([ys[0], x1]), (0, 2, 1, 3))

    def _thomas_up(self, t, ys, dr, P, gdt):
        """K⁻¹(r + δr ⊕ 0) − δr ⊕ 0 (L, M, C, nz) from the downward sweep of r and δr (2, M, C, nz) in ℓ ≤ 1 (the field
        correction): δr changes ρ_1 by δr_1 and ρ_0 by δr_0 − B_0 P_1 δr_1, then one upward sweep
        x_ℓ = y_ℓ − P_ℓ A_ℓ x_{ℓ−1}."""
        import jax.numpy as jnp
        L = self.L
        d = jnp.transpose(dr, (0, 2, 1, 3))                                          # (2, C, M, nz)
        q1 = self._pmul(P[1], d[1])
        xs = [ys[0] + self._pmul(P[0], d[0] + gdt * self._off(t, 0, 1, q1))]
        xs.append(ys[1] + q1 + gdt * self._pmul(P[1], self._off(t, 1, 0, xs[0])))
        for l in range(2, L):
            xs.append(ys[l] + gdt * self._pmul(P[l], self._off(t, l, l - 1, xs[l - 1])))
        return jnp.transpose(jnp.stack(xs).at[:2].add(-d), (0, 2, 1, 3))

    # -------------------------------------------------------------- gather / scatter
    def gather(self, f):
        """(…, N_x, N_ky, N_z) → tuple over groups of (…, C, nz): x → k_x (with the chains' physical phase), the members."""
        import jax.numpy as jnp
        lead = f.shape[:-3]
        F = (jnp.fft.fft(f, axis=-3) * self.chain_w[:, None, None]).reshape(lead + (self.Nx * self.Nky, self.Nz))
        Gm = jnp.take(F, self.ret, axis=-2)
        out, off = [], 0
        for g in self.groups:
            out.append(Gm[..., off:off + g.C * g.N, :].reshape(lead + (g.C, g.nz)))
            off += g.C * g.N
        return tuple(out)

    def scatter(self, pieces, lead):
        """The inverse of :meth:`gather` (modes in no chain come back zero)."""
        import jax.numpy as jnp
        Y = jnp.concatenate([p.reshape(lead + (g.C * g.N, self.Nz)) for p, g in zip(pieces, self.groups)]
                            + [jnp.zeros(lead + (1, self.Nz), dtype=pieces[0].dtype)], axis=-2)
        out = jnp.take(Y, self.back, axis=-2).reshape(lead + (self.Nx, self.Nky, self.Nz))
        return jnp.fft.ifft(out * jnp.conj(self.chain_w)[:, None, None], axis=-3)

    def project_out(self, f):
        """f minus its part on the chain modes (zero in normal operation: the state and every right-hand side live there)."""
        import jax.numpy as jnp
        if not self.x_dealias:
            return jnp.zeros_like(f)
        return f - jnp.fft.ifft(jnp.fft.fft(f, axis=-3) * self.x_mask, axis=-3)

    # -------------------------------------------------------------- field maps (local per chain point)
    def _symbols(self, theta=None):
        """The field solver's symbols (k_x FFT order): J (S, M, Nkx, Nky, Nz), Pinv, Ainv, and the δB∥ blocks."""
        fs = self.sim.fields
        rdt = self.rdtype
        if theta is None:
            out = dict(J=fs.J.astype(rdt), Pinv=fs.Pinv.astype(rdt), Ainv=None if fs.Ainv is None else fs.Ainv.astype(rdt))
            if self.bpar:
                out.update({k: getattr(fs, k).astype(rdt) for k in ("B11", "B12", "B21", "B22")},
                           bfac=fs.bfac.astype(rdt))
            return out
        J, _, Pinv, Ainv = fs.frame_symbols(theta, rdt)
        return dict(J=J, Pinv=Pinv, Ainv=Ainv)

    def _gather_sym(self, a, gi):
        """A real symbol (…, Nkx, Nky, Nz) → (…, C, nz) on group gi (no phase: symbols are per mode)."""
        import jax.numpy as jnp
        g = self.groups[gi]
        lead = a.shape[:-3]
        A = jnp.reshape(a, lead + (self.Nx * self.Nky, self.Nz))
        off = sum(h.C * h.N for h in self.groups[:gi])
        idx = self.ret[off:off + g.C * g.N]
        return jnp.take(A, idx, axis=-2).reshape(lead + (g.C, g.nz))

    def _field_maps(self, gi, sy):
        """(Q, C) of group gi: Q (S, C, F, nz, 2, M) moments (ℓ = 0, 1) → fields; C (S, C, 2, M, nz, F) fields → the
        field part (Z/T)χ of h.  Fields: φ, then A∥ (β > 0), then δB∥/B (physics.bpar)."""
        import jax.numpy as jnp
        fs, op = self.sim.fields, self.sim.op
        g = self.groups[gi]
        S, M, F = self.S, self.M, self.F
        rdt = self.rdtype
        J = self._gather_sym(sy["J"], gi)                                           # (S, M, C, nz)
        Pinv = self._gather_sym(sy["Pinv"], gi)                                     # (C, nz)
        qn, zt = fs.qn_w.astype(rdt), op.Z_over_T.astype(rdt)
        vt = op.vt.astype(rdt)
        Jt = jnp.transpose(J, (0, 2, 3, 1))                                         # (S, C, nz, M)
        Q = jnp.zeros((S, g.C, F, g.nz, 2, M), dtype=rdt)
        Cm = jnp.zeros((S, g.C, 2, M, g.nz, F), dtype=rdt)
        Kt = Jt - jnp.concatenate([jnp.zeros_like(Jt[..., :1]), Jt[..., :-1]], axis=-1)   # 𝒦_m = 𝒥_m − 𝒥_{m−1}
        if self.bpar:
            B11, B12, B21, B22 = (self._gather_sym(sy[k], gi) for k in ("B11", "B12", "B21", "B22"))
            bfac = jnp.broadcast_to(jnp.reshape(sy["bfac"], (1, 1, self.Nz)), (self.Nx, self.Nky, self.Nz))
            bfac = self._gather_sym(bfac, gi)
            pw = fs.perp_w.astype(rdt)
            rho_w = qn[:, None, None, None] * Jt                                   # (S, C, nz, M)
            sig_w = -bfac[None, :, :, None] * pw[:, None, None, None] * Kt
            Q = Q.at[:, :, 0, :, 0, :].set(B11[None, :, :, None] * rho_w + B12[None, :, :, None] * sig_w)
            Q = Q.at[:, :, F - 1, :, 0, :].set(B21[None, :, :, None] * rho_w + B22[None, :, :, None] * sig_w)
            Cm = Cm.at[:, :, 0, :, :, F - 1].set(jnp.transpose(Kt, (0, 1, 3, 2)))
        else:
            Q = Q.at[:, :, 0, :, 0, :].set(Pinv[None, :, :, None] * qn[:, None, None, None] * Jt)
        Cm = Cm.at[:, :, 0, :, :, 0].set(zt[:, None, None, None] * jnp.transpose(Jt, (0, 1, 3, 2)))
        if self.em:
            Ainv = self._gather_sym(sy["Ainv"], gi)
            aw = fs.amp_w.astype(rdt)
            Q = Q.at[:, :, 1, :, 1, :].set(Ainv[None, :, :, None] * aw[:, None, None, None] * Jt)
            Cm = Cm.at[:, :, 1, :, :, 1].set((zt * (-vt / SQRT2))[:, None, None, None] * jnp.transpose(Jt, (0, 1, 3, 2)))
        return Q, Cm

    # -------------------------------------------------------------- the operator A₀ on one group (chain space)
    def _dz(self, X, s):
        import jax.numpy as jnp
        return jnp.fft.ifft(jnp.fft.fft(X, axis=-1) * (1j * s).astype(X.dtype), axis=-1)

    def _apply_par(self, X, d):
        """A₀ of the implicit species on chain-space h, X (L, M, C, nz): streaming and mirror force (skew or advective
        form, the reference-surface coefficients), GX hypercollisions, end damping."""
        import jax.numpy as jnp
        from velcro.vlasov import apply_banded
        z = jnp.zeros_like(X[:1])
        VX = self.V_lo * jnp.concatenate([z, X[:-1]], axis=0) + self.V_up * jnp.concatenate([X[1:], z], axis=0)
        if self.skew:
            out = -0.5 * (d["c"] * self._dz(VX, d["sym"]) + d["ji"] * self._dz(d["jc"] * VX, d["sym"]))
        else:
            out = -d["c"] * self._dz(VX, d["sym"])
        out = out + d["cm"] * apply_banded(self.mir_off, self.mir_band, X)
        if self.hkz_e > 0 and self.L > 3:
            ak = jnp.fft.ifft(jnp.fft.fft(X[3:], axis=-1) * d["kabs"].astype(X.dtype), axis=-1)
            out = out.at[3:].add(-self.hkz_e * jnp.asarray(self.ell_p[3:], dtype=self.rdtype).reshape(-1, 1, 1, 1) * ak)
        return out - d["damp"] * X

    def _project(self, Y, g):
        """Π: the zonal Nyquist projection of a zonal group's output (along ζ) when op.zonal_nyq."""
        import jax.numpy as jnp
        if not (g.zonal and self.zonal_nyq):
            return Y
        u = (1.0 - 2.0 * (jnp.arange(g.nz) % 2)).astype(self.rdtype)
        return Y - jnp.mean(Y * u, axis=-1, keepdims=True) * u

    def f_imp(self, H):
        """f_I in Velcro space from h = G + (Z/T)χ (S·L, M, Nx, Nky, Nz)."""
        import jax.numpy as jnp
        S, L = self.S, self.L
        Hc = self.gather(H)
        out = []
        for gi, (g, d) in enumerate(zip(self.groups, self.gd)):
            X = Hc[gi].reshape((S, L) + Hc[gi].shape[1:])                             # (S, L, M, C, nz)
            Y = -d["damp"] * X
            Y = Y.at[self.e].set(self._apply_par(X[self.e], d))
            Y = self._project(Y, g) * d["gauge"]
            out.append(Y.reshape((S * L,) + Y.shape[2:]))
        return self.scatter(out, H.shape[:-3])

    def h_of(self, G, chi):
        """h = G + (Z/T)χ for the stacked state and χ (S, nrow, M, Nx, Nky, Nz) of kinetic_fields."""
        S, L = self.S, self.L
        op = self.sim.op
        Gs = G.reshape((S, L) + G.shape[1:])
        nrow = chi.shape[1]
        zt = op.Z_over_T.astype(self.rdtype).reshape((-1, 1, 1, 1, 1, 1))
        return Gs.at[:, :nrow].add(zt * chi).reshape(G.shape)

    # -------------------------------------------------------------- dense group matrices
    def _assemble(self, gi, gdt):
        """K_g = I − hγ Π A₀ (n × n, n = L·M·nz, index (ℓ, m, ζ)) of the implicit species on group gi."""
        import jax.numpy as jnp
        g, L, M = self.groups[gi], self.L, self.M
        nz = g.nz
        cdt = self.dtype
        D = _real_space_operator(1j * g.sym[self.e])                               # (nz, nz)
        tile = lambda p: np.tile(p, g.N)
        c, ji, jc = tile(self.prof_c), tile(self.prof_ji), tile(self.prof_j * self.prof_c)
        if self.skew:
            S4 = -0.5 * (c[:, None] * D + ji[:, None] * D * jc[None, :])
        else:
            S4 = -c[:, None] * D
        A = (jnp.asarray(self.Vm, cdt)[:, None, None, :, None, None] * jnp.eye(M, dtype=cdt)[None, :, None, None, :, None]
             * jnp.asarray(S4, cdt)[None, None, :, None, None, :])
        cm = jnp.asarray(tile(self.prof_cm), cdt)
        Iz = jnp.eye(nz, dtype=cdt)
        mir = jnp.asarray(self.mir.reshape(L, M, L, M), cdt)
        A = A + mir[:, :, None, :, :, None] * (cm[:, None] * Iz)[None, None, :, None, None, :]
        dmp = jnp.asarray(g.damp, cdt)
        A = A - (jnp.eye(L, dtype=cdt)[:, None, None, :, None, None] * jnp.eye(M, dtype=cdt)[None, :, None, None, :, None]
                 * (dmp[:, None] * Iz)[None, None, :, None, None, :])
        if self.hkz_e > 0:
            Kab = jnp.asarray(_real_space_operator(g.kabs), cdt)
            hl = jnp.asarray(self.hkz_e * self.ell_p, cdt)
            A = A - (jnp.diag(hl)[:, None, None, :, None, None] * jnp.eye(M, dtype=cdt)[None, :, None, None, :, None]
                     * Kab[None, None, :, None, None, :])
        if g.zonal and self.zonal_nyq:
            u = jnp.asarray(1.0 - 2.0 * (np.arange(nz) % 2), cdt)
            A = A - u[None, None, :, None, None, None] * jnp.einsum("w,lmwpqv->lmpqv", u, A)[:, :, None] / nz
        n = L * M * nz
        return jnp.eye(n, dtype=cdt) - gdt * A.reshape(n, n)

    def set_bytes(self, frame: bool = False) -> int:
        """Bytes of one factor set: the dense inverses and block-Thomas factors (``kinv_bytes``), plus in the shearing
        frame the Thomas groups' ℓ ≤ 1 blocks of K⁻¹ kept for the per-stage capacitance."""
        isz = 2 * self.rdtype.itemsize
        return int(self.kinv_bytes + (sum((2 * self.M * g.nz) ** 2 * isz for g, k in zip(self.groups, self.kind)
                                          if k == "thomas") if frame else 0))

    def _t01(self, gi, kinv):
        """K⁻¹'s ℓ ≤ 1 block of a dense group."""
        n = 2 * self.M * self.groups[gi].nz
        return kinv[:n, :n]

    def _kinv_fn(self, gi):
        import jax
        import jax.numpy as jnp
        if gi not in self._build:
            self._build[gi] = jax.jit(lambda gdt: jnp.linalg.inv(self._assemble(gi, gdt)))
        return self._build[gi]

    # -------------------------------------------------------------- capacitance
    def _cap(self, gi, T01, gdt, maps):
        """LU factors of the capacitance matrices of group gi: I + Σ_s Q_s(C_s − K_s⁻¹C_s), (C, F·nz, F·nz); T01 is the
        electrons' K⁻¹ restricted to ℓ ≤ 1 (2·M·nz square)."""
        import jax.numpy as jnp
        from jax.scipy.linalg import lu_factor
        g, M, F = self.groups[gi], self.M, self.F
        nz = g.nz
        Q, Cm = maps                                                                 # (S, C, F, nz, 2, M), (S, C, 2, M, nz, F)
        e = self.e
        loc = jnp.einsum("scfzam,scamzk->sczfk", Q, Cm)                             # (S, C, nz, F, F)
        w = gdt * self.gd[gi]["damp"] / (1.0 + gdt * self.gd[gi]["damp"])            # the other species: their damping
        wts = jnp.stack([jnp.ones_like(w) if s == e else w for s in range(self.S)])  # (S, nz)
        loc = jnp.einsum("sz,sczfk->czfk", wts, loc)
        T = T01.reshape(2, M, nz, 2, M, nz)
        X = jnp.einsum("cfzam,amzbnw->cfzbnw", Q[e].astype(T01.dtype), T)
        QTC = jnp.einsum("cfzbnw,cbnwk->cfzkw", X, Cm[e].astype(T01.dtype))
        cap = -QTC
        eyeF = jnp.eye(F, dtype=cap.dtype)
        cap = cap + (jnp.eye(nz, dtype=cap.dtype)[None, None, :, None, :] * (eyeF[None, :, None, :, None]
                     + jnp.transpose(loc, (0, 2, 1, 3))[:, :, :, :, None].astype(cap.dtype)))
        cap = cap.reshape(g.C, F * nz, F * nz)
        return lu_factor(cap)

    # -------------------------------------------------------------- factors per step size
    def taus(self):
        """The distinct Lawson propagation times (units of h) of the table, and per (i, j) the index into them."""
        c = self.tab.c
        s = self.tab.s
        vals = []
        for i in range(1, s):
            for j in range(i):
                vals.append(c[i] - c[j])
        vals += [1.0 - c[j] for j in range(s)]
        uniq = []
        for v in vals:
            if abs(v) > 1e-14 and not any(abs(v - u) < 1e-14 for u in uniq):
                uniq.append(v)
        return uniq

    def factors(self, dt, frame: bool = False):
        """The step-size data of :func:`make_step` (cached per dt): per group K_g⁻¹ (dense) or the block-Thomas P_ℓ,
        the capacitance LU per group (θ = 0; not in the shearing frame, which rebuilds it per stage from K⁻¹'s ℓ ≤ 1
        block, kept for the Thomas groups), the Lawson phases per propagation time."""
        import jax
        import jax.numpy as jnp
        try:
            key = (float(dt), frame)
        except Exception:                   # a tracer (a step inside someone else's jit): built in that trace
            return self._factors(dt, frame)
        if key not in self._cache:
            nmax = int(self.sim.cfg.numerics.imex_cache)
            try:                            # never more cached factor sets than ~45 % of the device holds, minus one
                lim = jax.devices()[0].memory_stats().get("bytes_limit", 0)       # for the next build's temporaries
                if lim:
                    nmax = min(nmax, max(1, int(0.45 * lim / max(self.set_bytes(frame), 1)) - 1))
            except Exception:  # noqa: BLE001
                pass
            while len(self._cache) >= max(nmax, 1):
                self._cache.pop(next(iter(self._cache)))
            t0 = time.time()
            fac = self._factors(jnp.asarray(dt, dtype=self.rdtype), frame)
            jax.block_until_ready(fac)
            self.last_factor_s = time.time() - t0
            self._cache[key] = fac
        return self._cache[key]

    def _factors(self, dt, frame):
        import jax.numpy as jnp
        op = self.sim.op
        gdt = (self.tab.gamma * dt).astype(self.rdtype) if hasattr(dt, "astype") else self.tab.gamma * dt
        g = jnp.asarray(gdt, dtype=self.rdtype)
        kinv, P, T = [], [], []
        for gi in range(len(self.groups)):
            if self.kind[gi] == "dense":
                kinv.append(self._kinv_fn(gi)(g))
                P.append(None)
                T.append(None)
            else:
                p, t = self._thomas_fn(gi)(g, self._tm[gi])
                kinv.append(None)
                P.append(p)
                T.append(t)
        out = dict(kinv=tuple(kinv), gdt=g)
        if "thomas" in self.kind:                   # the factors and the ζ-space blocks they are applied with
            out["P"] = tuple(P)
            out["tm"] = tuple(self._tm.get(gi) for gi in range(len(self.groups)))
        if not frame:
            out["cap"] = tuple(self._cap(gi, self._t01(gi, kinv[gi]) if T[gi] is None else T[gi], out["gdt"],
                                         self.fmaps0[gi]) for gi in range(len(self.groups)))
        elif "thomas" in self.kind:                 # the shearing frame rebuilds the capacitance per stage from these
            out["T"] = tuple(t if k == "thomas" else None for t, k in zip(T, self.kind))
        if op.base.split_drift_ky:
            out["phase"] = tuple(op.drift_phase(t * dt).astype(self.dtype) for t in self.taus())
        return out

    # -------------------------------------------------------------- the stage solve
    def solve(self, R, fac, theta=None):
        """X with X − hγ f_I(X) = R (R: the stacked state's shape)."""
        import jax.numpy as jnp
        from jax.scipy.linalg import lu_solve
        S, L, M, F = self.S, self.L, self.M, self.F
        e = self.e
        gdt = fac["gdt"]
        lead = R.shape[:-3]
        Rc = self.gather(R)
        sy = None if theta is None else self._symbols(theta)
        out = []
        for gi, (g, d) in enumerate(zip(self.groups, self.gd)):
            nz = g.nz
            kinv = fac["kinv"][gi]
            maps = self.fmaps0[gi] if theta is None else self._field_maps(gi, sy)
            if theta is None:
                lu = fac["cap"][gi]
            else:
                lu = self._cap(gi, self._t01(gi, kinv) if self.kind[gi] == "dense" else fac["T"][gi], gdt, maps)
            Q, Cm = maps
            X = Rc[gi].reshape((S, L, M, g.C, nz))
            y = X / (1.0 + gdt * d["damp"])                                         # the other species: damping only
            if self.kind[gi] == "thomas":                                           # block-Thomas along ℓ
                P, tm = fac["P"][gi], fac["tm"][gi]
                ys, y01 = self._thomas_down(tm, X[e], P, gdt)
                y01 = y[:, :2].at[e].set(y01)
                f = jnp.einsum("scfzam,samcz->cfz", Q.astype(y.dtype), y01)
                w = lu_solve(lu, f.reshape(g.C, F * nz, 1))[..., 0].reshape(g.C, F, nz)
                cw = jnp.einsum("scamzf,cfz->samcz", Cm.astype(y.dtype), w)
                Xo = y.at[:, :2].add(-cw * (gdt * d["damp"] / (1.0 + gdt * d["damp"])))
                Xo = Xo.at[e].set(self._thomas_up(tm, ys, cw[e], P, gdt))
                Xo = Xo * d["gauge"] + X * (1.0 - d["gauge"])
                out.append(Xo.reshape((S * L, M, g.C, nz)))
                continue
            Xe = jnp.transpose(X[e], (2, 0, 1, 3)).reshape(g.C, L * M * nz)
            ye = jnp.matmul(Xe, kinv.T)                                              # (C, n)
            y = y.at[e].set(jnp.transpose(ye.reshape(g.C, L, M, nz), (1, 2, 0, 3)))
            f = jnp.einsum("scfzam,samcz->cfz", Q.astype(y.dtype), y[:, :2])          # (C, F, nz)
            w = lu_solve(lu, f.reshape(g.C, F * nz, 1))[..., 0].reshape(g.C, F, nz)     # batched 2D solves
            cw = jnp.einsum("scamzf,cfz->samcz", Cm.astype(y.dtype), w)               # (S, 2, M, C, nz)
            # X = y + (K⁻¹C − C) w: electrons through K⁻¹'s ℓ = 0, 1 columns, the others through their damping
            ke_ = jnp.transpose(cw[e], (2, 0, 1, 3)).reshape(g.C, 2 * M * nz)
            corr = jnp.matmul(ke_, kinv[:, :2 * M * nz].T)                           # (C, n)
            Xo = y.at[:, :2].add(-cw * (gdt * d["damp"] / (1.0 + gdt * d["damp"])))
            Xo = Xo.at[e].set(y[e] + jnp.transpose(corr.reshape(g.C, L, M, nz), (1, 2, 0, 3))
                              - jnp.concatenate([cw[e], jnp.zeros((L - 2, M, g.C, nz), y.dtype)], axis=0))
            Xo = Xo * d["gauge"] + X * (1.0 - d["gauge"])                             # the k_x = k_y = 0 mode: identity
            out.append(Xo.reshape((S * L, M, g.C, nz)))
        return self.scatter(out, lead) + self.project_out(R)

    # -------------------------------------------------------------- the explicit part's CFL bound
    def explicit_rate(self, fields):
        """Sum of the per-term spectral radii of the explicit part (as kinetic.cfl_rate_kinetic), the maximum over
        species: everything but the implicit species' streaming, mirror force and hypercollisions (only the x-variation
        remainder of their parallel operator, ``x_resid`` of it) and both species' end damping."""
        import jax.numpy as jnp
        terms = iter(self.explicit_terms(fields).values())
        rate = next(terms)
        for v in terms:                             # summed in the order of the terms
            rate = rate + v
        return jnp.max(rate)

    def explicit_terms(self, fields, detail: bool = False) -> dict:
        """The terms of :meth:`explicit_rate`, each per species (v_t/R), in the order it sums them; ``detail`` adds the
        parallel term's parts (``.stream``, ``.mirror``, ``.hkz``) and the nonlinear term's (``.exb``, ``.flutter``),
        which are not summed (their keys start with a dot)."""
        import jax.numpy as jnp
        from velcro import kinetic as ke
        from velcro.vlasov import drift_abs
        op = self.sim.op
        b = op.base
        L, M = self.L, self.M
        _, _, chi, Pn_phi, Pn_apar = fields
        vts = jnp.abs(op.vt)
        gmax = jnp.max(jnp.abs(op.c_stream[0])) / vts[0]
        mmax = jnp.max(jnp.abs(op.c_mirror[0])) / vts[0]
        kz = op.stream_kmax if op.par_fourier else ke._FD4 / b.dz
        par = jnp.asarray([s in self.par for s in range(self.S)])
        stream_full = jnp.asarray(op.wave_factor) * vts * gmax * kz
        stream_expl = jnp.asarray(self.wave_expl) * vts * gmax * kz             # with the field: see __init__
        mirror = b.rad_mirror * vts * mmax
        hkz = op.hkz * jnp.max(op.ell_p) * jnp.max(op.kz_abs) if op.use_hkz else jnp.zeros_like(vts)
        t = {"parallel": jnp.where(par, self.x_resid * (stream_full + mirror), stream_expl + mirror + hkz)}
        drift_ky = 0.0 if b.split_drift_ky else drift_abs(b, "y") * b.kymax
        t["drift"] = self.rad_energy * jnp.abs(op.T_over_Z) * jnp.max(drift_ky + drift_abs(b, "x") * self.kx_drift)
        t["collisions"] = jnp.abs(op.nu) * ((L - 1 + 2 * (M - 1)) if not b.coll_dense else b.rad_coll)
        if op.ei:
            t["ei"] = jnp.abs(op.ei_nu) * (L - 1 + 2 * (M - 1))
        if op.landau_nu is not None:
            t["landau"] = jnp.abs(op.landau_nu) * op.landau_rad
        if op.flr_b is not None:
            t["flr"] = jnp.abs(op.nu) * op.flr_rad
            if op.flr_rad_ei is not None:
                t["flr_ei"] = jnp.abs(op.ei_nu if op.ei else op.landau_nu) * op.flr_rad_ei
        if op.khyp is not None:
            t["khyp"] = jnp.max(op.khyp)
        if getattr(op, "chain_hz", None) is not None:   # GX's hyperz along the chains (every species): up to gx_hyperz
            t["gx_hyperz"] = jnp.full_like(vts, op.gx_hyperz)
        if b.nonlinear:
            vy, vx = self._advection_speeds(Pn_phi)
            if detail:
                t[".exb"] = jnp.max(b.c_Cinv) * (vy * b.kymax + vx * self.kx_nl)
            if Pn_apar is not None:                 # the flutter part: v∥ b̃, |v∥| ≤ ρ(V) v̂t_s
                ay, ax = self._advection_speeds(Pn_apar)
                if detail:
                    t[".flutter"] = jnp.max(b.c_Cinv) * (b.rad_v * vts * ay * b.kymax + b.rad_v * vts * ax * self.kx_nl)
                vy, vx = vy + b.rad_v * vts * ay, vx + b.rad_v * vts * ax
            t["nonlinear"] = jnp.max(b.c_Cinv) * (vy * b.kymax + vx * self.kx_nl)
        if detail:
            t[".stream"] = jnp.where(par, self.x_resid * stream_full, stream_expl)
            t[".mirror"] = jnp.where(par, self.x_resid * mirror, mirror)
            t[".hkz"] = jnp.where(par, 0.0, hkz)
            return t
        return {k: v for k, v in t.items() if not k.startswith(".")}

    def _advection_speeds(self, P):
        """(max |∂_x P|, max |∂_y P|) per species of the node values P (S, Nmu, Nx, Nky, Nz): the E×B (or flutter) speeds
        along y and x.  ``imex_nl_bound = "sum"``: kinetic.cfl_rate_kinetic's Σ_ky |·| (an upper bound of the real-space
        values, FD4 ∂_x); ``"max"``: the real-space maxima on the bracket's own grid and ∂_x."""
        import jax.numpy as jnp
        from velcro.operators import _bracket_matrices, _ky_key, _to_real_pair, ddx, real_fourier_dx
        op = self.sim.op
        b = op.base
        if self.nl_bound == "sum":
            w = jnp.where(jnp.arange(P.shape[-2]) == 0, 1.0, 2.0)[:, None]
            vy = jnp.max(jnp.sum(w * jnp.abs(ddx(P, b.dx)), axis=-2), axis=(1, 2, 3))
            vx = jnp.max(jnp.sum(w * jnp.abs(b.ky * P), axis=-2), axis=(1, 2, 3))
            return vy, vx
        fwd2, _ = _bracket_matrices(P.shape[-2], b.Ny, _ky_key(b.ky))
        pr = _to_real_pair(P, fwd2)                                  # (S, Nmu, Nx, 2 Ny, Nz): [P | ∂_y P] in real y
        dxf = real_fourier_dx(op.kx_sym) if op.bracket_fourier else (lambda f: ddx(f, b.dx))
        vy = jnp.max(jnp.abs(dxf(pr[..., :b.Ny, :])), axis=(1, 2, 3, 4))
        vx = jnp.max(jnp.abs(pr[..., b.Ny:, :]), axis=(1, 2, 3, 4))
        return vy, vx


# ------------------------------------------------------------------------------------------------ the adaptive bound
def arnoldi_hessenberg(lin, v0, m: int):
    """The (m + 1) × m Hessenberg matrix of m Arnoldi steps of the real-linear map ``lin`` (complex arrays of v0's shape)
    from v0, in the real inner product Re Σ a*b (classical Gram–Schmidt, twice; jittable, m + 1 vectors held).  The
    projections take Re(V w̄), equal to Re(V̄ w) term by term: conjugating the basis materialised a second copy of it
    (6.6 of the 13.3 GiB on GX's JET box at (24,8)).  At an
    invariant subspace (the new vector's norm below 100 ε of the image's) the iteration stops: the rest of H stays 0."""
    import jax
    import jax.numpy as jnp
    shape, n = v0.shape, v0.size
    rdt = jnp.zeros((), v0.dtype).real.dtype
    tol = 100.0 * float(jnp.finfo(rdt).eps)
    v0 = v0.reshape(n)
    V = jnp.zeros((m + 1, n), v0.dtype).at[0].set(v0 / jnp.linalg.norm(v0))
    H = jnp.zeros((m + 1, m), rdt)
    idx = jnp.arange(m + 1)

    def body(j, carry):
        V, H = carry
        w = lin(V[j].reshape(shape)).reshape(n)
        w0 = jnp.linalg.norm(w)
        for _ in range(2):
            h = jnp.where(idx <= j, jnp.real(V @ jnp.conj(w)), 0.0)
            w = w - h.astype(V.dtype) @ V
            H = H.at[:, j].add(h)
        nrm = jnp.linalg.norm(w)
        ok = nrm > tol * w0
        H = H.at[j + 1, j].set(jnp.where(ok, nrm, 0.0))
        return V.at[j + 1].set(jnp.where(ok, w / jnp.where(ok, nrm, 1.0), 0.0)), H
    return jax.lax.fori_loop(0, m, body, (V, H))[1]


def ritz_values(H) -> np.ndarray:
    """The Ritz values of an Arnoldi Hessenberg matrix (m + 1) × m, truncated at the first invariant subspace (a zero
    subdiagonal entry), whose leading block's values are exact eigenvalues."""
    H = np.asarray(H, dtype=float)
    if not np.all(np.isfinite(H)):
        raise FloatingPointError("Arnoldi: non-finite Hessenberg entries (the state or its Jacobian is not finite)")
    m = H.shape[1]
    k = next((j + 1 for j in range(m) if H[j + 1, j] == 0.0), m)
    return np.linalg.eigvals(H[:k, :k])


class JacobianRate:
    """``numerics.imex_rate = "jacobian"``: the rate of ``dt_from_cfl`` from the explicit part's own spectral radius at the
    state instead of :meth:`ImexSystem.explicit_rate`, which sums per-term upper bounds and sits ≈ 2.7× above the radius
    at the saturated q 1.4 state (its flutter term alone is 4× its Jacobian's).

    Each call runs ``numerics.imex_rate_krylov`` = m steps of Arnoldi on J_E = ∂f_E/∂G at G (``jax.linearize`` of
    f_E = f − f_I: every explicit linear term and the E×B and flutter Jacobians at the state, in the shearing frame at the
    current θ) in the real inner product Re Σ a*b — the state space is real: the zonal row is real and the bracket
    couples k_y with −k_y, so J_E is real-linear, not complex-linear.  The start is seeded noise (one seed per call,
    reproducible) projected onto the evolved space (a real zonal row without its k_z Nyquist, the retained k_x, no
    k_x = k_y = 0 mode).  ρ̂ = the largest |Ritz value|: a lower estimate of the radius, close when the outer spectrum is
    a continuum.  The rate returned is max(ρ̂, ``imex_rate_floor`` × B), B the per-term bound, so the step is never longer
    than 1/floor times the bound's.  A linear run outside the shearing frame has a constant J_E: one estimate serves it.
    ``log`` keeps (B, ρ̂, Re λ̂, Im λ̂, rate) per call (λ̂ the Ritz value of largest modulus)."""

    def __init__(self, sim, imp: ImexSystem, f_full, fields, bound, frame: bool = False):
        import jax
        import jax.numpy as jnp
        from velcro import kinetic as ke
        num = sim.cfg.numerics
        op = sim.op
        self.m = int(num.imex_rate_krylov)
        self.floor = float(num.imex_rate_floor)
        self.bound = bound                          # (G, θ) → B
        self.const = not (op.base.nonlinear or frame)
        self.calls = 0
        self.log = []
        self._const = None
        zonal_nyq, x_dealias = bool(op.zonal_nyq), bool(op.x_dealias)

        def f_E(Y, th):
            F = fields(Y) if th is None else fields(Y, th)
            full = f_full(Y, F) if th is None else f_full(Y, F, th)
            return full - imp.f_imp(imp.h_of(Y, F[2]))

        def project(D):
            D = D.at[:, :, :, 0, :].set(jnp.real(D[:, :, :, 0, :]).astype(D.dtype))
            if zonal_nyq and D.shape[-1] % 2 == 0:
                u = (1.0 - 2.0 * (jnp.arange(D.shape[-1]) % 2)).astype(D.real.dtype)
                D = D.at[:, :, :, 0, :].add(-jnp.mean(D[:, :, :, 0, :] * u, axis=-1, keepdims=True) * u)
            if x_dealias:
                D = ke.x_project(D, op)
            return ke.no_gauge_state(D)

        self.f_E, self.project = f_E, project
        self.matvec = jax.jit(lambda G, th, v: jax.jvp(lambda Y: f_E(Y, th), (G,), (v,))[1])   # J_E v (tests, tools)
        self._hess = {}

    def ritz(self, G, theta=None, m: int | None = None, seed: int = 0) -> np.ndarray:
        """The Ritz values of an m-step Arnoldi run (default ``imex_rate_krylov``) at G from the start of ``seed``."""
        import jax
        m = int(m or self.m)
        if m not in self._hess:
            f_E, project = self.f_E, self.project

            def hessenberg(G, th, key):
                _, lin = jax.linearize(lambda Y: f_E(Y, th), G)
                return arnoldi_hessenberg(lin, project(jax.random.normal(key, G.shape, dtype=G.dtype)), m)
            self._hess[m] = jax.jit(hessenberg)
        return ritz_values(self._hess[m](G, theta, jax.random.PRNGKey(seed)))

    def radius(self, G, theta=None):
        """(ρ̂, λ̂): the largest |Ritz value| of the m-step Arnoldi run at G (θ in the shearing frame) and that value."""
        if self.const and self._const is not None:
            return self._const
        ev = self.ritz(G, theta, seed=1_000_003 + self.calls)
        self.calls += 1
        i = int(np.argmax(np.abs(ev)))
        out = (float(np.abs(ev[i])), complex(ev[i]))
        if self.const:
            self._const = out
        return out

    def __call__(self, G, theta=None) -> float:
        B = float(self.bound(G, theta))
        rho, lam = self.radius(G, theta)
        rate = max(rho, self.floor * B)
        self.log.append((B, rho, lam.real, lam.imag, rate))
        return rate


# ------------------------------------------------------------------------------------------------ the step
def make_step(sim, imp: ImexSystem, f_full, fields, frame: bool = False):
    """``step(G, dt, fac[, theta, rate]) → G`` of the Lawson ARK: ``f_full(G, F[, θ])`` is the run's right-hand side
    (with the binormal drift of G split out when op.base.split_drift_ky), ``fields(G[, θ])`` kinetic_fields."""
    import jax.numpy as jnp
    tab = imp.tab
    op = sim.op
    s = tab.s
    AE, AI, bE, bI, c = tab.AE, tab.AI, tab.bE, tab.bI, tab.c
    lawson = bool(op.base.split_drift_ky)
    taus = imp.taus()
    tidx = lambda tau: next(k for k, u in enumerate(taus) if abs(u - tau) < 1e-14)
    need_E = [bool(np.any(AE[i + 1:, i] != 0) or bE[i] != 0) for i in range(s)]
    need_I = [bool(np.any(AI[i + 1:, i] != 0) or bI[i] != 0) for i in range(s)]
    if tab.gsa:
        need_E = [bool(np.any(AE[i + 1:, i] != 0)) for i in range(s)]
        need_I = [bool(np.any(AI[i + 1:, i] != 0)) for i in range(s)]
    elif tab.stiffly_accurate:
        need_I = [bool(np.any(AI[i + 1:, i] != 0)) for i in range(s)]

    def prop(X, tau, fac):
        if not lawson or abs(tau) < 1e-14:
            return X
        return op.propagate(X, fac["phase"][tidx(tau)])

    deduce = bool(sim.cfg.numerics.imex_deduce_fi)

    def evals(Y, i, th, fI=None):
        """(f_E, f_I) at stage i; ``fI`` given = deduced from the stage equation, (Y_i − r_i)/(hγ), instead of evaluated."""
        if not (need_I[i] or need_E[i]):
            return None, None
        F = fields(Y) if th is None else fields(Y, th)
        if fI is None:
            fI = imp.f_imp(imp.h_of(Y, F[2]))
        fE = None
        if need_E[i]:
            full = f_full(Y, F) if th is None else f_full(Y, F, th)
            fE = full - fI
        return fE, (fI if need_I[i] else None)

    def step(G, dt, fac, th=None, rate=None):
        th_of = (lambda ci: None) if th is None else (lambda ci: th + ci * dt * rate)
        FE, FI = [None] * s, [None] * s
        Y = G
        FE[0], FI[0] = evals(G, 0, th_of(c[0]))
        for i in range(1, s):
            terms = {}
            for j in range(i):
                acc = None
                if AE[i, j] != 0 and FE[j] is not None:
                    acc = (dt * AE[i, j]) * FE[j]
                if AI[i, j] != 0 and FI[j] is not None:
                    t = (dt * AI[i, j]) * FI[j]
                    acc = t if acc is None else acc + t
                if j == 0:
                    acc = G if acc is None else G + acc
                if acc is None:
                    continue
                tau = c[i] - c[j]
                key = round(tau, 14)
                terms[key] = (tau, acc if key not in terms else terms[key][1] + acc)
            R = None
            for tau, acc in terms.values():
                v = prop(acc, tau, fac)
                R = v if R is None else R + v
            Y = imp.solve(R, fac, th_of(c[i]))
            fI = (Y - R) * (1.0 / (dt * AI[i, i])) if deduce else None
            FE[i], FI[i] = evals(Y, i, th_of(c[i]), fI)
        if tab.gsa:
            return Y
        if tab.stiffly_accurate:            # the f_I terms of the last stage's sum are bI's: only the explicit remainder
            out = Y
            for j in range(s):
                wgt = bE[j] - AE[s - 1, j]
                if wgt != 0:
                    out = out + prop((dt * wgt) * FE[j], 1.0 - c[j], fac)
            return out
        out = prop(G, 1.0, fac)
        for j in range(s):
            acc = (dt * bE[j]) * FE[j] if bE[j] != 0 else None
            if bI[j] != 0:
                acc = (dt * bI[j]) * FI[j] if acc is None else acc + (dt * bI[j]) * FI[j]
            if acc is not None:
                out = out + prop(acc, 1.0 - c[j], fac)
        return out

    return step
