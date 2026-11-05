"""Kinetic electrons: a second kinetic species, quasineutrality with both species kinetic, and electromagnetic A∥.

Model, normalisation and the time-stepping decision: the validation notes.  In brief (Velcro units,
v_t = √(2T/m); species s has charge Z_s, mass m̂_s, temperature T̂_s, density n̂_s, thermal speed v̂t_s = √(T̂_s/m̂_s)):

* the state is GX's g formulation, g_s = ⟨δf_s⟩ + Z_s v∥⟨A∥⟩F₀/T_s, so both fields are instantaneous functions of G and
  the state stays G alone.  The species are stacked along the Hermite axis: G has shape ``(S·L, M, N_x, N_ky, N_z)``,
  species-major (ions first), so every tool that slices k_y (axis 3) or x (axis 2) keeps working;
* χ = φ − v∥A∥ has the Hermite–Laguerre rows χ_s,0m = 𝒥_m(b_s)φ and χ_s,1m = −(v̂t_s/√2)𝒥_m(b_s)A∥, and
  H_s = G_s + (Z_s/T̂_s)χ_s.  Each species obeys the ion equation of :mod:`velcro.vlasov` with streaming and mirror force
  scaled by v̂t_s acting on H_s, the ∇B/curvature drift acting on (T̂_s/Z_s)H_s, the drive acting on χ_s, the bracket
  −½C⁻¹({J0φ, G_s} − v̂t_s V·{J0A∥, G_s}) (E×B and magnetic flutter), and Dougherty collisions on H_s;
* optional electron–ion collisions (``physics.ei_collisions``, GX's ``ei_colls = true``; GX device_funcs.cu rhs_linear):
  on the electrons ν_ei = Z_i ν_e is added to ν_e on the Dougherty diagonal −ν(ℓ + 2m) and in the energy restoring, and
  the momentum restoring pulls the electron flow toward the ion flow at the rate ν_ei:
      ∂_t H_e,10 ∋ −ν_ei H_e,10 + ν_ei (v̂t_i/v̂t_e) u_i,   u_i = Σ_m 𝒥_m(b_i) H_i,1m,
  with u_i = 0 when β = 0 (GX takes u_i from Ampère's law, k⊥²B²A∥/as_i − (Z n v̂t u)_e/(Z n v̂t)_i, which is exactly this
  u_i when β > 0 and which it sets to 0 when β = 0).  The ions feel no collisions with the electrons (as in GX).  As for
  Velcro's like-species operator the electron operator is the k⊥ = 0 Dougherty: GX's b_e-dependent pieces (−ν b_e on the
  diagonal, the u⊥ restoring, the 𝒥_ℓ(b_e) weights) are O(b_e), b_e = (m_e/m_i) b_i, and are dropped;
* quasineutrality Σ_s Z_s n̂_s Σ_m 𝒥_m(b_s)G_s,0m = [Σ_s (Z_s²n̂_s/T̂_s)(1 − Γ₀(b_s))] φ and Ampère
  [k⊥² + β Σ_s (Z_s²n̂_s/m̂_s)Σ_m𝒥_m(b_s)²] A∥ = √2 β Σ_s Z_s n̂_s v̂t_s Σ_m 𝒥_m(b_s)G_s,1m are solved with dense
  N_x × N_x operators per (k_y, z), as :mod:`velcro.fields` does; no adiabatic response, no flux-surface-average
  special case.  As in GX, the (k_x = 0, k_y = 0) component (Velcro's x mean at k_y = 0) carries no field and is not
  evolved;
* optional GX-type hypercollisions −ν_h,s ℓ^p |k_z| G_ℓ (ℓ ≥ 3), |k_z| exact on the Bloch-periodic field line at each
  (x, k_y).

Scope: flux tube (constant profiles), axisymmetric geometry, ``drive_gradB = False``, no ``ext_zonal``, no source; δB∥
(``physics.bpar``) and electron–ion collisions (``physics.ei_collisions``) are options.  Time stepping is explicit (Lawson RK4 with the binormal drift of every
species advanced exactly), so ``jax.jvp`` of a step needs nothing special.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from velcro import basis
from velcro.operators import (BRACKETS, _bracket_integrand, _bracket_matrices, _ky_key, _to_real_pair, _to_spec_pair, ddx,
                              ddz, gql_bracket, hyper_x, hyper_x_sheared, hyper_z, ifft_unscaled, kx_grid, to_real_y,
                              to_spec_y)
from velcro.vlasov import (VlasovOperators, apply_banded, apply_tridiag_l, contract, drift_abs, flr_increment_hat,
                           propagate_nodes)

SQRT2 = float(np.sqrt(2.0))
_FD4 = 1.3722          # max |symbol| of the 4th-order centred first derivative, in units of 1/Δ (as vlasov.py)


def _ifft_sym(sym, Fh, axis: int):
    """``ifft(sym ⊙ Fh)`` along ``axis`` with the 1/N folded into the symbol and the inverse taken as a forward FFT of
    the conjugate (:func:`velcro.operators.ifft_unscaled`): no separate scaling pass after the transform on the GPU.
    Every complex inverse FFT of the kinetic-electron step goes through here, and the step has no complex-to-real one:
    the bracket takes its x-derivatives on the spectral fields (:func:`_bracket`).  (Packing two real arrays into one
    complex FFT was rejected by the fp64 identity gate: the smaller array, A∥'s flux, inherits rounding error of its
    larger partner's size.)"""
    return ifft_unscaled(Fh * (sym * (1.0 / Fh.shape[axis])), axis=axis)


# ------------------------------------------------------------------------------------------------ species
@dataclass(frozen=True)
class Species:
    """A kinetic species in the flux tube (constant T̂, n̂); ``omega_n`` / ``omega_T`` are (N_x,) gradient profiles."""
    name: str
    Z: float
    mass: float
    temp: float
    dens: float
    nu: float
    omega_n: np.ndarray
    omega_T: np.ndarray

    @property
    def vt(self) -> float:
        return float(np.sqrt(self.temp / self.mass))


def profile_array(spec, x: np.ndarray, Lx: float) -> np.ndarray:
    """ω(x) of a :class:`velcro.config.ProfileSpec` (the logic of ``Simulation._profiles``)."""
    if spec.kind == "constant":
        return spec.value * np.ones_like(x)
    if spec.kind == "peaked":
        return spec.value / np.cosh(x / spec.width) ** 2
    if spec.kind == "plateau":
        xl, xr = -0.5 * Lx + spec.edge, 0.5 * Lx - spec.edge
        return spec.value * (1.0 - 1.0 / np.cosh((x - xl) / spec.width) ** 2 - 1.0 / np.cosh((x - xr) / spec.width) ** 2)
    xl, xr = x.min() + spec.edge, x.max() - spec.edge
    return spec.value * 0.5 * (np.tanh((x - xl) / spec.width) - np.tanh((x - xr) / spec.width))


def check_config(cfg, geom) -> None:
    """Refuse the set-ups the kinetic-electron path does not implement (instead of silently dropping physics)."""
    ph, nu = cfg.physics, cfg.numerics
    bad = []
    if ph.bpar and ph.beta <= 0:
        bad.append("physics.bpar without physics.beta > 0")
    if ph.Ti_profile or ph.n_profile:
        bad.append("T_i / n profiles (set physics.Ti_profile = physics.n_profile = false: flux tube only)")
    if not geom.axisymmetric:
        bad.append("non-axisymmetric geometry")
    if ph.drive_gradB:
        bad.append("physics.drive_gradB = true")
    if ph.ext_zonal is not None and ph.ext_zonal.amp != 0.0:
        bad.append("physics.ext_zonal")
    if ph.exb_shear is not None and (ph.exb_shear.gamma_E != 0.0 or ph.exb_shear.u_prime != 0.0):
        # the shearing frame with kinetic electrons (the validation notes): every symbol of the FFT field solver at
        # k_x − θ, the drift's ∂_x − iθ, the frame bracket for E×B and flutter, hyper_x / gx_hyper at k_x − θ, the
        # remap of both species; the chains and the parallel operators commute with e^{iθx} (unchanged).  What is not
        # carried to k_x − θ is refused rather than left at the grid k_x
        from velcro.vlasov import exb_frame_kind
        ex = ph.exb_shear
        if ex.gamma_E != 0.0 and exb_frame_kind(cfg) == "local":
            bad.append("physics.exb_shear with frame = 'local' (the kinetic path has the shearing frame only)")
        if ex.gamma_E != 0.0:
            if ph.bpar:
                bad.append("physics.exb_shear with physics.bpar (the coupled δB∥ solve is not evaluated at k_x − θ)")
            if ph.collision_flr:
                bad.append("physics.exb_shear with physics.collision_flr (its b_s is at the grid k_x)")
            if ph.ei_collisions and ph.beta > 0:
                bad.append("physics.exb_shear with ei_collisions at β > 0 (the ion flow's 𝒥_m(b_i) is at the grid k_x)")
    if nu.source_rate > 0:
        bad.append("numerics.source_rate > 0")
    if nu.field_solver == "lowrank":
        bad.append("numerics.field_solver = 'lowrank' (use 'dense' or 'fft')")
    if nu.kpar_floor > 0 and nu.parallel_derivative == "fd4":
        bad.append("numerics.kpar_floor with parallel_derivative = 'fd4' (the floor acts on the Fourier symbol)")
    if nu.gx_nyquist and nu.parallel_bc != "gx_chains":
        bad.append("numerics.gx_nyquist without parallel_bc = 'gx_chains'")
    if ph.ei_collisions and ph.collision_model not in ("dougherty", "landau", "gs2"):
        bad.append("physics.ei_collisions with collision_model ≠ 'dougherty' / 'landau' / 'gs2'")
    if ph.ei_collisions and ph.collision_model == "gs2" and ph.beta > 0:
        bad.append("physics.ei_collisions with collision_model = 'gs2' at β > 0 (GS2 then adds the electrons' drag toward "
                   "the ion flow, which velcro/gs2_collisions.py does not carry)")
    if ph.ei_collisions and ph.collision_model == "landau" and ph.beta > 0:
        bad.append("physics.ei_collisions with collision_model = 'landau' at β > 0 (the Landau e–i path has no drag toward "
                   "the ion flow — no e←i field block — so its friction would act on u_e instead of u_e − u_i; the "
                   "Dougherty path forms u_i at β > 0)")
    if bad:
        raise ValueError("physics.electrons = 'kinetic' does not support: " + "; ".join(bad))


def species_list(cfg, prof: dict, x: np.ndarray) -> list:
    """[ions, electrons] from the config: the ions take the physics.* gradients and ν, the electrons physics.electron."""
    ph = cfg.physics
    e = ph.electron
    ions = Species("ion", 1.0, 1.0, 1.0, 1.0, float(ph.nu), np.asarray(prof["omega_n"]), np.asarray(prof["omega_T"]))
    on = prof["omega_n"] if e.omega_n is None else profile_array(e.omega_n, x, cfg.grid.Lx)
    oT = prof["omega_T"] if e.omega_T is None else profile_array(e.omega_T, x, cfg.grid.Lx)
    elec = Species(e.name, float(e.Z), float(e.mass), float(e.temp), float(e.dens), float(e.nu),
                   np.asarray(on), np.asarray(oT))
    return [ions, elec]


# ------------------------------------------------------------------------------------------------ fields
def _quantiser(geom, ky: np.ndarray, Lx: float):
    """The Kohn–Nirenberg quantisation of :func:`velcro.fields.build_field_solver`: a real symbol (N_x, N_z, N_kx, N_ky)
    evaluated at the output point x → Hermitian N_x × N_x operators (N_ky, N_z, N_x, N_x); plus the metric k⊥²."""
    Nx = len(geom.x)
    kx = kx_grid(Nx, Lx)
    gxx, gxy, gyy, B = (geom.y_averaged(a)[:, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    x = geom.x
    E = np.exp(1j * kx[None, None, :] * (x[:, None, None] - x[None, :, None])) / Nx
    Er, Ei = np.ascontiguousarray(E.real), np.ascontiguousarray(E.imag)
    kperp2 = (gxx[:, :, None, None] * kx[None, None, :, None] ** 2
              + 2 * gxy[:, :, None, None] * kx[None, None, :, None] * ky[None, None, None, :]
              + gyy[:, :, None, None] * ky[None, None, None, :] ** 2)                          # (Nx, Nz, Nkx, Nky)

    def quantise(symbol):
        op = np.matmul(Er[:, None], symbol) + 1j * np.matmul(Ei[:, None], symbol)               # (x, z, w, k)
        op = op.transpose(3, 1, 0, 2)
        return 0.5 * (op + np.conj(np.swapaxes(op, -1, -2)))

    return quantise, np.maximum(kperp2, 0.0), B


@dataclass
class KineticFieldSolver:
    """Quasineutrality (and Ampère's law when β > 0) with every species kinetic; operators as dense N_x × N_x matrices per
    (k_y, z) with the batch indices leading (the layout of :class:`velcro.fields.FieldSolver`)."""
    gyro: jnp.ndarray            # (S, Nky, Nz, M, Nx, Nx)   𝒥_m(b_s)
    Pinv: jnp.ndarray            # (Nky, Nz, Nx, Nx)          inverse polarisation, regularised on the k_x = 0 gauge at k_y = 0
    qn_w: jnp.ndarray            # (S,)                       Z_s n̂_s
    fsw: jnp.ndarray             # (Nx, Nz)
    M: int
    S: int
    L: int
    Ainv: Optional[jnp.ndarray] = None       # (Nky, Nz, Nx, Nx) inverse Ampère operator (β > 0)
    amp_w: Optional[jnp.ndarray] = None      # (S,)  √2 β Z_s n̂_s v̂t_s
    gyro_nodes: Optional[jnp.ndarray] = None  # (S, Nky, Nz, Nmu, Nx, Nx)  J0(√(2b_sμ_j))
    # δB∥ (physics.bpar): the coupled (φ, δB∥/B) inverse, the perpendicular-pressure weights and the μ-node operator
    Binv: Optional[jnp.ndarray] = None       # (Nky, Nz, 2Nx, 2Nx)
    perp_w: Optional[jnp.ndarray] = None     # (S,)  n̂_s T̂_s
    bfac: Optional[jnp.ndarray] = None       # (Nx, 1, Nz)  β/(2B²)
    gyroK_nodes: Optional[jnp.ndarray] = None  # (S, Nky, Nz, Nmu, Nx, Nx)  μ_j 2J1(a)/a, a = √(2b_sμ_j)

    @property
    def bpar(self) -> bool:
        return self.Binv is not None

    @staticmethod
    def _no_gauge(f):
        """Remove the x mean of the k_y = 0 row (GX's masked k_x = k_y = 0 mode)."""
        return f.at[:, 0, :].add(-jnp.mean(f[:, 0, :], axis=0))

    def moment(self, Gl, w):
        """Σ_s w_s Σ_m 𝒥_m(b_s) G_s,ℓm for the ℓ rows ``Gl`` (S, M, Nx, Nky, Nz) → (Nx, Nky, Nz)."""
        gt = jnp.transpose(Gl, (0, 3, 4, 1, 2))                                   # (S, Nky, Nz, M, Nx)
        out = jnp.einsum("skzmxw,skzmw->skzx", self.gyro, gt)
        out = jnp.einsum("s,skzx->kzx", w.astype(out.real.dtype), out)
        return jnp.transpose(out, (2, 0, 1))

    def _invert(self, Minv, rhs):
        rhs = self._no_gauge(rhs)
        return self._no_gauge(jnp.einsum("kzxw,wkz->xkz", Minv, rhs))

    def solve(self, G):
        """(φ, A∥) from the state G (S·L, M, Nx, Nky, Nz); A∥ is None when β = 0 (φ includes the δB∥ coupling)."""
        phi, apar, _ = self.solve_all(G)
        return phi, apar

    def solve_all(self, G):
        """(φ, A∥, δB∥/B); the last two are None when absent."""
        Gs = G.reshape((self.S, self.L) + G.shape[1:])
        apar = None if self.Ainv is None else self._invert(self.Ainv, self.moment(Gs[:, 1], self.amp_w))
        if not self.bpar:
            return self._invert(self.Pinv, self.moment(Gs[:, 0], self.qn_w)), apar, None
        # quasineutrality  P φ − Q b = Σ Zn Σ𝒥 G_0;  pressure balance  f Q φ + (1 + f ΣnT Σ𝒦²) b = −f ΣnT Σ𝒦 G_0 (f = β/2B²)
        rho = self._no_gauge(self.moment(Gs[:, 0], self.qn_w))
        sig = self._no_gauge(-self.bfac * self.momentK(Gs[:, 0], self.perp_w))
        Nx = G.shape[2]
        rs = jnp.concatenate([rho, sig], axis=0)                                  # (2Nx, Nky, Nz)
        out = jnp.einsum("kzxw,wkz->xkz", self.Binv, rs)
        return self._no_gauge(out[:Nx]), apar, self._no_gauge(out[Nx:])

    def momentK(self, Gl, w):
        """Σ_s w_s Σ_m 𝒦_m G_s,ℓm with 𝒦_m = 𝒥_m − 𝒥_{m−1} (the Laguerre coefficients of μ̂·2J1(a)/a):
        Σ_m 𝒦_m G_m = Σ_m 𝒥_m (G_m − G_{m+1})."""
        return self.moment(Gl - jnp.concatenate([Gl[:, 1:], jnp.zeros_like(Gl[:, :1])], axis=1), w)

    def gyroaverage(self, f):
        """𝒥_m(b_s) f per species → (S, M, Nx, Nky, Nz)."""
        out = jnp.einsum("skzmxw,kzw->skzmx", self.gyro, jnp.transpose(f, (1, 2, 0)))
        return jnp.transpose(out, (0, 3, 4, 1, 2))

    def gyroaverageK(self, f):
        """𝒦_m(b_s) f = 𝒥_m f − 𝒥_{m−1} f per species → (S, M, Nx, Nky, Nz)."""
        J = self.gyroaverage(f)
        return J - jnp.concatenate([jnp.zeros_like(J[:, :1]), J[:, :-1]], axis=1)

    def gyroaverageK_nodes(self, f):
        """μ_j 2J1(a)/a f at the Gauss–Laguerre nodes per species → (S, Nmu, Nx, Nky, Nz)."""
        out = jnp.einsum("skzmxw,kzw->skzmx", self.gyroK_nodes, jnp.transpose(f, (1, 2, 0)))
        return jnp.transpose(out, (0, 3, 4, 1, 2))

    def gyroaverage_nodes(self, f):
        """J0(√(2b_sμ_j)) f at the Gauss–Laguerre nodes per species → (S, Nmu, Nx, Nky, Nz)."""
        out = jnp.einsum("skzmxw,kzw->skzmx", self.gyro_nodes, jnp.transpose(f, (1, 2, 0)))
        return jnp.transpose(out, (0, 3, 4, 1, 2))


def _K_nodes(b, mu):
    """μ·2J1(a)/a with a = √(2bμ) (→ μ as a → 0): the δB∥ gyroaverage at a Gauss–Laguerre node."""
    from scipy.special import j1
    a = np.sqrt(2.0 * b * mu)
    return mu * np.where(a > 1e-8, 2.0 * j1(a) / np.where(a > 1e-8, a, 1.0), 1.0 - a ** 2 / 8.0)


def build_kinetic_field_solver(geom, ky: np.ndarray, species: list, M: int, L: int, Lx: float, beta: float,
                               dtype=np.complex128, node_gyro: bool = False, bpar: bool = False) -> KineticFieldSolver:
    """Precompute the species gyro-operators, the polarisation inverse, (β > 0) the Ampère inverse and (``bpar``) the
    coupled quasineutrality / perpendicular-pressure-balance inverse."""
    Nx = len(geom.x)
    quantise, kperp2, B = _quantiser(geom, ky, Lx)
    S = len(species)
    gyro, nodes, knodes, pol, skin, qK, kk = [], [], [], 0.0, 0.0, 0.0, 0.0
    mu_nodes, _ = basis.laguerre_quadrature(M)
    for sp in species:
        b = 0.5 * (sp.temp * sp.mass / sp.Z ** 2) / (B ** 2)[:, :, None, None] * kperp2       # (Nx, Nz, Nkx, Nky)
        Jm = basis.gyro_coefficients(b, M)
        gyro.append(np.stack([quantise(j) for j in Jm], axis=2))                                  # (Nky, Nz, M, Nx, Nx)
        if node_gyro:
            from scipy.special import j0
            nodes.append(np.stack([quantise(j0(np.sqrt(2.0 * b * mu))) for mu in mu_nodes], axis=2))
            if bpar:
                knodes.append(np.stack([quantise(_K_nodes(b, mu)) for mu in mu_nodes], axis=2))
        pol = pol + (sp.Z ** 2 * sp.dens / sp.temp) * (1.0 - basis.gamma0(b))
        skin = skin + (sp.Z ** 2 * sp.dens / sp.mass) * np.sum(Jm ** 2, axis=0)
        Km = Jm - np.concatenate([np.zeros_like(Jm[:1]), Jm[:-1]], axis=0)
        qK = qK + sp.Z * sp.dens * np.sum(Jm * Km, axis=0)
        kk = kk + sp.dens * sp.temp * np.sum(Km ** 2, axis=0)
    P = quantise(pol)
    ones = np.ones((Nx, Nx)) / Nx
    P[0] = P[0] + ones[None]                            # k_y = 0: the k_x = 0 (x-mean) gauge, projected out in solve()
    Pinv = np.linalg.inv(P)
    Ainv = amp_w = Binv = perp_w = bfac = None
    if beta > 0:
        Ainv = jnp.asarray(np.linalg.inv(quantise(kperp2 + beta * skin)), dtype=dtype)
        amp_w = jnp.asarray([SQRT2 * beta * sp.Z * sp.dens * sp.vt for sp in species])
    if bpar:
        if beta <= 0:
            raise ValueError("physics.bpar needs physics.beta > 0")
        f = (beta / (2.0 * B ** 2)).T                                   # (Nz, Nx): the row factor at the output point
        Q = quantise(qK)
        Kq = quantise(kk)
        I = np.eye(Nx)
        R = f[None, :, :, None] * Q                                     # f ⊙ Q: (1/f)·R = Q, so the field matrix is symmetric
        Sb = I[None, None] + f[None, :, :, None] * Kq
        Bm = np.block([[P, -Q], [R, Sb]])                               # P already carries the k_y = 0 gauge regulariser
        Binv = jnp.asarray(np.linalg.inv(Bm), dtype=dtype)
        perp_w = jnp.asarray([sp.dens * sp.temp for sp in species])
        bfac = jnp.asarray((beta / (2.0 * B ** 2))[:, None, :])
    return KineticFieldSolver(
        gyro=jnp.asarray(np.stack(gyro), dtype=dtype), Pinv=jnp.asarray(Pinv, dtype=dtype),
        qn_w=jnp.asarray([sp.Z * sp.dens for sp in species]), fsw=jnp.asarray(geom.fs_weights()), M=M, S=S, L=L,
        Ainv=Ainv, amp_w=amp_w, gyro_nodes=jnp.asarray(np.stack(nodes), dtype=dtype) if node_gyro else None,
        Binv=Binv, perp_w=perp_w, bfac=bfac,
        gyroK_nodes=jnp.asarray(np.stack(knodes), dtype=dtype) if (node_gyro and bpar) else None)


@dataclass
class KineticFFTFieldSolver:
    """The flux-tube limit of :class:`KineticFieldSolver`: every symbol taken at the reference surface (x = 0) and applied
    as a multiplier on the FFT in x (the dense operators are then circulant: ½(A + A†) = F⁻¹ diag(a) F).  Differs from the
    dense solver only by Velcro's residual radial variation at ρ* = 1e-5 (≈ 4e-4 in B across a 95 ρ_ref box); stores
    O(N_x) numbers per (k_y, z) instead of N_x²."""
    J: jnp.ndarray               # (S, M, Nkx, Nky, Nz)   𝒥_m(b_s), real
    Pinv: jnp.ndarray            # (Nkx, Nky, Nz)          1/Σ_s (Z²n/T)(1 − Γ₀), 0 at k_x = k_y = 0
    qn_w: jnp.ndarray
    fsw: jnp.ndarray
    M: int
    S: int
    L: int
    Ainv: Optional[jnp.ndarray] = None      # (Nkx, Nky, Nz)  1/(k⊥² + β Σ (Z²n/m) Σ𝒥²), 0 at k_x = k_y = 0
    amp_w: Optional[jnp.ndarray] = None
    Jn: Optional[jnp.ndarray] = None        # (S, Nmu, Nkx, Nky, Nz)  J0(√(2b_sμ_j))
    # δB∥: the 2 × 2 inverse per mode, [[P, −Q], [fQ, 1 + fΣnTΣ𝒦²]]⁻¹ as its four entries (Nkx, Nky, Nz), 0 at k_x = k_y = 0
    B11: Optional[jnp.ndarray] = None
    B12: Optional[jnp.ndarray] = None
    B21: Optional[jnp.ndarray] = None
    B22: Optional[jnp.ndarray] = None
    perp_w: Optional[jnp.ndarray] = None
    bfac: Optional[jnp.ndarray] = None      # (1, 1, Nz)  β/(2B²) at x = 0
    Kn: Optional[jnp.ndarray] = None        # (S, Nmu, Nkx, Nky, Nz)  μ_j 2J1(a)/a
    # the shearing frame of physics.exb_shear (the validation notes): the data to evaluate every symbol at k_x − θ_ky
    # inside the jitted step — None without the frame (the precomputed arrays above are the θ = 0 symbols)
    frame: Optional[dict] = None

    @property
    def bpar(self) -> bool:
        return self.B11 is not None

    # ------------------------------------------------------------------ the frame's symbols at k_x − θ
    def frame_b(self, theta, dtype):
        """b_s(k_x − θ_ky, k_y, z) ≥ 0 → (S, Nkx, Nky, Nz); ``theta`` (Nky,) in 1/ρ_ref."""
        fr = self.frame
        kxs = fr["kx"].astype(dtype)[:, None] - jnp.reshape(theta, (-1,)).astype(dtype)[None, :]     # (Nkx, Nky)
        kxs = kxs[:, :, None]
        ky = fr["ky"].astype(dtype)[None, :, None]
        k2 = fr["gxx"].astype(dtype) * kxs ** 2 + 2 * fr["gxy"].astype(dtype) * kxs * ky + fr["gyy"].astype(dtype) * ky ** 2
        k2 = jnp.maximum(k2, 0.0)
        return fr["rho2"].astype(dtype)[:, None, None, None] * (k2 / fr["B2"].astype(dtype))[None], k2

    def frame_symbols(self, theta, dtype, nodes: bool = False):
        """(𝒥 (S, M, Nkx, Nky, Nz), J0 at the μ nodes (S, Nmu, …) or None, 1/polarisation, 1/Ampère or None), each at
        k_x − θ_ky; the k_x = k_y = 0 entries 0 (the gauge, as the precomputed arrays: θ_0 = 0 on the k_y = 0 row).
        With ``frame["qn_grid"]`` (physics.exb_shear.gx_qn_grid) the two denominators are the grid's (θ = 0, GX's
        qneutFacPhi computed once)."""
        from velcro.fields import _j0
        fr = self.frame
        b, k2 = self.frame_b(theta, dtype)                                          # (S, Nkx, Nky, Nz)
        m = jnp.arange(self.M, dtype=dtype).reshape((1, self.M, 1, 1, 1))
        bb = b[:, None]
        pos = bb > 0
        logb2 = jnp.log(jnp.where(pos, bb, 1.0) / 2.0)
        J = jnp.exp(m * logb2 - bb / 2.0 - jax.scipy.special.gammaln(m + 1))
        J = jnp.where(pos, J, jnp.where(m == 0, 1.0, 0.0))                         # 𝒥_m(0) = δ_m0
        gauge = fr["gauge"].astype(dtype)                                          # (Nkx, Nky, 1): 0 at k_x = k_y = 0
        Jn = None
        if nodes:
            mu = fr["mu"].astype(dtype).reshape((1, -1, 1, 1, 1))
            Jn = _j0(jnp.sqrt(2.0 * bb * mu))
        if fr["qn_grid"]:
            Pinv = self.Pinv.astype(dtype)
            Ainv = None if self.Ainv is None else self.Ainv.astype(dtype)
            return J, Jn, Pinv, Ainv
        w = fr["pol_w"].astype(dtype)[:, None, None, None]                          # Z²n/T per species
        pol = jnp.sum(w * (1.0 - jax.scipy.special.i0e(b)), axis=0)
        Pinv = gauge / jnp.where(gauge > 0, pol, 1.0)
        Ainv = None
        if self.Ainv is not None:
            skin = jnp.sum(fr["skin_w"].astype(dtype)[:, None, None, None] * jnp.sum(J ** 2, axis=1), axis=0)
            Ainv = gauge / jnp.where(gauge > 0, k2 + fr["beta"] * skin, 1.0)
        return J, Jn, Pinv, Ainv

    def _moment_hat(self, Gl, w, J=None):
        """Σ_s w_s Σ_m 𝒥_m Ĝ_s,ℓm in k_x space, (Nkx, Nky, Nz)."""
        Gh = jnp.fft.fft(Gl, axis=-3)
        J = self.J if J is None else J
        return jnp.einsum("s,smqkz->qkz", w.astype(Gh.real.dtype), J.astype(Gh.real.dtype) * Gh)

    def moment(self, Gl, w):
        return _ifft_sym(1.0, self._moment_hat(Gl, w), 0)

    def solve(self, G, theta=None):
        phi, apar, _ = self.solve_all(G, theta)
        return phi, apar

    def solve_all(self, G, theta=None):
        if theta is not None:
            return self._solve_all_frame(G, theta)
        Gs = G.reshape((self.S, self.L) + G.shape[1:])
        apar = None if self.Ainv is None else _ifft_sym(self.Ainv, self._moment_hat(Gs[:, 1], self.amp_w), 0)
        rho = self._moment_hat(Gs[:, 0], self.qn_w)
        if not self.bpar:
            return _ifft_sym(self.Pinv, rho, 0), apar, None
        G0 = Gs[:, 0]
        GK = G0 - jnp.concatenate([G0[:, 1:], jnp.zeros_like(G0[:, :1])], axis=1)      # Σ_m 𝒦_m G_m = Σ_m 𝒥_m (G_m − G_{m+1})
        sig = -self.bfac * self._moment_hat(GK, self.perp_w)
        phi = _ifft_sym(1.0, self.B11 * rho + self.B12 * sig, 0)
        bpar = _ifft_sym(1.0, self.B21 * rho + self.B22 * sig, 0)
        return phi, apar, bpar

    def _solve_all_frame(self, G, theta):
        """:meth:`solve_all` in the shearing frame: numerator and denominators at k_x − θ_ky (δB∥ refused)."""
        if self.frame is None:
            raise ValueError("this field solver was built without the shearing frame (physics.exb_shear)")
        if self.bpar:
            raise ValueError("physics.exb_shear with physics.bpar is not implemented")
        Gs = G.reshape((self.S, self.L) + G.shape[1:])
        J, _, Pinv, Ainv = self.frame_symbols(theta, G.real.dtype)
        apar = None if self.Ainv is None else _ifft_sym(Ainv, self._moment_hat(Gs[:, 1], self.amp_w, J), 0)
        return _ifft_sym(Pinv, self._moment_hat(Gs[:, 0], self.qn_w, J), 0), apar, None

    def gyroaverage(self, f, theta=None):
        fh = jnp.fft.fft(f, axis=0)
        J = self.J if theta is None else self.frame_symbols(theta, fh.real.dtype)[0]
        return _ifft_sym(J.astype(fh.real.dtype), fh[None, None], -3)

    def gyroaverageK(self, f):
        J = self.gyroaverage(f)
        return J - jnp.concatenate([jnp.zeros_like(J[:, :1]), J[:, :-1]], axis=1)

    def gyroaverageK_nodes(self, f):
        fh = jnp.fft.fft(f, axis=0)
        return _ifft_sym(self.Kn.astype(fh.real.dtype), fh[None, None], -3)

    def gyroaverage_nodes(self, f, theta=None):
        fh = jnp.fft.fft(f, axis=0)
        Jn = self.Jn if theta is None else self.frame_symbols(theta, fh.real.dtype, nodes=True)[1]
        return _ifft_sym(Jn.astype(fh.real.dtype), fh[None, None], -3)


def _damping_factors(geom, ky: np.ndarray, Lx: float, species: list, beta: float):
    """(D, f_qn, f_amp) at the reference surface, (Nz, Nkx, Nky) each, k_x in FFT order: the polarisation
    D = Σ_s (Z²n/T)(1 − Γ₀(b_s)), f_qn = Σ_s (Z²n/T) Γ₀(b_s)/D and f_amp = βΣ(Z²n/m)Γ₀/(k⊥² + βΣ(Z²n/m)Γ₀)."""
    Nx = len(geom.x)
    i0 = Nx // 2
    kx = kx_grid(Nx, Lx)
    gxx, gxy, gyy, B = (geom.y_averaged(a)[i0, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    k2 = np.maximum(gxx[:, None, None] * kx[None, :, None] ** 2 + 2 * gxy[:, None, None] * kx[None, :, None] * ky[None, None, :]
                    + gyy[:, None, None] * ky[None, None, :] ** 2, 0.0)
    pol, num, skin = 0.0, 0.0, 0.0
    for sp in species:
        b = 0.5 * (sp.temp * sp.mass / sp.Z ** 2) / (B ** 2)[:, None, None] * k2
        g0 = basis.gamma0(b)
        pol = pol + (sp.Z ** 2 * sp.dens / sp.temp) * (1.0 - g0)
        num = num + (sp.Z ** 2 * sp.dens / sp.temp) * g0
        skin = skin + beta * (sp.Z ** 2 * sp.dens / sp.mass) * g0
    fqn = np.where(pol > 0, num / np.where(pol > 0, pol, 1.0), 0.0)
    famp = np.where(k2 + skin > 0, skin / np.where(k2 + skin > 0, k2 + skin, 1.0), 0.0)
    return pol, fqn, famp


def kinetic_frame_data(geom, ky: np.ndarray, species: list, Lx: float, beta: float, M: int, qn_grid: bool = False) -> dict:
    """The data of :meth:`KineticFFTFieldSolver.frame_symbols` (the shearing frame of physics.exb_shear): the metric and
    B² at the reference surface (as :func:`build_kinetic_fft_solver`), k_x (FFT order) and k_y, per species ½T̂m̂/Z²,
    Z²n̂/T̂, Z²n̂/m̂, the Laguerre nodes and the k_x = k_y = 0 gauge mask."""
    Nx = len(geom.x)
    i0 = Nx // 2
    gxx, gxy, gyy, B = (np.asarray(geom.y_averaged(a))[i0, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    gauge = np.ones((Nx, len(ky), 1))
    gauge[0, 0, 0] = 0.0
    mu, _ = basis.laguerre_quadrature(M)
    r = lambda a: jnp.asarray(np.asarray(a, dtype=float))
    return dict(kx=r(kx_grid(Nx, Lx)), ky=r(ky), gxx=r(gxx), gxy=r(gxy), gyy=r(gyy), B2=r(B ** 2),
                rho2=r([0.5 * sp.temp * sp.mass / sp.Z ** 2 for sp in species]),
                pol_w=r([sp.Z ** 2 * sp.dens / sp.temp for sp in species]),
                skin_w=r([sp.Z ** 2 * sp.dens / sp.mass for sp in species]), beta=float(beta), mu=r(mu),
                gauge=r(gauge), qn_grid=bool(qn_grid))


def build_kinetic_fft_solver(geom, ky: np.ndarray, species: list, M: int, L: int, Lx: float, beta: float,
                             dtype=np.complex128, node_gyro: bool = False, bpar: bool = False) -> KineticFFTFieldSolver:
    """The symbols of :func:`build_kinetic_field_solver` at the reference surface (x index N_x/2, x = 0)."""
    Nx = len(geom.x)
    _, kperp2, B = _quantiser(geom, ky, Lx)
    i0 = Nx // 2
    k2 = kperp2[i0]                                                    # (Nz, Nkx, Nky)
    B0 = B[i0]                                                         # (Nz,)
    mu_nodes, _ = basis.laguerre_quadrature(M)
    J, Jn, Kn, pol, skin, qK, kk = [], [], [], 0.0, 0.0, 0.0, 0.0
    for sp in species:
        b = 0.5 * (sp.temp * sp.mass / sp.Z ** 2) / (B0 ** 2)[:, None, None] * k2
        Jm = basis.gyro_coefficients(b, M)                             # (M, Nz, Nkx, Nky)
        J.append(Jm)
        if node_gyro:
            from scipy.special import j0
            Jn.append(np.stack([j0(np.sqrt(2.0 * b * mu)) for mu in mu_nodes]))
            if bpar:
                Kn.append(np.stack([_K_nodes(b, mu) for mu in mu_nodes]))
        pol = pol + (sp.Z ** 2 * sp.dens / sp.temp) * (1.0 - basis.gamma0(b))
        skin = skin + (sp.Z ** 2 * sp.dens / sp.mass) * np.sum(Jm ** 2, axis=0)
        Km = Jm - np.concatenate([np.zeros_like(Jm[:1]), Jm[:-1]], axis=0)
        qK = qK + sp.Z * sp.dens * np.sum(Jm * Km, axis=0)
        kk = kk + sp.dens * sp.temp * np.sum(Km ** 2, axis=0)
    to_qkz = lambda a: np.moveaxis(a, -3, -1)                          # (..., Nz, Nkx, Nky) → (..., Nkx, Nky, Nz)
    gauge = np.ones_like(pol)
    gauge[:, 0, 0] = 0.0                                               # k_x = k_y = 0 (index 0 in FFT order) carries no field
    Pinv = to_qkz(np.where(gauge > 0, 1.0 / np.where(pol > 0, pol, 1.0), 0.0))
    Ainv = amp_w = None
    if beta > 0:
        Ainv = jnp.asarray(to_qkz(gauge / (k2 + beta * skin)))
        amp_w = jnp.asarray([SQRT2 * beta * sp.Z * sp.dens * sp.vt for sp in species])
    bk = {}
    if bpar:
        if beta <= 0:
            raise ValueError("physics.bpar needs physics.beta > 0")
        f = (beta / (2.0 * B0 ** 2))[:, None, None]                    # (Nz, 1, 1)
        a11, a12, a21, a22 = pol, -qK, f * qK, 1.0 + f * kk
        det = np.where(gauge > 0, a11 * a22 - a12 * a21, 1.0)
        bk = dict(B11=a22 / det * gauge, B12=-a12 / det * gauge, B21=-a21 / det * gauge, B22=a11 / det * gauge)
        bk = {k: jnp.asarray(to_qkz(v)) for k, v in bk.items()}
        bk.update(perp_w=jnp.asarray([sp.dens * sp.temp for sp in species]),
                  bfac=jnp.asarray((beta / (2.0 * B0 ** 2))[None, None, :]),
                  Kn=jnp.asarray(to_qkz(np.stack(Kn))) if node_gyro else None)
    return KineticFFTFieldSolver(
        J=jnp.asarray(to_qkz(np.stack(J))), Pinv=jnp.asarray(Pinv), qn_w=jnp.asarray([sp.Z * sp.dens for sp in species]),
        fsw=jnp.asarray(geom.fs_weights()), M=M, S=len(species), L=L, Ainv=Ainv, amp_w=amp_w,
        Jn=jnp.asarray(to_qkz(np.stack(Jn))) if node_gyro else None, **bk)


# ------------------------------------------------------------------------------------------------ operators
class KineticOperators(eqx.Module):
    """The shared :class:`velcro.vlasov.VlasovOperators` (geometry, velocity matrices, dissipation; built for the ions)
    plus the per-species coefficients, stacked on a leading species axis."""
    base: VlasovOperators
    c_stream: Any       # (S, Nx, 1, Nz)  v̂t_s b·∇z
    c_mirror: Any       # (S, Nx, 1, Nz)  v̂t_s b·∇z ∂_z ln B
    T_over_Z: Any       # (S,)
    Z_over_T: Any       # (S,)
    omega_n: Any        # (S, Nx, 1, 1)
    omega_T: Any        # (S, Nx, 1, 1)
    nu: Any             # (S,)
    vt: Any             # (S,)
    hkz: Any            # (S,)   GX hypercollision coefficient (0 = off)
    dens_temp: Any      # (S,)   n̂_s T̂_s (flux weights)
    dens: Any           # (S,)
    drive_T1: Any       # (L, M, M)   (v̂² + μ̂ − 3/2) acting on the ℓ = 1 row of χ
    ell_p: Any          # (L, 1, 1, 1, 1)  ℓ^p for ℓ ≥ 3, else 0
    kz_abs: Any = None  # (Nx, Nky, Nz)  |n − θ/2π| on the FFT modes of the Bloch-periodic field line
    bloch: Any = None   # (Nx, Nky, Nz)  e^{iθ z/2π}
    dz_sym: Any = None  # (S, Nx, Nky, Nz)  s(k_z) of the Fourier parallel derivative i s(k_z), per species
    kx_sym: Any = None  # (Nx, 1, 1)     k_x of the Fourier radial drift derivative, Nyquist 0 (numerics.radial_drift)
    x_mask: Any = None  # (Nx, 1, 1)     1 on the retained k_x (FFT order) of numerics.x_dealias, else 0
    khyp: Any = None    # (Nx, Nky, 1)   GX's hyperviscosity rate per (k_x FFT order, k_y) (numerics.gx_hyper), or None
    # ... and its pieces for the shearing frame (physics.exb_shear): the rate D ((k_x − θ)² + k_y²)/norm)^p at k_x − θ
    khyp_kx: Any = None  # (Nx, 1, 1)
    khyp_ky: Any = None  # (1, Nky, 1)
    khyp_D: float = eqx.field(static=True, default=0.0)
    khyp_norm: float = eqx.field(static=True, default=1.0)
    khyp_p: int = eqx.field(static=True, default=1)
    energy_band_k: Any = None  # numerics.gx_hermite_closure: base.energy_band with GX's last-row v̂² (same offsets)
    U_v_k: Any = None          # ... and the Hermite factor of the drift propagator: eigenvectors of (V V)[:L,:L]
    lam_v_k: Any = None        # ... and 2 × its eigenvalues
    vpar2_band_k: Any = None   # ... and base.vpar2_band (β' ≠ 0: 2v̂², the curvature excess's factor) with the same v̂²
    chain_w: Any = None      # (Nx,)  e^{−i k_m x_0}: DFT coefficients → physical ones (x origin on a rational surface)
    chain_sym: Any = None    # tuple over chain classes of (S, N·Nz) parallel symbols (numerics.parallel_bc = "gx_chains");
                             # (S, C, N·Nz), per chain, with numerics.gx_nyquist
    chain_abs: Any = None    # tuple over chain classes of (N·Nz,) |k_z| (GX's hypercollisions on the chains)
    chain_damp: Any = None   # tuple over chain classes of (C, N·Nz) GX end-damping rates ν(ξ) A (0 on k_y = 0 chains)
    chain_hz: Any = None     # tuple over chain classes of (N·Nz,) |k_z/k_z,max|^p: GX's hyperz (numerics.gx_hyperz), or None
    gx_hyperz: float = eqx.field(static=True, default=0.0)   # its rate (v_t/R)
    # physics.collision_model = "landau" with ei_collisions: the cross-species blocks (L, M, L, M) in ν̂ units and their
    # rates.  Only electrons ← ions' test block is filled (C_T,ei on H_e, rate ν̂_ei = Z_i ν_e: a stationary ion
    # Maxwellian, i.e. GX's β = 0 model); the field blocks and the ions' blocks are zero (_landau_cross_operators); None
    # without the Landau model
    landau_T: Any = None      # (S, L, M, L, M): the test block of species s with the other species' Maxwellian
    landau_F: Any = None      # (S, L, M, L, M): the field block acting on the other species' H (zero, see above)
    landau_nu: Any = None     # (S,) the cross rates
    landau_rad: float = eqx.field(static=True, default=0.0)  # ‖C_T,ei‖₂ (ν̂ units): the e–i block's own radius, for the CFL
    # physics.collision_flr (velcro/collisions_flr.py): per species s the FLR increment C(b_s) − C(0) in k_x space, b_s at
    # the reference surface; flr_vec (S, 4, M, Nkx, Nky, Nz) Dougherty; Landau the Bessel maps flr_q0 (S, K₀, M, …) /
    # flr_q1 (S, K₁, M, …) and the shared like-species matrices flr_K, flr_CF, flr_F0, flr_F1 (the same in every species'
    # own units); flr_Kei (S, L, M, L, M) the e–i test block's classical diffusion (electrons; zero on the ions); flr_rad
    # (S,) / flr_rad_ei (S,) CFL bounds
    flr_b: Any = None
    flr_vec: Any = None
    flr_q0: Any = None
    flr_q1: Any = None
    flr_F0: Any = None
    flr_F1: Any = None
    flr_K: Any = None
    flr_CF: Any = None
    flr_Kei: Any = None
    flr_rad: Any = None
    flr_rad_ei: Any = None
    flr_kind: str = eqx.field(static=True, default="")
    ei_nu: Any = None        # (S,) ν_ei per species (Z_i ν_e on the electrons, 0 on the ions): physics.ei_collisions
    ei_vr: Any = None        # (S,) v̂t_i/v̂t_s: the ion flow in each species' thermal units
    ei_J: Any = None         # the ions' 𝒥_m(b_i) for u_i (β > 0): (M, Nkx, Nky, Nz) FFT multipliers or (Nky, Nz, M, Nx, Nx)
    S: int = eqx.field(static=True, default=2)
    L: int = eqx.field(static=True, default=1)
    em: bool = eqx.field(static=True, default=False)
    flutter: bool = eqx.field(static=True, default=False)
    use_hkz: bool = eqx.field(static=True, default=False)
    kinetic: bool = eqx.field(static=True, default=True)
    wave_factor: tuple = eqx.field(static=True, default=())   # per species max(ρ(V), ω_H factor), see build_kinetic_operators
    fused_flutter: bool = eqx.field(static=True, default=True)  # E×B + flutter brackets with shared transforms (_bracket_em)
    par_fourier: bool = eqx.field(static=True, default=False)   # ∂_z of streaming by :func:`fourier_dz` instead of FD4
    stream_kmax: float = eqx.field(static=True, default=0.0)    # max |symbol| of that ∂_z (per unit z), for the CFL bound
    drift_fourier: bool = eqx.field(static=True, default=False)  # ∂_x of the radial drift by :func:`fourier_dx`
    x_dealias: bool = eqx.field(static=True, default=False)      # right-hand side projected by :func:`x_project`
    zonal_nyq: bool = eqx.field(static=True, default=False)      # k_z Nyquist of the k_y = 0 row projected out
    bracket_fourier: bool = eqx.field(static=True, default=False)  # ∂_x inside the bracket by FFT (numerics.bracket_dx)
    chains: tuple = eqx.field(static=True, default=())           # ((N, ((flat index, …) per chain, …)), …) per class
    damp_cfl: float = eqx.field(static=True, default=0.0)        # bound on the end damping's rate acting through h, for the CFL
    chain_ret: tuple = eqx.field(static=True, default=())        # flat indices of every chain member, class after class
    chain_back: tuple = eqx.field(static=True, default=())       # per flat index: its position in chain_ret, or len(chain_ret)
    ei: bool = eqx.field(static=True, default=False)             # physics.ei_collisions
    ei_flow: str = eqx.field(static=True, default="")            # how u_i is formed: "" (u_i = 0, β = 0), "fft", "dense"

    # attributes external tools read from ``sim.op``
    split_drift_ky = property(lambda self: self.base.split_drift_ky)
    nonlinear = property(lambda self: self.base.nonlinear)
    gql_cutoff = property(lambda self: self.base.gql_cutoff)
    bracket_form = property(lambda self: self.base.bracket_form)
    axisymmetric = property(lambda self: self.base.axisymmetric)
    dx = property(lambda self: self.base.dx)
    dz = property(lambda self: self.base.dz)
    ky = property(lambda self: self.base.ky)
    Ny = property(lambda self: self.base.Ny)
    phase = property(lambda self: self.base.phase)
    fsw = property(lambda self: self.base.fsw)
    c_Cinv = property(lambda self: self.base.c_Cinv)
    lag_to_nodes = property(lambda self: self.base.lag_to_nodes)
    nodes_to_lag = property(lambda self: self.base.nodes_to_lag)

    def drift_phase(self, h):
        """exp(−i h (2v_i² + μ_j) k_y K̂_y T̂_s/Z_s) → (S, Nx, Nky, Nz, L, M) (node indices minor-most, the layout of
        :func:`velcro.vlasov.propagate_nodes`): the exact binormal drift of every species."""
        b = self.base
        theta = h * b.ky * b.c_Ky                                                   # (Nx, Nky, Nz)  (T̂_i = 1)
        lam_v = b.lam_v if self.lam_v_k is None else self.lam_v_k
        lam = lam_v[:, None] + b.lam_mu[None, :]
        tz = self.T_over_Z.reshape((-1, 1, 1, 1, 1, 1))
        arg = theta[None, ..., None, None] * lam
        if b.c_Ky_cv is not None:   # β' ≠ 0: the curvature excess rotates the v∥² nodes only
            arg = arg + (h * b.ky * b.c_Ky_cv)[None, ..., None, None] * lam_v[:, None]
        return jnp.exp(-1j * tz * arg)

    def propagate(self, G, phase):
        """The drift propagator of :func:`drift_phase` applied species by species (separable Hermite ⊗ Laguerre nodes,
        as the fused passes of :func:`velcro.vlasov.propagate_nodes`)."""
        b = self.base
        Gs = G.reshape((self.S, self.L) + G.shape[1:])
        Uv = b.U_v if self.U_v_k is None else self.U_v_k
        return propagate_nodes(Gs, phase, Uv, b.U_mu).reshape(G.shape)


def _bloch_kz(geom, ky: np.ndarray, Nz: int):
    """k_z on the field line at each (x, k_y): f(z + 2π) = e^{−iθ} f(z) with θ = k_y τ(x) (τ = twist_phase), so
    f = e^{−iθz/2π} p with p 2π-periodic and the Fourier modes of p carry k_z = n − θ/2π (signed; FFT order).  θ is taken
    in (−π, π]."""
    theta = np.angle(np.exp(-1j * ky[None, :] * geom.twist_phase[:, None]))              # arg of the ghost factor = −θ
    theta = -theta                                                                         # θ ∈ [−π, π)
    n = np.fft.fftfreq(Nz, d=1.0 / Nz)                                                     # integer modes, FFT order
    kz = n[None, None, :] - theta[:, :, None] / (2 * np.pi)                                # (Nx, Nky, Nz)
    z = geom.z
    bloch = np.exp(1j * theta[:, :, None] * z[None, None, :] / (2 * np.pi))
    return kz, bloch


def abs_kz(G, op: KineticOperators):
    """|k_z| G along the Bloch-periodic field line at each (x, k_y) (z is the last axis)."""
    p = jnp.fft.fft(G * op.bloch, axis=-1)
    return _ifft_sym(op.kz_abs, p, -1) * jnp.conj(op.bloch)


def fourier_dz(f, op: KineticOperators, sym):
    """∂_z f along the Bloch-periodic field line at each (x, k_y): i s(k_z) on its Fourier modes k_z = n − θ/2π, with
    s = ``sym`` (one species' row of ``op.dz_sym``): s(k) = k ('fourier') or sign(k) min(|k|, 1.3722/Δz)
    ('fourier_capped'), optionally floored (``kpar_floor``).  s is real and odd, so the operator is anti-Hermitian on every
    field line (as the FD4 stencil); on the k_y = 0 row (θ = 0) the Nyquist mode, its own partner, has s = 0, which keeps
    that row real."""
    p = jnp.fft.fft(f * op.bloch, axis=-1)
    return _ifft_sym(1j * sym, p, -1) * jnp.conj(op.bloch)


from velcro.operators import x_dealias_mask  # noqa: E402  (shared with the adiabatic path; re-exported by name)


def x_project(f, op: KineticOperators):
    """The 2/3-rule projection along x (axis −3) of numerics.x_dealias."""
    return _ifft_sym(op.x_mask, jnp.fft.fft(f, axis=-3), -3)


def gx_chain_classes(Nx: int, Nky: int, shift: int):
    """GX's linked chains of the retained x modes (|m| ≤ (Nx−1)//3), grouped by length: for k_y index n the twist links
    m → m + n·shift; a chain starts where m − n·shift is not retained.  Returns ((N, chains), …) with every chain a tuple
    of flat indices ikx·Nky + n (ikx the FFT-ordered position of m); every retained (m, n) is in exactly one chain."""
    mmax = (Nx - 1) // 3
    classes: dict = {}
    for n in range(Nky):
        s = n * shift
        for m in range(-mmax, mmax + 1):
            if s != 0 and abs(m - s) <= mmax:
                continue                                  # not a chain start
            members = [m]
            while s != 0 and abs(members[-1] + s) <= mmax:
                members.append(members[-1] + s)
            classes.setdefault(len(members), []).append(tuple((mm % Nx) * Nky + n for mm in members))
    return tuple((N, tuple(ch)) for N, ch in sorted(classes.items()))


def chain_layout(chains, Nx: int, Nky: int):
    """(ret, back) for :func:`chain_apply`: ``ret`` lists the flat indices of all chain members, class after class, chain
    after chain; ``back[f]`` is the position of flat index f in ``ret``, or len(ret) (a zero row) if f is not retained."""
    ret = [i for _, chs in chains for ch in chs for i in ch]
    back = np.full(Nx * Nky, len(ret), dtype=np.int64)
    back[np.asarray(ret, dtype=np.int64)] = np.arange(len(ret))
    return tuple(int(i) for i in ret), tuple(int(i) for i in back)


def chain_apply(f, op: KineticOperators, syms):
    """Apply the diagonal symbols ``syms`` (one (N·Nz,) array per chain class) along GX's linked chains: x → k_x (physical
    phase), one gather of every chain member (class after class, each chain its extended z line member after member), an
    FFT along each chain, the symbol, back, and one gather to the k_x grid (the modes outside the retained set are zero)."""
    Nx, Nky, Nz = f.shape[-3:]
    lead = f.shape[:-3]
    F = (jnp.fft.fft(f, axis=-3) * op.chain_w[:, None, None]).reshape(lead + (Nx * Nky, Nz))
    G = jnp.take(F, np.asarray(op.chain_ret, dtype=np.int32), axis=-2)
    pieces, off = [], 0
    for (N, chains), s in zip(op.chains, syms):
        C = len(chains)
        g = G[..., off:off + C * N, :].reshape(lead + (C, N * Nz))
        g = _ifft_sym(s, jnp.fft.fft(g, axis=-1), -1)
        pieces.append(g.reshape(lead + (C * N, Nz)))
        off += C * N
    pieces.append(jnp.zeros(lead + (1, Nz), dtype=G.dtype))
    Y = jnp.concatenate(pieces, axis=-2)
    out = jnp.take(Y, np.asarray(op.chain_back, dtype=np.int32), axis=-2).reshape(lead + (Nx, Nky, Nz))
    return _ifft_sym(jnp.conj(op.chain_w)[:, None, None], out, -3)


def _chain_apply_reference(f, op: KineticOperators, syms):
    """:func:`chain_apply` with one gather and one scatter per class (the first implementation; tests compare the two)."""
    Nx, Nky, Nz = f.shape[-3:]
    lead = f.shape[:-3]
    F = (jnp.fft.fft(f, axis=-3) * op.chain_w[:, None, None]).reshape(lead + (Nx * Nky, Nz))
    out = jnp.zeros_like(F)
    for (N, chains), s in zip(op.chains, syms):
        ii = np.asarray(chains, dtype=np.int32).reshape(-1)
        C = len(chains)
        g = jnp.take(F, ii, axis=-2).reshape(lead + (C, N * Nz))
        g = jnp.fft.ifft(jnp.fft.fft(g, axis=-1) * s, axis=-1)
        out = out.at[..., ii, :].set(g.reshape(lead + (C * N, Nz)))
    out = out.reshape(lead + (Nx, Nky, Nz)) * jnp.conj(op.chain_w)[:, None, None]
    return jnp.fft.ifft(out, axis=-3)


def chain_multiply(f, op: KineticOperators, profiles):
    """Multiply along GX's chains by ``profiles`` (one (C, N·Nz) array per class, in the chains' extended coordinate):
    x → k_x, one gather, the product, one gather back, k_x → x.  The modes outside the retained set come back zero."""
    Nx, Nky, Nz = f.shape[-3:]
    lead = f.shape[:-3]
    F = (jnp.fft.fft(f, axis=-3) * op.chain_w[:, None, None]).reshape(lead + (Nx * Nky, Nz))
    G = jnp.take(F, np.asarray(op.chain_ret, dtype=np.int32), axis=-2)
    pieces, off = [], 0
    for (N, chains), pr in zip(op.chains, profiles):
        C = len(chains)
        g = G[..., off:off + C * N, :].reshape(lead + (C, N * Nz)) * pr
        pieces.append(g.reshape(lead + (C * N, Nz)))
        off += C * N
    pieces.append(jnp.zeros(lead + (1, Nz), dtype=G.dtype))
    Y = jnp.concatenate(pieces, axis=-2)
    out = jnp.take(Y, np.asarray(op.chain_back, dtype=np.int32), axis=-2).reshape(lead + (Nx, Nky, Nz))
    return _ifft_sym(jnp.conj(op.chain_w)[:, None, None], out, -3)


def gx_end_profile(N: int, Nz: int, widthfrac: float) -> np.ndarray:
    """GX's dampEnds_linked profile along an extended chain of N·Nz points: ν = 1 − 2ξ²/(1+ξ⁴) with ξ = idzp/width at the
    left end, (N·Nz − idzp)/width at the right end, width = int(Nz·N·widthfrac), 0 elsewhere."""
    nzl = N * Nz
    width = int(Nz * N * widthfrac)
    nu = np.zeros(nzl)
    if width <= 0:
        return nu
    for i in range(nzl):
        if i <= width:
            xx = i / width
        elif i >= nzl - width:
            xx = (nzl - i) / width
        else:
            continue
        nu[i] = 1 - 2 * xx * xx / (1 + xx ** 4)
    return nu


def fourier_dx(f, op: KineticOperators):
    """∂_x f by FFT along x (axis −3; the flux tube is periodic in x): i k_x on every mode, 0 on the Nyquist mode (so real
    rows stay real and the operator is anti-Hermitian)."""
    return _ifft_sym(1j * op.kx_sym, jnp.fft.fft(f, axis=-3), -3)


def parallel_symbol(kz: np.ndarray, kind: str, dz: float, floor: np.ndarray | None = None) -> np.ndarray:
    """s(k_z) of :func:`fourier_dz` from the signed k_z of :func:`_bloch_kz` (Nx, Nky, Nz); ``floor`` (Nky,) is a lower
    bound on |s| per k_y (0 on the k_y = 0 row; k_z = 0 keeps s = 0)."""
    cap = _FD4 / dz if kind == "fourier_capped" else np.inf
    mag = np.abs(kz) if floor is None else np.maximum(np.abs(kz), floor[None, :, None])
    sym = np.sign(kz) * np.minimum(mag, cap)
    Nz = kz.shape[-1]
    if Nz % 2 == 0:
        sym[:, 0, Nz // 2] = 0.0             # k_y = 0 (θ = 0): the mode n = −Nz/2 pairs with itself
    return sym


def twist_shear(cfg, geom) -> float:
    """The shear S of the twist e^{−i k_y 2π S x}: GX's geometry carries its own (``geom.extra["twist_shat"]``, kind
    gx_file); otherwise ``geometry.shat`` of the config."""
    if "twist_shat" in geom.extra:
        return float(geom.extra["twist_shat"])
    return float(getattr(cfg.geometry, "shat", 0.0))


def _landau_cross_operators(cfg, species: list) -> dict:
    """The Landau cross-species blocks (two species: ions first, electrons second) for ``collision_model = "landau"``
    with ``ei_collisions``: species 1 (electrons) collide with 0 (ions) at ν̂_10 = Z_0 ν_1 (quasineutral n, |Z_1| = 1);
    the ions' blocks are zero."""
    from velcro.landau import pair_matrices
    L, M = cfg.grid.L, cfg.grid.M
    if len(species) != 2:
        raise ValueError("collision_model = 'landau' with ei_collisions needs exactly two species")
    a, b = species
    if not (a.Z > 0 and b.Z < 0):
        raise ValueError("collision_model = 'landau' with ei_collisions needs the ions first and the electrons (Z < 0) "
                         f"second (got Z = {a.Z}, {b.Z})")
    # the electrons' test block only (b ← a's Maxwellian: pitch-angle scattering and the O(m_e/m_i) energy exchange,
    # closed-form potentials, accurate at every order).  The field blocks (the drag toward the ion flow, and the ions'
    # response to the electrons) are reliable only at low order with the monomial route (velcro/landau.py) and are not
    # used: with β = 0 this is exactly GX's model structure (u_i = 0), and the ions feel no collisions with the electrons
    T_ba, _ = pair_matrices(L, M, b.mass, b.temp, a.mass, a.temp)
    Z0 = abs(a.Z)
    nus = [0.0, Z0 * b.nu]
    r = lambda m: jnp.asarray(np.asarray(m).reshape(L, M, L, M))
    zero = jnp.zeros((L, M, L, M))
    # the CFL bound needs the e–i block's own radius: ‖C_T,ei‖₂ is 1.8–1.9 × ρ(C_ee) at (8, 4)–(16, 8) (pitch-angle
    # scattering ∝ v⁻³ on the slow electrons), so the like-species radius under-counts ν_e C_ee + ν_ei C_T,ei by 32–37 %
    return dict(landau_T=jnp.stack([zero, r(T_ba)]), landau_F=jnp.stack([zero, zero]),
                landau_nu=jnp.asarray(np.asarray(nus)), landau_rad=float(np.linalg.norm(np.asarray(T_ba), 2)))


def _gs2_cross_operators(cfg, species: list) -> dict:
    """GS2's electron–ion collisions for ``collision_model = "gs2"`` with ``ei_collisions`` (two species, ions first):
    the electrons' pitch-angle scattering off ions at rest, ½(Z_i ν_e/v̂³)L (GS2's zeff term in vnew; velcro/
    gs2_collisions.py), through the cross-block slots of the Landau path (test block only; no field block, no ion block:
    at β = 0 GS2 has no drag toward the ion flow and no e–i collisions on the ions)."""
    from velcro.gs2_collisions import ei_matrix
    L, M = cfg.grid.L, cfg.grid.M
    if len(species) != 2:
        raise ValueError("collision_model = 'gs2' with ei_collisions needs exactly two species")
    a, b = species
    if not (a.Z > 0 and b.Z < 0):
        raise ValueError("collision_model = 'gs2' with ei_collisions needs the ions first and the electrons (Z < 0) "
                         f"second (got Z = {a.Z}, {b.Z})")
    T = ei_matrix(L, M, cfg.physics.gs2_collision_terms, cfg.physics.gs2_const_v)
    r = lambda m: jnp.asarray(np.asarray(m).reshape(L, M, L, M))
    zero = jnp.zeros((L, M, L, M))
    return dict(landau_T=jnp.stack([zero, r(T)]), landau_F=jnp.stack([zero, zero]),
                landau_nu=jnp.asarray(np.asarray([0.0, abs(a.Z) * b.nu])), landau_rad=float(np.linalg.norm(T, 2)))


def _ei_operators(cfg, species: list, fields) -> dict:
    """The electron–ion collision coefficients of :class:`KineticOperators` (``physics.ei_collisions``), GX's form: the
    electrons (Z < 0) get ν_ei = Z_i ν_e with Z_i the charge of the first species (GX's species_h[0], the main ions).
    With ``collision_model = "landau"`` the cross terms are the Landau test and field blocks instead
    (:func:`_landau_cross_operators`)."""
    if not cfg.physics.ei_collisions:
        return {}
    if cfg.physics.collision_model == "landau":
        return _landau_cross_operators(cfg, species)
    if cfg.physics.collision_model == "gs2":
        return _gs2_cross_operators(cfg, species)
    ion = species[0]
    nu_ei = [ion.Z * sp.nu if sp.Z < 0 else 0.0 for sp in species]
    out = dict(ei=True, ei_nu=jnp.asarray(np.asarray(nu_ei)), ei_vr=jnp.asarray(np.asarray([ion.vt / sp.vt for sp in species])))
    if cfg.physics.beta > 0:                    # u_i = Σ_m 𝒥_m(b_i) H_i,1m (GX: from Ampère's law, the same thing)
        if fields is None:
            raise ValueError("physics.ei_collisions with β > 0 needs the field solver (build_kinetic_operators(fields=...))")
        if hasattr(fields, "J"):
            out.update(ei_flow="fft", ei_J=fields.J[0])
        else:
            out.update(ei_flow="dense", ei_J=fields.gyro[0])
    return out


def ion_flow(H1, op: KineticOperators):
    """u_i = Σ_m 𝒥_m(b_i) H_i,1m from the ions' ℓ = 1 rows H1 (M, Nx, Nky, Nz) → (Nx, Nky, Nz)."""
    if op.ei_flow == "fft":
        Hh = jnp.fft.fft(H1, axis=-3)
        return _ifft_sym(1.0, jnp.sum(op.ei_J.astype(Hh.real.dtype) * Hh, axis=0), -3)
    out = jnp.einsum("kzmxw,kzmw->kzx", op.ei_J, jnp.transpose(H1, (2, 3, 0, 1)))
    return jnp.transpose(out, (2, 0, 1))


def _flr_operators(cfg, geom, species: list, ky: np.ndarray) -> dict:
    """The :class:`KineticOperators` entries of ``physics.collision_flr``: per species b_s = ½(T̂_s m̂_s/Z_s²) k⊥²/B² at the
    reference surface (the FFT solver's symbols) and the data of velcro.collisions_flr.flr_data; with ei_collisions the
    e–i increments (Dougherty: −b_e and the energy restoring, no momentum; Landau: the test block's −b_e K_ei)."""
    if not cfg.physics.collision_flr:
        return {}
    from velcro import collisions_flr as cf
    from velcro.vlasov import flr_reference_b
    cf.check_config(cfg)
    L, M, model = cfg.grid.L, cfg.grid.M, cfg.physics.collision_model
    r = lambda a: jnp.asarray(np.asarray(a))
    data = [cf.flr_data(model, L, M, flr_reference_b(geom, ky, cfg.grid.Lx, len(geom.x), 0.5 * sp.temp * sp.mass / sp.Z ** 2))
            for sp in species]
    out = dict(flr_kind=model, flr_b=r(np.stack([d["b"] for d in data])), flr_rad=r([d["rad"] for d in data]))
    if model == "dougherty":
        out["flr_vec"] = r(np.stack([d["vec"] for d in data]))
    elif model == "landau":
        out.update(flr_q0=r(np.stack([d["q0"] for d in data])), flr_q1=r(np.stack([d["q1"] for d in data])),
                   flr_K=r(data[0]["K"]), flr_CF=r(data[0]["CF"]), flr_F0=r(data[0]["F0"]), flr_F1=r(data[0]["F1"]))
    if cfg.physics.ei_collisions:
        bmax = np.array([float(d["b"].max()) for d in data])
        if model == "landau":                       # the electrons' test block C_T,ei (ions first, electrons second)
            a, e = species
            Kei = cf.landau_ei_flr_matrix(L, M, e.mass, e.temp, a.mass, a.temp)
            out["flr_Kei"] = r(np.stack([np.zeros((L, M, L, M)), np.asarray(Kei).reshape(L, M, L, M)]))
            out["flr_rad_ei"] = r(bmax * float(np.linalg.norm(Kei, 2)))
        else:
            out["flr_rad_ei"] = r([d["rad"] for d in data])
    return out


def build_kinetic_operators(base: VlasovOperators, geom, cfg, species: list, ky: np.ndarray, dtype,
                            fields=None) -> KineticOperators:
    """``fields`` (the kinetic field solver) is needed only by ``physics.ei_collisions`` with β > 0 (the ion flow)."""
    L, M = cfg.grid.L, cfg.grid.M
    r = lambda a: jnp.asarray(np.asarray(a))
    V, Mu = basis.hermite_v_matrix(L), basis.laguerre_mu_matrix(M)
    drive_T = (np.kron(V @ V, np.eye(M)) + np.kron(np.eye(L), Mu) - 1.5 * np.eye(L * M)).reshape(L, M, L, M)
    p = cfg.numerics.hyper_kz_p
    ell = np.arange(L, dtype=float)
    ell_p = np.where(ell >= 3, ell ** p, 0.0).reshape(L, 1, 1, 1, 1)
    gpar = np.asarray(geom.gpar)
    gref = float(np.mean(np.abs(gpar)))                          # ⟨|b·∇z|⟩ (= 1/q for GX's s-α)
    hk = cfg.numerics.hyper_kz
    # GX (linear.cu): ν_hyp = nu_hyper_m (p+½)/(L−1)^(p+½) · 2.3 · v_ts |∇∥z|, applied as −ν_hyp ℓ^p |k_z| g; in Velcro units
    # a GX rate carries 1.9642 = (R/a)/√2 and ∇∥z_GX = a/(qR), so the rate is the same expression with v̂t_s ⟨|b·∇z|⟩/√2.
    hkz = [hk * (p + 0.5) / max(L - 1, 1) ** (p + 0.5) * 2.3 * sp.vt * gref / SQRT2 for sp in species]
    kz_abs = bloch = dz_sym = None
    pd = cfg.numerics.parallel_derivative
    stream_kmax = _FD4 / float(base.dz)
    if hk > 0 or pd != "fd4":
        kz, bl = _bloch_kz(geom, ky, len(geom.z))
        bloch = jnp.asarray(bl, dtype=dtype)
        if hk > 0:
            kz_abs = r(np.abs(kz))
        if pd != "fd4":
            # the sub-grid floor acts on the lightest species only: its non-adiabatic layer at a rational surface,
            # ≈ ωqR/(ŝ k_y v_ts), is ≈ √(m_i/m_e) times thinner than the ions' (which the radial grid resolves)
            fl = None
            if cfg.numerics.kpar_floor > 0:
                shat = twist_shear(cfg, geom)
                fl = cfg.numerics.kpar_floor * abs(shat) * np.asarray(ky, dtype=float) * (cfg.grid.Lx / cfg.grid.Nx)
                fl[0] = 0.0
            m_min = min(sp.mass for sp in species)
            syms = np.stack([parallel_symbol(kz, pd, float(base.dz), fl if sp.mass == m_min else None) for sp in species])
            dz_sym, stream_kmax = r(syms), float(np.max(np.abs(syms)))
    x_mask = r(x_dealias_mask(cfg.grid.Nx).reshape(-1, 1, 1)) if cfg.numerics.x_dealias else None
    chains, chain_w, chain_sym, chain_abs, chain_damp, damp_cfl = (), None, None, None, None, 0.0
    chain_hz = None
    if cfg.numerics.gx_hyperz > 0 and cfg.numerics.parallel_bc != "gx_chains":
        raise ValueError("numerics.gx_hyperz (GX's hyperz) acts along GX's linked chains: it needs parallel_bc = 'gx_chains'")
    chain_ret = chain_back = ()
    if cfg.numerics.parallel_bc == "gx_chains":
        if not cfg.numerics.x_dealias or pd == "fd4":
            raise ValueError("numerics.parallel_bc = 'gx_chains' needs numerics.x_dealias and a Fourier parallel_derivative")
        Nx, Nz, Lx = cfg.grid.Nx, len(geom.z), cfg.grid.Lx
        x = np.asarray(geom.x)
        shat = twist_shear(cfg, geom)
        s1 = shat * float(ky[1]) * Lx if len(ky) > 1 else 0.0
        shift = int(round(s1))
        if abs(s1 - shift) > 1e-6 or shift == 0:
            raise ValueError(f"gx_chains needs an integer twist shift ŝ k_y,min L_x (got {s1})")
        # e^{−iθ_B(x)} from the float64 geometry (the run dtype's phase is fp32 in fp32 runs), against θ_B = 2π ŝ k_y x
        ph = np.exp(-1j * np.asarray(ky, dtype=float)[None, :] * np.asarray(geom.twist_phase, dtype=float)[:, None])
        want = np.exp(-2j * np.pi * shat * np.asarray(ky, dtype=float)[None, :] * x[:, None])
        if not np.allclose(ph, want, atol=1e-6):
            raise ValueError("gx_chains needs θ_B(0) ≡ 0 (q0 n0 an integer) and a linear q(x)")
        chains = gx_chain_classes(Nx, len(ky), shift)
        m = np.fft.fftfreq(Nx, d=1.0 / Nx)
        chain_w = jnp.asarray(np.exp(-1j * 2 * np.pi * m / Lx * x[0]), dtype=dtype)
        cap = _FD4 / float(base.dz) if pd == "fourier_capped" else np.inf
        syms, abss = [], []
        for N, chs in chains:
            kz = np.fft.fftfreq(N * Nz, d=1.0 / (N * Nz)) / N                 # j/N per radian, FFT order
            sym = np.sign(kz) * np.minimum(np.abs(kz), cap)
            sym[(N * Nz) // 2] = 0.0 if (N * Nz) % 2 == 0 else sym[(N * Nz) // 2]
            if cfg.numerics.gx_nyquist and (N * Nz) % 2 == 0:
                # GX's init_kzLinked: the Nyquist entry is +nzL/2/nLinks on every chain; here on the k_y > 0 chains only
                sym = np.tile(sym, (len(chs), 1))
                sym[[ch[0] % len(ky) != 0 for ch in chs], (N * Nz) // 2] = min(Nz / 2.0, cap)
            syms.append(r(np.stack([1j * sym for _ in species])).astype(dtype))
            abss.append(r(np.abs(kz)))
        chain_sym, chain_abs = tuple(syms), tuple(abss)
        if cfg.numerics.gx_hyperz > 0:           # GX's hyperkzLinked: |k_z/k_z,max|^p, k_z,max = nz/zp/2 (zp = 1)
            chain_hz = tuple(r((np.abs(np.fft.fftfreq(N * Nz, d=1.0 / (N * Nz)) / N) / (Nz / 2.0)) ** cfg.numerics.gx_hyperz_p)
                             for N, _ in chains)
        damp_on = cfg.numerics.gx_damp_ends > 0 or cfg.numerics.gx_damp_amp > 0
        if damp_on:
            damp = []
            # The damping acts on h = G + (Z/T)χ, so its rate is A‖1 + (Z/T)χ(·)‖.  Quasineutrality bounds the φ part by
            # f_qn = Σ_s (Z_s²n_s/T_s) Γ₀(b_s) / D (D = Σ (Z²n/T)(1 − Γ₀), the polarisation); Ampère bounds the A∥ part by
            # f_amp = βΣ(Z²n/m)Γ₀ / (k⊥² + βΣ(Z²n/m)Γ₀) ≤ 1.  damp_cfl = A (1 + max over the damped points of f_qn + f_amp).
            pol, fqn, famp = _damping_factors(geom, ky, Lx, species, cfg.physics.beta)     # (Nz, Nkx, Nky) at x = 0
            fmax = 0.0
            for N, chs in chains:
                prof = gx_end_profile(N, Nz, cfg.numerics.gx_damp_width)
                on = np.array([0.0 if ch[0] % len(ky) == 0 else 1.0 for ch in chs])        # k_y = 0 chains stay periodic
                damp.append(r(on[:, None] * prof[None, :]))                                 # ν; Simulation sets the rate
                for ch, o in zip(chs, on):
                    if o == 0:
                        continue
                    for idzp in np.flatnonzero(prof > 0):
                        i = ch[idzp // Nz]
                        iz, ikx, n = idzp % Nz, i // len(ky), i % len(ky)
                        if pol[iz, ikx, n] > 0:
                            fmax = max(fmax, fqn[iz, ikx, n] + famp[iz, ikx, n])
            chain_damp = tuple(damp)
            damp_cfl = float(1.0 + fmax)                    # × the rate: the damping's largest real eigenvalue bound
        chain_ret, chain_back = chain_layout(chains, Nx, len(ky))
        stream_kmax = max(float(np.max(np.abs(np.asarray(s_)))) for s_ in syms)
    kx_sym = None
    if cfg.numerics.radial_drift == "fourier" or cfg.numerics.bracket_dx == "fourier":
        kx = kx_grid(cfg.grid.Nx, cfg.grid.Lx)
        if cfg.grid.Nx % 2 == 0:
            kx[cfg.grid.Nx // 2] = 0.0
        kx_sym = r(kx.reshape(-1, 1, 1))
    energy_band_k = U_v_k = lam_v_k = vpar2_band_k = None
    if cfg.numerics.gx_hermite_closure:        # GX's v̂² on the last Hermite row: (V V)[:L,:L] with V of L+1 modes
        from velcro.vlasov import _band
        V1 = basis.hermite_v_matrix(L + 1)
        A = (V1 @ V1)[:L, :L]
        Mu = basis.laguerre_mu_matrix(M)
        off, band = _band(2.0 * np.kron(A, np.eye(M)) + np.kron(np.eye(L), Mu), L, M)
        if tuple(off) != tuple(base.energy_off):
            raise ValueError("gx_hermite_closure changed the energy matrix's band structure")
        energy_band_k = tuple(r(c) for c in band)
        a, Ua = np.linalg.eigh(A)
        U_v_k, lam_v_k = r(Ua), r(2.0 * a)
        if base.c_Ky_cv is not None:           # β' ≠ 0: the curvature excess's 2v̂² with the same last-row v̂²
            voff, vb = _band(2.0 * np.kron(A, np.eye(M)), L, M)
            if tuple(voff) != tuple(base.vpar2_off):
                raise ValueError("gx_hermite_closure changed the 2v̂² matrix's band structure")
            vpar2_band_k = tuple(r(c) for c in vb)
    khyp, khyp_frame = None, {}
    if cfg.numerics.gx_hyper > 0:              # GX's hyperdiff: D ((k_x² + k_y²)/(k_x,max² + k_y,max²))^p, every moment
        kxh = kx_grid(cfg.grid.Nx, cfg.grid.Lx)
        kxmax = ((cfg.grid.Nx - 1) // 3) * 2 * np.pi / cfg.grid.Lx
        kyv = np.asarray(ky, dtype=float)
        kyref = kyv[-1]
        if cfg.numerics.gx_hyper_ref_ny > 0:    # a coarser grid's k_y,max: the same damping at every k the two grids share
            nref = (cfg.numerics.gx_hyper_ref_ny - 1) // 3 + 1
            if nref > len(kyv):
                raise ValueError(f"gx_hyper_ref_ny {cfg.numerics.gx_hyper_ref_ny} is finer than the grid's N_y {cfg.grid.Ny}")
            kyref = kyv[nref - 1]
        ratio = (kxh[:, None] ** 2 + kyv[None, :] ** 2) / (kxmax ** 2 + kyref ** 2)
        khyp = r((cfg.numerics.gx_hyper * ratio ** cfg.numerics.gx_hyper_p)[:, :, None])
        if base.exb_frame:                      # the shearing frame evaluates the same rate at k_x − θ every stage
            khyp_frame = dict(khyp_kx=r(kxh.reshape(-1, 1, 1)), khyp_ky=r(kyv.reshape(1, -1, 1)),
                              khyp_D=float(cfg.numerics.gx_hyper), khyp_norm=float(kxmax ** 2 + kyref ** 2),
                              khyp_p=int(cfg.numerics.gx_hyper_p))
    # The field-coupled parallel wave of the lightest species (ω_H mode, or the kinetic Alfvén wave when β > 0):
    # ω ≈ k∥ (v̂t_s/√2) / √(β/(2m̂_s) + b_min) with b_min the smallest ion b of the box (GX's linear.cu bound).  In the
    # electrostatic limit at small k⊥ it is several times faster than free streaming at the largest Hermite node.
    Bmax = float(np.max(np.asarray(geom.B)))
    kx_min, ky_min = 2 * np.pi / cfg.grid.Lx, (float(ky[1]) if len(ky) > 1 else np.inf)
    gxx, gyy = float(np.min(np.asarray(geom.gxx))), float(np.min(np.asarray(geom.gyy)))
    b_min = 0.5 * min(kx_min ** 2 * gxx, ky_min ** 2 * gyy) / Bmax ** 2
    m_min = min(sp.mass for sp in species)
    rad_v = float(np.abs(np.linalg.eigvalsh(V)).max())
    wave = tuple(float(max(rad_v, (1.0 / (SQRT2 * np.sqrt(cfg.physics.beta / (2 * sp.mass) + b_min))) if sp.mass == m_min else 0.0))
                 for sp in species)
    stack = lambda f: r(np.stack([f(sp) for sp in species]))
    return KineticOperators(
        base=base,
        c_stream=stack(lambda sp: sp.vt * gpar), c_mirror=stack(lambda sp: sp.vt * gpar * np.asarray(geom.dlnBdz)),
        T_over_Z=r([sp.temp / sp.Z for sp in species]), Z_over_T=r([sp.Z / sp.temp for sp in species]),
        omega_n=stack(lambda sp: sp.omega_n[:, None, None]), omega_T=stack(lambda sp: sp.omega_T[:, None, None]),
        nu=r([sp.nu for sp in species]), vt=r([sp.vt for sp in species]), hkz=r(hkz),
        dens_temp=r([sp.dens * sp.temp for sp in species]), dens=r([sp.dens for sp in species]),
        drive_T1=r(drive_T[:, :, 1, :]), ell_p=r(ell_p), kz_abs=kz_abs, bloch=bloch,
        S=len(species), L=L, em=cfg.physics.beta > 0, flutter=bool(cfg.physics.beta > 0 and cfg.physics.flutter),
        use_hkz=hk > 0, wave_factor=wave, dz_sym=dz_sym, par_fourier=pd != "fd4", stream_kmax=stream_kmax,
        kx_sym=kx_sym, drift_fourier=cfg.numerics.radial_drift == "fourier", x_mask=x_mask, khyp=khyp,
        energy_band_k=energy_band_k, U_v_k=U_v_k, lam_v_k=lam_v_k, vpar2_band_k=vpar2_band_k, **khyp_frame,
        x_dealias=bool(cfg.numerics.x_dealias), bracket_fourier=cfg.numerics.bracket_dx == "fourier",
        zonal_nyq=bool(cfg.numerics.zonal_nyquist_project and len(geom.z) % 2 == 0),
        chains=chains, chain_w=chain_w, chain_sym=chain_sym, chain_abs=chain_abs, chain_damp=chain_damp, damp_cfl=damp_cfl,
        chain_hz=chain_hz, gx_hyperz=float(cfg.numerics.gx_hyperz),
        chain_ret=chain_ret if chains else (), chain_back=chain_back if chains else (),
        **_ei_operators(cfg, species, fields), **_flr_operators(cfg, geom, species, ky))


# ------------------------------------------------------------------------------------------------ right-hand side
def kinetic_fields(G, fs: KineticFieldSolver, op: KineticOperators, exact_nodes: bool, bpar_part: str = "all",
                   theta=None):
    """(φ, A∥, χ, J0φ at the μ nodes, J0A∥ at the μ nodes): χ (S, 1 or 2, M, Nx, Nky, Nz) holds the ℓ = 0 (and ℓ = 1)
    rows of χ_s; the node values are None when the nonlinearity (or the flutter part) is off.  ``bpar_part`` selects
    what the ℓ = 0 row of χ and the φ node values carry when δB∥ is on: "all" (the run's: J0φ + (T/Z)𝒦 δB∥/B),
    "phi" (J0φ only) or "bpar" (the δB∥ part only) — for the term-resolved diagnostics; without δB∥ "bpar" gives
    zeros and "phi" is "all".  ``theta`` (Nky,): the shearing-frame phase of physics.exb_shear (every symbol at
    k_x − θ; the FFT solver only)."""
    if theta is not None:
        return _kinetic_fields_frame(G, fs, op, exact_nodes, theta)
    phi, apar, bpar = fs.solve_all(G)
    Jphi = fs.gyroaverage(phi)
    if bpar is not None:        # χ ∋ (T/Z) μ̂ (2J1(a)/a) δB∥/B, Laguerre coefficients (T/Z) 𝒦_m
        tz = op.T_over_Z.reshape((-1, 1, 1, 1, 1)).astype(Jphi.real.dtype)
        JK = tz * fs.gyroaverageK(bpar)
        Jphi_K = Jphi + JK if bpar_part == "all" else (Jphi if bpar_part == "phi" else JK)
    else:
        Jphi_K = Jphi if bpar_part != "bpar" else jnp.zeros_like(Jphi)
    chi = Jphi_K[:, None]
    Pn_phi = Pn_apar = None
    b = op.base
    if b.nonlinear:
        if exact_nodes:
            Pn_phi = fs.gyroaverage_nodes(phi)
            if bpar is not None:
                PK = tz * fs.gyroaverageK_nodes(bpar)
                Pn_phi = Pn_phi + PK if bpar_part == "all" else (Pn_phi if bpar_part == "phi" else PK)
            elif bpar_part == "bpar":
                Pn_phi = jnp.zeros_like(Pn_phi)
        else:
            Pn_phi = contract(b.lag_to_nodes, Jphi_K, 1)
    if apar is not None:
        Ja = fs.gyroaverage(apar)
        chi = jnp.concatenate([chi, (-op.vt / SQRT2).reshape((-1, 1, 1, 1, 1)).astype(Ja.real.dtype)[:, None] * Ja[:, None]],
                              axis=1)
        if op.flutter and b.nonlinear:
            Pn_apar = fs.gyroaverage_nodes(apar) if exact_nodes else contract(b.lag_to_nodes, Ja, 1)
    return phi, apar, chi, Pn_phi, Pn_apar


def _kinetic_fields_frame(G, fs, op: KineticOperators, exact_nodes: bool, theta):
    """:func:`kinetic_fields` in the shearing frame (no δB∥): the solve, the gyroaverages and the node values with the
    symbols at k_x − θ_ky, computed once per call (one set of symbols for the solve and every gyroaverage)."""
    if not isinstance(fs, KineticFFTFieldSolver) or fs.frame is None:
        raise ValueError("the shearing frame needs the kinetic FFT field solver built with its frame data")
    if fs.bpar:
        raise ValueError("physics.exb_shear with physics.bpar is not implemented")
    b = op.base
    rdt = G.real.dtype
    nodes = b.nonlinear and exact_nodes
    J, Jn, Pinv, Ainv = fs.frame_symbols(theta, rdt, nodes=nodes)
    Gs = G.reshape((fs.S, fs.L) + G.shape[1:])
    phi = _ifft_sym(Pinv, fs._moment_hat(Gs[:, 0], fs.qn_w, J), 0)
    apar = None if fs.Ainv is None else _ifft_sym(Ainv, fs._moment_hat(Gs[:, 1], fs.amp_w, J), 0)
    ga = lambda f, K: _ifft_sym(K, jnp.fft.fft(f, axis=0)[None, None], -3)
    Jphi = ga(phi, J)
    chi = Jphi[:, None]
    Pn_phi = Pn_apar = None
    if b.nonlinear:
        Pn_phi = ga(phi, Jn) if exact_nodes else contract(b.lag_to_nodes, Jphi, 1)
    if apar is not None:
        Ja = ga(apar, J)
        chi = jnp.concatenate([chi, (-op.vt / SQRT2).reshape((-1, 1, 1, 1, 1)).astype(rdt)[:, None] * Ja[:, None]], axis=1)
        if op.flutter and b.nonlinear:
            Pn_apar = ga(apar, Jn) if exact_nodes else contract(b.lag_to_nodes, Ja, 1)
    return phi, apar, chi, Pn_phi, Pn_apar


def _fourier_skew(b, op) -> bool:
    """The skew bracket with numerics.bracket_dx = "fourier" (no GQL): ∂_x on the spectral fields (:func:`_bracket`)."""
    return op is not None and op.bracket_fourier and b.bracket_form == "skew" and b.gql_cutoff < 0


def _integrand_fx(P, pb, rbx, Ny, fwd2, op):
    """The real-space pieces of the skew bracket {P, g} with ∂_x taken on the spectral fields: ``P`` (Nmu, Nx, Nky, Nz)
    the potential at the μ nodes, ``pb`` = [g | g_y] (:func:`velcro.operators._to_real_pair`), ``rbx`` = g_x on the real
    y grid.  Returns [a_x g_y − a_y g_x | a_x g] (the integrand of :func:`velcro.operators._bracket_integrand` without
    its flux term) and the flux product a_y g, whose ∂_x :func:`_bracket_fx` takes after the forward y transform."""
    a = P[None]
    pa = _to_real_pair(a, fwd2)
    ra, ray = pa[..., :Ny, :], pa[..., Ny:, :]
    rax = to_real_y(fourier_dx(a, op), Ny)
    rb, rby = pb[..., :Ny, :], pb[..., Ny:, :]
    return jnp.concatenate([rax * rby - ray * rbx, rax * rb], axis=-2), ray * rb


def _bracket_fx(I, F, inv2, nky, op):
    """½ [to_spec_pair(I) − ∂_x to_spec_y(F)]: the bracket from the pieces of :func:`_integrand_fx`."""
    return 0.5 * (_to_spec_pair(I, inv2) - fourier_dx(to_spec_y(F, nky), op))


def _bracket(b, P, Gn, op=None, Gnx=None):
    """{P, G} at the μ nodes: P (Nmu, Nx, Nky, Nz), Gn (L, Nmu, Nx, Nky, Nz).  The skew bracket with
    numerics.bracket_dx = "fourier" takes every x-derivative on the spectral (k_y) fields, by complex FFTs with the 1/N
    folded (:func:`fourier_dx`): a_x and g_x before the inverse y transforms, the flux term ∂_x(a_y g) after the forward
    one (∂_x commutes with the y transforms).  No complex-to-real transform, and the x-FFTs act on N_ky complex values
    instead of N_y real ones.  ``Gnx`` = ∂_x Gn when the caller has it
    (:func:`_rhs_species`: from the radial drift's ∂_x G); computed here otherwise."""
    if _fourier_skew(b, op):
        nky = Gn.shape[-2]
        fwd2, inv2 = _bracket_matrices(nky, b.Ny, _ky_key(b.ky))
        rgx = to_real_y(fourier_dx(Gn, op) if Gnx is None else Gnx, b.Ny)
        return _bracket_fx(*_integrand_fx(P, _to_real_pair(Gn, fwd2), rgx, b.Ny, fwd2, op), inv2, nky, op)
    if b.gql_cutoff >= 0:
        return gql_bracket(b.bracket_form, P[None], Gn, b.ky, b.dx, b.Ny, b.gql_cutoff)
    return BRACKETS[b.bracket_form](P[None], Gn, b.ky, b.dx, b.Ny)


def _bracket_em(b, P_phi, P_apar, Gn, vt, op=None, Gnx=None):
    """{P_φ, G} − v̂t V·{P_A, G} with one inverse and one forward y-transform of G: the skew bracket's real-space
    integrand is linear in its second slot, and V (the Hermite v̂ recurrence) acts on ℓ only, so it commutes with the
    y-transforms and moves inside, V·I(P_A, g) = I(P_A, V·g).  Equal to two :func:`_bracket` calls to round-off.  With
    numerics.bracket_dx = "fourier" as :func:`_bracket` (``Gnx`` likewise): g_x once (V·g_x = (V·g)_x), and one ∂_x of
    the summed flux products."""
    nky = Gn.shape[-2]
    fwd2, inv2 = _bracket_matrices(nky, b.Ny, _ky_key(b.ky))
    pg = _to_real_pair(Gn, fwd2)                                          # (L, Nmu, Nx, 2Ny, Nz)
    V = lambda f: apply_tridiag_l(b.V_lo, b.V_up, f)
    if op is not None and op.bracket_fourier:
        rgx = to_real_y(fourier_dx(Gn, op) if Gnx is None else Gnx, b.Ny)
        I, F = _integrand_fx(P_phi, pg, rgx, b.Ny, fwd2, op)
        IA, FA = _integrand_fx(P_apar, V(pg), V(rgx), b.Ny, fwd2, op)
        return _bracket_fx(I - vt * IA, F - vt * FA, inv2, nky, op)
    I = _bracket_integrand(_to_real_pair(P_phi[None], fwd2), pg, b.dx, b.Ny)
    I = I - vt * _bracket_integrand(_to_real_pair(P_apar[None], fwd2), V(pg), b.dx, b.Ny)
    return 0.5 * _to_spec_pair(I, inv2)


# The terms of :func:`_rhs_species`, for ``terms=`` (a partition of the run's right-hand side).  ``drift_ky`` is the k_y
# drift of (T/Z) G that a Lawson run advances in its propagator: never in the run's rhs, only when asked for by name.
RHS_TERMS = ("parallel", "drift", "drive", "exb", "flutter", "collisions", "hyper", "end_damp", "hkz", "krook")


def _rhs_species(G, chi, Pn_phi, Pn_apar, sp: dict, op: KineticOperators, terms=None, theta=None):
    """∂_t G_s of one species: G (L, M, Nx, Nky, Nz); χ (1 or 2, M, Nx, Nky, Nz); ``sp`` = that species' coefficients.
    ``terms`` (a set of :data:`RHS_TERMS` names, and/or ``drift_ky``) keeps only those terms; None = all (the run's).
    ``theta`` (1, Nky, 1): the shearing-frame phase (physics.exb_shear; G is then the frame representation p = e^{iθx}G):
    the radial drift's ∂_x − iθ, the frame bracket of the physical fields (operators.frame_bracket, E×B and flutter),
    hyper_x and gx_hyper at the physical k_x − θ (unless gx_hyper_grid); the parallel operators, the chains and their
    end damping, the k_y drift, the collisions and the drive commute with e^{iθx} and are unchanged."""
    b = op.base
    ky = b.ky
    nrow = chi.shape[0]
    use = lambda k: terms is None or k in terms
    frame = theta is not None
    H = G.at[:nrow].add(sp["Z_over_T"] * chi)
    c_stream, c_mirror, tz = sp["c_stream"], sp["c_mirror"], sp["T_over_Z"]
    if op.chains:
        dz = lambda f: chain_apply(f, op, sp["chain_sym"])
    elif op.par_fourier:
        dz = lambda f: fourier_dz(f, op, sp["dz_sym"])
    else:
        dz = lambda f: ddz(f, b.dz, b.phase)
    gx = []                 # ∂_x G by FFT (op.kx_sym), taken once: the radial drift's and the bracket's g_x

    def dGx():
        if not gx:
            gx.append(fourier_dx(G, op))
        return gx[0]
    out = jnp.zeros_like(G)
    # --- parallel streaming + E∥ (the ∇∥φ and ∇∥A∥ parts of H) + mirror force, on H
    if use("parallel") and b.parallel_skew:
        VH = apply_tridiag_l(b.V_lo, b.V_up, H)
        out = -0.5 * (c_stream * dz(VH) + b.inv_jac * dz(b.jac * c_stream * VH))
        out = out + c_mirror * apply_banded(b.mirror_skew_off, b.mirror_skew_band, H)
    elif use("parallel"):
        out = -c_stream * apply_tridiag_l(b.V_lo, b.V_up, dz(H))
        out = out + c_mirror * apply_banded(b.mirror_off, b.mirror_band, H)
    # --- magnetic drift on (T/Z) H = (T/Z) G + χ
    band = b.energy_band if op.energy_band_k is None else op.energy_band_k
    vband = b.vpar2_band if op.vpar2_band_k is None else op.vpar2_band_k
    if use("drift"):
        dx = (lambda f: fourier_dx(f, op)) if op.drift_fourier else (lambda f: ddx(f, b.dx))
        dTHdx = (tz * (dGx() if op.drift_fourier else dx(G))).at[:nrow].add(dx(chi))
        if frame:                   # ∂_x → ∂_x − iθ_ky on the frame representation (the physical k_x − θ)
            dTHdx = dTHdx - 1j * theta * (tz * G).at[:nrow].add(chi)
        if b.split_drift_ky:    # the k_y drift of (T/Z) G is advanced exactly by the Lawson factor; keep its action on χ
            drift = (b.c_Kx * dTHdx).at[:nrow].add(b.c_Ky * (1j * ky) * chi)
        else:
            drift = b.c_Kx * dTHdx + b.c_Ky * (1j * ky) * (tz * G).at[:nrow].add(chi)
        out = out - apply_banded(b.energy_off, band, drift)
        if b.c_Ky_cv is not None:   # β' ≠ 0: the curvature drift's excess over the ∇B drift, through 2v̂² only
            if b.split_drift_ky:
                drift = (b.c_Kx_cv * dTHdx).at[:nrow].add(b.c_Ky_cv * (1j * ky) * chi)
            else:
                drift = b.c_Kx_cv * dTHdx + b.c_Ky_cv * (1j * ky) * (tz * G).at[:nrow].add(chi)
            out = out - apply_banded(b.vpar2_off, vband, drift)
    if use("drift") and b.c_Ky_co is not None:     # GS2's Coriolis drift: v̂ (T/Z)H / v̂t_s, on all of (T/Z)H (not split)
        TH = (tz * G).at[:nrow].add(chi)
        co = (b.c_Kx_co * dTHdx + b.c_Ky_co * (1j * ky) * TH) / sp["vt"]
        out = out - apply_tridiag_l(b.V_lo, b.V_up, co)
    if terms is not None and "drift_ky" in terms and b.split_drift_ky:
        out = out - apply_banded(b.energy_off, band, b.c_Ky * (1j * ky) * (tz * G))
        if b.c_Ky_cv is not None:
            out = out - apply_banded(b.vpar2_off, vband, b.c_Ky_cv * (1j * ky) * (tz * G))
    # --- drive: −½ C⁻¹ i k_y [ω_n + ω_T (v̂² + μ̂ − 3/2)] χ
    if use("drive"):
        chiL = jnp.pad(chi, ((0, G.shape[0] - nrow),) + ((0, 0),) * (chi.ndim - 1))      # χ as an (L, M, …) field
        dT = apply_banded(b.drive_off, b.drive_band, chiL)       # (v̂² + μ̂ − 3/2) on the ℓ = 0 (and ℓ = 1) rows of χ
        drv = (sp["omega_T"] * dT).at[:nrow].add(sp["omega_n"] * chi)
        if b.pvg_T0 is not None:    # parallel-velocity-gradient drive 2v̂ ω_u,s χ, ω_u,s = ω_u/v̂t_s (physics.exb_shear.u_prime)
            pv = (2.0 * sp["pvg_u"]) * apply_tridiag_l(b.V_lo, b.V_up, chiL)
            drv = drv + (pv if b.pvg_prof is None else b.pvg_prof.astype(pv.real.dtype) * pv)
        out = out - b.c_Cinv * (1j * ky) * drv
    # --- E×B and magnetic flutter: −½ C⁻¹ ({J0φ, G} − v̂t V·{J0A∥, G}), pseudo-spectral in μ
    fl = Pn_apar is not None and use("flutter")
    if b.nonlinear and frame and (use("exb") or fl):    # the shearing frame: the bracket of the physical fields at any θ
        from velcro.operators import frame_bracket
        Gn = contract(b.lag_to_nodes, G, 1)
        dxp = (lambda f: fourier_dx(f, op)) if op.bracket_fourier else (lambda f: ddx(f, b.dx))
        fb = lambda P: frame_bracket(P[None], Gn, ky, b.dx, b.Ny, theta, b.exb_xg, dxp)
        br = fb(Pn_phi) if use("exb") else 0.0
        if fl:
            br = br - sp["vt"] * apply_tridiag_l(b.V_lo, b.V_up, fb(Pn_apar))
        out = out - b.c_Cinv * contract(b.nodes_to_lag, br, 1)
    elif b.nonlinear and (use("exb") or fl):
        Gn = contract(b.lag_to_nodes, G, 1)
        Gnx = contract(b.lag_to_nodes, dGx(), 1) if _fourier_skew(b, op) else None    # g_x at the μ nodes
        if fl and use("exb") and b.bracket_form == "skew" and b.gql_cutoff < 0 and op.fused_flutter:
            br = _bracket_em(b, Pn_phi, Pn_apar, Gn, sp["vt"], op, Gnx)
        else:
            br = _bracket(b, Pn_phi, Gn, op, Gnx) if use("exb") else 0.0
            if fl:
                br = br - sp["vt"] * apply_tridiag_l(b.V_lo, b.V_up, _bracket(b, Pn_apar, Gn, op, Gnx))
        out = out - b.c_Cinv * contract(b.nodes_to_lag, br, 1)
    # --- collisions (Dougherty, or the dense Landau matrix) on H, as GX
    if use("collisions"):
        if b.coll_dense:
            cH = jnp.einsum("lmpq,pq...->lm...", b.coll.astype(H.real.dtype), H)
        else:
            cH = b.coll_diag * H
            for n, (li, mi, lj, mj) in enumerate(b.coll_off_idx):
                cH = cH.at[li, mi].add(b.coll_off_vals[n] * H[lj, mj])
        out = out + sp["nu"] * cH
        if op.landau_T is not None:     # Landau cross-species: the test block on this species' H, the field block on the other's
            cross = jnp.einsum("lmpq,pq...->lm...", sp["landau_T"], H) + jnp.einsum("lmpq,pq...->lm...", sp["landau_F"], sp["H_other"])
            out = out + sp["landau_nu"] * cross
        if op.ei:       # e–i (GX ei_colls): ν_ei × Dougherty without its own momentum restoring, + drag toward u_i
            cei = cH.at[1, 0].add(-H[1, 0]) if G.shape[0] > 1 else cH
            if G.shape[0] > 1 and op.ei_flow:
                cei = cei.at[1, 0].add(sp["ei_u"])
            out = out + sp["ei_nu"] * cei
        if op.flr_b is not None:                    # physics.collision_flr: the FLR increments C(b_s) − C(0), in k_x
            Hh = jnp.fft.fft(H, axis=2)
            sectors = ((op.flr_F0, sp["flr_q0"], 1.0), (op.flr_F1, sp["flr_q1"], 2.0)) if op.flr_kind == "landau" else ()
            dC = sp["nu"] * flr_increment_hat(Hh, op.flr_kind, sp["flr_b"], sp.get("flr_vec"), op.flr_K, op.flr_CF, sectors)
            if op.ei:           # e–i: −b_e and the energy restoring dressed, no momentum restoring; the drag on 𝒥_m(b_e)
                dC = dC + sp["ei_nu"] * flr_increment_hat(Hh, "dougherty", sp["flr_b"], sp["flr_vec"], momentum=False)
                if G.shape[0] > 1 and op.ei_flow:
                    uh = jnp.fft.fft(sp["ei_u"], axis=0)
                    dC = dC.at[1].add(sp["ei_nu"] * SQRT2 * sp["flr_vec"][0].astype(uh.real.dtype) * uh[None])
                    dC = dC.at[1, 0].add(-sp["ei_nu"] * uh)
            if op.flr_Kei is not None:                      # the Landau e–i test block's classical diffusion
                dC = dC - sp["landau_nu"] * sp["flr_b"] * jnp.einsum("lmpq,pq...->lm...", sp["flr_Kei"], Hh)
            out = out + _ifft_sym(1.0, dC, 2)
    # --- numerical dissipation (on G)
    if use("hyper"):
        sheared = frame and not b.exb_hyper_grid    # the frame: dissipation at the physical k_x − θ
        if b.hyper_x > 0:
            out = out - b.hyper_x * (hyper_x_sheared(G, theta, b.dx) if sheared else hyper_x(G))
        if b.hyper_z > 0:
            out = out - b.hyper_z * hyper_z(G, b.phase)
        if b.hyper_ky > 0:
            out = out - b.hyper_ky * (ky / b.kymax) ** 4 * G
        out = out - b.hypv_diag * G
        if op.khyp is not None:                     # GX's k-space hyperviscosity (diagonal in k_x, k_y)
            rdt = G.real.dtype
            if sheared:
                kxs = op.khyp_kx.astype(rdt) - theta.astype(rdt)                       # (Nkx, Nky, 1)
                rate = op.khyp_D * ((kxs ** 2 + op.khyp_ky.astype(rdt) ** 2) / op.khyp_norm) ** op.khyp_p
            else:
                rate = op.khyp.astype(rdt)
            out = out - _ifft_sym(rate, jnp.fft.fft(G, axis=-3), -3)
        if op.chain_hz is not None:                 # GX's hyperz: −ν |k_z/k_z,max|^p along the chains, every moment
            out = out - op.gx_hyperz * chain_apply(G, op, op.chain_hz)
    if op.chain_damp is not None and use("end_damp"):   # GX's absorbing chain ends act on h = G + (Z/T)χ
        out = out - chain_multiply(H, op, op.chain_damp)
    if op.use_hkz and G.shape[0] > 3 and use("hkz"):
        akz = chain_apply(G[3:], op, op.chain_abs) if op.chains else abs_kz(G[3:], op)
        out = out.at[3:].add(-sp["hkz"] * op.ell_p[3:] * akz)
    return out - b.krook * G if use("krook") else out


def rhs_kinetic(G, fields, op: KineticOperators, terms=None, theta=None):
    """∂_t G for the stacked state G (S·L, M, Nx, Nky, Nz) given ``fields`` from :func:`kinetic_fields`; ``terms`` as
    :func:`_rhs_species` (the post-processing, x projection and the gauge, is linear, so the terms add up).  ``theta``
    (Nky,): the shearing-frame phase of physics.exb_shear, required when the frame is on (``fields`` must then come
    from :func:`kinetic_fields` at the same θ)."""
    if op.base.exb_frame and theta is None:
        # without θ every frame term would silently be the θ = 0 operator: right only before the first time step
        raise ValueError("rhs_kinetic: the shearing-frame operator (physics.exb_shear) needs the frame phase theta — pass "
                         "sim.exb_theta at call time")
    th = None if theta is None else jnp.reshape(theta, (1, -1, 1))
    _, _, chi, Pn_phi, Pn_apar = fields
    Gs = G.reshape((op.S, op.L) + G.shape[1:])
    rdt = G.real.dtype
    sp = dict(c_stream=op.c_stream.astype(rdt), c_mirror=op.c_mirror.astype(rdt), T_over_Z=op.T_over_Z.astype(rdt),
              Z_over_T=op.Z_over_T.astype(rdt), omega_n=op.omega_n.astype(rdt), omega_T=op.omega_T.astype(rdt),
              nu=op.nu.astype(rdt), vt=op.vt.astype(rdt), hkz=op.hkz.astype(rdt),
              dz_sym=op.dz_sym.astype(rdt) if op.par_fourier else None,
              chain_sym=op.chain_sym if op.chains else None)
    if op.ei:
        sp["ei_nu"] = op.ei_nu.astype(rdt)
        if op.ei_flow and op.L > 1:             # u_i from the ions' H (species 0), in each species' thermal units
            H1 = Gs[0, 1] + op.Z_over_T[0].astype(rdt) * chi[0, 1]
            sp["ei_u"] = op.ei_vr.astype(rdt).reshape((-1, 1, 1, 1)) * ion_flow(H1, op)[None]
    if op.landau_T is not None:                 # the other species' H for the Landau field blocks (k⊥ = 0: no gyroaverage)
        nrow = chi.shape[1]
        Hs = Gs.at[:, :nrow].add(op.Z_over_T.astype(rdt).reshape((-1, 1, 1, 1, 1, 1)) * chi)
        sp["H_other"] = Hs[::-1]
        sp["landau_T"], sp["landau_F"], sp["landau_nu"] = op.landau_T.astype(rdt), op.landau_F.astype(rdt), op.landau_nu.astype(rdt)
    if op.flr_b is not None:                    # physics.collision_flr: per-species data (velcro/collisions_flr.py)
        sp["flr_b"] = op.flr_b.astype(rdt)
        for k in ("flr_vec", "flr_q0", "flr_q1", "flr_Kei"):
            if getattr(op, k) is not None:
                sp[k] = getattr(op, k).astype(rdt)
    if op.base.pvg_T0 is not None:              # the PVG drive per species: ω_u,s = ω_u v_tref/v_ts (u∥ shared by the species)
        sp["pvg_u"] = (op.base.pvg_u / op.vt).astype(rdt)
    out = jax.vmap(lambda g, c, pp, pa, s: _rhs_species(g, c, pp, pa, s, op, terms, th))(Gs, chi, Pn_phi, Pn_apar, sp)
    out = out.reshape(G.shape)
    if op.x_dealias:
        out = x_project(out, op)
    if op.zonal_nyq:                                  # the zonal row's k_z Nyquist ((−1)^j along z) projected out
        sgn = (1.0 - 2.0 * (jnp.arange(out.shape[-1]) % 2)).astype(out.real.dtype)
        z0 = out[:, :, :, 0, :]
        out = out.at[:, :, :, 0, :].add(-jnp.mean(z0 * sgn, axis=-1, keepdims=True) * sgn)
    return out.at[:, :, :, 0, :].add(-jnp.mean(out[:, :, :, 0, :], axis=2, keepdims=True))   # k_x = k_y = 0 not evolved


def no_gauge_state(G):
    """G with the x mean of its k_y = 0 row removed (the mode :func:`rhs_kinetic` does not evolve)."""
    return G.at[:, :, :, 0, :].add(-jnp.mean(G[:, :, :, 0, :], axis=2, keepdims=True))


# ------------------------------------------------------------------------------------------------ time step, diagnostics
def cfl_rate_kinetic(fields, op: KineticOperators, L: int, M: int):
    """Sum of per-term spectral radii (as :func:`velcro.vlasov.cfl_rates_spectral`) for each species; the maximum over
    species.  Streaming and mirror force scale with v̂t_s (the electrons set the step), the drift with |T̂_s/Z_s|."""
    b = op.base
    _, _, chi, Pn_phi, Pn_apar = fields
    vts = jnp.abs(op.vt)
    gmax = jnp.max(jnp.abs(op.c_stream[0])) / vts[0]
    mmax = jnp.max(jnp.abs(op.c_mirror[0])) / vts[0]
    if op.par_fourier:                                                       # free streaming, or the ω_H / KAW branch
        stream = jnp.asarray(op.wave_factor) * vts * gmax * op.stream_kmax
    else:
        stream = jnp.asarray(op.wave_factor) * vts * gmax * _FD4 / b.dz
    mirror = b.rad_mirror * vts * mmax
    drift_ky = 0.0 if b.split_drift_ky else drift_abs(b, "y") * b.kymax
    if op.drift_fourier:
        drift = b.rad_energy * jnp.abs(op.T_over_Z) * jnp.max(drift_ky + drift_abs(b, "x") * float(np.pi / b.dx))
    else:
        drift = b.rad_energy * jnp.abs(op.T_over_Z) * jnp.max(drift_ky + drift_abs(b, "x") * _FD4 / b.dx)
    if b.c_Ky_co is not None:                       # the Coriolis drift: ρ(V) |c_co| (k_y + k_x) |T̂_s/Z_s| / v̂t_s
        kxb = float(np.pi / b.dx) if op.drift_fourier else _FD4 / b.dx
        drift = drift + b.rad_v * jnp.abs(op.T_over_Z) / jnp.abs(op.vt) * jnp.max(
            jnp.abs(b.c_Ky_co) * b.kymax + jnp.abs(b.c_Kx_co) * kxb)
    rate = stream + mirror + drift
    rate = rate + jnp.abs(op.nu) * ((L - 1 + 2 * (M - 1)) if not b.coll_dense else b.rad_coll)
    if op.ei:
        rate = rate + jnp.abs(op.ei_nu) * (L - 1 + 2 * (M - 1))
    if op.landau_nu is not None:                    # the e–i block's own radius, not the like-species one
        rate = rate + jnp.abs(op.landau_nu) * op.landau_rad
    if op.flr_b is not None:                        # physics.collision_flr: the FLR increments' radius bounds
        rate = rate + jnp.abs(op.nu) * op.flr_rad
        if op.flr_rad_ei is not None:
            rate = rate + jnp.abs(op.ei_nu if op.ei else op.landau_nu) * op.flr_rad_ei
    if op.use_hkz:                                  # −ν ℓ^p |k_z|: real negative eigenvalues up to ν (L−1)^p max|k_z|
        rate = rate + op.hkz * jnp.max(op.ell_p) * jnp.max(op.kz_abs)
    if op.chain_hz is not None:                     # GX's hyperz: real negative eigenvalues down to −gx_hyperz
        rate = rate + op.gx_hyperz
    if b.nonlinear:
        w = jnp.where(jnp.arange(Pn_phi.shape[-2]) == 0, 1.0, 2.0)[:, None]
        vy = jnp.max(jnp.sum(w * jnp.abs(ddx(Pn_phi, b.dx)), axis=-2), axis=(1, 2, 3))          # per species
        vx = jnp.max(jnp.sum(w * jnp.abs(b.ky * Pn_phi), axis=-2), axis=(1, 2, 3))
        if Pn_apar is not None:
            vy = vy + b.rad_v * vts * jnp.max(jnp.sum(w * jnp.abs(ddx(Pn_apar, b.dx)), axis=-2), axis=(1, 2, 3))
            vx = vx + b.rad_v * vts * jnp.max(jnp.sum(w * jnp.abs(b.ky * Pn_apar), axis=-2), axis=(1, 2, 3))
        rate = rate + jnp.max(b.c_Cinv) * (vy * b.kymax + vx * _FD4 / b.dx)
    return jnp.max(rate)


def kinetic_fluxes(G, fields, op: KineticOperators, Ehalf, fs=None):
    """Heat and particle fluxes per species, split by field: the E×B (φ), magnetic-flutter (A∥) and compressional (δB∥)
    parts: Q_s = n̂_s T̂_s ⟨(v̂² + μ̂) G_s v_x[χ_s]⟩ with v_x[χ] = −½C⁻¹ ∂_y χ, box average as ``Simulation``'s Q.
    Returns (Q, Γ), each (S, 3): column 0 = φ part, 1 = A∥ part (0 without A∥), 2 = δB∥ part (0 without δB∥; the φ part
    is then χ's whole ℓ = 0 row, so columns 0 and 1 are the former two).  ``fs`` (the field solver) is needed for the
    δB∥ split: the ℓ = 0 row of χ minus J0φ."""
    b = op.base
    phi, _, chi, _, _ = fields
    Gs = G.reshape((op.S, op.L) + G.shape[1:])
    nrow = chi.shape[1]
    bpar_on = fs is not None and getattr(fs, "bpar", False)
    if bpar_on:
        Jphi = fs.gyroaverage(phi)
        chi = jnp.concatenate([Jphi[:, None], chi[:, 1:], (chi[:, 0] - Jphi)[:, None]], axis=1)   # φ | A∥ | δB∥ rows
        nrow = chi.shape[1]
    vx = -b.c_Cinv * (1j * b.ky) * chi                                       # (S, nrow, M, Nx, Nky, Nz)
    EG = jnp.einsum("lmpq,spq...->slm...", Ehalf, Gs)
    EG = EG[:, :nrow] if not bpar_on else jnp.concatenate([EG[:, :1], EG[:, 1:2], EG[:, :1]], axis=1)
    GG = Gs[:, :nrow] if not bpar_on else jnp.concatenate([Gs[:, :1], Gs[:, 1:2], Gs[:, :1]], axis=1)
    w = jnp.where(jnp.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, None, :, None]
    fsw = b.fsw

    def box(a):                                                              # (S, nrow, M, Nx, Nky, Nz) → (S, nrow)
        s = jnp.sum(jnp.real(a) * w, axis=(2, 4))                            # (S, nrow, Nx, Nz)
        return jnp.sum(s * fsw, axis=(2, 3)) / fsw.shape[0]
    Q = box(jnp.conj(vx) * EG) * op.dens_temp[:, None]
    Gam = box(jnp.conj(vx) * GG) * op.dens[:, None]
    while Q.shape[1] < 3:
        Q, Gam = jnp.concatenate([Q, jnp.zeros_like(Q[:, :1])], axis=1), jnp.concatenate([Gam, jnp.zeros_like(Gam[:, :1])], axis=1)
    return Q, Gam
