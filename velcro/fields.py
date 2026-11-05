"""Gyroaverage operators and the gyrokinetic Poisson (quasi-neutrality) solve (§2.2, §4.3 K4).

Real-space-in-``x`` operators are precomputed from the Fourier symbol of each
Laguerre gyro-coefficient ``𝒥_m(b)`` and of the polarisation ``1 − Γ_0(b)``
with ``b = ρ̂²(x,z) k_⊥² / 2`` (``ρ̂ = v_t/Ω``, ``v_t = sqrt(2T/m)``), ``k_⊥² = g^{xx} k_x² + 2 g^{xy} k_x k_y + g^{yy} k_y²``
(Kohn–Nirenberg quantisation, symmetrised), and applied as dense
``(N_x × N_x)`` matrices per ``(k_y, z)``.  This is the “real-space gyroaverage
with precomputed weights” route of Risk 3.  For non-axisymmetric geometry the
``y``-averaged metric is used inside these operators (exact for tokamaks; the
position-dependent 2-D kernel is the Risk-3 follow-up).

Quasi-neutrality with adiabatic electrons, normalised (single ion species, Z=1):

    (1/T̂_i)(1 − Γ_0) φ + (1/T̂_e)(φ − ⟨φ⟩_FS δ_{k_y,0}) = Σ_m 𝒥_m G_{0m}

The ``k_y = 0`` flux-surface-average coupling across ``z`` is handled by a
Woodbury (rank-``N_x``) correction; the ``k_y = 0, k_x = 0`` gauge mode is
projected out.

Performance note: the gyro operators are stored with the batch indices leading,
``(N_ky, N_z, M, N_x, N_x)``, and applied as batched matrix–vector products.  With the
``(M, N_ky, N_z, N_x, N_x)`` layout XLA transposed the whole (memory-bound) tensor on every
call; the batched form is ~1.8x faster on CPU and identical to roundoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from velcro import basis
from velcro.operators import kx_grid


@dataclass
class FieldSolver:
    gyro: jnp.ndarray      # (Nky, Nz, M, Nx, Nx)  real-space 𝒥_m operators, batch indices leading
    Ainv: jnp.ndarray      # (Nky, Nz, Nx, Nx)     inverse of the polarisation operator (k_y ≠ 0 rows; k_y=0 row = A_z⁻¹)
    W: jnp.ndarray         # (Nx, Nx)  Woodbury matrix for the k_y = 0 flux-surface coupling
    fsw: jnp.ndarray       # (Nx, Nz)  flux-surface-average weights
    inv_Te: jnp.ndarray    # (Nx,)
    M: int
    gyro_nodes: jnp.ndarray | None = None   # (Nky, Nz, Nmu, Nx, Nx)  J0(√(2bμ_j)) at the Gauss–Laguerre nodes

    def gyroaverage_nodes(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Gyroaveraged potential at the μ nodes, ``J0(√(2bμ_j)) φ`` → ``(Nmu, Nx, Nky, Nz)``.  This is
        the closed form of ``Σ_m 𝒥_m(b) L_m(μ_j)`` summed to infinity; the sum truncated at ``M``
        diverges at the outer nodes once ``M ≳ 16`` (|Φ| 300x too large at μ = 51.7) and that
        spurious E×B velocity sets the explicit stability limit."""
        pt = jnp.transpose(phi, (1, 2, 0))
        out = jnp.einsum("kzmxw,kzw->kzmx", self.gyro_nodes, pt)
        return jnp.transpose(out, (2, 3, 0, 1))

    def gyroaverage(self, phi: jnp.ndarray) -> jnp.ndarray:
        """``Φ_m(x, k_y, z) = Σ_x' 𝒥_m[x, x'] φ(x', k_y, z)`` → shape ``(M, Nx, Nky, Nz)``."""
        pt = jnp.transpose(phi, (1, 2, 0))                                # (Nky, Nz, Nx)
        out = jnp.einsum("kzmxw,kzw->kzmx", self.gyro, pt)               # batched matvec
        return jnp.transpose(out, (2, 3, 0, 1))

    def gyro_density(self, G0: jnp.ndarray) -> jnp.ndarray:
        """``Σ_m 𝒥_m G_{0m}`` with ``G0`` of shape ``(M, Nx, Nky, Nz)`` (adjoint = same symmetric matrix)."""
        gt = jnp.transpose(G0, (2, 3, 0, 1))                              # (Nky, Nz, M, Nx)
        out = jnp.sum(jnp.einsum("kzmxw,kzmw->kzmx", self.gyro, gt), axis=2)   # (Nky, Nz, Nx)
        return jnp.transpose(out, (2, 0, 1))

    def solve(self, G0: jnp.ndarray) -> jnp.ndarray:
        """Solve quasi-neutrality for ``φ(x, k_y, z)`` given the ``ℓ = 0`` moments."""
        rhs = self.gyro_density(G0)                          # (Nx, Nky, Nz)
        # k_y ≠ 0 (and the A_z^{-1} part for k_y = 0)
        phi = jnp.einsum("kzxw,wkz->xkz", self.Ainv, rhs)
        # k_y = 0: Woodbury correction for the flux-surface-average term
        rhs0 = rhs[:, 0, :]
        rhs0 = rhs0 - jnp.sum(rhs0 * self.fsw) / jnp.sum(self.fsw)    # remove the gauge/volume mode
        p0 = jnp.einsum("zxw,wz->xz", self.Ainv[0], rhs0)             # A_z⁻¹ rhs
        avg = jnp.sum(p0 * self.fsw, axis=1)                          # ⟨A⁻¹ rhs⟩_FS
        phibar = self.W @ avg                                         # ⟨φ⟩_FS
        corr = jnp.einsum("zxw,w->xz", self.Ainv[0], self.inv_Te * phibar)
        phi0 = p0 + corr
        phi0 = phi0 - jnp.mean(phi0)
        return phi.at[:, 0, :].set(phi0)


def build_field_solver(geom, ky: np.ndarray, Ti: np.ndarray, Te: np.ndarray, M: int,
                       Lx: float, dtype=np.complex128, node_gyro: bool = False) -> FieldSolver:
    """Precompute gyro-operators and the polarisation inverse.

    Parameters
    ----------
    geom : Geometry
    ky : (Nky,) retained binormal wavenumbers
    Ti, Te : (Nx,) normalised ion / electron temperature profiles
    M : number of Laguerre moments
    """
    Nx, Nz = len(geom.x), len(geom.z)
    Nky = len(ky)
    kx = kx_grid(Nx, Lx)
    # y-averaged metric and field (exact for axisymmetric geometry)
    gxx, gxy, gyy, B = (geom.y_averaged(a)[:, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    rho2 = 0.5 * Ti[:, None] / B**2                           # ρ̂_i²(x, z)/2:  b = k⊥² ρ_i²/2 for v_t = sqrt(2T/m)
    x = geom.x
    # phase matrix E[x, x', kx] = exp(i kx (x - x')) / Nx
    E = np.exp(1j * kx[None, None, :] * (x[:, None, None] - x[None, :, None])) / Nx
    Er, Ei = np.ascontiguousarray(E.real), np.ascontiguousarray(E.imag)
    b = (rho2[:, :, None, None] * (gxx[:, :, None, None] * kx[None, None, :, None] ** 2
                                   + 2 * gxy[:, :, None, None] * kx[None, None, :, None] * ky[None, None, None, :]
                                   + gyy[:, :, None, None] * ky[None, None, None, :] ** 2))  # (Nx, Nz, Nkx, Nky)
    b = np.maximum(b, 0.0)
    Jm = basis.gyro_coefficients(b, M)                        # (M, Nx, Nz, Nkx, Nky)
    G0b = basis.gamma0(b)                                      # (Nx, Nz, Nkx, Nky)

    def quantise(symbol):  # symbol (Nx, Nz, Nkx, Nky) evaluated at output point x -> op (Nky, Nz, Nx, Nx)
        # = einsum("xwq,xzqk->kzxw", E, symbol) as BLAS matmuls batched over (x, z), real and imaginary part of E
        # apart (symbol is real): the plain einsum is single-threaded loops, 20 s a call at Nx=384 (17 calls a build,
        # the GPU idle meanwhile); this is 0.75 s, equal to 2e-15
        op = np.matmul(Er[:, None], symbol) + 1j * np.matmul(Ei[:, None], symbol)   # (x, z, w, k)
        op = op.transpose(3, 1, 0, 2)
        return 0.5 * (op + np.conj(np.swapaxes(op, -1, -2)))  # symmetrise (Hermitian)

    def quantise_all(symbols):  # → (Nky, Nz, len, Nx, Nx), written in place: np.stack + a C-order copy cost 2 passes
        out = np.empty((len(ky), Nz, len(symbols), Nx, Nx), dtype=complex)
        for m, s in enumerate(symbols):
            out[:, :, m] = quantise(s)
        return out

    gyro = quantise_all(Jm)                                    # (Nky, Nz, M, Nx, Nx)
    pol = quantise(1.0 - G0b)
    gyro_nodes = None
    if node_gyro:
        from scipy.special import j0
        mu_nodes, _ = basis.laguerre_quadrature(M)
        gyro_nodes = jnp.asarray(quantise_all([j0(np.sqrt(2.0 * b * mu)) for mu in mu_nodes]), dtype=dtype)
    inv_Ti, inv_Te = 1.0 / Ti, 1.0 / Te
    I = np.eye(Nx)
    A = inv_Ti[None, None, :, None] * pol + inv_Te[None, None, :, None] * I[None, None]
    Ainv = np.linalg.inv(A)
    # Woodbury for k_y = 0:  (I − K) ⟨φ⟩ = ⟨A⁻¹ rhs⟩ ,  K = Σ_z w_z A_z⁻¹ diag(1/T_e)
    fsw = geom.fs_weights()                                    # (Nx, Nz)
    K = np.einsum("xz,zxw->xw", fsw, Ainv[0]) * inv_Te[None, :]
    ones = np.ones((Nx, 1))
    W = np.linalg.inv(I - K + ones @ ones.T / Nx)             # regularised on the gauge mode
    return FieldSolver(
        gyro=jnp.asarray(gyro, dtype=dtype), Ainv=jnp.asarray(Ainv, dtype=dtype),
        W=jnp.asarray(W.real), fsw=jnp.asarray(fsw), inv_Te=jnp.asarray(inv_Te), M=M,
        gyro_nodes=gyro_nodes,
    )


# ------------------------------------------------------------------------------------------------ flux-tube FFT solver
@dataclass
class FFTFieldSolver:
    """The flux-tube limit of :class:`FieldSolver` (``numerics.field_solver = "fft"`` with adiabatic electrons): the
    symbols at the reference surface x = 0 applied as multipliers on the FFT in x, evaluated at k_x* = k_x − θ_ky inside
    the jitted step (``theta`` = the shearing-frame phase of ``physics.exb_shear``, 0 without it), so the gyroaverages
    and the polarisation follow the sheared frame.  Quasineutrality per (k_x, k_y, z):
    [(1/T̂_i)(1 − Γ₀(b)) + 1/T̂_e] φ = Σ_m 𝒥_m G_0m for k_y ≠ 0; at k_y = 0 the flux-surface average couples z
    (Sherman–Morrison per k_x) and the k_x = 0 gauge mode carries no field.  Differs from the dense solver only by
    Velcro's residual radial variation (ρ* x) of T̂_i, B and the metric."""
    kx: jnp.ndarray          # (Nkx,) FFT order
    ky: jnp.ndarray          # (Nky,)
    gxx: jnp.ndarray; gxy: jnp.ndarray; gyy: jnp.ndarray; rho2: jnp.ndarray   # (Nz,) at x = 0 (rho2 = ½ T̂_i/B²)
    inv_Ti: float; inv_Te: float
    w: jnp.ndarray           # (Nz,) flux-surface weights at x = 0, Σ = 1
    mu_nodes: jnp.ndarray    # (Nmu,)
    M: int
    node_gyro: bool = False

    def b(self, theta):
        """b(k_x − θ, k_y, z) ≥ 0 → (Nkx, Nky, Nz); ``theta`` (Nky,) or 0."""
        kxs = self.kx[:, None] - jnp.zeros_like(self.ky)[None, :] - jnp.asarray(theta, self.kx.dtype)[None, :] \
            if jnp.ndim(theta) else self.kx[:, None] - theta
        kxs = kxs[:, :, None]
        kyy = self.ky[None, :, None]
        k2 = self.gxx * kxs ** 2 + 2 * self.gxy * kxs * kyy + self.gyy * kyy ** 2
        return jnp.maximum(self.rho2 * k2, 0.0)

    def gyro_coefficients(self, b):
        """𝒥_m(b) = (b/2)^m e^{−b/2}/m! → (M, Nkx, Nky, Nz), in log space (as basis.gyro_coefficients)."""
        m = jnp.arange(self.M, dtype=b.dtype).reshape((self.M, 1, 1, 1))
        # b = 0 at the k_x = k_y = 0 symbol: no floor (1e-300 is 0 in float32, and finfo.tiny/2 is a subnormal that
        # XLA flushes to zero, so log(0)·0 = NaN either way) — the b > 0 branch is evaluated on a safe argument and the
        # b = 0 value, 𝒥_m(0) = δ_m0, is set explicitly
        pos = b > 0
        logb2 = jnp.log(jnp.where(pos, b, 1.0) / 2.0)
        out = jnp.exp(m * logb2 - b / 2.0 - jax.scipy.special.gammaln(m + 1))
        return jnp.where(pos, out, jnp.where(m == 0, 1.0, 0.0))

    def gyroaverage(self, phi, theta=0.0):
        """Φ_m = 𝒥_m(b*) φ → (M, Nx, Nky, Nz)."""
        fh = jnp.fft.fft(phi, axis=0)
        J = self.gyro_coefficients(self.b(theta)).astype(fh.real.dtype)
        return jnp.fft.ifft(J * fh[None], axis=1)

    def gyroaverage_nodes(self, phi, theta=0.0):
        """J0(√(2 b* μ_j)) φ at the Gauss–Laguerre nodes → (Nmu, Nx, Nky, Nz)."""
        fh = jnp.fft.fft(phi, axis=0)
        a = jnp.sqrt(2.0 * self.b(theta)[None] * self.mu_nodes.reshape((-1, 1, 1, 1)))
        J0 = _j0(a).astype(fh.real.dtype)
        return jnp.fft.ifft(J0 * fh[None], axis=1)

    def gyro_density(self, G0, theta=0.0):
        Gh = jnp.fft.fft(G0, axis=1)
        J = self.gyro_coefficients(self.b(theta)).astype(Gh.real.dtype)
        return jnp.fft.ifft(jnp.sum(J * Gh, axis=0), axis=0)

    def solve(self, G0, theta=0.0, theta_den=None):
        """φ(x, k_y, z) from the ℓ = 0 moments (M, Nx, Nky, Nz).  ``theta_den`` (physics.exb_shear.gx_qn_grid): the
        phase of the polarisation denominator when it differs from the numerator's (None: ``theta``)."""
        Gh = jnp.fft.fft(G0, axis=1)
        b = self.b(theta)
        J = self.gyro_coefficients(b).astype(Gh.real.dtype)
        rho = jnp.sum(J * Gh, axis=0)                                            # (Nkx, Nky, Nz)
        bd = b if theta_den is None else self.b(theta_den)
        d = self.inv_Ti * (1.0 - jax.scipy.special.i0e(bd)) + self.inv_Te          # the diagonal symbol
        phi = rho / d
        # k_y = 0: [d − (1/T̂_e) 1 wᵀ] φ = ρ per k_x (Sherman–Morrison).  At k_x = 0 the polarisation vanishes and the
        # system is singular on the flux-surface average: as the dense solver, the FSA of ρ is removed and the FSA of φ
        # set to 0, keeping the z-varying part φ = (ρ − ⟨ρ⟩)/(1/T̂_e) (GX masks the whole mode instead)
        r0, d0 = rho[:, 0, :], d[:, 0, :]
        wd = jnp.sum(self.w[None, :] / d0, axis=1)                               # wᵀ D⁻¹ 1
        wr = jnp.sum(self.w[None, :] * r0 / d0, axis=1)                          # wᵀ D⁻¹ ρ
        den = 1.0 - self.inv_Te * wd
        gauge = (jnp.arange(r0.shape[0]) != 0)
        avg = jnp.where(gauge, wr / jnp.where(gauge, den, 1.0), 0.0)
        phi0 = (r0 + self.inv_Te * avg[:, None]) / d0
        r00 = r0[0] - jnp.sum(self.w * r0[0])
        phi0 = phi0.at[0].set(r00 / self.inv_Te)
        phi = phi.at[:, 0, :].set(phi0)
        return jnp.fft.ifft(phi, axis=0)


def _hankel_coefficients(n: int = 14) -> np.ndarray:
    """a_k(ν = 0) = Π_{j ≤ k} (−(2j − 1)²) / (k! 8^k), k < n: Hankel's asymptotic expansion of J₀ (A&S 9.2.5–9.2.10)."""
    a = [1.0]
    for k in range(1, n):
        a.append(a[-1] * (-(2 * k - 1) ** 2) / (k * 8.0))
    return np.asarray(a)


_HANKEL = _hankel_coefficients()
_J0_SWITCH = 25.0      # bessel_jn's Miller recurrence (n_iter 60) is exact to 3e-16 below; it fails beyond ~40 (|err| 3e-3 at 40–55)


def _j0(x):
    """J₀(x) for x ≥ 0.  float64: jax.scipy.special.bessel_jn below x = 25 (Miller's backward recurrence; NaN at exactly
    0, the k_x = k_y = 0 node, so guarded) and Hankel's asymptotic expansion with 14 terms above (|error| < 1e-14 there;
    the recurrence with n_iter 60 alone loses accuracy beyond x ≈ 40 and returns O(1) garbage beyond 55, arguments the node
    gyroaverage √(2bμ_j) reaches at high k⊥ or large M).  float32: the recurrence overflows, so the rational approximations
    of Numerical Recipes' bessj0 (|error| < 1e-8, Abramowitz & Stegun 9.4.1/9.4.3)."""
    if np.dtype(x.dtype) == np.dtype("float64"):               # (jnp.float64 is a scalar type, not equal to a dtype)
        safe = jnp.where((x > 0) & (x < _J0_SWITCH), x, 1.0)
        small = jax.scipy.special.bessel_jn(safe, v=0, n_iter=60)[0]
        xl = jnp.maximum(x, _J0_SWITCH)                          # the unused branch on a safe argument (no NaN in the jvp)
        P = sum(((-1) ** k) * _HANKEL[2 * k] / xl ** (2 * k) for k in range(len(_HANKEL) // 2))
        Q = sum(((-1) ** k) * _HANKEL[2 * k + 1] / xl ** (2 * k + 1) for k in range(len(_HANKEL) // 2))
        w = xl - 0.25 * np.pi
        large = jnp.sqrt(2.0 / (np.pi * xl)) * (P * jnp.cos(w) - Q * jnp.sin(w))
        return jnp.where(x <= 0, 1.0, jnp.where(x < _J0_SWITCH, small, large))
    ax = jnp.abs(x)
    y = jnp.minimum(ax, 8.0) ** 2                                # each branch's argument clamped to its own range: no
    axl = jnp.maximum(ax, 8.0)                                   # overflow in the unused branch (its NaN would leak
    z = 8.0 / axl                                                # through jnp.where into the jvp)
    small = (57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))) \
        / (57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))))
    y2 = z * z
    xx = axl - 0.785398164
    p1 = 1.0 + y2 * (-0.1098628627e-2 + y2 * (0.2734510407e-4 + y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
    p2 = -0.1562499995e-1 + y2 * (0.1430488765e-3 + y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 - y2 * 0.934935152e-7)))
    large = jnp.sqrt(0.636619772 / axl) * (jnp.cos(xx) * p1 - z * jnp.sin(xx) * p2)
    return jnp.where(ax < 8.0, small, large)


def build_fft_field_solver(geom, ky: np.ndarray, Ti: np.ndarray, Te: np.ndarray, M: int, Lx: float,
                           node_gyro: bool = False) -> FFTFieldSolver:
    """:class:`FFTFieldSolver` from the geometry at the reference surface (x index N_x/2)."""
    Nx = len(geom.x)
    i0 = Nx // 2
    kx = kx_grid(Nx, Lx)
    gxx, gxy, gyy, B = (np.asarray(geom.y_averaged(a))[i0, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    mu_nodes, _ = basis.laguerre_quadrature(M)
    w = geom.fs_weights()[i0]
    return FFTFieldSolver(kx=jnp.asarray(kx), ky=jnp.asarray(np.asarray(ky, float)), gxx=jnp.asarray(gxx),
                          gxy=jnp.asarray(gxy), gyy=jnp.asarray(gyy), rho2=jnp.asarray(0.5 * float(Ti[i0]) / B ** 2),
                          inv_Ti=float(1.0 / Ti[i0]), inv_Te=float(1.0 / Te[i0]), w=jnp.asarray(w / w.sum()),
                          mu_nodes=jnp.asarray(mu_nodes), M=M, node_gyro=node_gyro)
