"""Matrix-free gyroaverage operators and field solve: low-rank symbols applied with FFTs (the memory wall of §2.2).

:mod:`velcro.fields` quantises each Fourier symbol ``a(x, k_x)`` (a Laguerre gyro-coefficient ``𝒥_m(b)``, ``J_0`` at a
μ node, or the polarisation ``1 − Γ_0``) into a dense ``(N_x × N_x)`` matrix per ``(k_y, z)``,

    (A φ)(x) = Σ_q a(x, q) e^{iqx} φ̂(q),      op = ½ (A + A†),

which costs ``N_ky N_z (M + N_μ + 1) N_x²`` numbers — 3 GB at N_x = 256 on the (16,8) flux tube and more than the
15 GB host can build.  Here the symbol is factorised once, ``a(x, q) = Σ_r U_r(x) V_r(q)`` (a truncated SVD of the
``(N_x × N_kx)`` table, exact at rank 1 when the metric and profiles do not depend on ``x``), so that

    A φ  = Σ_r U_r ⊙ ifft(V_r ⊙ fft φ),        A† φ = Σ_r ifft(V_r ⊙ fft(U_r ⊙ φ))

(``U``, ``V`` real; numpy FFT conventions) and the storage is ``R (N_x + N_kx)`` per operator instead of ``N_x²``.
``R`` is the largest rank any ``(k_y, z, m)`` needs at the requested tolerance; the build reports it.

Quasi-neutrality for ``k_y ≠ 0`` is then solved iteratively.  With a temperature profile the operator
``T_i⁻¹ (1 − Γ_0) + T_e⁻¹`` is not Hermitian, so the equation is multiplied by ``T_i``:

    [Pol + diag(T_i / T_e)] φ = T_i ⊙ Σ_m 𝒥_m G_{0m}

whose operator is Hermitian positive definite, and solved by preconditioned conjugate gradients, batched over
``(k_y, z)``, with the ``x``-averaged symbol inverted in Fourier space as the preconditioner (exact in a flux tube,
so one iteration there).  The ``k_y = 0`` row keeps the dense ``(N_z, N_x, N_x)`` inverse and the Woodbury flux-surface
coupling of :mod:`velcro.fields`: it is ``N_z N_x²`` numbers (10 MB at N_x = 256), and its operator is the awkward one.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from velcro import basis
from velcro.operators import kx_grid


def _factor(symbol: np.ndarray, tol: float):
    """SVD of a real ``(Nx, Nkx)`` symbol table → ``U S`` (Nx, r), ``V`` (Nkx, r), keeping σ_r > tol σ_1.

    ``V`` is copied: a view of ``Vt`` would keep the whole (Nkx, Nkx) matrix alive for every slice (≈ 116 GB at
    N_x = 512 with 171 k_y).  Large tables go through a randomised range finder (p probe vectors, then an SVD of the
    p × Nkx projection), accepted only if the rank found leaves ≥ 4 spare probes and the residual is within the bound
    of the exact truncation (‖A − U Vᵀ‖_F ≤ tol σ_1 √Nkx); otherwise the full SVD."""
    n, m = symbol.shape
    p = 16
    if min(n, m) > 4 * p:
        Q, _ = np.linalg.qr(symbol @ np.random.default_rng(0).standard_normal((m, p)))
        Ub, S, Vt = np.linalg.svd(Q.T @ symbol, full_matrices=False)
        r = max(1, int(np.sum(S > tol * S[0])))
        if r <= p - 4:
            U, V = (Q @ Ub[:, :r]) * S[None, :r], Vt[:r].T.copy()
            if np.linalg.norm(symbol - U @ V.T) <= tol * S[0] * np.sqrt(m):
                return U, V, r
    U, S, Vt = np.linalg.svd(symbol, full_matrices=False)
    r = max(1, int(np.sum(S > tol * S[0])))
    return U[:, :r] * S[None, :r], Vt[:r].T.copy(), r


def _pad(mats, R):
    """Stack a list of (n, r_i) factors into (len, n, R), zero-padded."""
    out = np.zeros((len(mats), mats[0].shape[0], R))
    for i, m in enumerate(mats):
        out[i, :, : m.shape[1]] = m
    return out


@dataclass
class LowRankFieldSolver:
    """Same interface as :class:`velcro.fields.FieldSolver`; arrays ``(Nky, Nz, n_op, N, R)`` with batch indices leading."""
    gyro_U: jnp.ndarray        # (Nky, Nz, M, Nx, R)    U·S of 𝒥_m
    gyro_V: jnp.ndarray        # (Nky, Nz, M, Nkx, R)
    pol_U: jnp.ndarray         # (Nky, Nz, Nx, R)       U·S of 1 − Γ_0
    pol_V: jnp.ndarray         # (Nky, Nz, Nkx, R)
    precond: jnp.ndarray       # (Nky, Nz, Nkx)  1 / (⟨1−Γ0⟩_x(q) + ⟨T_i/T_e⟩_x)
    Ainv0: jnp.ndarray         # (Nz, Nx, Nx)    dense inverse of the k_y = 0 polarisation operator (T_i⁻¹ Pol + T_e⁻¹)
    W: jnp.ndarray             # (Nx, Nx)        Woodbury matrix for the k_y = 0 flux-surface coupling
    fsw: jnp.ndarray           # (Nx, Nz)
    inv_Te: jnp.ndarray        # (Nx,)
    Ti: jnp.ndarray            # (Nx,)
    M: int
    rank: int
    cg_tol: float
    cg_maxiter: int
    nodes_U: jnp.ndarray | None = None   # (Nky, Nz, Nmu, Nx, R)
    nodes_V: jnp.ndarray | None = None

    # ---------------------------------------------------------------- operator application
    @staticmethod
    def _apply(U, V, pt):
        """Hermitian part of the quantised symbol on ``pt`` (Nky, Nz, Nx) → (Nky, Nz, n_op, Nx)."""
        # A φ = Σ_r U_r ⊙ ifft(V_r ⊙ fft φ)
        ph = jnp.fft.fft(pt, axis=-1)                                              # (k, z, q)
        t = jnp.fft.ifft(V[..., :, :] * ph[:, :, None, :, None], axis=-2)         # (k, z, m, x, r)
        a = jnp.sum(U * t, axis=-1)                                                # (k, z, m, x)
        # A† φ = Σ_r ifft(V_r ⊙ fft(U_r ⊙ φ))
        s = jnp.fft.fft(U * pt[:, :, None, :, None], axis=-2)                      # (k, z, m, q, r)
        b = jnp.fft.ifft(jnp.sum(V * s, axis=-1), axis=-1)                         # (k, z, m, x)
        return 0.5 * (a + b)

    def gyroaverage(self, phi: jnp.ndarray) -> jnp.ndarray:
        pt = jnp.transpose(phi, (1, 2, 0))
        return jnp.transpose(self._apply(self.gyro_U, self.gyro_V, pt), (2, 3, 0, 1))      # (M, Nx, Nky, Nz)

    def gyroaverage_nodes(self, phi: jnp.ndarray) -> jnp.ndarray:
        pt = jnp.transpose(phi, (1, 2, 0))
        return jnp.transpose(self._apply(self.nodes_U, self.nodes_V, pt), (2, 3, 0, 1))    # (Nmu, Nx, Nky, Nz)

    def gyro_density(self, G0: jnp.ndarray) -> jnp.ndarray:
        """``Σ_m 𝒥_m G_{0m}``: each 𝒥_m is Hermitian, so apply it to G_{0m} and sum."""
        gt = jnp.transpose(G0, (2, 3, 0, 1))                                               # (k, z, m, x)
        ph = jnp.fft.fft(gt, axis=-1)
        t = jnp.fft.ifft(self.gyro_V * ph[..., None], axis=-2)                             # (k, z, m, x, r)
        a = jnp.sum(self.gyro_U * t, axis=(-1, -3))                                        # (k, z, x)
        s = jnp.fft.fft(self.gyro_U * gt[..., None], axis=-2)                              # (k, z, m, q, r)
        b = jnp.fft.ifft(jnp.sum(self.gyro_V * s, axis=(-1, -3)), axis=-1)                 # (k, z, x)
        return jnp.transpose(0.5 * (a + b), (2, 0, 1))                                     # (Nx, Nky, Nz)

    def polarisation(self, pt: jnp.ndarray) -> jnp.ndarray:
        """``[Pol + diag(T_i/T_e)] φ`` on ``pt`` (Nky, Nz, Nx), Hermitian positive definite."""
        p = self._apply(self.pol_U[:, :, None], self.pol_V[:, :, None], pt)[:, :, 0]
        return p + (self.Ti * self.inv_Te) * pt

    # ---------------------------------------------------------------- field solve
    def _pcg(self, rhs):
        """Batched preconditioned CG over the leading (Nky, Nz) blocks; stops when every block's residual is below tol."""
        prec = lambda r: jnp.fft.ifft(jnp.fft.fft(r, axis=-1) * self.precond, axis=-1)
        dot = lambda a, b: jnp.sum(jnp.conj(a) * b, axis=-1).real                          # (k, z)
        b2 = jnp.maximum(dot(rhs, rhs), 1e-300)
        x = prec(rhs)
        r = rhs - self.polarisation(x)
        zt = prec(r)
        p = zt
        rz = dot(r, zt)

        def cond(state):
            x, r, p, rz, it = state
            return jnp.logical_and(it < self.cg_maxiter, jnp.max(dot(r, r) / b2) > self.cg_tol ** 2)

        def body(state):
            x, r, p, rz, it = state
            Ap = self.polarisation(p)
            alpha = rz / jnp.maximum(dot(p, Ap), 1e-300)
            x = x + alpha[..., None] * p
            r = r - alpha[..., None] * Ap
            zt = prec(r)
            rz_new = dot(r, zt)
            p = zt + (rz_new / jnp.maximum(rz, 1e-300))[..., None] * p
            return x, r, p, rz_new, it + 1

        x, r, p, rz, it = jax.lax.while_loop(cond, body, (x, r, p, rz, 0))
        return x

    def solve(self, G0: jnp.ndarray) -> jnp.ndarray:
        rhs = self.gyro_density(G0)                                                        # (Nx, Nky, Nz)
        rt = jnp.transpose(rhs, (1, 2, 0))                                                 # (k, z, x)
        # k_y ≠ 0: T_i-weighted Hermitian system by PCG (row 0 is solved too, then overwritten)
        phi_t = self._pcg(self.Ti * rt)
        phi = jnp.transpose(phi_t, (2, 0, 1))
        # k_y = 0: dense inverse + Woodbury, exactly as velcro.fields
        rhs0 = rhs[:, 0, :]
        rhs0 = rhs0 - jnp.sum(rhs0 * self.fsw) / jnp.sum(self.fsw)
        p0 = jnp.einsum("zxw,wz->xz", self.Ainv0, rhs0)
        avg = jnp.sum(p0 * self.fsw, axis=1)
        phibar = self.W @ avg
        corr = jnp.einsum("zxw,w->xz", self.Ainv0, self.inv_Te * phibar)
        phi0 = p0 + corr
        phi0 = phi0 - jnp.mean(phi0)
        return phi.at[:, 0, :].set(phi0)


def build_lowrank_field_solver(geom, ky: np.ndarray, Ti: np.ndarray, Te: np.ndarray, M: int, Lx: float,
                               dtype=np.complex128, node_gyro: bool = False, tol: float = 1e-8,
                               cg_tol: float = 1e-6, cg_maxiter: int = 50) -> LowRankFieldSolver:
    """Factorise every symbol slice by slice (memory ``O(M N_x N_kx)`` at any time) and keep the k_y = 0 row dense."""
    from scipy.special import j0
    Nx, Nz, Nky = len(geom.x), len(geom.z), len(ky)
    kx = kx_grid(Nx, Lx)
    gxx, gxy, gyy, B = (geom.y_averaged(a)[:, 0, :] for a in (geom.gxx, geom.gxy, geom.gyy, geom.B))
    rho2 = 0.5 * Ti[:, None] / B ** 2                                                     # (Nx, Nz)
    mu_nodes = basis.laguerre_quadrature(M)[0] if node_gyro else None
    gU, gV, pU, pV, nU, nV, pre = [], [], [], [], [], [], []
    ranks = []
    pol0 = np.zeros((Nz, Nx, Nx))
    # k_y = 0 row: op[x, w] = (1/N_x) Σ_q pol[x, q] exp(i k_q (x − w)) = ifft(pol[x])[(x − w) mod N_x]  (x on the FFT grid)
    shift = (np.arange(Nx)[:, None] - np.arange(Nx)[None, :]) % Nx
    for k in range(Nky):
        for z in range(Nz):
            b = rho2[:, z, None] * (gxx[:, z, None] * kx[None, :] ** 2 + 2 * gxy[:, z, None] * kx[None, :] * ky[k]
                                    + gyy[:, z, None] * ky[k] ** 2)                       # (Nx, Nkx)
            b = np.maximum(b, 0.0)
            Jm = basis.gyro_coefficients(b, M)                                             # (M, Nx, Nkx)
            pol = 1.0 - basis.gamma0(b)
            us, vs = [], []
            for m in range(M):
                u, v, r = _factor(Jm[m], tol); us.append(u); vs.append(v); ranks.append(r)
            gU.append(us); gV.append(vs)
            u, v, r = _factor(pol, tol); pU.append([u]); pV.append([v]); ranks.append(r)
            pre.append(1.0 / (pol.mean(axis=0) + np.mean(Ti / Te)))
            if node_gyro:
                us, vs = [], []
                for mu in mu_nodes:
                    u, v, r = _factor(j0(np.sqrt(2.0 * b * mu)), tol); us.append(u); vs.append(v); ranks.append(r)
                nU.append(us); nV.append(vs)
            if k == 0:
                op = np.take_along_axis(np.fft.ifft(pol, axis=1), shift, axis=1)
                pol0[z] = 0.5 * (op + op.conj().T).real
    R = max(ranks)
    stack = lambda lst: np.array([_pad(s, R) for s in lst]).reshape(Nky, Nz, -1, lst[0][0].shape[0], R)
    rd = np.float64 if dtype == np.complex128 else np.float32
    # k_y = 0 row: dense inverse and Woodbury, as in velcro.fields
    inv_Ti, inv_Te = 1.0 / Ti, 1.0 / Te
    I = np.eye(Nx)
    A0 = inv_Ti[None, :, None] * pol0 + inv_Te[None, :, None] * I[None]
    Ainv0 = np.linalg.inv(A0)
    fsw = geom.fs_weights()
    K = np.einsum("xz,zxw->xw", fsw, Ainv0) * inv_Te[None, :]
    ones = np.ones((Nx, 1))
    W = np.linalg.inv(I - K + ones @ ones.T / Nx)
    fs = LowRankFieldSolver(
        gyro_U=jnp.asarray(stack(gU), dtype=rd), gyro_V=jnp.asarray(stack(gV), dtype=rd),
        pol_U=jnp.asarray(stack(pU)[:, :, 0], dtype=rd), pol_V=jnp.asarray(stack(pV)[:, :, 0], dtype=rd),
        precond=jnp.asarray(np.array(pre).reshape(Nky, Nz, Nx), dtype=rd),
        Ainv0=jnp.asarray(Ainv0, dtype=dtype), W=jnp.asarray(W.real, dtype=rd), fsw=jnp.asarray(fsw, dtype=rd),
        inv_Te=jnp.asarray(inv_Te, dtype=rd), Ti=jnp.asarray(Ti, dtype=rd), M=M, rank=R,
        cg_tol=cg_tol, cg_maxiter=cg_maxiter,
        nodes_U=jnp.asarray(stack(nU), dtype=rd) if node_gyro else None,
        nodes_V=jnp.asarray(stack(nV), dtype=rd) if node_gyro else None,
    )
    return fs
