"""Post-processing helpers: mode fitting (Prony), plasma dispersion function, reference dispersion relations."""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve
from scipy.special import wofz


def Z(zeta):
    """Plasma dispersion function ``Z(ζ) = i√π w(ζ)``."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def landau_ion_acoustic_root(tau: float, kpar: float = 1.0):
    """Least-damped root of ``1 + 1/τ + ζ Z(ζ) = 0`` (ion Landau damping of the ion
    acoustic wave with adiabatic electrons, ``v_t = sqrt(2T_i/m)``).  Returns
    ``ω = ζ k∥ v_t`` (complex; ``Im ω < 0`` is damping)."""
    def f(v):
        z = v[0] + 1j * v[1]
        r = 1 + 1 / tau + z * Z(z)
        return [r.real, r.imag]
    best = None
    for zr in np.linspace(0.3, 3.5, 17):
        for zi in np.linspace(-1.5, -0.02, 9):
            s, info, ier, _ = fsolve(f, [zr, zi], full_output=True)
            if ier == 1 and s[1] < 0 and s[0] > 0:
                if best is None or s[1] > best[1]:
                    best = s
    return (best[0] + 1j * best[1]) * kpar


def prony(t: np.ndarray, p: np.ndarray, order: int = 2):
    """Fit ``p(t) ≈ Σ_k a_k exp(-i ω_k t)`` on uniform samples by Prony's method.
    Returns complex frequencies ``ω_k`` (``Im ω > 0`` = growth) and amplitudes, sorted by |a|."""
    t, p = np.asarray(t), np.asarray(p)
    dt = t[1] - t[0]
    n = len(p)
    # linear prediction: p[k] = Σ_j c_j p[k-j]
    A = np.column_stack([p[order - j - 1: n - j - 1] for j in range(order)])
    c = np.linalg.lstsq(A, p[order:], rcond=None)[0]
    roots = np.roots(np.concatenate([[1.0], -c]))
    omega = 1j * np.log(roots) / dt
    # amplitudes
    Vd = np.exp(-1j * np.outer(t, omega))
    a = np.linalg.lstsq(Vd, p, rcond=None)[0]
    idx = np.argsort(-np.abs(a))
    return omega[idx], a[idx]


def landau_root_with_flr(tau: float, b: float, kpar: float = 1.0):
    """Least-damped root of ``1 + (1 − Γ0(b) + 1/τ)/Γ0(b) + ζ Z(ζ) = 0`` — the slab
    ion-acoustic/Landau dispersion relation including the polarisation and
    gyroaverage factors at finite ``b = k_⊥² ρ_i²``."""
    from velcro.basis import gamma0
    G0 = float(gamma0(np.array(b)))
    c = (1 - G0 + 1 / tau) / G0

    def f(v):
        z = v[0] + 1j * v[1]
        r = 1 + c + z * Z(z)
        return [r.real, r.imag]
    best = None
    for zr in np.linspace(0.3, 3.5, 17):
        for zi in np.linspace(-1.5, -0.02, 9):
            s, info, ier, _ = fsolve(f, [zr, zi], full_output=True)
            if ier == 1 and s[1] < 0 and s[0] > 0:
                if best is None or s[1] > best[1]:
                    best = s
    return (best[0] + 1j * best[1]) * kpar


def linear_matrix(sim, ky_index: int):
    """Dense matrix of the linearised Velcro operator restricted to one ``k_y`` block.

    Returns ``A`` such that ``∂_t g = A g`` for the flattened block ``G[:, :, :, ky_index, :]``
    (nonlinear term disabled).  Sizes must be small (dense Jacobian)."""
    import jax
    import jax.numpy as jnp
    from velcro.vlasov import rhs
    import equinox as eqx
    import dataclasses
    if getattr(sim, "kinetic", False):
        return _linear_matrix_kinetic(sim, ky_index)
    op = dataclasses.replace(sim.op, nonlinear=False)
    shape = sim.G.shape
    L, M, Nx, Nky, Nz = shape

    def f(gblock):
        G = jnp.zeros(shape, dtype=sim.dtype).at[:, :, :, ky_index, :].set(gblock.reshape(L, M, Nx, Nz))
        phi = sim.fields.solve(G[0])
        Phi = sim.fields.gyroaverage(phi)
        out = rhs(G, phi, Phi, op)
        return out[:, :, :, ky_index, :].reshape(-1)

    n = L * M * Nx * Nz
    A = jax.jacfwd(f, holomorphic=True)(jnp.zeros(n, dtype=sim.dtype))
    return np.asarray(A)


def _linear_matrix_kinetic(sim, ky_index: int, chunk: int = 128):
    """:func:`linear_matrix` for a kinetic-electron sim (velcro/kinetic.py): the stacked (species, ℓ, m, x, z) block of one
    k_y, nonlinearity off.  Used with ``integrator = "rk4"`` (the Lawson right-hand side omits the k_y drift of G).  The
    operator is linear, so its columns are images of unit vectors, built ``chunk`` at a time: one ``jacfwd`` over all n
    columns holds n copies of every intermediate (≈ 50 GB at n = 6144)."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    from velcro.kinetic import kinetic_fields, rhs_kinetic
    op = dataclasses.replace(sim.op, base=dataclasses.replace(sim.op.base, nonlinear=False))
    shape = sim.G.shape
    SL, M, Nx, Nky, Nz = shape

    def f(gblock):
        G = jnp.zeros(shape, dtype=sim.dtype).at[:, :, :, ky_index, :].set(gblock.reshape(SL, M, Nx, Nz))
        out = rhs_kinetic(G, kinetic_fields(G, sim.fields, op, False), op)
        return out[:, :, :, ky_index, :].reshape(-1)

    n = SL * M * Nx * Nz
    fv = jax.jit(jax.vmap(f))
    cols = []
    for i in range(0, n, chunk):
        c = min(chunk, n - i)
        E = np.zeros((c, n), dtype=complex)
        E[np.arange(c), i + np.arange(c)] = 1.0
        cols.append(np.asarray(fv(jnp.asarray(E, dtype=sim.dtype))))
    return np.concatenate(cols, axis=0).T


def eigen_modes(A: np.ndarray, n: int = 5):
    """Eigenvalues ``λ`` of ``A`` sorted by growth rate (``Re λ`` descending).
    Complex frequency ``ω = i λ`` (so ``Re ω`` = real frequency, ``Im ω`` = growth)."""
    lam = np.linalg.eigvals(A)
    idx = np.argsort(-lam.real)
    return 1j * lam[idx[:n]], lam[idx]
