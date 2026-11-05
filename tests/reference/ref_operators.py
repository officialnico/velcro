"""Spatial discretisation (§3.2, §3.4): finite differences in ``x`` and ``z``,
Fourier pseudo-spectral in ``y`` with 2/3-rule de-aliasing, twist-and-shift
parallel boundary conditions, and the Orszag-style Poisson bracket.

Field arrays are complex with layout ``(..., Nx, Nky, Nz)`` for spectral-``y``
data.  ``Nky`` counts the retained (de-aliased) non-negative ``k_y`` modes; the
real-space ``y`` grid has ``Ny`` points with ``Nky = (Ny − 1) // 3 + 1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# 4th-order centred stencils
_D1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0          # first derivative
_D4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0])                  # (Δ)^4 ∂^4 (hyperdiffusion)


def n_ky(Ny: int) -> int:
    """Number of retained non-negative ``k_y`` modes: the 2/3 rule requires
    ``Ny ≥ 3 K + 1`` for products of modes ``|k| ≤ K`` to be alias-free."""
    return (Ny - 1) // 3 + 1


def ky_grid(Ny: int, Ly: float) -> np.ndarray:
    """Retained non-negative ``k_y`` modes after the 2/3 rule."""
    return 2 * np.pi * np.arange(n_ky(Ny)) / Ly


def kx_grid(Nx: int, Lx: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)


# --------------------------------------------------------------------------
# x direction: periodic 4th-order finite differences (buffer zones make the
# periodicity inert; see §3.2 and the Krook buffers in vlasov.py)
# --------------------------------------------------------------------------
def ddx(f: jnp.ndarray, dx: float, axis: int = -3) -> jnp.ndarray:
    r = lambda s: jnp.roll(f, -s, axis=axis)
    return (_D1[0] * r(-2) + _D1[1] * r(-1) + _D1[3] * r(1) + _D1[4] * r(2)) / dx


def hyper_x(f: jnp.ndarray, axis: int = -3) -> jnp.ndarray:
    """Grid-scale fourth difference ``Δx⁴ ∂⁴_x f`` (positive-definite damping when subtracted)."""
    r = lambda s: jnp.roll(f, -s, axis=axis)
    return r(-2) - 4 * r(-1) + 6 * f - 4 * r(1) + r(2)


# --------------------------------------------------------------------------
# z direction: twist-and-shift.  f(x, k_y, z + 2π) = f(x, k_y, z) exp(-i k_y τ(x))
# with τ(x) = 2π q(x) C_y provided by the geometry.
# --------------------------------------------------------------------------
def z_ghosts(f: jnp.ndarray, phase: jnp.ndarray, n: int = 2):
    """Return ``f`` padded with ``n`` twist-and-shift ghost cells on both ends of the
    last axis.  ``phase`` has shape ``(Nx, Nky, 1)`` and holds ``exp(-i k_y τ(x))``."""
    top = f[..., :n] * phase          # values at z ≥ π  (continue from z = -π side)
    bot = f[..., -n:] * jnp.conj(phase)  # values at z < -π
    return jnp.concatenate([bot, f, top], axis=-1)


def ddz(f: jnp.ndarray, dz: float, phase: jnp.ndarray) -> jnp.ndarray:
    g = z_ghosts(f, phase, 2)
    n = f.shape[-1]
    return (_D1[0] * g[..., 0:n] + _D1[1] * g[..., 1:n + 1] + _D1[3] * g[..., 3:n + 3] + _D1[4] * g[..., 4:n + 4]) / dz


def hyper_z(f: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    g = z_ghosts(f, phase, 2)
    n = f.shape[-1]
    return g[..., 0:n] - 4 * g[..., 1:n + 1] + 6 * f - 4 * g[..., 3:n + 3] + g[..., 4:n + 4]


# --------------------------------------------------------------------------
# y direction: real <-> spectral with 2/3 de-aliasing
# --------------------------------------------------------------------------
def to_real_y(f: jnp.ndarray, Ny: int, axis: int = -2) -> jnp.ndarray:
    """Inverse real FFT of de-aliased modes onto the ``Ny``-point real grid (zero padded)."""
    nky = f.shape[axis]
    nfull = Ny // 2 + 1
    pad = [(0, 0)] * f.ndim
    pad[axis] = (0, nfull - nky)
    fp = jnp.pad(f, pad)
    return jnp.fft.irfft(fp, n=Ny, axis=axis) * Ny   # unnormalised: Σ_k f_k e^{iky}


def to_spec_y(f: jnp.ndarray, nky: int, axis: int = -2) -> jnp.ndarray:
    """Forward real FFT and truncation to the retained ``k_y`` modes."""
    Ny = f.shape[axis]
    F = jnp.fft.rfft(f, axis=axis) / Ny
    return jnp.take(F, jnp.arange(nky), axis=axis)


def multiply_real_y(coef: jnp.ndarray, f: jnp.ndarray, Ny: int) -> jnp.ndarray:
    """Pseudo-spectral product ``coef(x,y,z) f(x,k_y,z)`` → spectral, de-aliased.
    ``coef`` is real with shape broadcastable to ``(Nx, Ny, Nz)``."""
    nky = f.shape[-2]
    fr = to_real_y(f, Ny)
    return to_spec_y(coef * fr, nky)


def apply_coef(coef: jnp.ndarray, f: jnp.ndarray, Ny: int, axisymmetric: bool) -> jnp.ndarray:
    """Multiply a spectral field by a geometry coefficient (``(Nx,1,Nz)`` for
    axisymmetric geometry → direct; ``(Nx,Ny,Nz)`` → pseudo-spectral)."""
    if axisymmetric:
        return coef * f
    return multiply_real_y(coef, f, Ny)


# --------------------------------------------------------------------------
# Poisson bracket {a, b} = ∂_x a ∂_y b − ∂_y a ∂_x b, pseudo-spectral in y
# --------------------------------------------------------------------------
def bracket(a: jnp.ndarray, b: jnp.ndarray, ky: jnp.ndarray, dx: float, Ny: int) -> jnp.ndarray:
    """De-aliased pseudo-spectral Poisson bracket of spectral fields ``a``, ``b``
    with layout ``(..., Nx, Nky, Nz)``.  ``ky`` has shape ``(1, Nky, 1)``."""
    nky = a.shape[-2]
    ax, bx = ddx(a, dx), ddx(b, dx)
    ay, by = 1j * ky * a, 1j * ky * b
    r = lambda f: to_real_y(f, Ny)
    prod = r(ax) * r(by) - r(ay) * r(bx)
    return to_spec_y(prod, nky)
