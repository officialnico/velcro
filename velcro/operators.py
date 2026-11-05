"""Spatial discretisation (§3.2, §3.4): finite differences in ``x`` and ``z``,
Fourier pseudo-spectral in ``y`` with 2/3-rule de-aliasing, twist-and-shift
parallel boundary conditions, and the Orszag-style Poisson bracket.

Field arrays are complex with layout ``(..., Nx, Nky, Nz)`` for spectral-``y``
data.  ``Nky`` counts the retained (de-aliased) non-negative ``k_y`` modes; the
real-space ``y`` grid has ``Ny`` points with ``Nky = (Ny − 1) // 3 + 1``.

Performance note: the ``y`` transforms are short (``Ny`` = 24–96) and sit on a non-final
axis, so they are applied as dense real DFT matrices (one real matmul each way) rather than
through ``jnp.fft``; on CPU that is 2–3x faster for these lengths and identical to roundoff.
"""
from __future__ import annotations

from functools import lru_cache

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


def hyper_x_sheared(p: jnp.ndarray, theta: jnp.ndarray, dx: float) -> jnp.ndarray:
    """:func:`hyper_x` of the physical field in the shearing frame of ``physics.exb_shear``:
    ``p`` (…, Nx, Nky, Nz) is the frame representation G = e^{−iθx} p, so its grid mode k_x stands for the physical
    k_x − θ_ky.  The stencil's symbol 16 sin⁴(κ/2), κ = (k_x − θ_ky) Δx, is applied spectrally along x: at θ = 0 this is
    the stencil itself (to rounding), and at θ = Δk_x it is the stencil on the periodic sheared field e^{−iΔk_x x} p.
    ``theta`` (1, Nky, 1) in the units of k_x."""
    n = p.shape[-3]
    kx = jnp.asarray(2.0 * np.pi * np.fft.fftfreq(n, d=dx), dtype=p.real.dtype)[:, None, None]
    sym = 16.0 * jnp.sin(0.5 * dx * (kx - theta)) ** 4                        # (Nx, Nky, 1)
    return jnp.fft.ifft(sym * jnp.fft.fft(p, axis=-3), axis=-3)


def fourier_kx(Nx: int, dx: float) -> np.ndarray:
    """``k_x`` of the FFT-ordered x modes with the Nyquist mode set to 0 — the symbol of :func:`fourier_dx`
    (``numerics.radial_drift`` / ``numerics.bracket_dx = "fourier"``): ``i k_x`` is then anti-Hermitian and real rows stay
    real."""
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    if Nx % 2 == 0:
        kx[Nx // 2] = 0.0
    return kx


def ifft_unscaled(X: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """``N · ifft(X)`` along ``axis`` (the unnormalised inverse DFT), computed as ``conj(fft(conj(X)))``.  On the GPU XLA
    runs an inverse FFT as cuFFT's unnormalised transform followed by a separate cuBLAS ``scal`` pass over the output
    (the 1/N); a forward FFT has none, and the two conjugations fuse into the neighbouring kernels.  Callers fold the 1/N
    into the multiplier they apply just before (the scal passes
    were 0.9–1.0 ms of a kinetic-electron step).  Equal to ``N · ifft`` up to rounding."""
    return jnp.conj(jnp.fft.fft(jnp.conj(X), axis=axis))


def fourier_dx(f: jnp.ndarray, kx_sym: jnp.ndarray) -> jnp.ndarray:
    """``∂_x f`` by FFT along x (axis −3; the box is periodic in x): ``i k_x`` on every mode with ``kx_sym`` (Nx, 1, 1)
    from :func:`fourier_kx`."""
    return jnp.fft.ifft(jnp.fft.fft(f, axis=-3) * (1j * kx_sym), axis=-3)


def real_fourier_dx(kx_sym: jnp.ndarray):
    """The ``∂_x`` of :func:`fourier_dx` for real arrays (x on axis −3), by rfft — the bracket's real-space factors."""
    Nx = kx_sym.shape[0]
    kx = kx_sym[: Nx // 2 + 1]
    return lambda f: jnp.fft.irfft(jnp.fft.rfft(f, axis=-3) * (1j * kx.astype(f.dtype)), n=Nx, axis=-3)


def x_dealias_mask(Nx: int) -> np.ndarray:
    """1 on |m| ≤ (Nx−1)//3 of the FFT-ordered x modes (GX's retained k_x for the same N_x), 0 elsewhere: the 2/3 rule
    of ``numerics.x_dealias``."""
    m = np.abs(np.fft.fftfreq(Nx, d=1.0 / Nx))
    return (m <= (Nx - 1) // 3).astype(float)


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
# y direction: real <-> spectral with 2/3 de-aliasing, as dense DFT matrices.
#
#   real:  f(y_j) = Σ_k w_k Re(f_k e^{i k y_j}),  w_0 = 1, w_k = 2 (Nyquist, if present: 1)
#   spec:  f_k    = (1/Ny) Σ_j f(y_j) e^{-i k y_j},  k = 0 … nky−1
#
# These are exactly ``irfft`` of the zero-padded modes (times Ny) and ``rfft / Ny`` truncated.
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _dft_matrices(nky: int, Ny: int):
    """``(fwd, inv)``: ``fwd`` is ``(2 nky, Ny)`` acting on stacked ``[Re f; Im f]``;
    ``inv`` is ``(Ny, 2 nky)`` producing stacked ``[Re f; Im f]`` from real samples."""
    nfull = Ny // 2 + 1
    if nky > nfull:
        raise ValueError(f"nky={nky} exceeds the {nfull} modes representable on Ny={Ny} points")
    y = 2 * np.pi * np.arange(Ny) / Ny
    k = np.arange(nky)
    C, S = np.cos(np.outer(k, y)), np.sin(np.outer(k, y))     # (nky, Ny)
    w = np.where((k == 0) | (2 * k == Ny), 1.0, 2.0)[:, None]
    fwd = np.concatenate([w * C, -w * S], axis=0)             # Re f cos − Im f sin
    inv = np.concatenate([C.T, -S.T], axis=1) / Ny            # (Ny, 2 nky)
    return fwd, inv


def to_real_y(f: jnp.ndarray, Ny: int, axis: int = -2) -> jnp.ndarray:
    """Inverse real transform of de-aliased modes onto the ``Ny``-point real grid
    (unnormalised: ``Σ_k f_k e^{iky}`` with the Hermitian partner modes included)."""
    axis = axis % f.ndim
    fwd, _ = _dft_matrices(f.shape[axis], Ny)
    fr = jnp.real(f)
    fwd = jnp.asarray(fwd, dtype=fr.dtype)
    stacked = jnp.concatenate([fr, jnp.imag(f)], axis=axis)   # (..., 2 nky, ...)
    if axis == f.ndim - 2:
        return jnp.einsum("...kz,ky->...yz", stacked, fwd)
    out = jnp.einsum("...k,ky->...y", jnp.moveaxis(stacked, axis, -1), fwd)
    return jnp.moveaxis(out, -1, axis)


def to_spec_y(f: jnp.ndarray, nky: int, axis: int = -2) -> jnp.ndarray:
    """Forward real transform (normalised by ``Ny``) truncated to the retained ``k_y`` modes."""
    axis = axis % f.ndim
    _, inv = _dft_matrices(nky, f.shape[axis])
    inv = jnp.asarray(inv, dtype=f.dtype)
    if axis == f.ndim - 2:
        out = jnp.einsum("...yz,yk->...kz", f, inv)
    else:
        out = jnp.moveaxis(jnp.einsum("...y,yk->...k", jnp.moveaxis(f, axis, -1), inv), -1, axis)
    re, im = jnp.split(out, 2, axis=axis)
    return jax.lax.complex(re, im)


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
def bracket_advective(a: jnp.ndarray, b: jnp.ndarray, ky: jnp.ndarray, dx: float, Ny: int) -> jnp.ndarray:
    """Advective form ``a_x b_y − a_y b_x`` (Orszag): de-aliased in ``y``, 4th-order FD in ``x``.

    Conserves ``Σ b {a, b}`` only to the truncation order of the ``x`` stencil: the residual
    is proportional to the grid-scale content of the velocity ``a_y``, and it is *positive* for
    grid-scale structure, so it can drive a finite-time blow-up at high amplitude.  Kept for
    comparison; the default is :func:`bracket`."""
    nky = a.shape[-2]
    ax, bx = ddx(a, dx), ddx(b, dx)
    ay, by = 1j * ky * a, 1j * ky * b
    r = lambda f: to_real_y(f, Ny)
    prod = r(ax) * r(by) - r(ay) * r(bx)
    return to_spec_y(prod, nky)


@lru_cache(maxsize=None)
def _bracket_matrices(nky: int, Ny: int, ky: tuple):
    """DFT matrices of :func:`bracket`: ``fwd2`` (2 nky, 2 Ny) maps stacked ``[Re f; Im f]`` to real-space
    ``[f | ∂_y f]``; ``inv2`` (2 Ny, 2 nky) maps real-space ``[P | Q]`` to the stacked spectral ``P̂ + i k_y Q̂``.
    On stacked coefficients ``i k_y`` is the real matrix ``J = [[0, −K], [K, 0]]``."""
    fwd, inv = _dft_matrices(nky, Ny)
    K = np.diag(np.asarray(ky, dtype=np.float64))
    Z = np.zeros((nky, nky))
    J = np.block([[Z, -K], [K, Z]])
    return np.concatenate([fwd, J.T @ fwd], axis=1), np.concatenate([inv, inv @ J.T], axis=0)


def _to_real_pair(f: jnp.ndarray, fwd2: np.ndarray) -> jnp.ndarray:
    """``(..., nky, Nz)`` → ``(..., 2 Ny, Nz)``: real-space ``f`` in the first ``Ny`` rows, ``∂_y f`` in the rest."""
    stacked = jnp.concatenate([jnp.real(f), jnp.imag(f)], axis=-2)
    return jnp.einsum("...kz,ky->...yz", stacked, jnp.asarray(fwd2, dtype=stacked.dtype))


def _to_spec_pair(PQ: jnp.ndarray, inv2: np.ndarray) -> jnp.ndarray:
    """``(..., 2 Ny, Nz)`` real ``[P | Q]`` → spectral ``to_spec_y(P) + i k_y to_spec_y(Q)``."""
    out = jnp.einsum("...yz,yk->...kz", PQ, jnp.asarray(inv2, dtype=PQ.dtype))
    re, im = jnp.split(out, 2, axis=-2)
    return jax.lax.complex(re, im)


def _bracket_integrand(pa: jnp.ndarray, pb: jnp.ndarray, dx: float, Ny: int, dxf=None) -> jnp.ndarray:
    """Real-space integrand of the skew bracket from the pairs ``[a | a_y]``, ``[b | b_y]`` of :func:`_to_real_pair`:
    ``[a_x b_y − a_y b_x − ∂_x(a_y b) | a_x b]``, whose :func:`_to_spec_pair` is ``2 {a, b}``.  ``∂_x`` (the FD4 stencil
    along the ``x`` axis, or ``dxf`` — :func:`real_fourier_dx` for ``numerics.bracket_dx = "fourier"``) commutes with
    the ``y`` transforms, so ``a_x``, ``b_x`` and the flux term are taken in real space."""
    d = (lambda f: ddx(f, dx)) if dxf is None else dxf
    ra, ray = pa[..., :Ny, :], pa[..., Ny:, :]
    rb, rby = pb[..., :Ny, :], pb[..., Ny:, :]
    rax, rbx = d(ra), d(rb)
    return jnp.concatenate([rax * rby - ray * rbx - d(ray * rb), rax * rb], axis=-2)


def _ky_key(ky) -> tuple:
    return tuple(float(k) for k in np.asarray(ky).reshape(-1))


def bracket(a: jnp.ndarray, b: jnp.ndarray, ky: jnp.ndarray, dx: float, Ny: int, dxf=None) -> jnp.ndarray:
    """Skew-symmetric (energy-conserving) Poisson bracket of spectral fields ``a``, ``b`` with
    layout ``(..., Nx, Nky, Nz)``; ``ky`` has shape ``(1, Nky, 1)`` (a concrete array)::

        {a, b} = ½ [a_x b_y + ∂_y(a_x b)] − ½ [a_y b_x + ∂_x(a_y b)]

    Each half is the average of the advective and the flux form of one advection term.  In the
    continuum the extra terms ``½ a_xy b`` cancel; on the grid they make ``Σ_xy b {a, b} = 0``
    hold to round-off for *any* ``a`` and ``b``, because the ``y`` products are de-aliased (so
    integration by parts in ``y`` is exact) and the ``x`` stencil is antisymmetric.

    Performance note: the four de-aliased products are formed from one inverse transform per field
    (``[f | ∂_y f]`` in one matmul, ``∂_x`` taken in real space) and returned through one forward
    transform (``P̂ + i k_y Q̂`` in one matmul, ``∂_x`` of the flux term taken in real space) — two
    matmuls on the large field instead of seven, identical to round-off (``tests/test_perf_equivalence.py``).
    ``dxf`` replaces the FD4 ``∂_x`` (:func:`real_fourier_dx`: the pseudo-spectral bracket, still exactly skew)."""
    fwd2, inv2 = _bracket_matrices(a.shape[-2], Ny, _ky_key(ky))
    PQ = _bracket_integrand(_to_real_pair(a, fwd2), _to_real_pair(b, fwd2), dx, Ny, dxf)
    return 0.5 * _to_spec_pair(PQ, inv2)


def frame_bracket(a: jnp.ndarray, b: jnp.ndarray, ky: jnp.ndarray, dx: float, Ny: int, theta: jnp.ndarray,
                  x: jnp.ndarray, dxp) -> jnp.ndarray:
    """:func:`bracket` in the shearing frame of ``physics.exb_shear``, for ANY frame phase θ_ky (Nky,).

    ``a``, ``b`` are frame representations: row k_y stands for the physical field A = e^{−iθ_ky x} a.  The plain bracket of
    the representations equals the physical bracket only when θ ∝ k_y (the frame is then a coordinate shear); the remap
    moves each row's θ by Δk_x on its own, after which the eddy–eddy triads of the plain bracket land one k_x off and miss
    the (θ₁k_y2 − k_y1θ₂) term.  Here, as GX's continuous flow-shear algorithm (nonlinear.cu: ∂_x at i·kxstar and the
    phase factor e^{i(kx*−kx̄)x} on every field in the FFT, the inverse phase on the result): the physical fields and their
    x-derivatives e^{−iθx}(∂_x − iθ)a are formed row by row, multiplied pointwise in real space, and the result is mapped
    back to the frame (e^{+iθx}; the flux term's ∂_x as ∂_x − iθ on the frame representation).  ``dxp`` is the periodic
    ∂_x along axis −3 of the frame representation (Fourier: exact, so the result is gauge covariant to rounding; FD4: to
    the stencil's error); ``x`` (Nx,) the grid.  At θ = 0 this is :func:`bracket` with the same ∂_x, to rounding."""
    nky = a.shape[-2]
    th = jnp.reshape(theta, (1, -1, 1)).astype(jnp.real(a).dtype)
    E = jnp.exp(-1j * th * jnp.reshape(x, (-1, 1, 1)).astype(th.dtype))            # (Nx, Nky, 1)
    A, B = E * a, E * b
    Ax = E * (dxp(a) - 1j * th * a)
    Bx = E * (dxp(b) - 1j * th * b)
    fwd2, inv2 = _bracket_matrices(nky, Ny, _ky_key(ky))
    pA, pB = _to_real_pair(A, fwd2), _to_real_pair(B, fwd2)
    rA, rAy = pA[..., :Ny, :], pA[..., Ny:, :]
    rB, rBy = pB[..., :Ny, :], pB[..., Ny:, :]
    rAx, rBx = to_real_y(Ax, Ny), to_real_y(Bx, Ny)
    main = _to_spec_pair(jnp.concatenate([rAx * rBy - rAy * rBx, rAx * rB], axis=-2), inv2)
    s = jnp.conj(E) * to_spec_y(rAy * rB, nky)                                        # frame rep. of A_y B
    return 0.5 * (jnp.conj(E) * main - (dxp(s) - 1j * th * s))


BRACKETS = {"skew": bracket, "advective": bracket_advective}


def gql_bracket(form: str, a: jnp.ndarray, b: jnp.ndarray, ky: jnp.ndarray, dx: float, Ny: int, cutoff: int) -> jnp.ndarray:
    """Generalised quasilinear bracket (Marston, Chini & Tobias 2016): modes with k_y index ≤ ``cutoff`` are "low"
    (``cutoff = 0``: the zonal modes only = the quasilinear approximation), the rest "high".  Kept: low·low → low,
    high·high → low, low·high → high.  Dropped: high·high → high (eddy–eddy scattering), low·low → high, low·high → low.
    The kept triads are closed under the quadratic invariants, so with the skew bracket Σ b {a, b} = 0 still holds.
    Each high k_y then evolves linearly in the instantaneous low (zonal) fields.

    For the skew form the bracket is bilinear and the transforms are linear, so the low and high parts of each field
    are transformed once and the real-space integrands are summed per destination before the forward transform:
    two inverse and two forward matmuls on the large field instead of the sixteen of four separate brackets."""
    low = (jnp.arange(a.shape[-2]) <= cutoff)[:, None]
    if form != "skew":
        f = BRACKETS[form]
        al, ah, bl, bh = a * low, a * (~low), b * low, b * (~low)
        to_low = f(al, bl, ky, dx, Ny) + f(ah, bh, ky, dx, Ny)
        to_high = f(al, bh, ky, dx, Ny) + f(ah, bl, ky, dx, Ny)
        return to_low * low + to_high * (~low)
    fwd2, inv2 = _bracket_matrices(a.shape[-2], Ny, _ky_key(ky))
    pal, pbl = _to_real_pair(a * low, fwd2), _to_real_pair(b * low, fwd2)
    pah, pbh = _to_real_pair(a, fwd2) - pal, _to_real_pair(b, fwd2) - pbl
    to_low = _to_spec_pair(_bracket_integrand(pal, pbl, dx, Ny) + _bracket_integrand(pah, pbh, dx, Ny), inv2)
    to_high = _to_spec_pair(_bracket_integrand(pal, pbh, dx, Ny) + _bracket_integrand(pah, pbl, dx, Ny), inv2)
    return 0.5 * (to_low * low + to_high * (~low))
