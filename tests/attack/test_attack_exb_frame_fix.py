"""The fix of the shearing-frame bracket (operators.frame_bracket, wired into vlasov.rhs when the frame is on).

Truths: (1) at θ = 0 it is the plain skew bracket with the same ∂_x; (2) it conserves Σ b{a, b} for ANY θ (the skew
structure: the frame ∂_x − iθ is anti-Hermitian per row); (3) for per-row integer θ_ky = n_ky Δk_x with n NOT ∝ k_y the
physical fields e^{−in Δk_x x}a are periodic, so the plain Fourier bracket of the physical fields is computable exactly and
mapped back must equal the frame bracket; (4) the frame right-hand side commutes with the remap (gauge covariance) to
rounding with the Fourier ∂_x, to the stencil's error with FD4.
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from velcro.operators import bracket, ddx, fourier_dx, fourier_kx, frame_bracket, real_fourier_dx

from test_attack_exb_frame import FOURIER_X, _gauge, _noise, _slab, _tube

Nx, Ny, Nz, Lx = 32, 16, 3, 20.0
NKY = (Ny - 1) // 3 + 1
DX = Lx / Nx
X = -0.5 * Lx + DX * np.arange(Nx)
KY = jnp.asarray(2 * np.pi * np.arange(NKY) / 30.0)[None, :, None]
KXS = jnp.asarray(fourier_kx(Nx, DX).reshape(-1, 1, 1))


def _field(seed, kmax=4, lead=(2,)):
    r = np.random.default_rng(seed)
    f = r.normal(size=lead + (Nx, NKY, Nz)) + 1j * r.normal(size=lead + (Nx, NKY, Nz))
    fh = np.fft.fft(f, axis=-3)
    m = np.abs(np.fft.fftfreq(Nx, d=1.0 / Nx))
    fh[..., m > kmax, :, :] = 0.0
    f = np.fft.ifft(fh, axis=-3)
    f[..., :, 0, :] = f[..., :, 0, :].real
    return jnp.asarray(f)


DXP = {"fourier": lambda f: fourier_dx(f, KXS), "fd4": lambda f: ddx(f, DX)}
DXF = {"fourier": real_fourier_dx(KXS), "fd4": None}


@pytest.mark.parametrize("kind", ["fourier", "fd4"])
def test_frame_bracket_at_theta0_is_the_plain_bracket(kind):
    a, b = _field(1), _field(2)
    want = np.asarray(bracket(a, b, KY, DX, Ny, dxf=DXF[kind]))
    got = np.asarray(frame_bracket(a, b, KY, DX, Ny, jnp.zeros(NKY), jnp.asarray(X), DXP[kind]))
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"{kind}: θ = 0 frame bracket vs plain: {err:.2e}")
    assert err < 1e-13


@pytest.mark.parametrize("kind", ["fourier", "fd4"])
def test_frame_bracket_conserves_the_free_energy_at_any_theta(kind):
    a, b = _field(3, kmax=9), _field(4, kmax=9)
    w = np.where(np.arange(NKY) == 0, 1.0, 2.0)[None, :, None]
    for seed in range(3):
        th = jnp.asarray(np.random.default_rng(seed).uniform(-0.5, 0.5, NKY) * 2 * np.pi / Lx * (np.arange(NKY) > 0))
        br = np.asarray(frame_bracket(a, b, KY, DX, Ny, th, jnp.asarray(X), DXP[kind]))
        e = np.sum(w * np.real(np.conj(np.asarray(b)) * br))
        scale = np.sum(w * np.abs(np.asarray(b)) * np.abs(br))
        print(f"{kind} seed {seed}: Σ b{{a,b}} / Σ|b||{{a,b}}| = {e / scale:.2e}")
        assert abs(e) < 1e-13 * scale


def test_frame_bracket_is_the_physical_bracket_for_arbitrary_integer_theta():
    """θ_ky = n_ky Δk_x with n = (0, 1, 0, −1, 2, 1): the physical fields are periodic; their plain Fourier bracket mapped
    back by e^{+iθx} must equal the frame bracket.  The old path (plain bracket of the frame representations) fails it."""
    dkx = 2 * np.pi / Lx
    n = np.array([0, 1, 0, -1, 2, 1])[:NKY]
    th = jnp.asarray(n * dkx)
    a, b = _field(5, kmax=3), _field(6, kmax=3)                    # sheared content ≤ 3 + 2: products ≤ 10 < 16
    E = np.exp(-1j * np.asarray(th)[None, :, None] * X[:, None, None])
    A, B = jnp.asarray(np.asarray(a) * E), jnp.asarray(np.asarray(b) * E)
    want = np.asarray(bracket(A, B, KY, DX, Ny, dxf=DXF["fourier"])) / E
    got = np.asarray(frame_bracket(a, b, KY, DX, Ny, th, jnp.asarray(X), DXP["fourier"]))
    old = np.asarray(bracket(a, b, KY, DX, Ny, dxf=DXF["fourier"]))
    err, err_old = np.abs(got - want).max() / np.abs(want).max(), np.abs(old - want).max() / np.abs(want).max()
    print(f"integer θ not ∝ k_y: frame bracket error {err:.2e}; the old plain bracket of the representations {err_old:.2e}")
    assert err < 1e-12 and err_old > 1e-2


@pytest.mark.parametrize("s", [0.3, 0.45, 0.8])
def test_fixed_frame_rhs_commutes_with_the_remap_slab(s):
    sim = _slab(nonlinear=True, numerics=FOURIER_X)
    p = _noise(sim, 1, scale=10.0, kmax=4, zflat=True)
    theta, p2, th2, ph, n = _gauge(sim, p, s)
    F = np.asarray(sim.rhs_fn(p, theta))
    F2 = np.asarray(sim.rhs_fn(p2, th2))
    err = np.abs(F2 - F * ph).max() / np.abs(F).max()
    print(f"fixed, slab, s={s}: n_j={n.tolist()} gauge error {err:.2e}")
    assert err < 1e-11


@pytest.mark.parametrize("numerics,tol", [(FOURIER_X, 1e-11), (None, 2e-2)])
def test_fixed_frame_rhs_commutes_with_the_remap_tube(numerics, tol):
    """Every term on; with FD4 (the default bracket_dx/radial_drift) the remaining error is the stencil's non-covariance
    on this coarse low-k_x state (it was 0.25 with the old bracket)."""
    sim = _tube(nonlinear=True, numerics=numerics)
    p = _noise(sim, 3, scale=30.0, kmax=6)
    theta, p2, th2, ph, n = _gauge(sim, p, 0.35)
    F = np.asarray(sim.rhs_fn(p, theta))
    F2 = np.asarray(sim.rhs_fn(p2, th2))
    err = np.abs(F2 - F * ph).max() / np.abs(F).max()
    print(f"fixed, tube, {'fourier' if numerics else 'fd4'}: n_j={n.tolist()} gauge error {err:.2e}")
    assert err < tol
