import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from velcro import operators as ops


def test_ddx_fourth_order_convergence():
    errs = []
    for Nx in (16, 32, 64):
        Lx = 2 * np.pi
        x = np.arange(Nx) * Lx / Nx
        f = jnp.asarray(np.sin(2 * x))[:, None, None] * jnp.ones((1, 1, 1), dtype=complex)
        d = ops.ddx(f, Lx / Nx)
        errs.append(float(jnp.max(jnp.abs(d[:, 0, 0] - 2 * np.cos(2 * x)))))
    order = np.log(errs[0] / errs[1]) / np.log(2), np.log(errs[1] / errs[2]) / np.log(2)
    assert min(order) > 3.8


def test_ddz_with_twist_phase():
    """A function periodic up to the twist phase must be differentiated exactly to 4th-order accuracy."""
    Nz = 64
    dz = 2 * np.pi / Nz
    z = -np.pi + dz * np.arange(Nz)
    tau = 0.7                       # twist phase per unit k_y
    ky = np.array([0.0, 1.0])
    # f(z) = exp(i (n - ky τ/(2π)) z): satisfies f(z+2π) = f(z) exp(-i ky τ)
    keff = 3 - ky * tau / (2 * np.pi)
    f = np.exp(1j * keff[None, :, None] * z[None, None, :]) * np.ones((2, 1, 1))
    phase = np.exp(-1j * ky * tau)[None, :, None] * np.ones((2, 1, 1))
    d = ops.ddz(jnp.asarray(f), dz, jnp.asarray(phase))
    exact = 1j * keff[None, :, None] * f
    # 4th-order truncation error ≈ k⁵ dz⁴ / 30 ≈ 7.5e-4 for k = 3, Nz = 64; uniform across z
    err = np.abs(np.asarray(d) - exact)
    assert np.max(err) < 1e-3
    assert np.std(err[0, 1]) < 1e-6  # no boundary artefacts from the twist ghosts


def test_fft_roundtrip_and_dealiasing():
    Ny = 24
    nky = ops.n_ky(Ny)
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, nky, 2)) + 1j * rng.normal(size=(3, nky, 2))
    f[:, 0] = f[:, 0].real
    g = ops.to_spec_y(ops.to_real_y(jnp.asarray(f), Ny), nky)
    assert np.allclose(np.asarray(g), f, atol=1e-12)


def _smooth_fields(Nx, Ny, Nz, Lx, Ly, seed=1):
    nky = ops.n_ky(Ny)
    x = np.arange(Nx) * Lx / Nx
    rng = np.random.default_rng(seed)
    def make():
        f = np.zeros((Nx, nky, Nz), dtype=complex)
        for n in range(-2, 3):
            for k in range(nky):
                c = rng.normal() + 1j * rng.normal()
                f[:, k, :] += c * np.exp(2j * np.pi * n * x / Lx)[:, None]
        f[:, 0] = f[:, 0].real
        return f
    return make(), make()


def _inner(a, b, w, Nx):
    return np.sum(np.real(np.conj(np.asarray(a)) * np.asarray(b)) * w) / Nx


def _norm(a, w, Nx):
    return np.sqrt(np.sum(np.abs(np.asarray(a)) ** 2 * w) / Nx)


def test_bracket_advective_antisymmetry_and_conservation():
    """Original Orszag form: {a,b} = −{b,a} exactly; ∫ a {a,b} → 0 at 4th order in Δx."""
    Ny, Nz, Lx, Ly = 24, 2, 10.0, 12.0
    nky = ops.n_ky(Ny)
    ky = jnp.asarray(2 * np.pi * np.arange(nky) / Ly)[None, :, None]
    w = np.where(np.arange(nky) == 0, 1.0, 2.0)[None, :, None]
    errs = []
    for Nx in (16, 32, 64):
        a, b = _smooth_fields(Nx, Ny, Nz, Lx, Ly)
        a, b = jnp.asarray(a), jnp.asarray(b)
        ab = ops.bracket_advective(a, b, ky, Lx / Nx, Ny)
        ba = ops.bracket_advective(b, a, ky, Lx / Nx, Ny)
        assert np.allclose(np.asarray(ab + ba), 0, atol=1e-12)
        errs.append(abs(_inner(a, ab, w, Nx)) / (_norm(a, w, Nx) * _norm(ab, w, Nx)))
    order = np.log(errs[0] / errs[2]) / np.log(4)
    assert errs[2] < 1e-4 and order > 3.5, (errs, order)


def test_bracket_skew_conserves_energy_exactly():
    """Skew form: Σ b {a,b} = 0 to round-off for smooth *and* for grid-scale (random) fields,
    and it agrees with the advective form at 4th order for smooth fields."""
    Ny, Nz, Lx, Ly = 24, 2, 10.0, 12.0
    nky = ops.n_ky(Ny)
    ky = jnp.asarray(2 * np.pi * np.arange(nky) / Ly)[None, :, None]
    w = np.where(np.arange(nky) == 0, 1.0, 2.0)[None, :, None]
    rng = np.random.default_rng(7)
    diffs = []
    for Nx in (16, 32, 64):
        dx = Lx / Nx
        a, b = _smooth_fields(Nx, Ny, Nz, Lx, Ly)
        a, b = jnp.asarray(a), jnp.asarray(b)
        ab = ops.bracket(a, b, ky, dx, Ny)
        assert abs(_inner(b, ab, w, Nx)) / (_norm(b, w, Nx) * _norm(ab, w, Nx)) < 1e-12
        adv = ops.bracket_advective(a, b, ky, dx, Ny)
        diffs.append(_norm(ab - adv, w, Nx) / _norm(adv, w, Nx))
        # grid-scale (white-noise) fields: the advective form injects energy, the skew form does not
        na = rng.normal(size=(Nx, nky, Nz)) + 1j * rng.normal(size=(Nx, nky, Nz))
        nb = rng.normal(size=(Nx, nky, Nz)) + 1j * rng.normal(size=(Nx, nky, Nz))
        na[:, 0], nb[:, 0] = na[:, 0].real, nb[:, 0].real
        na, nb = jnp.asarray(na), jnp.asarray(nb)
        skew = ops.bracket(na, nb, ky, dx, Ny)
        assert abs(_inner(nb, skew, w, Nx)) / (_norm(nb, w, Nx) * _norm(skew, w, Nx)) < 1e-12
        adv = ops.bracket_advective(na, nb, ky, dx, Ny)
        assert abs(_inner(nb, adv, w, Nx)) / (_norm(nb, w, Nx) * _norm(adv, w, Nx)) > 1e-3
    order = np.log(diffs[0] / diffs[2]) / np.log(4)
    assert diffs[2] < 1e-3 and order > 3.5, (diffs, order)


@pytest.mark.parametrize("form", ["skew", "advective"])
def test_bracket_value_against_the_analytic_bracket(form):
    """a = A(x) cos k₁y, b = B(x) sin k₁y:  {a, b} = a_x b_y − a_y b_x = (k₁/2)(AB)' + (k₁/2)(A'B − AB') cos 2k₁y.
    The k_y = 0 part is the zonal (Reynolds-stress-like) drive, so its *magnitude* matters, not only the
    symmetry and conservation checked above; nothing may leak into other k_y."""
    Nx, Lx, Ny, Ly, Nz = 256, 40.0, 24, 2 * np.pi * 10.0, 2
    x, dx = np.arange(Nx) * Lx / Nx, Lx / Nx
    ky = ops.ky_grid(Ny, Ly); k1 = ky[1]; p = 2 * np.pi * 3 / Lx
    A, Ap = np.sin(p * x), p * np.cos(p * x)
    B, Bp = 1.5 + np.cos(2 * p * x), -2 * p * np.sin(2 * p * x)
    a = np.zeros((Nx, len(ky), Nz), complex); b = np.zeros_like(a)
    a[:, 1, :] = (A / 2)[:, None]                      # A cos k₁y
    b[:, 1, :] = (B / 2j)[:, None]                     # B sin k₁y
    y = np.arange(Ny) * Ly / Ny
    assert np.allclose(np.asarray(ops.to_real_y(jnp.asarray(a), Ny))[:, :, 0], A[:, None] * np.cos(k1 * y)[None, :], atol=1e-13)
    exact0, exact2 = 0.5 * k1 * (Ap * B + A * Bp), 0.25 * k1 * (Ap * B - A * Bp)
    out = np.asarray(ops.BRACKETS[form](jnp.asarray(a), jnp.asarray(b), jnp.asarray(ky)[None, :, None], dx, Ny))[:, :, 0]
    assert abs(np.vdot(exact0, out[:, 0].real) / np.vdot(exact0, exact0) - 1) < 5e-4      # 4th-order stencil: (3p·Δx)⁴/30 ≈ 1e-4
    assert abs(np.vdot(exact2, out[:, 2].real) / np.vdot(exact2, exact2) - 1) < 5e-4
    assert np.abs(out[:, 0] - exact0).max() < 5e-4 * np.abs(exact0).max()
    assert np.abs(out[:, 2] - exact2).max() < 5e-4 * np.abs(exact2).max()
    assert np.abs(np.delete(out, [0, 2], axis=1)).max() < 1e-13


def _rand_fields(Nx=32, nky=6, Nz=2, seed=0):
    r = np.random.default_rng(seed)
    f = lambda: r.normal(size=(Nx, nky, Nz)) + 1j * r.normal(size=(Nx, nky, Nz))
    a, b = f(), f(); a[:, 0] = a[:, 0].real; b[:, 0] = b[:, 0].real
    return jnp.asarray(a), jnp.asarray(b)


def test_gql_bracket_keeps_the_zonal_drive_and_drops_eddy_eddy():
    """a = A cos k₁y, b = B sin k₁y (both 'high' for cutoff 0): the quasilinear bracket keeps the analytic k_y = 0 part
    (k₁/2)(AB)' and drops the 2k₁ part; with every mode 'low' it is the full bracket."""
    Nx, Lx, Ny, Ly, Nz = 256, 40.0, 24, 2 * np.pi * 10.0, 2
    x, dx = np.arange(Nx) * Lx / Nx, Lx / Nx
    ky = ops.ky_grid(Ny, Ly); k1 = ky[1]; p = 2 * np.pi * 3 / Lx
    A, Ap, B, Bp = np.sin(p * x), p * np.cos(p * x), 1.5 + np.cos(2 * p * x), -2 * p * np.sin(2 * p * x)
    a = np.zeros((Nx, len(ky), Nz), complex); b = np.zeros_like(a)
    a[:, 1, :] = (A / 2)[:, None]; b[:, 1, :] = (B / 2j)[:, None]
    kyj = jnp.asarray(ky)[None, :, None]
    ql = np.asarray(ops.gql_bracket("skew", jnp.asarray(a), jnp.asarray(b), kyj, dx, Ny, 0))[:, :, 0]
    exact0 = 0.5 * k1 * (Ap * B + A * Bp)
    assert np.abs(ql[:, 0] - exact0).max() < 5e-4 * np.abs(exact0).max()
    assert np.abs(ql[:, 1:]).max() < 1e-13
    full = np.asarray(ops.bracket(jnp.asarray(a), jnp.asarray(b), kyj, dx, Ny))
    assert np.allclose(np.asarray(ops.gql_bracket("skew", jnp.asarray(a), jnp.asarray(b), kyj, dx, Ny, len(ky) - 1)), full, atol=1e-14)


@pytest.mark.parametrize("cutoff", [0, 2])
def test_gql_bracket_conserves_the_quadratic_invariant(cutoff):
    """Σ_xy b {a, b}_GQL = 0 to round-off for white noise, as for the full skew bracket."""
    Nx, Ny, nky = 32, 16, 6
    a, b = _rand_fields(Nx, nky)
    ky = jnp.asarray(2 * np.pi * np.arange(nky) / 20.0)[None, :, None]
    out = np.asarray(ops.gql_bracket("skew", a, b, ky, 0.3, Ny, cutoff))
    w = np.where(np.arange(nky) == 0, 1.0, 2.0)[None, :, None]
    inner = np.sum(w * np.real(np.conj(np.asarray(b)) * out))
    assert abs(inner) < 1e-12 * np.sum(w * np.abs(np.asarray(b)) * np.abs(out))
    assert np.abs(out).max() > 0
