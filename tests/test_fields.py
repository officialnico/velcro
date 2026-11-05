import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from velcro import basis, geometry
from velcro.fields import build_field_solver
from velcro.operators import ky_grid, n_ky


def _slab_solver(Nx=32, Lx=20.0, Ny=12, Nz=4, M=4, tau=1.0):
    g = geometry.slab(Nx, Lx, Ny, Ly=2 * np.pi * 5, Nz=Nz)
    ky = ky_grid(Ny, g.Ly)
    Ti = np.ones(Nx); Te = tau * np.ones(Nx)
    return g, ky, build_field_solver(g, ky, Ti, Te, M, Lx)


def test_gyroaverage_matches_fourier_symbol():
    """In a uniform slab the real-space operator must reproduce 𝒥_m(b) e^{ikx x} exactly."""
    g, ky, fs = _slab_solver()
    Nx = len(g.x); Lx = 20.0
    for n in (0, 1, 3):
        kx = 2 * np.pi * n / Lx
        phi = np.exp(1j * kx * g.x)[:, None, None] * np.ones((1, len(ky), len(g.z)))
        Phi = np.asarray(fs.gyroaverage(jnp.asarray(phi)))
        b = 0.5 * (kx**2 + ky**2)
        J = basis.gyro_coefficients(b, fs.M)  # (M, Nky)
        expect = J[:, None, :, None] * phi[None]
        assert np.allclose(Phi, expect, atol=1e-10)


def test_field_solve_manufactured_solution():
    """Choose φ, build the density Σ𝒥_m G_0m that quasi-neutrality implies, solve back."""
    g, ky, fs = _slab_solver(tau=2.0)
    Nx, Nky, Nz = len(g.x), len(ky), len(g.z)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(Nx, Nky, Nz)) + 1j * rng.normal(size=(Nx, Nky, Nz))
    phi[:, 0] = phi[:, 0].real
    phi[:, 0] -= phi[:, 0].mean()                                  # gauge: zero mean for k_y=0
    # Build the LHS density: (1-Γ0)φ/T_i + (φ - <φ>_FS δ_ky0)/T_e   in spectral x for slab
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=20.0 / Nx)
    phik = np.fft.fft(phi, axis=0)
    b = 0.5 * (kx[:, None] ** 2 + ky[None, :] ** 2)
    G0b = basis.gamma0(b)[:, :, None]
    rhs_k = (1 - G0b) * phik / 1.0 + phik / 2.0
    # flux-surface average of the ky=0 mode (uniform weights in slab)
    fsa = phi[:, 0, :].mean(axis=1, keepdims=True)
    rhs = np.fft.ifft(rhs_k, axis=0)
    rhs[:, 0, :] -= fsa / 2.0
    # Now find G_0m so that Σ_m 𝒥_m G_0m = rhs: simplest choice G_00 = rhs / 𝒥_0 with others 0
    J0 = basis.gyro_coefficients(b, 1)[0][:, :, None]
    G00 = np.fft.ifft(np.fft.fft(rhs, axis=0) / J0, axis=0)
    G0 = np.zeros((fs.M, Nx, Nky, Nz), dtype=complex)
    G0[0] = G00
    phi_num = np.asarray(fs.solve(jnp.asarray(G0)))
    assert np.allclose(phi_num, phi, atol=1e-9)


def test_polarisation_operator_symmetric_and_positive():
    g, ky, fs = _slab_solver()
    A = np.linalg.inv(np.asarray(fs.Ainv))
    for k in range(len(ky)):
        Ak = A[k, 0]
        assert np.allclose(Ak, Ak.conj().T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(Ak)) > 0
