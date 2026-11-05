import numpy as np
import pytest

from velcro import geometry as G


def test_fourier_tokamak_matches_s_alpha_at_large_aspect_ratio():
    r0, q0, shat = 0.02, 1.4, 0.8
    qs = lambda s: q0 * (1 + shat * (np.sqrt(s) / 0.5 - 1))
    g = G.s_alpha(Nx=8, Lx=4, Ny=1, Nz=16, rho_star=1 / 2000, r0=r0, q0=q0, shat=shat)
    eq = G.circular_tokamak_fourier(R0=1.0, a=r0 / 0.5, B0=1.0, q_of_s=qs)
    gf = eq.build(Nx=8, Lx=4, Ny=1, Nz=16, s0=0.25, n0=1, rho_star=1 / 2000)
    i = 4
    for k, tol in [("B", 0.05), ("gpar", 0.05), ("gxx", 0.05), ("gxy", 0.06), ("gyy", 0.06),
                   ("Kx", 0.06), ("Ky", 0.06), ("Cinv", 0.05), ("twist_phase", 0.01)]:
        a, b = np.asarray(getattr(g, k)), np.asarray(getattr(gf, k))
        a = a[i] if a.ndim >= 1 and a.shape[0] == 8 else a
        b = b[i] if b.ndim >= 1 and b.shape[0] == 8 else b
        rel = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)
        assert rel < tol, (k, rel)


def test_twist_phase_consistent_with_shear_metric():
    """d(twist phase)/dx per unit k_y must equal 2π × (the coefficient of z in g^{xy})."""
    g = G.s_alpha(Nx=32, Lx=40, Ny=1, Nz=8, rho_star=1 / 180, r0=0.18, q0=1.4, shat=0.8)
    dphase = np.gradient(g.twist_phase, g.x)
    z = g.z
    # g^{xy} = s(x) z  → s(x) = g^{xy}(z=z1)/z1
    k = np.argmax(np.abs(z))
    s_local = g.gxy[:, 0, k] / z[k]
    assert np.allclose(dphase, 2 * np.pi * s_local, rtol=1e-6)


def test_s_alpha_reference_values():
    g = G.s_alpha(Nx=8, Lx=10, Ny=1, Nz=8, rho_star=1 / 180, r0=0.18, q0=1.4, shat=0.8)
    i = 4  # x = 0
    assert np.isclose(g.B[i, 0, np.argmin(np.abs(g.z))], 1 / 1.18)
    assert np.isclose(g.gpar[i, 0, np.argmin(np.abs(g.z))], -1 / (1.4 * 1.18))
    assert np.isclose(g.Cinv[i, 0, 0], 1.0)
    assert np.isclose(g.Ky[i, 0, np.argmin(np.abs(g.z))], -1 / 1.18)
