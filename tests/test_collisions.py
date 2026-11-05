import numpy as np
from velcro.collisions import collision_matrix


def _moments(vec, L, M):
    v = vec.reshape(L, M)
    return v[0, 0], v[1, 0], v[2, 0] / np.sqrt(2) - v[0, 1]


def test_dougherty_conserves_density_momentum_energy():
    L, M = 8, 5
    C = collision_matrix("dougherty", L, M)
    rng = np.random.default_rng(0)
    g = rng.normal(size=L * M)
    assert np.allclose(_moments(C @ g, L, M), 0.0, atol=1e-12)
    # negative semi-definite on the non-conserved subspace
    w = np.linalg.eigvals(C)
    assert np.max(w.real) < 1e-12


def test_lenard_bernstein_conserves_density_only():
    L, M = 6, 4
    C = collision_matrix("lenard-bernstein", L, M)
    g = np.ones(L * M)
    n, u, T = _moments(C @ g, L, M)
    assert n == 0 and u != 0
