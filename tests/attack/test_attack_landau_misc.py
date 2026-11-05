"""Smaller attacks on velcro/landau.py: GKX's table normalisation read element by element, the lru_cache aliasing of
pair_matrices, small truncations (M = 1, L ≤ 3), the e–i test part at T_e ≠ T_i against the reference, and the e–i block's
dissipativity at T_e ≠ T_i.  Run with -s."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import landau_ref as ref  # noqa: E402

from velcro import landau as la  # noqa: E402

MR = 2.7e-4


def test_gkx_table_normalisation_element_by_element():
    """tests/test_landau_operator.py fits one overall scale between GKX's tables and this operator (2.000000 in job 1770).
    GKX's docs (operators page) list three literature normalisations ν = C n q⁴ lnΛ/(m^½ T^{3/2}): Sugama C = 4.4429
    (= √2π: Helander & Sigmar's ν̂, this code's unit), Frei et al. 2021 C = 2.3633 (= 4√π/3), Frei–Hoffmann–Ricci 2022
    C = 0.7523; the tables cite Frei et al. 2021.  In Frei's unit the like-species test-part friction element T(1,0)(1,0) is
    −(4/3√π)(1/√2)·(4.4429/2.3633) = −1 exactly; in ν̂ units −0.5319.  Print the table's elements."""
    gkx = pytest.importorskip("gkx")
    import json
    from importlib import resources
    root = resources.files("gkx").joinpath("data")
    stem, L, M = "finite_wavelength_coulomb", 4, 2
    meta = json.loads(root.joinpath(stem + ".json").read_text())
    with root.joinpath(stem + ".npz").open("rb") as fh:
        tab = np.load(fh)
        T, F = np.asarray(tab["test_matrix"])[0], np.asarray(tab["field_matrix"])[0]
    CT, CF = la.test_particle_matrix(L, M), la.field_particle_matrix(L, M)
    i = 1 * M + 0
    print(f"\ngkx {getattr(gkx, '__version__', '?')}, {meta.get('source')}: T[(1,0),(1,0)] {T[i, i]:.12f}, F {F[i, i]:.12f}; "
          f"velcro (ν̂ units) C_T {CT[i, i]:.12f}, C_F {CF[i, i]:.12f}; ratio T {T[i, i] / CT[i, i]:.9f}, F {F[i, i] / CF[i, i]:.9f}")
    print(f"  expected ratio with Frei 2021's ν (C = 4√π/3): {np.sqrt(2) * np.pi / (4 * np.sqrt(np.pi) / 3):.9f}; with ν̂/2: 2")
    for name, A, B in (("T", T, CT), ("F", F, CF)):
        D = np.diag([(-1.0) ** m for l in range(L) for m in range(M)])
        best = min(((np.abs(s * B - G).max() / np.abs(G).max(), s) for G in (A, D @ A @ D)
                    for s in [float(np.sum(G * B) / np.sum(B * B))]))
        print(f"  {name}: best fitted scale {best[1]:.9f}, residual {best[0]:.2e}")


def test_pair_matrices_hands_out_the_cached_arrays():
    """pair_matrices / like_species_matrix return lru_cache'd arrays: a caller writing one in place would poison every
    later call.  Since the fix they are read-only (the write raises) and a second call sees the original values."""
    A, F = la.pair_matrices(4, 2, 1.0, 1.0, 4.0, 2.0)
    a00 = float(A[0, 0])
    for X in (A, F, la.like_species_matrix(4, 2), la.test_particle_matrix(4, 2), la.field_particle_matrix(4, 2)):
        with pytest.raises(ValueError):
            X[0, 0] += 1.0
    A2, _ = la.pair_matrices(4, 2, 1.0, 1.0, 4.0, 2.0)
    assert float(A2[0, 0]) == a00


@pytest.mark.parametrize("L,M", [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (8, 1), (16, 1), (1, 8)])
def test_small_truncations(L, M):
    """Small (L, M), in particular M = 1 (the drift-kinetic-in-μ runs some tests use): no crash; conservation of the moments
    the truncated space can represent; dissipativity; against the reference."""
    C = la.like_species_matrix(L, M)
    R = ref.like_reference_full(L, M)
    mom = la.moment_functionals(L, M)
    cons = np.abs(mom @ C).max(axis=1)
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    print(f"\n({L},{M}): |C − ref| {np.abs(C - R).max():.1e}; |n,u,E · C| {cons}; max eig {w.max():+.2e}, min {w.min():+.4f}")
    assert np.abs(C - R).max() < 1e-12 and w.max() < 1e-12


@pytest.mark.parametrize("Te", [0.5, 2.0, 3.0])
def test_ei_test_part_at_unequal_temperatures(Te):
    """C_T,ei at T_e/T_i ≠ 1 (α = √(T_e m_i/(T_i m_e)) = 43–105) against the reference at (12, 6); and its spectrum: with
    T_e ≠ T_i the test part is not self-adjoint — any eigenvalue with a positive real part is a spurious collisional
    instability of the electrons."""
    L, M = 12, 6
    RT, _ = ref.pair_reference(L, M, MR, Te, 1.0, 1.0)
    VT, _ = la.pair_matrices(L, M, MR, Te, 1.0, 1.0)
    err = np.abs(VT - RT).max() / np.abs(RT).max()
    ev = np.linalg.eigvals(VT)
    Cee = la.like_species_matrix(L, M)
    ev2 = np.linalg.eigvals(Cee + VT)
    sym = np.linalg.eigvalsh(0.5 * (VT + VT.T))
    print(f"\nT_e/T_i {Te}: |ΔC_T,ei| {err:.1e}; max Re eig C_T,ei {ev.real.max():+.3e}, of C_ee + C_T,ei {ev2.real.max():+.3e}; "
          f"max eig of the symmetric part {sym.max():+.3e} (energy-exchange row: {VT[2 * M, 0]:+.3e} (2,0)←(0,0))")
    assert err < 1e-9


def test_gkx_closed_form_six_moment_coulomb_against_this_operator():
    """GKX (src/gkx/operators/linear/dissipation.py, drift_kinetic_coulomb_six_moment_contribution, 'Frei, Ernst & Ricci
    (2022), equations (C9a)–(C9f)') hard-codes the drift-kinetic Coulomb operator on the thermal block T = {(ℓ=2,m=0),
    (0,1)} and the heat-flux block Q = {(3,0),(1,1)} (its state is [Laguerre, Hermite]; signed Laguerre convention):
    T: (−16√2/15, 16/15, −8√2/15)/√π, Q: (−8√2/5, 8/(5√3), −28√2/15)/√π, times its ν.  The ratio of each element to this
    operator's (ν̂ units) is GKX's ν̂/ν_GKX if both are the same operator; one ratio for all six elements is the check.
    GKX's docs: Frei et al. 2021 and FER 2022 use ν = (4√π/3) n q⁴lnΛ/(m^½T^{3/2}) → ratio √2π/(4√π/3) = 1.8800;
    ν̂/2 (the factor fitted in job 1770) → 2."""
    L, M = 4, 2
    C = la.like_species_matrix(L, M)
    s2, s3, sp = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(np.pi)
    sgn = lambda m1, m2: (-1.0) ** (m1 + m2)                         # signed-Laguerre convention
    blocks = {"T": ([(2, 0), (0, 1)], (-16 * s2 / 15, 16 / 15, -8 * s2 / 15)),
              "Q": ([(3, 0), (1, 1)], (-8 * s2 / 5, 8 / (5 * s3), -28 * s2 / 15))}
    ratios = []
    for name, ((a, b), (c00, c01, c11)) in blocks.items():
        ia, ib = a[0] * M + a[1], b[0] * M + b[1]
        g = {(ia, ia): c00 / sp, (ia, ib): c01 / sp, (ib, ia): c01 / sp, (ib, ib): c11 / sp}
        for (i, j), gv in g.items():
            v = C[i, j] * sgn(divmod(i, M)[1], divmod(j, M)[1])
            ratios.append(gv / v)
            print(f"  {name} ({divmod(i, M)})←({divmod(j, M)}): GKX/FER {gv:+.10f}, velcro (signed) {v:+.10f}, ratio {gv / v:.8f}")
    r = np.array(ratios)
    print(f"\nGKX six-moment Coulomb / velcro: ratios {np.round(r, 8)}; spread {r.max() - r.min():.2e}; "
          f"Frei's ν predicts {np.sqrt(2) * np.pi / (4 * np.sqrt(np.pi) / 3):.6f}, ν̂/2 predicts 2")
    assert r.max() - r.min() < 1e-9 * abs(r.mean()), "GKX's closed-form Coulomb block is not a multiple of this operator"
