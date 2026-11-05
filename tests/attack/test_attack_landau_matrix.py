"""Adversarial checks of velcro/landau.py's matrix elements against an independent reference (tests/attack/landau_ref.py:
the symmetrised Landau form with exact polynomial quadrature in centre-of-mass/relative velocities — no Rosenbluth
potentials, no monomials, no radial grid).  Run with -s: every test prints its numbers.

What velcro/landau.py claims (module docstring): the like-species matrix symmetric to 1e-14 up to (12, 6), to 1e-5 at
(16, 8), wrong at (20, 10) because of the monomial expansion of the field part; the electron←ion blocks (test and field)
accurate; the ion←electron field block wrong beyond degree 4.  Here each block is compared element by element with the
reference, and the error at (16, 8)/(20, 10) is split between its two possible sources (the v grid nv = 240, vmax = 9
of the outer quadrature, and the monomial field part) by re-running velcro's own functions with a larger grid.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import landau_ref as ref  # noqa: E402

from velcro import landau as la  # noqa: E402

MR = 2.7e-4


def _err(A, R):
    """(max |A − R| / max |R|, the worst element's (row, col))."""
    d = np.abs(A - R)
    i = np.unravel_index(np.argmax(d), d.shape)
    return d.max() / np.abs(R).max(), i


def _lm(i, M):
    return (int(i) // M, int(i) % M)


def test_reference_rules_are_exact_and_the_reference_conserves():
    """The reference against itself and closed forms: more quadrature nodes change nothing (the rules are exact), C_T + C_F
    is symmetric and conserves n, u∥, E, the like-species test-part (1,0)(1,0) element is −(4/3√π)/√2 (Hirshman–Sigmar
    M⁰⁰ = −(1 + m_a/m_b)(1 + x²)^{−3/2} at x = 1, times 4/(3√π) = 1/(ν̂τ)), the Lorentz (1,0)(1,0) element −4/(3√π), and
    the one-product form −¼⟨ΔUΔ⟩ equals C_T + C_F."""
    L, M = 6, 3
    t0 = time.time()
    CT, CF = ref.like_reference(L, M, extra=0)
    CT2, CF2 = ref.like_reference(L, M, extra=5)
    C = CT + CF
    mom = la.moment_functionals(L, M)
    e_rules = max(_err(CT, CT2)[0], _err(CF, CF2)[0])
    asym = np.abs(C - C.T).max() / np.abs(C).max()
    cons = np.abs(mom @ C).max() / np.abs(C).max()
    full = ref.like_reference_full(L, M)
    Lor = ref.lorentz_reference(ref.hl_eval(L, M), L * M, (L - 1) + 2 * (M - 1))
    hs = -4 / (3 * np.sqrt(np.pi)) / np.sqrt(2.0)
    print(f"\nreference (6,3): rules exact to {e_rules:.1e}; C symmetric to {asym:.1e}; n,u,E conserved to {cons:.1e}; "
          f"one-product form {_err(full, C)[0]:.1e}; CT[10,10] = {CT[M, M]:.15f} (closed form {hs:.15f}); "
          f"Lorentz[10,10] = {Lor[M, M]:.15f} (−4/3√π = {-4 / (3 * np.sqrt(np.pi)):.15f}); {time.time() - t0:.1f} s")
    assert e_rules < 1e-13 and asym < 1e-13 and cons < 1e-13 and _err(full, C)[0] < 1e-13
    assert abs(CT[M, M] - hs) < 1e-13 and abs(Lor[M, M] + 4 / (3 * np.sqrt(np.pi))) < 1e-13
    # Hirshman–Sigmar M⁰⁰ for unlike species (test part, any mass and temperature ratio): −(4/3√π)(1 + m_a/m_b)/(1 + x²)^{3/2},
    # x = v_tb/v_ta — the reference's test part against the closed form
    for (ma, Ta, mb, Tb) in ((1.0, 1.0, 4.0, 2.0), (1.0, 1.0, 1.0, 3.0), (MR, 1.0, 1.0, 1.0), (1.0, 1.0, MR, 1.0)):
        CTab, _ = ref.pair_reference(4, 2, ma, Ta, mb, Tb)
        x2 = (Tb / mb) / (Ta / ma)
        want = -4 / (3 * np.sqrt(np.pi)) * (1 + ma / mb) / (1 + x2) ** 1.5
        print(f"  HS M00 pair {(ma, Ta, mb, Tb)}: reference {CTab[2, 2]:.12e}, closed form {want:.12e}")
        assert abs(CTab[2, 2] - want) < 1e-11 * abs(want)


@pytest.mark.parametrize("L,M", [(4, 2), (8, 4), (12, 6), (16, 8)])
def test_like_species_blocks_against_the_reference(L, M):
    """C_T, C_F and C = C_T + C_F of velcro/landau.py against the reference, element by element (relative to max|C|),
    and the low-order block (ℓ ≤ 3, m ≤ 1: flow, heat flux, anisotropy) separately."""
    t0 = time.time()
    RT, RF = ref.like_reference(L, M)
    t_ref = time.time() - t0
    VT, VF = la.test_particle_matrix(L, M), la.field_particle_matrix(L, M)
    R, V = RT + RF, VT + VF
    scale = np.abs(R).max()
    eT = np.abs(VT - RT).max() / scale
    eF = np.abs(VF - RF).max() / scale
    eC, iC = _err(V, R)
    low = [l * M + m for l in range(min(L, 4)) for m in range(min(M, 2))]
    eLow = np.abs(V[np.ix_(low, low)] - R[np.ix_(low, low)]).max() / scale
    asymV = np.abs(V - V.T).max() / scale
    print(f"\n({L},{M}) reference {t_ref:.1f} s, max|C| {scale:.3f}: |ΔC_T| {eT:.2e}, |ΔC_F| {eF:.2e}, |ΔC| {eC:.2e} at "
          f"{_lm(iC[0], M)}←{_lm(iC[1], M)} (velcro {V[iC]:.10f}, reference {R[iC]:.10f}); low block {eLow:.2e}; "
          f"velcro asymmetry {asymV:.2e}")
    ev = np.linalg.eigvalsh(0.5 * (R + R.T))
    print(f"  reference spectrum: min {ev.min():.6f} (spectral radius), 3 null directions: {np.sum(np.abs(ev) < 1e-10) == 3}")
    # the Rosenbluth route (now the cross-check) holds 1e-10 to (12, 6) and its documented 8.8e-6 at (16, 8); the
    # production matrix (the exact quadrature) equals the reference at every (L, M)
    tol = 1e-10 if L <= 12 else 2e-5
    assert eC < tol, (eC, _lm(iC[0], M), _lm(iC[1], M))
    P = la.like_species_matrix(L, M)
    assert np.abs(P - R).max() / scale < 1e-12


def test_error_budget_at_16_8_and_20_10():
    """The two candidate error sources at the edge of the range: velcro's own functions with the default v grid
    (nv 240, vmax 9) and with larger grids (nv 400, vmax 12; nv 600, vmax 14), test and field parts separately.  If the
    error falls with the grid, the cause is the outer quadrature's truncation at v = 9 — not the monomial expansion."""
    for (L, M) in ((16, 8), (20, 10)):
        t0 = time.time()
        RT, RF = ref.like_reference(L, M)
        scale = np.abs(RT + RF).max()
        print(f"\n({L},{M}) reference in {time.time() - t0:.1f} s, max|C| {scale:.3f}")
        for nv, vmax in ((240, 9.0), (400, 12.0), (600, 14.0)):
            VT = la.test_particle_matrix(L, M, 1.0, 1.0, nv, vmax)
            VF = la.field_particle_matrix(L, M, 1.0, 1.0, nv, vmax)
            V = VT + VF
            eT = np.abs(VT - RT).max() / scale
            eF = np.abs(VF - RF).max() / scale
            asym = np.abs(V - V.T).max() / scale
            w = np.linalg.eigvalsh(0.5 * (V + V.T))
            print(f"  nv {nv} vmax {vmax}: |ΔC_T| {eT:.2e}  |ΔC_F| {eF:.2e}  |ΔC| {np.abs(V - RT - RF).max() / scale:.2e}  "
                  f"asymmetry {asym:.2e}  max eigenvalue {w.max():.2e}")


@pytest.mark.parametrize("L,M,pair", [(8, 4, (1.0, 1.0, 4.0, 2.0)), (8, 4, (1.0, 1.0, 1.0, 3.0)),
                                      (4, 2, (MR, 1.0, 1.0, 1.0)), (6, 3, (MR, 1.0, 1.0, 1.0)), (8, 4, (MR, 1.0, 1.0, 1.0)),
                                      (12, 6, (MR, 1.0, 1.0, 1.0)),
                                      (16, 8, (MR, 1.0, 1.0, 1.0)),
                                      (4, 2, (1.0, 1.0, MR, 1.0)), (6, 3, (1.0, 1.0, MR, 1.0)), (8, 4, (1.0, 1.0, MR, 1.0))])
def test_cross_blocks_against_the_reference(L, M, pair):
    """C_T,ab and C_F,ab of pair_matrices (internals called directly where _check_range would refuse nothing here) against
    the reference: mass ratio 4 at T_b = 2T_a, T_b = 3T_a at equal mass, electrons←ions (m_e/m_i = 2.7e-4; the module
    says both blocks are accurate; kinetic.py uses only C_T,ei, calling C_F,ei unreliable) and ions←electrons (the module
    says C_F,ie is wrong beyond degree 4)."""
    ma, Ta, mb, Tb = pair
    t0 = time.time()
    RT, RF = ref.pair_reference(L, M, ma, Ta, mb, Tb)
    VT, VF = la.pair_matrices(L, M, ma, Ta, mb, Tb)
    eT, iT = _err(VT, RT)
    eF, iF = _err(VF, RF)
    print(f"\n{pair} ({L},{M}) [{time.time() - t0:.1f} s]: C_T rel err {eT:.2e} at {_lm(iT[0], M)}←{_lm(iT[1], M)} "
          f"(max|C_T| {np.abs(RT).max():.4e}); C_F rel err {eF:.2e} at {_lm(iF[0], M)}←{_lm(iF[1], M)} "
          f"(max|C_F| {np.abs(RF).max():.4e}; velcro {VF[iF]:.6e}, reference {RF[iF]:.6e})")
    # momentum/energy exchange identity on the reference itself (Landau symmetry), the reference's own sanity
    mom = la.moment_functionals(L, M)
    RT2, RF2 = ref.pair_reference(L, M, mb, Tb, ma, Ta)
    u = Tb * (mom[1] @ RT) + Ta * (mom[1] @ RF2)
    print(f"  reference Landau symmetry (momentum): {np.abs(u).max() / np.abs(mom[1] @ RT).max():.1e}")
    assert eT < 1e-8, ("C_T", eT)
    assert eF < 1e-8, ("C_F", eF)
