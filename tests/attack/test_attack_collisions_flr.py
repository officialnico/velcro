"""The gyrokinetic FLR collision terms against independent references (the validation notes, rules F7–F9):
the exact gyroaverage of the like-species Landau operator (velcro.landau_exact.like_exact_gk, every Bessel order), GKX's
finite-wavelength Coulomb tables of Frei et al. 2021 (tests/data/gkx), and a linear ITG growth rate.  Run with -s: every
test prints its numbers."""
import dataclasses
import functools
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from velcro import basis
from velcro import collisions_flr as cf
from velcro.landau import like_species_matrix
from velcro.landau_exact import like_exact_gk, pair_exact

GKX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gkx")
B_GKX = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _rel(A, R):
    A, R = np.asarray(A), np.asarray(R)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(R)), "non-finite values"   # job 2132: max(0.0, nan) = 0.0
    return float(np.abs(A - R).max() / np.abs(R).max())


# ------------------------------------------------------------------ F7: the exact reference
def test_exact_gk_reference_properties():
    """b = 0 is the exact k⊥ = 0 matrix; symmetric; negative semi-definite; converged in its t / cos θ_u rules (the Bessel
    factors make them inexact: 40 vs 60 extra nodes); the density element is O(b²)."""
    L, M = 6, 3
    C0 = like_species_matrix(L, M)
    d0 = _rel(like_exact_gk(L, M, 0.0), C0)
    print(f"\nb 0: |C_gk(0) − C(0)| {d0:.1e}")
    assert d0 < 1e-13
    for b in (0.1, 2.0, 8.0):
        A = like_exact_gk(L, M, b)
        B = like_exact_gk(L, M, b, extra_t=60, extra_c=60)
        conv = _rel(A, B)
        sym = float(np.abs(A - A.T).max() / np.abs(A).max())
        emax = float(np.linalg.eigvalsh(A).max()) / np.abs(A).max()
        print(f"b {b}: convergence 40 vs 60 extra nodes {conv:.1e}; max eigenvalue/max|C| {emax:+.2e}; C_00 {A[0, 0]:+.6e}")
        assert conv < 1e-11 and sym < 1e-13 and emax < 1e-12
    c1, c2 = like_exact_gk(L, M, 1e-3)[0, 0], like_exact_gk(L, M, 2e-3)[0, 0]
    print(f"C_00(1e-3) {c1:+.4e} (/b² {c1 / 1e-6:+.5f}), C_00(2e-3)/C_00(1e-3) {c2 / c1:.4f}")
    assert abs(c2 / c1 - 4.0) < 0.02


@pytest.mark.parametrize("L,M", [(4, 2), (6, 3)])
def test_exact_reference_test_part_is_affine_in_b(L, M):
    """The phase-free products of the exact 6-D form (the test part) equal C_T(0) − b K with K from the 2-D speed-function
    route (velcro.collisions_flr.classical_diffusion_matrix): the derivation's claim that the gyroaveraged test part is
    exactly affine in b, by two independent quadratures."""
    CT, _ = pair_exact(L, M, 1.0, 1.0, 1.0)
    K = cf.classical_diffusion_matrix(L, M)
    for b in (0.0, 0.05, 0.5, 2.0, 8.0):
        T = like_exact_gk(L, M, b, part="test")
        err = _rel(T, np.asarray(CT) - b * K)
        print(f"\n({L},{M}) b {b}: |T_6D − (C_T(0) − b K)| / max {err:.1e}")
        assert err < 1e-11


# ------------------------------------------------------------------ F8: GKX's tables at finite b
def _gkx(stem):
    with open(os.path.join(GKX_DIR, stem + ".json")) as f:
        meta = json.load(f)
    t = np.load(os.path.join(GKX_DIR, stem + ".npz"))
    return meta, np.asarray(t["test_matrix"]) + np.asarray(t["field_matrix"])


def test_exact_gk_against_gkx_tables():
    """GKX's like-species finite-wavelength Coulomb tables (Frei et al. 2021 eqs. 3.47–3.50, 60 digits; (P, J) = (3, 1) and
    (5, 2)) against the exact gyroaverage at b = B²/2, after GKX's conventions (ν̂/2: tables = 2 × Velcro; signed Laguerre
    D = (−1)^m).  The exact Galerkin elements do not depend on the truncation; GKX's two tables disagree with each other
    at finite B (its kernel sums are truncated with the table), so the rule is relative to that sensitivity: on the
    (p ≤ 1) block, |Velcro − GKX(5,2)| ≤ |GKX(3,1) − GKX(5,2)| + 1e-8 at every B, and ≤ 1e-6 of max|C| at B = 0.125."""
    m8, C8 = _gkx("finite_wavelength_coulomb")
    m18, C18 = _gkx("finite_wavelength_coulomb_18")
    assert m18["bessel_argument_grid"] == list(B_GKX) and m18["maximum_hermite_order"] == 5
    L, M = 6, 3
    D = np.diag([(-1.0) ** m for l in range(L) for m in range(M)])
    blk = [(p, j) for p in range(2) for j in range(2)]
    i18 = [p * 3 + j for p, j in blk]
    i8 = [p * 2 + j for p, j in blk]
    i18_31 = [p * 3 + j for p in range(4) for j in range(2)]
    bad = []
    for k, B in enumerate(B_GKX):
        b = 0.5 * B * B
        V = 2.0 * D @ like_exact_gk(L, M, b) @ D
        Vm = 2.0 * D @ cf.landau_gk_model(L, M, b) @ D
        assert np.all(np.isfinite(V)) and np.all(np.isfinite(Vm)), B
        scale = np.abs(C18[k]).max()
        dV = np.abs(V[np.ix_(i18, i18)] - C18[k][np.ix_(i18, i18)]).max() / scale
        dM = np.abs(Vm[np.ix_(i18, i18)] - C18[k][np.ix_(i18, i18)]).max() / scale
        dG = np.abs(C8[k][np.ix_(i8, i8)] - C18[k][np.ix_(i18, i18)]).max() / scale
        d31 = np.abs(V[np.ix_(i18_31, i18_31)] - C18[k][np.ix_(i18_31, i18_31)]).max() / scale
        dfull = np.abs(V - C18[k]).max() / scale
        print(f"\nB {B:5.3f} (b {b:.4f}): p ≤ 1 block — |exact − GKX(5,2)| {dV:.2e}, |model − GKX(5,2)| {dM:.2e}, "
              f"GKX's own |(3,1) − (5,2)| {dG:.2e}; (p ≤ 3, j ≤ 1) block {d31:.2e}; whole (5,2) table {dfull:.2e}")
        if not dV <= dG + 1e-8:
            bad.append(B)
    _, C18b = _gkx("finite_wavelength_coulomb_18")
    k = 1
    small = np.abs(2.0 * D @ like_exact_gk(L, M, 0.5 * B_GKX[k] ** 2) @ D - C18b[k])[np.ix_(i18, i18)].max() / np.abs(C18b[k]).max()
    assert small < 1e-6, small
    assert not bad, f"exact vs GKX(5,2) beyond GKX's own truncation sensitivity at B = {bad}"


# ------------------------------------------------------------------ F9: the Landau model against the exact gyroaverage
@pytest.mark.parametrize("L,M", [(6, 3), (8, 4)])
def test_landau_flr_model_against_the_exact_gyroaverage(L, M):
    """The model (exact test part, n = 0 gyroaverage of the field part, J₁ perpendicular-momentum restoring) against the
    exact gyroaverage: the FLR change ΔC = C(b) − C(0) of the model vs the exact one, r(b) = ‖ΔC_model − ΔC_exact‖ /
    ‖ΔC_exact‖ (max norm) — the share of the FLR change the model misses (registered for the first model, F9: ≤ 0.35 at
    every b — failed; for the sector form, G3: r/b ≤ 2 at b ≤ 0.05 and r ≤ 0.35 at every b); also the field part's own
    share, the density element's b² coefficient and the least-damped eigenvalues (G3: within 10 % at b ≤ 0.2)."""
    C0 = like_species_matrix(L, M)
    K = cf.classical_diffusion_matrix(L, M)
    worst, small, eig_bad = 0.0, 0.0, 0.0
    ev = lambda A: np.sort(np.linalg.eigvalsh(0.5 * (A + A.T)))[::-1][1:3]           # the two least damped after n
    for b in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        E = like_exact_gk(L, M, b)
        Mo = cf.landau_gk_model(L, M, b)
        assert np.all(np.isfinite(E)) and np.all(np.isfinite(Mo)), b
        dE, dM = E - C0, Mo - C0
        r = float(np.abs(dM - dE).max() / np.abs(dE).max())
        fE = dE + b * K                                                     # the exact field part's FLR change
        rf = float(np.abs(dM + b * K - fE).max() / np.abs(fE).max())
        worst = max(worst, r)
        if b <= 0.05:
            small = max(small, r / b)
        eE, eM = ev(E), ev(Mo)
        if b <= 0.2:
            eig_bad = max(eig_bad, float(np.abs(eM / eE - 1).max()))
        print(f"\n({L},{M}) b {b}: r {r:.3f} (of the field part's FLR change {rf:.3f}); |ΔC_exact| {np.abs(dE).max():.3e}, "
              f"|b K| {b * np.abs(K).max():.3e}; C_00 exact {E[0, 0]:+.4e} model {Mo[0, 0]:+.4e}; least damped exact {eE} "
              f"model {eM}")
    print(f"max r {worst:.3f}; max r/b at b ≤ 0.05 {small:.3f}; least-damped eigenvalues at b ≤ 0.2 within {eig_bad:.1%}")
    assert worst <= 0.35 and small <= 2.0 and eig_bad <= 0.10, (worst, small, eig_bad)


# ------------------------------------------------------------------ the ITG growth rate with FLR collisions (slow)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# G5 = F10's judged run (job 2146, code 2a727eb, COLL_FLR.md §2.9), reproduced to every printed digit by job 2446 on
# d5e9bf0: γ at k_y index 2 and 3 for (ν, collision_flr)
_G5 = {"dougherty": {(0.0, False): (0.17304, 0.26369), (0.05, False): (0.15705, 0.24198),
                     (0.05, True): (0.15377, 0.23423), (0.2, False): (0.14102, 0.21200), (0.2, True): (0.12276, 0.18167)},
       "landau": {(0.0, False): (0.17304, 0.26369), (0.05, False): (0.17023, 0.26043),
                  (0.05, True): (0.16926, 0.25758), (0.2, False): (0.16409, 0.25439), (0.2, True): (0.16034, 0.24346)}}


@functools.lru_cache(maxsize=None)
def _itg_flr_rates(model):
    """{(ν, collision_flr, k_y index): (the most unstable eigenvalue, k_y)} of F10's set-up, once per model (the two
    tests below share it); prints the table."""
    from test_attack_landau_paths import _acfg, _adiabatic_maps, _block
    from velcro.config import VelcroConfig
    from velcro.simulation import Simulation
    res = {}
    for nu in (0.0, 0.05, 0.2):
        for flr in ((False,) if nu == 0 else (False, True)):
            d = _acfg("tube", L=6, M=3, Nx=8, Ny=10, Nz=12, nu=nu, model=model).model_dump()
            d["physics"]["collision_flr"] = flr
            sim = Simulation(VelcroConfig.model_validate(d))
            full, _, _, _ = _adiabatic_maps(sim)
            for k in (2, 3):
                ev = np.linalg.eigvals(_block(sim, k, full))
                res[(nu, flr, k)] = (ev[np.argmax(ev.real)], float(np.asarray(sim.ky)[k]))
    for k in (2, 3):
        g0 = res[(0.0, False, k)][0].real
        print(f"\n{model} k_y {res[(0.0, False, k)][1]:.3f}: γ(ν = 0) {g0:+.5f}")
        for nu in (0.05, 0.2):
            goff, gon = res[(nu, False, k)][0], res[(nu, True, k)][0]
            dflr, dcol = gon.real - goff.real, goff.real - g0
            print(f"  ν {nu}: γ k⊥=0 {goff.real:+.5f} (ω {goff.imag:+.5f}), γ FLR {gon.real:+.5f} (ω {gon.imag:+.5f}); "
                  f"collisional shift {dcol:+.5f}, FLR change {dflr:+.5f} ({dflr / dcol:+.1%} of it)")
    return res


@pytest.mark.slow
@pytest.mark.parametrize("model", ["dougherty", "landau"])
def test_itg_growth_rate_with_flr_collisions(model):
    """A CBC-like ITG eigenmode (s-α tube, adiabatic electrons, R/L_T 6.9, R/L_n 2.2, (6, 3), collisions on h): the
    growth rate without collisions, with the k⊥ = 0 operator and with the FLR terms, ν = 0.05 and 0.2, k_y indices 2, 3.
    Registered (COLL_FLR.md §2.7 F10): (a) the FLR terms lower γ at every (ν, k_y); (b) the FLR change is smaller than
    half the k⊥ = 0 collisional shift; (c) it is larger at the larger k_y.  G5 (§2.9): (a) and (c) held, (b) failed (the
    FLR change is 21–117 % of the shift).  Holds (a), (c) and G5's growth rates to 2e-5; (b) is the strict xfail below."""
    res = _itg_flr_rates(model)
    for nu in (0.05, 0.2):
        d2, d3 = (res[(nu, True, k)][0].real - res[(nu, False, k)][0].real for k in (2, 3))
        print(f"  ν {nu}: FLR change at k_y index 2 {d2:+.5f}, index 3 {d3:+.5f}")
        assert d2 < 0 and d3 < 0, (nu, d2, d3)                                                       # (a)
        assert abs(d3) > abs(d2), (nu, d2, d3)                                                       # (c)
    for (nu, flr), g in _G5[model].items():
        for k, gk in zip((2, 3), g):
            assert abs(res[(nu, flr, k)][0].real - gk) < 2e-5, (nu, flr, k, res[(nu, flr, k)][0].real, gk)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="registered F10 clause (b) falsified by G5 (COLL_FLR.md §2.9): the FLR change of "
                                       "γ is 21–117 % of the k⊥ = 0 collisional shift, predicted < 50 %")
@pytest.mark.parametrize("model", ["dougherty", "landau"])
def test_itg_flr_change_is_below_half_the_collisional_shift(model):
    """F10 clause (b) as registered: |FLR change| < ½ |k⊥ = 0 collisional shift| at every (ν, k_y)."""
    res = _itg_flr_rates(model)
    for k in (2, 3):
        g0 = res[(0.0, False, k)][0].real
        for nu in (0.05, 0.2):
            dflr = res[(nu, True, k)][0].real - res[(nu, False, k)][0].real
            dcol = res[(nu, False, k)][0].real - g0
            assert abs(dflr) < 0.5 * abs(dcol), (k, nu, dflr, dcol)
