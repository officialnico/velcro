"""Classical transport coefficients from the Landau operator, pushed to convergence (tier-2 attack on velcro/landau.py).

tests/test_landau_operator.py checks σ/σ_L = 0.5816 (rel 5e-3), σ = 1.975 (5e-3), κ_e = 3.203 (1e-2), κ_i = 3.906 (2e-2)
at (16, 8) only.  Those tolerances can hide an O(1 %) normalisation or field-part error, and κ_i = 3.906 = 125/32 is
Braginskii's low-order Sonine value, not a converged one.  Here:

1. the *exact* operator's coefficients (tests/attack/landau_ref.py on the l = 1 Sonine basis v_z L^{(3/2)}_k(v²), k ≤ K,
   the like-species Landau operator by exact quadrature, the e–i operator the Lorentz limit in closed form): the K
   sequence shows the converged value of each coefficient for *this* operator, to compare with the literature
   (Spitzer & Härm 1953 γ_E(Z=1) = 0.5816; Epperlein & Haines 1986 α₀(Z=1) = 0.5061 → σ = 1.9759 n e²τ_e/m_e,
   κ₀ = 3.2030; Braginskii 1965 α₀ = 0.5129, κ_e 3.1616, κ_i 3.906);
2. velcro's own matrices at (L, M) = (4, 2) … (16, 8) (Lorentz e–i and the m_e/m_i = 2.7e-4 test part), the truncation
   error against the converged values;
3. the reference operator in velcro's basis at the same (L, M) (to separate operator error from basis truncation).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import landau_ref as ref  # noqa: E402

from velcro import landau as la  # noqa: E402

MR = 2.7e-4
LIT = dict(sigma_ratio_SH=0.5816, sigma_EH=1 / 0.5061, kappa_e_EH=3.2030, sigma_Brag=1 / 0.5129, kappa_e_Brag=3.1616,
           kappa_i_Brag=3.906, sigma_L=32 / (3 * np.pi))
_CONV = {}


def _converged(K=22):
    if K not in _CONV:
        t0 = time.time()
        Cs = ref.sonine_like(K)
        Ls = ref.sonine_lorentz(K)
        _CONV[K] = (Cs, Ls, time.time() - t0)
    return _CONV[K]


def test_exact_operator_converged_coefficients():
    """The Sonine sequence K = 1 … 22 for the exact operator: σ/τ_e, σ/σ_L, κ_e (zero current), κ_i."""
    K = 22
    Cs, Ls, dt = _converged(K)
    print(f"\nexact like-species operator on the l = 1 Sonine basis, K = {K}: {dt:.1f} s")
    asym = np.abs(Cs - Cs.T).max() / np.abs(Cs).max()
    print(f"  symmetric to {asym:.1e}; C[0,0] (momentum) {Cs[0, 0]:.2e}, C[0,1:] max {np.abs(Cs[0, 1:]).max():.2e}")
    print("     K   sigma/tau_e    sigma/sigma_L    kappa_e        kappa_i")
    rows = {}
    for k in list(range(1, 13)) + [14, 16, 18, 20, 22]:
        o = ref.sonine_transport(Cs, Ls, k)
        rows[k] = o
        print(f"  {k:4d}  {o['sigma']:.7f}    {o['sigma_ratio']:.7f}       {o['kappa_e']:.7f}    {o['kappa_i']:.7f}")
    o = rows[K]
    print(f"  σ_L/τ_e = {o['sigma_L']:.8f} (32/3π = {LIT['sigma_L']:.8f})")
    print(f"  literature: SH σ/σ_L 0.5816 → σ {0.5816 * LIT['sigma_L']:.5f}; EH σ {LIT['sigma_EH']:.5f} (σ/σ_L "
          f"{LIT['sigma_EH'] / LIT['sigma_L']:.5f}), κ_e 3.2030; Braginskii σ {LIT['sigma_Brag']:.4f}, κ_e 3.1616, κ_i 3.906")
    print(f"  converged − literature: σ/σ_L − SH {o['sigma_ratio'] - 0.5816:+.5f}; σ − EH {o['sigma'] - LIT['sigma_EH']:+.5f}; "
          f"κ_e − EH {o['kappa_e'] - 3.2030:+.5f}; κ_i − Braginskii {o['kappa_i'] - 3.906:+.5f} ({o['kappa_i'] / 3.906 - 1:+.3%})")
    assert asym < 1e-12
    assert abs(o["sigma_L"] / LIT["sigma_L"] - 1) < 1e-6
    for key in ("sigma", "kappa_e", "kappa_i"):
        assert abs(rows[20][key] / rows[22][key] - 1) < 5e-5, (key, rows[20][key], rows[22][key])   # converged (κ_e: 2.2e-5)
    assert abs(o["sigma"] / LIT["sigma_EH"] - 1) < 2e-3
    assert abs(o["kappa_e"] / LIT["kappa_e_EH"] - 1) < 2e-3


def test_velcro_coefficients_and_their_truncation_error():
    """velcro's matrices at (L, M): with the Lorentz e–i operator (the classical problem) and with velcro's e–i test part
    at m_e/m_i = 2.7e-4 (what tests/test_landau_operator.py uses); and the reference operator in the same basis."""
    Cs, Ls, _ = _converged(22)
    conv = ref.sonine_transport(Cs, Ls, 22)
    print(f"\nconverged: σ {conv['sigma']:.6f}, σ/σ_L {conv['sigma_ratio']:.6f}, κ_e {conv['kappa_e']:.6f}, κ_i {conv['kappa_i']:.6f}")
    print("  (L,M)   source            sigma      sigma/sigma_L   kappa_e     kappa_i    | rel. err. vs converged (σ, κ_e, κ_i)")
    for (L, M) in ((4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (14, 7), (16, 8)):
        t0 = time.time()
        Cee = la.like_species_matrix(L, M)
        Lor = la.lorentz_matrix(L, M)
        CTei = la.test_particle_matrix(L, M, float(np.sqrt(1 / MR)), MR)
        Rfull = ref.like_reference_full(L, M)
        Rlor = ref.lorentz_reference(ref.hl_eval(L, M), L * M, (L - 1) + 2 * (M - 1))
        for name, (A, B) in (("velcro Lorentz", (Cee, Lor)), ("velcro e-i MR", (Cee, CTei)), ("reference Lor.", (Rfull, Rlor))):
            o = ref.hl_transport(A, B, L, M)
            err = [o[k] / conv[k] - 1 for k in ("sigma", "kappa_e", "kappa_i")]
            print(f"  ({L:2d},{M})  {name:16s}  {o['sigma']:.6f}   {o['sigma_ratio']:.6f}      {o['kappa_e']:.6f}    "
                  f"{o['kappa_i']:.6f}   | {err[0]:+.2e} {err[1]:+.2e} {err[2]:+.2e}")
        print(f"          [{time.time() - t0:.1f} s]")


def test_braginskii_two_polynomial_values_from_velcro_matrices():
    """Braginskii's (1965) coefficients are the K = 2 Sonine values of the exact operator (the reference's K = 2 row: σ
    1.9498597 → α₀ = 0.51286 (Braginskii 0.5129), κ_e 3.1616302 (3.1616), κ_i 125/32 = 3.90625 (3.906)).  velcro's own
    matrices projected on v_z L^{(3/2)}_k(v²), k ≤ 2 (exactly representable at (6, 3)), must give them to rounding: a sharp
    absolute test of the l = 1 block that tests/test_landau_operator.py's 2 % tolerance on 3.906 cannot be."""
    from velcro import basis
    L, M, K = 6, 3, 2
    P = np.zeros((L * M, K + 1))
    for k in range(K + 1):
        f = lambda v, mu, k=k: v * ref.lag_table(K + 1, v ** 2 + mu, 1.5)[0][k]
        P[:, k] = basis.project(f, L, M, nv=40, nmu=40).reshape(-1)
    Cee = P.T @ la.like_species_matrix(L, M) @ P
    Lor = P.T @ la.lorentz_matrix(L, M) @ P
    got = ref.sonine_transport(Cee, Lor, K)
    want = ref.sonine_transport(ref.sonine_like(K), ref.sonine_lorentz(K), K)
    print(f"\nvelcro (6,3) on the K = 2 Sonine space: σ {got['sigma']:.10f} (α₀ {1 / got['sigma']:.6f}), κ_e {got['kappa_e']:.10f}, "
          f"κ_i {got['kappa_i']:.10f}; exact operator: σ {want['sigma']:.10f}, κ_e {want['kappa_e']:.10f}, κ_i {want['kappa_i']:.10f}; "
          f"Braginskii: α₀ 0.5129, κ_e 3.1616, κ_i 3.906 (= 125/32 = 3.90625)")
    for key in ("sigma", "kappa_e", "kappa_i"):
        assert abs(got[key] / want[key] - 1) < 1e-11, (key, got[key], want[key])
    assert abs(want["kappa_i"] - 125 / 32) < 1e-11 and abs(1 / want["sigma"] - 0.5129) < 5e-5 and abs(want["kappa_e"] - 3.1616) < 5e-5
