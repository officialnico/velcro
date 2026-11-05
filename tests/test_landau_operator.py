"""The linearised Landau operator in the Hermite–Laguerre basis (velcro/landau.py).  KE_VALIDATION E9-0.

Truths used (all closed forms or classical constants, none from this code):
* conservation of particles, parallel momentum and energy by the like-species operator to rounding; between species
  the Landau symmetry: species b's momentum/energy loss equals species a's gain for every perturbation of either
  (exact for moderate mass ratios; the e–i field blocks only at (6, 3), the documented limit);
* self-adjointness in the F₀-weighted inner product (symmetric matrix) and the H-theorem (negative semi-definite, exactly
  three null directions: n, u∥, E) — pitch-angle scattering alone has two (n, E);
* the Maxwellian potentials' derivatives against finite differences of their closed forms; the Legendre expansion of
  the Rosenbluth potentials against a direct 3-D quadrature for an anisotropic source (to the direct quadrature's
  1e-3, set by the 1/|v − v′| singularity);
* the Lorentz limit: C_T,ei → (1/2v³) ∂_ξ(1 − ξ²)∂_ξ as m_e/m_i → 0, whose (1,0) element is −4/(3√π) in closed form;
* Spitzer–Härm (1953) / Epperlein–Haines (1986), Z = 1: σ/σ_Lorentz = 0.58192, σ = 1.97580 n e²τ_e/m_e,
  σ_Lorentz = (32/3π) n e²τ_e/m_e, κ_e = 3.20295 n T τ_e/m_e at zero current; the ions κ_i = 3.9502081 n T τ_i/m_i
  (Braginskii's two-polynomial 3.906; converged with the exact operator, tier-2 review); with τ_e = (3√π/4)/ν̂_ee and
  τ_i = (3√π/2√2)/ν̂_ii;
* GKX's tabulated drift-kinetic Coulomb matrices (Frei, Ball, Hoffmann, Jorge, Ricci & Stenger 2021, at B = 0; the
  package's own data, if ``gkx`` is importable) agree up to their Laguerre sign convention and the collision-frequency
  normalisation (fitted, reported);
* Gate 0: ``collision_matrix("dougherty")`` is untouched.
"""
import numpy as np
import pytest

from velcro import landau as la
from velcro.collisions import collision_matrix

MR = 2.7e-4                         # m_e/m_i of the GX decks
TAU_E = 3 * np.sqrt(np.pi) / 4      # ν̂_ee τ_e (Braginskii's τ_e)
TAU_I = 3 * np.sqrt(np.pi) / (2 * np.sqrt(2))


def _e(L, M, l, m):
    return np.eye(L * M)[l * M + m]


def _solve_constrained(C, drive, constraints):
    """f with C f = −drive on the complement of ``constraints`` (the exact invariants, rows)."""
    n = C.shape[0]
    Q, _ = np.linalg.qr(np.asarray(constraints).T)
    P = np.eye(n) - Q @ Q.T
    A = P @ C @ P + (np.eye(n) - P)
    return P @ np.linalg.solve(A, P @ (-drive))


# ------------------------------------------------------------------ building blocks
def test_maxwellian_potential_derivatives_are_closed_forms():
    v = np.linspace(0.3, 6.0, 40)
    for alpha in (1.0, 0.3, 7.0):
        H, Hp, Gp, Gpp = la.maxwellian_potentials(v, alpha)
        from scipy.special import erf
        G = lambda x: (x + 1 / (2 * alpha ** 2 * x)) * erf(alpha * x) + np.exp(-(alpha * x) ** 2) / (alpha * np.sqrt(np.pi))
        Hf = lambda x: erf(alpha * x) / x
        h = 1e-5
        assert np.allclose(Hp, (Hf(v + h) - Hf(v - h)) / (2 * h), rtol=1e-7, atol=1e-9)
        assert np.allclose(Gp, (G(v + h) - G(v - h)) / (2 * h), rtol=1e-7, atol=1e-9)
        h2 = 1e-4                                                     # the second difference: round-off 4ε/h² ≈ 4e-8
        assert np.allclose(Gpp, (G(v + h2) - 2 * G(v) + G(v - h2)) / h2 ** 2, rtol=1e-5, atol=1e-6)
        assert np.allclose(H, Hf(v))


def test_source_potentials_match_direct_3d_quadrature():
    """Φ = vG′ − G + H of the source ψ₁L₀F₀ (∝ v∥ e^{−v²}: the l = 1 harmonic) and of ψ₀L₁F₀ (l = 0, 2) against a
    direct Gauss quadrature of ∫f(v′)/|v − v′| and ∫f(v′)|v − v′| at a few points."""
    from scipy.special import roots_legendre
    L, M = 3, 2
    v = np.array([0.4, 1.1, 2.3])
    Phi = la.source_potentials(L, M, 1.0, v, 1.0)                     # (L, M, lmax+1, nv)
    xq, wq = roots_legendre(160)
    vp = 4.5 * (xq + 1)                                              # v′ ∈ (0, 9)
    wvp = 4.5 * wq
    xi_q, wxi = roots_legendre(80)
    ph_q = np.pi * (xq + 1)
    wph = np.pi * wq
    VP, XIQ, PH = np.meshgrid(vp, xi_q, ph_q, indexing="ij")
    W3 = (wvp[:, None, None] * VP ** 2) * wxi[None, :, None] * wph[None, None, :]
    for (l0, m0) in ((1, 0), (0, 1)):
        from velcro import basis
        f = basis.psi(l0, VP * XIQ) * basis.laguerre(m0, VP ** 2 * (1 - XIQ ** 2)) * np.exp(-VP ** 2) / np.pi ** 1.5
        for i, vi in enumerate(v):
            for xi0 in (0.3, -0.8):                                   # the field point on the z-axis tilted: (v, ξ0)
                s0 = np.sqrt(1 - xi0 ** 2)
                # |v − v′|: v = v(s0, 0, ξ0), v′ = v′(√(1−ξ′²)cosφ′, √(1−ξ′²)sinφ′, ξ′)
                d = np.sqrt(vi ** 2 + VP ** 2 - 2 * vi * VP * (s0 * np.sqrt(1 - XIQ ** 2) * np.cos(PH) + xi0 * XIQ))
                H = np.sum(W3 * f / d)
                G = np.sum(W3 * f * d)
                h = 1e-4
                dp = np.sqrt((vi + h) ** 2 + VP ** 2 - 2 * (vi + h) * VP * (s0 * np.sqrt(1 - XIQ ** 2) * np.cos(PH) + xi0 * XIQ))
                dm = np.sqrt((vi - h) ** 2 + VP ** 2 - 2 * (vi - h) * VP * (s0 * np.sqrt(1 - XIQ ** 2) * np.cos(PH) + xi0 * XIQ))
                Gp = (np.sum(W3 * f * dp) - np.sum(W3 * f * dm)) / (2 * h)
                want = vi * Gp - G + H
                from scipy.special import eval_legendre
                got = sum(Phi[l0, m0, l, i] * eval_legendre(l, xi0) for l in range(Phi.shape[2]))
                # the direct quadrature integrates the 1/|v − v′| singularity: its own accuracy is ~1e-3
                assert abs(got - want) < 1e-3 * max(abs(want), 1e-2), (l0, m0, vi, xi0, got, want)


# ------------------------------------------------------------------ the operator's properties
@pytest.mark.parametrize("L,M,sym", [(4, 2, 1e-12), (8, 4, 1e-12), (12, 6, 1e-12), (16, 8, 1e-12), (20, 10, 1e-12)])
def test_like_species_conserves_is_self_adjoint_and_dissipative(L, M, sym):
    """The production matrix (exact quadrature, velcro/landau_exact.py) is symmetric, conserving and dissipative to
    rounding at every (L, M) — (20, 10) included, where the Rosenbluth route had a positive eigenvalue; beyond (24, 12) the
    construction is refused (time, not accuracy)."""
    C = la.like_species_matrix(L, M)
    mom = la.moment_functionals(L, M)
    assert np.abs(mom @ C).max() < 1e-11 * np.abs(C).max()           # n, u∥, E conserved to rounding
    assert np.abs(C - C.T).max() < sym * np.abs(C).max()             # self-adjoint
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    assert w.max() < 1e-11 and np.sum(np.abs(w) < 1e-8) == 3         # H-theorem; null space n, u∥, E
    with pytest.raises(ValueError, match="built for"):
        la.like_species_matrix(26, 13)
    # the test part alone conserves particles only; pitch-angle scattering (Lorentz) conserves n and E
    CT = la.test_particle_matrix(L, M)
    assert np.abs(mom[0] @ CT).max() < 1e-12 and np.abs(mom[1] @ CT).max() > 1e-3
    Lor = la.lorentz_matrix(L, M)
    assert np.abs(mom[[0, 2]] @ Lor).max() < 1e-12 * np.abs(Lor).max() and np.abs(mom[1] @ Lor).max() > 1e-3


@pytest.mark.parametrize("L,M,pair", [(8, 4, (1.0, 1.0, 4.0, 2.0)), (12, 6, (1.0, 1.0, 4.0, 2.0)), (8, 4, (1.0, 1.0, 1.0, 3.0))])
def test_unlike_species_exchange_momentum_and_energy_symmetrically(L, M, pair):
    """Landau symmetry: with n_a = n_b the momentum species a loses to b's Maxwellian (C_T,ab) is what b's Maxwellian
    gains from a's perturbation (C_F,ba): T_b (C_T,ab)_u + T_a (C_F,ba)_u = 0, and for the energy
    (C_T,ab)_E/√(m_a T_a) + (C_F,ba)_E/√(m_b T_b) = 0 (dP_a/dt ∝ n_a m_a v_ta ν̂_ab ∝ 1/T_a, dE_a/dt ∝ 1/√(m_a T_a));
    the field part conserves a's particles.  Exact to rounding for a mass ratio 4 at T_b = 2 T_a up to (12, 6) and for
    T_b = 3 T_a at equal mass; for the e–i pair (m_e/m_i = 2.7e-4: sources spread over v ≫ v_ti or concentrated at
    v ≪ v_te) the monomial route holds only at low order, the documented limit of the cross field blocks, which the
    solver does not use (the electrons' e–i block is the test part: the Lorentz-limit test)."""
    ma, Ta, mb, Tb = pair
    mom = la.moment_functionals(L, M)
    CT_ab, CF_ab = la.pair_matrices(L, M, ma, Ta, mb, Tb)
    CT_ba, CF_ba = la.pair_matrices(L, M, mb, Tb, ma, Ta)
    u = Tb * (mom[1] @ CT_ab) + Ta * (mom[1] @ CF_ba)
    assert np.abs(u).max() < 1e-9 * np.abs(mom[1] @ CT_ab).max(), (ma, Ta, mb, Tb)
    en = (mom[2] @ CT_ab) / np.sqrt(ma * Ta) + (mom[2] @ CF_ba) / np.sqrt(mb * Tb)
    assert np.abs(en).max() < 1e-9 * np.abs((mom[2] @ CT_ab) / np.sqrt(ma * Ta)).max(), (ma, Ta, mb, Tb)
    assert np.abs(mom[0] @ CF_ab).max() < 1e-12 and np.abs(mom[0] @ CT_ab).max() < 1e-12


def test_ei_test_part_is_the_lorentz_operator_at_small_mass_ratio():
    L, M = 8, 4
    Lor = la.lorentz_matrix(L, M)
    assert Lor[M, M] == pytest.approx(-4 / (3 * np.sqrt(np.pi)), rel=1e-10)      # ⟨ψ₁, (1/2v³)∂_ξ(1−ξ²)∂_ξ ψ₁F₀⟩
    for mr, tol in ((MR, 2e-3), (MR / 10, 2e-4)):
        CT = la.test_particle_matrix(L, M, float(np.sqrt(1 / mr)), mr)
        assert np.abs(CT - Lor).max() / np.abs(Lor).max() < tol


def test_spitzer_harm_and_braginskii_coefficients():
    """Electrons with e–e (full Landau) and e–i (test part, m_e/m_i = 2.7e-4) at Z = 1, ions with i–i: the classical
    transport coefficients from the constrained steady responses, at (16, 8)."""
    L, M = 16, 8
    Cee = la.like_species_matrix(L, M)
    CTei = la.test_particle_matrix(L, M, float(np.sqrt(1 / MR)), MR)
    Lor = la.lorentz_matrix(L, M)
    mom = la.moment_functionals(L, M)
    e10 = _e(L, M, 1, 0)
    dE = np.sqrt(2.0) * e10                                            # drive 2v̂∥F₀ per unit eE/(m_e v_te)
    j = lambda f: f @ e10 / np.sqrt(2.0)                                # ∫v̂∥ f
    s_lor = j(_solve_constrained(Lor, dE, mom[[0, 2]]))
    s_full = j(_solve_constrained(Cee + CTei, dE, mom[[0, 2]]))
    # σ = n e² τ_e/m_e × coef with τ_e = TAU_E/ν̂_ee: coef = (∫v̂∥ f per unit drive)/τ_e; the Lorentz closed form fixes it
    assert s_lor / TAU_E == pytest.approx(32 / (3 * np.pi), rel=2e-5)            # σ_Lorentz = (32/3π) n e²τ_e/m_e (5e-6 at (16, 8))
    # Spitzer & Härm (1953) / Epperlein & Haines (1986), Z = 1, converged with the exact operator on the l = 1 Sonine basis
    # to K = 22 (tier-2 review): σ/σ_L = 0.58192 (the oft-quoted 0.5816 is 5.5e-4 low), σ = 1.97580 n e²τ_e/m_e (EH:
    # 1/0.5061 = 1.97589); Velcro's (16, 8) truncation −5e-4 (Lorentz e–i), −8.5e-4 with the m_e/m_i test part
    assert s_full / s_lor == pytest.approx(0.58192, rel=1.5e-3)
    assert s_full / TAU_E == pytest.approx(1.97580, rel=1.5e-3)
    # thermal conductivity at zero current: drive v̂∥(v̂² − 5/2)F₀ (∇T), f = f_T + c f_E with ∫v̂∥ f = 0
    v1 = e10 / np.sqrt(2)
    v3 = (np.sqrt(48) * _e(L, M, 3, 0) + 12 * e10 / np.sqrt(2)) / 8          # v̂∥³ = (H₃ + 12 v̂∥)/8
    vmu = (e10 - _e(L, M, 1, 1)) / np.sqrt(2)                                   # v̂∥ μ̂, μ̂ = 1 − L₁
    dT = v3 + vmu - 2.5 * v1
    Ce = Cee + CTei
    fT, fE = _solve_constrained(Ce, dT, mom[[0, 2]]), _solve_constrained(Ce, dE, mom[[0, 2]])
    f = fT - (v1 @ fT) / (v1 @ fE) * fE
    assert (f @ dT) * 2 / TAU_E == pytest.approx(3.20295, rel=3e-3)             # Epperlein–Haines 3.2030 (Braginskii's two-
    #                                                                              polynomial 3.1616); (16, 8) truncation −1.3e-3
    fi = _solve_constrained(Cee, dT, mom)
    # the ions: Braginskii's 3.906 is his two-polynomial value (125/32 = 3.90625 exactly); the converged coefficient of the
    # exact Landau operator is 3.9502081 (8 digits by K = 22), and (16, 8) reproduces it to 1e-8
    assert (fi @ dT) * 2 / TAU_I == pytest.approx(3.9502081, rel=1e-6)


def test_gkx_coulomb_tables_agree_up_to_their_conventions():
    """GKX's drift-kinetic Coulomb (test + field) matrix at B = 0 (its finite-wavelength table, p ≤ 3, j ≤ 1 and p ≤ 5,
    j ≤ 2) against this operator at the same (L, M): equal after the signed-Laguerre convention (D C D, D = (−1)^m)
    and one overall collision-frequency scale (fitted by least squares, reported)."""
    gkx = pytest.importorskip("gkx")
    import json
    from importlib import resources
    root = resources.files("gkx").joinpath("data")
    for stem, L, M in (("finite_wavelength_coulomb", 4, 2), ("finite_wavelength_coulomb_18", 6, 3)):
        meta = json.loads(root.joinpath(stem + ".json").read_text())
        with root.joinpath(stem + ".npz").open("rb") as fh:
            tab = np.load(fh)
            T, F = np.asarray(tab["test_matrix"])[0], np.asarray(tab["field_matrix"])[0]     # B = 0
        assert meta["maximum_hermite_order"] == L - 1 and meta["maximum_laguerre_order"] == M - 1
        C = la.like_species_matrix(L, M)
        D = np.diag([(-1.0) ** m for l in range(L) for m in range(M)])
        best = None
        for sign in (np.eye(L * M), D):
            G = sign @ (T + F) @ sign
            scale = float(np.sum(G * C) / np.sum(C * C))
            res = np.abs(scale * C - G).max() / np.abs(G).max()
            best = (res, scale) if best is None or res < best[0] else best
        print(f"GKX {stem}: residual {best[0]:.2e} at scale {best[1]:.6f}")
        assert best[0] < 1e-6, best
        # the scale is exactly 2: GKX / Frei, Ernst & Ricci 2022 measure collision frequencies in ν̂/2 (their hard-coded
        # six-moment Coulomb coefficients are 2.000000 × these elements, tier-2 review job 2005)
        assert best[1] == pytest.approx(2.0, rel=1e-6)


def test_gate0_dougherty_untouched():
    """The Dougherty matrix is what it was: −ν(ℓ + 2m) on the diagonal, +1 at (1,0) (momentum restoring), and the energy
    restoring 2·(2/3) c_i c_j on the (2,0)/(0,1) block with c = (1/√2, −1)."""
    L, M = 6, 3
    C = collision_matrix("dougherty", L, M)
    l = np.arange(L)[:, None] * np.ones((1, M)); m = np.ones((L, 1)) * np.arange(M)[None, :]
    ref = np.diag(-(l + 2 * m).reshape(-1))
    ref[M, M] += 1.0
    idx = [(2 * M, 1 / np.sqrt(2.0)), (1, -1.0)]
    for i, ci in idx:
        for j, cj in idx:
            ref[i, j] += 2.0 * (2.0 / 3.0) * ci * cj
    assert np.array_equal(C, ref)
    assert collision_matrix("landau", L, M).shape == (L * M, L * M) and np.allclose(collision_matrix("landau", L, M), la.like_species_matrix(L, M))
