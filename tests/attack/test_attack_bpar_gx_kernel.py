"""Adversarial check of KE_VALIDATION E10-A2: is GX's δB∥ heat-flux kernel really not the projection of the δB∥ weight?

Pure numpy/scipy (no Velcro import), so the evidence can be handed to the GX developers as is.  GX's helpers and kernels
are transcribed verbatim from gx ``src/device_funcs.cu`` at 3865a537 (= origin/gx, what the cards run; unchanged on
origin/next 3a88e7b2, origin/sundials-nvector 90062f86, origin/pk/imex_full, origin/feature/gx-v2):

    127  Jflr(l, b)   = (−b/2)^l / l! · e^{−b/2}   (0 for l < 0 or l > 30; enforce_JL_0 defaults to false)
    141  JflrA(l, b)  = l Jflr(l−1) + 2l Jflr(l) + (l+1) Jflr(l+1)
    146  JflrB(l, b)  = Jflr(l) + Jflr(l−1)
    151  Jfac(l, b)   = 1.5 Jflr(l) + JflrA(l)
   1621  p_bar  += Jfac(l) G_l0 + Jflr(l) G_l2/√2                               (ES heat flux;   also 1656)
   1622  q_bar  += Jfac(l) G_l1 + Jflr(l) (√1.5 G_l3 + G_l1)                    (A∥ heat flux;   also 1689)
   1623  qB_bar += (Jfac(l) + Jfac(l−1)) G_l0 + JflrB(l) G_l2/√2                 (δB∥ heat flux;  also 1722)
   1631  Q = Re[conj(i k_y φ) p_bar − v_t conj(i k_y A∥) q_bar + (T/Z) conj(i k_y δB∥) qB_bar] · 2 flxJac · n T
   1765  n_bar += Jflr G_l0;  u_bar += Jflr G_l1;  uB_bar += JflrB G_l0            (particle fluxes, same prefactors)
   3181  rhs_linear's gradient drive, m = 0:  i k_y φ [Jflr(l−1) l τ' + Jflr(l)(κ' + 2l τ') + Jflr(l+1)(l+1) τ']
                                          + (i k_y/zt) δB∥ [JflrB(l−1) l τ' + JflrB(l)(κ' + 2l τ') + JflrB(l+1)(l+1) τ']
         m = 1: −v_t i k_y A∥ [Jflr(l−1) l τ' + Jflr(l)(κ' + (2l+1)τ') + Jflr(l+1)(l+1)τ'];  m = 2: i k_y (φ Jflr + δB∥ JflrB/zt) τ'/√2;
         m = 3: −v_t i k_y A∥ √1.5 τ' Jflr          (κ' = fprim, τ' = tprim)
   1927  H_l0 = g_l0 + zt Jflr φ + JflrB δB∥,   H_l1 = g_l1 − zt v_t Jflr A∥   (h from the evolved g)

Velocity conventions, re-derived here independently of the internal quadrature check (which uses physicists' Hermite): GX's
v_t = √(T/m), x = v∥/v_t, normalised probabilists' Hermite ĥ_m = He_m(x)/√m! (weight e^{−x²/2}/√(2π): v∥ = v_t ĥ_1 as
in the m = 1 rows above), signed Laguerre ψ_l = (−1)^l L_l(μ̂) with μ̂ = μB/T (weight e^{−μ̂}), E/T = x²/2 + μ̂,
a = k⊥v⊥/Ω = √(2bμ̂).  A moment array G[l, m] (GX's Gh_(·, l, m)) represents h = F_M Σ G_lm ĥ_m(x) ψ_l(μ̂).
The gyrokinetic potential is ⟨χ⟩ = J0 φ − v∥ J0 A∥ + (T/Z) μ̂ (2J1(a)/a) δB∥/B, so the δB∥ heat flux needs the moment
∫ (E/T) · K · h with K = μ̂ 2J1(a)/a.

Run: pytest -s tests/attack/test_attack_bpar_gx_kernel.py (seconds, numbers printed).
"""
import math

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.laguerre import laggauss
from scipy.special import eval_hermitenorm, eval_laguerre, j0, j1, jv

R2, R15 = math.sqrt(2.0), math.sqrt(1.5)
BS = (0.0, 0.01, 0.1, 0.3, 1.0, 2.0, 4.0)


# ------------------------------------------------------------------ GX's helpers, verbatim (device_funcs.cu:127–154)
def Jflr(l, b):
    if l > 30 or l < 0:
        return 0.0
    return (-0.5 * b) ** l / math.factorial(l) * math.exp(-0.5 * b)


def JflrA(l, b):
    return l * Jflr(l - 1, b) + 2 * l * Jflr(l, b) + (l + 1) * Jflr(l + 1, b)


def JflrB(l, b):
    return Jflr(l, b) + Jflr(l - 1, b)


def Jfac(l, b):
    return 1.5 * Jflr(l, b) + JflrA(l, b)


# ------------------------------------------------------------------ GX's moment kernels (one (x, y, z) point)
def gx_p_bar(G, b):
    return sum(Jfac(l, b) * G[l, 0] + Jflr(l, b) * G[l, 2] / R2 for l in range(G.shape[0]))


def gx_q_bar(G, b):
    return sum(Jfac(l, b) * G[l, 1] + Jflr(l, b) * (R15 * G[l, 3] + G[l, 1]) for l in range(G.shape[0]))


def gx_qB_bar(G, b):
    return sum((Jfac(l, b) + Jfac(l - 1, b)) * G[l, 0] + JflrB(l, b) * G[l, 2] / R2 for l in range(G.shape[0]))


def gx_n_bar(G, b):
    return sum(Jflr(l, b) * G[l, 0] for l in range(G.shape[0]))


def gx_u_bar(G, b):
    return sum(Jflr(l, b) * G[l, 1] for l in range(G.shape[0]))


def gx_uB_bar(G, b):
    return sum(JflrB(l, b) * G[l, 0] for l in range(G.shape[0]))


def JfacB(l, b):
    """The energy weight GX's own drive applies to JflrB (rhs_linear, m = 0 row + the 1.5 of v̂²'s ĥ_0 part):
    1.5 JflrB(l) + l JflrB(l−1) + 2l JflrB(l) + (l+1) JflrB(l+1) — Jfac with Jflr → JflrB."""
    return 0.0 if l < 0 else 1.5 * JflrB(l, b) + l * JflrB(l - 1, b) + 2 * l * JflrB(l, b) + (l + 1) * JflrB(l + 1, b)


def fixed_qB_bar(G, b):
    return sum(JfacB(l, b) * G[l, 0] + JflrB(l, b) * G[l, 2] / R2 for l in range(G.shape[0]))


# ------------------------------------------------------------------ quadrature in GX's variables (independent of GX's algebra)
def _grid(nx=40, nmu=150):
    x, wx = hermegauss(nx)
    mu, wmu = laggauss(nmu)
    return x, wx / math.sqrt(2.0 * math.pi), mu, wmu


GRID = _grid()


def h_over_FM(G, x, mu):
    """Σ G_lm ĥ_m(x) ψ_l(μ̂) at the nodes, (nx, nmu)."""
    nl, nm = G.shape
    hm = np.array([eval_hermitenorm(m, x) / math.sqrt(math.factorial(m)) for m in range(nm)])
    pl = np.array([(-1) ** l * eval_laguerre(l, mu) for l in range(nl)])
    return np.einsum("lm,mx,lu->xu", G, hm, pl)


def quad(w, G, b, grid=None):
    """∫ d(x) dμ̂ (Maxwellian weights) w(x, μ̂; b) · h/F_M."""
    x, wx, mu, wmu = grid or GRID
    W = w(x[:, None], mu[None, :], b)
    return np.einsum("x,u,xu,xu->", wx, wmu, W * np.ones((len(x), len(mu))), h_over_FM(G, x, mu))


def J0w(mu, b):
    return j0(np.sqrt(2.0 * b * mu))


def Kw(mu, b):
    """μ̂ 2J1(a)/a, a = √(2bμ̂) (→ μ̂ as a → 0): the δB∥ part of ⟨χ⟩ per (T/Z) δB∥/B."""
    a = np.sqrt(2.0 * b * mu)
    safe = np.where(a > 1e-8, a, 1.0)
    return mu * np.where(a > 1e-8, 2.0 * j1(safe) / safe, 1.0 - a ** 2 / 8.0)


E = lambda x, mu: 0.5 * x ** 2 + mu                       # E/T in GX's variables
W_ES = lambda x, mu, b: E(x, mu) * J0w(mu, b)             # p_bar's integrand (ĥ_0, ĥ_2 parts)
W_APAR = lambda x, mu, b: E(x, mu) * x * J0w(mu, b)        # q_bar's integrand: v∥/v_t = x
W_BPAR = lambda x, mu, b: E(x, mu) * Kw(mu, b)             # the exact δB∥ heat-flux integrand


def W_GXEQ(x, mu, b):
    """The weight GX's qB_bar actually integrates (the E10-A2 claim): (E/T) K − (2/b) μ̂ J2(a) (→ x²μ̂/2 + μ̂²/2 at b → 0)."""
    if b == 0.0:
        return 0.5 * x ** 2 * mu + 0.5 * mu ** 2
    return E(x, mu) * Kw(mu, b) - (2.0 / b) * mu * jv(2, np.sqrt(2.0 * b * mu))


def _random_G(nl, nm, seed, complex_=False):
    r = np.random.default_rng(seed)
    G = r.normal(size=(nl, nm))
    if complex_:
        G = G + 1j * r.normal(size=(nl, nm))
    return G


def _unit(nl, nm, l, m):
    G = np.zeros((nl, nm))
    G[l, m] = 1.0
    return G


# ================================================================== 1. the basis conventions behind GX's helpers
@pytest.mark.parametrize("b", BS + (8.0,))
def test_gx_helpers_are_the_signed_laguerre_coefficients(b):
    """Jflr(l) = ⟨J0 ψ_l⟩, JflrB(l) = ⟨μ̂ 2J1(a)/a ψ_l⟩ (the μ̂ is inside JflrB), JflrA(l) = ⟨(μ̂ − 1) J0 ψ_l⟩,
    Jfac(l) = ⟨(½ + μ̂) J0 ψ_l⟩: GX's helpers are exact signed-Laguerre coefficients with a = √(2bμ̂)."""
    nl, err = 9, 0.0
    for l in range(nl):
        u = _unit(nl, 1, l, 0)
        rows = ((lambda x, mu, bb: J0w(mu, bb), Jflr(l, b)),
                (lambda x, mu, bb: Kw(mu, bb), JflrB(l, b)),
                (lambda x, mu, bb: (mu - 1.0) * J0w(mu, bb), JflrA(l, b)),
                (lambda x, mu, bb: (0.5 + mu) * J0w(mu, bb), Jfac(l, b)))
        for w, ref in rows:
            err = max(err, abs(quad(w, u, b) - ref))
    print(f"\n  b {b:4.2f}: max_l |quadrature − GX helper| = {err:.1e}")
    assert err < 1e-11
    if b > 0:     # the alternative reading 'JflrB = ⟨2J1(a)/a ψ_l⟩' (no μ̂) is false
        no_mu = [quad(lambda x, mu, bb: Kw(mu, bb) / np.maximum(mu, 1e-300), _unit(nl, 1, l, 0), b) for l in range(3)]
        assert max(abs(no_mu[l] - JflrB(l, b)) for l in range(3)) > 1e-3


# ================================================================== 2. control: GX's other flux kernels are exact projections
@pytest.mark.parametrize("b", BS)
def test_control_gx_es_apar_and_particle_kernels_are_exact(b):
    """The same machinery finds GX's ES and A∥ heat-flux kernels and all three particle-flux kernels exact (≤ 1e-11):
    the method is not biased against GX."""
    G = _random_G(8, 4, 11, complex_=True)
    checks = dict(p_bar=(gx_p_bar(G, b), quad(W_ES, G, b)),
                  q_bar=(gx_q_bar(G, b), quad(W_APAR, G, b)),
                  n_bar=(gx_n_bar(G, b), quad(lambda x, mu, bb: J0w(mu, bb), G, b)),
                  u_bar=(gx_u_bar(G, b), quad(lambda x, mu, bb: x * J0w(mu, bb), G, b)),
                  uB_bar=(gx_uB_bar(G, b), quad(lambda x, mu, bb: Kw(mu, bb), G, b)))
    line = "  ".join(f"{k} {abs(a - q):.1e}" for k, (a, q) in checks.items())
    print(f"\n  b {b:4.2f}: |GX kernel − quadrature|: {line}")
    for k, (a, q) in checks.items():
        assert abs(a - q) < 1e-11 * max(1.0, abs(q)), k


# ================================================================== 3. the δB∥ heat-flux kernel
@pytest.mark.parametrize("b", BS)
def test_gx_bpar_heat_kernel_is_not_the_projection_of_E_times_K(b):
    """GX's qB_bar ≠ ∫(E/T)K h; it equals ∫[(E/T)K − (2/b)μ̂J2(a)] h; the difference is Σ (JflrB(l) + JflrB(l−1)) G_l0;
    the fix (JfacB: Jfac with Jflr → JflrB, the weight GX's drive already uses) is the exact projection."""
    G = _random_G(6, 4, 3)                              # the same test distribution as the internal quadrature check
    exact = quad(W_BPAR, G, b)
    gx = gx_qB_bar(G, b)
    gxeq = quad(W_GXEQ, G, b)
    missing = sum((JflrB(l, b) + JflrB(l - 1, b)) * G[l, 0] for l in range(G.shape[0]))
    fixed = fixed_qB_bar(G, b)
    claim_fix = sum((Jfac(l, b) + Jfac(l - 1, b) + JflrB(l, b) + JflrB(l - 1, b)) * G[l, 0] + JflrB(l, b) * G[l, 2] / R2
                    for l in range(G.shape[0]))
    print(f"\n  b {b:4.2f}: exact ∫(E/T)K h {exact:+.9f} | GX qB_bar {gx:+.9f} | ∫[(E/T)K − (2/b)μ̂J2]h {gxeq:+.9f} | "
          f"Σ(JflrB_l + JflrB_l−1)G_l0 {missing:+.9f} | fixed kernel {fixed:+.9f}")
    assert abs(gx - gxeq) < 1e-11
    assert abs(exact - gx - missing) < 1e-11
    assert abs(fixed - exact) < 1e-11 and abs(claim_fix - exact) < 1e-11
    assert abs(gx - exact) > 0.2 * abs(exact)           # not a rounding matter: tens of per cent or a sign


def test_hand_example_density_perturbation_at_long_wavelength():
    """b → 0, h = F_M (G_00 = 1): the δB∥-drift heat flux weight is ⟨(E/T) μ̂⟩ = ⟨x²/2⟩⟨μ̂⟩ + ⟨μ̂²⟩ = 1/2 + 2 = 5/2;
    GX's kernel gives Jfac(0) = 3/2 (the ES energy ⟨E/T⟩, as if K were 1).  For G_10 = 1 (h = F_M ψ_1, ψ_1 = μ̂ − 1):
    exact ⟨(x²/2 + μ̂) μ̂ (μ̂ − 1)⟩ = 1/2 + 4 = 9/2, GX Jfac(1) + Jfac(0) = 5/2."""
    for (l, exact_hand, gx_hand) in ((0, 2.5, 1.5), (1, 4.5, 2.5)):
        G = _unit(3, 3, l, 0)
        q, gx = quad(W_BPAR, G, 0.0), gx_qB_bar(G, 0.0)
        print(f"\n  b = 0, G_{l}0 = 1: exact {q:.12f} (hand {exact_hand}), GX qB_bar {gx:.12f} (hand {gx_hand}), "
              f"fixed {fixed_qB_bar(G, 0.0):.12f}")
        assert abs(q - exact_hand) < 1e-10 and abs(gx - gx_hand) < 1e-14 and abs(fixed_qB_bar(G, 0.0) - exact_hand) < 1e-14
    # GKX issue #326 (uwplasma/GKX, 2026-10-02, the same kernel ported): G_00 = 1 at b = 0.27053411424286467 gives
    # 1.1920704940 with the kernel and 2.0655531048 by quadrature; in closed form e^{−b/2}(3/2 − b/2) and e^{−b/2}(5/2 − b/2)
    b = 0.27053411424286467
    G = _unit(3, 3, 0, 0)
    q, gx, fx = quad(W_BPAR, G, b), gx_qB_bar(G, b), fixed_qB_bar(G, b)
    print(f"  b = {b}, G_00 = 1: exact {q:.10f} (GKX #326: 2.0655531048), GX qB_bar {gx:.10f} (GKX #326: 1.1920704940), "
          f"closed forms {math.exp(-b / 2) * (2.5 - b / 2):.10f} / {math.exp(-b / 2) * (1.5 - b / 2):.10f}")
    assert abs(q - 2.0655531048) < 1e-9 and abs(fx - 2.0655531048) < 1e-9 and abs(gx - 1.1920704940) < 1e-9


# ================================================================== 4. GX against itself: its drive uses the exact weight
def gx_drive(G_shape, b, phi, apar, bpar, iky, kap, tau, zt, vt):
    """rhs_linear's gradient-drive rows (device_funcs.cu:3181–3205), one point, as an (nl, nm) array."""
    nl, nm = G_shape
    out = np.zeros((nl, nm), dtype=complex)
    for l in range(nl):
        out[l, 0] = (iky * phi * (Jflr(l - 1, b) * l * tau + Jflr(l, b) * (kap + 2 * l * tau) + Jflr(l + 1, b) * (l + 1) * tau)
                     + iky / zt * bpar * (JflrB(l - 1, b) * l * tau + JflrB(l, b) * (kap + 2 * l * tau)
                                          + JflrB(l + 1, b) * (l + 1) * tau))
        out[l, 1] = -vt * iky * apar * (Jflr(l - 1, b) * l * tau + Jflr(l, b) * (kap + (2 * l + 1) * tau)
                                         + Jflr(l + 1, b) * (l + 1) * tau)
        out[l, 2] = iky * phi * Jflr(l, b) / R2 * tau + iky / zt * bpar * JflrB(l, b) / R2 * tau
        out[l, 3] = -vt * iky * apar * R15 * tau * Jflr(l, b)
    return out


@pytest.mark.parametrize("b", (0.0, 0.1, 1.0, 3.0))
def test_gx_drive_rows_are_exact_projections_including_the_bpar_rows(b):
    """GX's drive for δB∥ is the projection of [κ' + τ'(E/T − 3/2)] K: GX's own linear operator expands E·K as
    (μ̂-recurrence)∘JflrB = E∘(S𝒥), not as S(E∘𝒥) like its heat-flux kernel."""
    nl, nm, kap, tau = 7, 4, 0.8, 2.5
    for field, fac in (("phi", lambda x, mu, bb: J0w(mu, bb)), ("bpar", lambda x, mu, bb: Kw(mu, bb)),
                       ("apar", lambda x, mu, bb: x * J0w(mu, bb))):
        D = gx_drive((nl, nm), b, phi=1.0 if field == "phi" else 0.0, apar=1.0 if field == "apar" else 0.0,
                     bpar=1.0 if field == "bpar" else 0.0, iky=1.0, kap=kap, tau=tau, zt=1.0, vt=1.0)
        if field == "apar":
            D = -D                                        # the row carries −v_t i k_y A∥ (v∥/v_t = x)
        err = 0.0
        x, wx, mu, wmu = GRID
        Wv = (kap + tau * (E(x[:, None], mu[None, :]) - 1.5)) * fac(x[:, None], mu[None, :], b) * np.ones((len(x), len(mu)))
        for l in range(nl):
            for m in range(nm):
                # ⟨[κ' + τ'(E/T − 3/2)]·kernel, ĥ_m ψ_l⟩ by quadrature against the basis function
                hm = eval_hermitenorm(m, x) / math.sqrt(math.factorial(m))
                pl = (-1) ** l * eval_laguerre(l, mu)
                proj = np.einsum("x,u,xu,x,u->", wx, wmu, Wv, hm, pl)
                err = max(err, abs(D[l, m] - proj))
        print(f"\n  b {b:3.1f} {field:4s}: max |GX drive row − projection of [κ' + τ'(E/T − 3/2)]·kernel| = {err:.1e}")
        assert err < 1e-10


@pytest.mark.parametrize("b", (0.0, 0.1, 1.0, 3.0))
def test_gx_free_energy_injection_matches_the_heat_flux_only_with_the_exact_bpar_kernel(b):
    """Σ_lm Re[conj(H_lm) · drive_lm] = κ' Γ + τ'(Q − 1.5 Γ) for GX's own drive rows, with Q and Γ in
    heat_flux_summand's / particle_flux_summand's form evaluated on the same H: an algebraic identity when the δB∥
    heat kernel is the exact one (the drive's weight); GX's qB_bar breaks it by τ'(T/Z) Re[conj(i k_y δB∥) Σ (JflrB_l +
    JflrB_l−1) H_l0]."""
    nl, nm = 8, 4
    r = np.random.default_rng(int(10 * b) + 5)
    H = r.normal(size=(nl, nm)) + 1j * r.normal(size=(nl, nm))
    phi, apar, bpar = (complex(r.normal(), r.normal()) for _ in range(3))
    iky, kap, tau, zt, vt = 1j * 0.7, 0.8, 2.5, 1.0 / 1.3, 0.9
    tzs = 1.0 / zt
    D = gx_drive((nl, nm), b, phi, apar, bpar, iky, kap, tau, zt, vt)
    inj = float(np.sum(np.real(np.conj(H) * D)))
    vP, vA, vB = iky * phi, iky * apar, iky * bpar

    def Q(qB):
        return float(np.real(np.conj(vP) * gx_p_bar(H, b) - vt * np.conj(vA) * gx_q_bar(H, b) + tzs * np.conj(vB) * qB))
    Gam = float(np.real(np.conj(vP) * gx_n_bar(H, b) - vt * np.conj(vA) * gx_u_bar(H, b) + tzs * np.conj(vB) * gx_uB_bar(H, b)))
    rhs_exact = kap * Gam + tau * (Q(fixed_qB_bar(H, b)) - 1.5 * Gam)
    rhs_gx = kap * Gam + tau * (Q(gx_qB_bar(H, b)) - 1.5 * Gam)
    print(f"\n  b {b:3.1f}: drive injection {inj:+.10f} | κ'Γ + τ'(Q − 1.5Γ) with the exact δB∥ kernel {rhs_exact:+.10f} "
          f"| with GX's qB_bar {rhs_gx:+.10f}")
    assert abs(inj - rhs_exact) < 1e-10 * max(1.0, abs(inj))
    assert abs(inj - rhs_gx) > 1e-3 * max(1.0, abs(inj))


@pytest.mark.parametrize("b", (0.0, 0.1, 1.0))
def test_gx_total_heat_flux_depends_on_g_versus_h_only_through_the_bpar_kernel(b):
    """h = g + (Z/T)⟨χ⟩F_M (GX's H_ macro) carries no flux: Re[conj(i k_y ⟨χ⟩)⟨χ⟩] = 0 at every velocity, so any exact
    kernel gives Q_total(H) = Q_total(G).  GX's heat_flux_summand changes by k_y Im(φ̄ δB∥)[Σ Jfac JflrB − Σ (Jfac_l +
    Jfac_l−1) Jflr] — its reported total flux is not even invariant under adding the flux-free adiabatic part."""
    nl, nm = 12, 4
    r = np.random.default_rng(7)
    G = r.normal(size=(nl, nm)) + 1j * r.normal(size=(nl, nm))
    phi, apar, bpar = (complex(r.normal(), r.normal()) for _ in range(3))
    iky, zt, vt = 1j * 0.6, 1.0 / 1.7, 0.8
    tzs = 1.0 / zt
    H = G.copy()
    for l in range(nl):
        H[l, 0] += zt * Jflr(l, b) * phi + JflrB(l, b) * bpar
        H[l, 1] += -zt * vt * Jflr(l, b) * apar
    vP, vA, vB = iky * phi, iky * apar, iky * bpar

    def Q(X, kern):
        return float(np.real(np.conj(vP) * gx_p_bar(X, b) - vt * np.conj(vA) * gx_q_bar(X, b) + tzs * np.conj(vB) * kern(X, b)))
    d_exact = Q(H, fixed_qB_bar) - Q(G, fixed_qB_bar)
    d_gx = Q(H, gx_qB_bar) - Q(G, gx_qB_bar)
    print(f"\n  b {b:3.1f}: Q_total(h) − Q_total(g): exact δB∥ kernel {d_exact:+.2e} | GX's kernel {d_gx:+.6f} "
          f"(|Q_total(g)| = {abs(Q(G, gx_qB_bar)):.3f})")
    assert abs(d_exact) < 1e-12
    assert abs(d_gx) > 1e-3


# ================================================================== 5. GX's μ-grid branch: the same long-wavelength error, nodal
@pytest.mark.parametrize("b", (0.0, 0.1, 1.0))
def test_gx_mu_grid_branch_bpar_weight(b):
    """origin/feature/mu-gauss-laguerre-grid (d2f8c560, device_funcs.cu:2231 heat_flux_summand_mu) and GX v2
    (origin/feature/gx-v2 5c213ac6, src/diagnostics.cu:966–971): at each μ node, p_bar's m = 0 weight is (μ̂ + ½) J0
    (exact) but qB_bar's is ½(μ̂ + 1) JB with JB = 2μ̂ J1(α)/α = K: the weight (x²/2 + μ̂/2) K — the Laguerre kernel's
    b → 0 limit (x²μ̂/2 + μ̂²/2) carried over, not (E/T) K.  The nodal fix: (μ̂ + ½) JB."""
    G = _random_G(6, 4, 3)
    nodal_gx = quad(lambda x, mu, bb: 0.5 * x ** 2 * Kw(mu, bb) + 0.5 * mu * Kw(mu, bb), G, b)   # ½(μ+1)K ĥ_0 + K ĥ_2/√2
    # the branch's formula literally, with h/F_M's Hermite moments at the nodes: Σ_j w_j [½(μ_j + 1) K G0(μ_j) + K G2(μ_j)/√2]
    x, wx, mu, wmu = GRID
    pl = np.array([(-1) ** l * eval_laguerre(l, mu) for l in range(G.shape[0])])
    G0, G2 = G[:, 0] @ pl, G[:, 2] @ pl
    branch = float(np.sum(wmu * (0.5 * (mu + 1.0) * Kw(mu, b) * G0 + Kw(mu, b) * G2 / R2)))
    branch_es = float(np.sum(wmu * ((mu + 0.5) * J0w(mu, b) * G0 + J0w(mu, b) * G2 / R2)))
    exact, gx_lag = quad(W_BPAR, G, b), gx_qB_bar(G, b)
    print(f"\n  b {b:3.1f}: μ-grid branch qB_bar {branch:+.9f} (= ∫(x²/2 + μ̂/2)K h {nodal_gx:+.9f}) | exact {exact:+.9f} | "
          f"Laguerre GX {gx_lag:+.9f} | μ-grid ES p_bar vs exact {abs(branch_es - quad(W_ES, G, b)):.1e}")
    assert abs(branch - nodal_gx) < 1e-11 and abs(branch_es - quad(W_ES, G, b)) < 1e-11
    assert abs(branch - exact) > 0.1 * abs(exact)
    if b == 0.0:
        assert abs(branch - gx_lag) < 1e-11
    else:
        assert abs(branch - gx_lag) > 1e-4


def test_quadrature_is_converged():
    """The quadrature grid against a coarser one at the largest b used."""
    G = _random_G(6, 4, 3)
    for w in (W_BPAR, W_GXEQ, W_ES):
        a, c = quad(w, G, 4.0), quad(w, G, 4.0, grid=_grid(30, 110))
        assert abs(a - c) < 1e-11, (a, c)
