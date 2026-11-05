"""The linearised Landau (Fokker–Planck, Coulomb) collision operator in the Hermite–Laguerre basis at k⊥ = 0
(``physics.collision_model = "landau"``).

For species a colliding with species b (Rosenbluth, MacDonald & Judd 1957; Helander & Sigmar 2002 §3):

    C_ab(f_a, f_b) = Γ_ab ∂_i [ ½ (∂_i∂_j G_b) ∂_j f_a − (m_a/m_b) (∂_i H_b) f_a ],      Γ_ab = e_a² e_b² ln Λ/(4π ε₀² m_a²),

with the Rosenbluth potentials H_b = ∫ f_b(v′)/|v − v′| d³v′ and G_b = ∫ f_b(v′)|v − v′| d³v′.  Linearised about
Maxwellians, C_ab = C_ab(f_a1, F_b0) + C_ab(F_a0, f_b1): the *test-particle* part (the Maxwellian's potentials, closed
forms with erf) and the *field-particle* part (the potentials of the perturbation of species b acting on species a's
Maxwellian).  Velocities are measured in v_ta = √(2T_a/m_a) and the rate in ν̂_ab = Γ_ab n_b/v_ta³ (Helander & Sigmar's
ν̂; ν̂_ei = Z_i ν̂_ee for quasineutral n, ν̂_ie = ν̂_ii/Z_i); Braginskii's τ_e = (3√π/4)/ν̂_ee, τ_i = (3√π/(2√2))/ν̂_ii.

Matrix elements ⟨ψ_ℓ L_m, C[ψ_ℓ′ L_m′ F₀]⟩ (the basis of :mod:`velcro.basis`: ψ_ℓ(v̂∥) orthonormal under e^{−v̂²}/√π,
L_m(μ̂) under e^{−μ̂}, μ̂ = v̂⊥²) are computed exactly up to quadrature:

* test part, integrated by parts once: −∫ [½(G″ g_v f_v + (G′/v³)(1 − ξ²) g_ξ f_ξ) − (m_a/m_b) H′ g_v f] d³v, with the
  isotropic Maxwellian potentials of species b in a's units (α = v_ta/v_tb): H = erf(αv)/v,
  G = (v + 1/(2α²v)) erf(αv) + e^{−α²v²}/(α√π);
* field part, integrated by parts twice: −∫ Φ_b ∇·(F_a0 ∇g) d³v with Φ_b = v∂_vG_b − G_b + (m_a/m_b) H_b (so that
  ½∂_i∂_jG_b ∂_jF_a0 − (m_a/m_b)∂_iH_b F_a0 = −F_a0 ∂_iΦ_b for ∂_jF_a0 = −2v_jF_a0), the source ψ_ℓ′L_m′F_b0 expanded in
  Legendre harmonics f_l(v) = (polynomial in v) × e^{−α²v²}, whose radial Rosenbluth integrals are incomplete gamma
  functions (closed form; no radial grid);
* the (v, ξ) quadrature is Gauss–Legendre, exact in ξ (polynomial integrands) and converged in v.

The operator conserves particles, parallel momentum and energy (like species exactly; between species with the
Landau symmetry, tests/test_landau_operator.py), is self-adjoint and negative semi-definite (H-theorem) in the
F₀-weighted inner product, reduces to the Lorentz pitch-angle operator for e–i at m_e/m_i → 0, and gives the
Spitzer–Härm transport coefficients.  These matrices are the k⊥ = 0 operator; the finite-Larmor-radius terms are
``physics.collision_flr`` (velcro/collisions_flr.py: the exact gyroaveraged test part C_T(0) − bK, the field part's
J₀-Galerkin average and the J₁ perpendicular-momentum restoring; the exact gyroaverage with Bessel functions of every
order, Pan & Ernst 2019, is velcro.landau_exact.like_exact_gk, the reference; the validation notes).  The Sugama
model operator is not implemented (the full Landau field part is available).

Production route (since the tier-2 review, 2026-10-03): :func:`like_species_matrix` and :func:`pair_matrices` come from
:mod:`velcro.landau_exact`, an exact quadrature of the symmetrised Landau form in centre-of-mass/relative velocities — exact
to rounding at any (L, M) (Hirshman–Sigmar friction to 12–13 digits, the Lorentz element to 1e-15, symmetric and
conserving to 2e-16), cached, read-only; the range is capped at (24, 12) only for the construction time ((16, 8) ≈ 14 s,
(24, 12) ≈ 100 s on one core).  The Rosenbluth-potential route below (:func:`test_particle_matrix`,
:func:`field_particle_matrix`: closed-form Maxwellian potentials, a field part expanded in monomials v^k with
incomplete-gamma radial integrals, Gauss–Legendre in v ∈ (0, 9]) is kept as the independent cross-check: it agrees with the
exact route to 1e-13 up to (12, 6), then loses precision — at (16, 8) its field part is off by 8.8e-6 of max|C| (0.8 % of
the worst, highest-order element; the test part 2e-8, from vmax = 9), at (20, 10) by 2.4 × max|C| with a positive
eigenvalue; its e←i field block C_F,ei is right only to (6, 3) (1.5, 1.4e2 and 3.0e3 × its own maximum off at (8, 4),
(12, 6), (16, 8)) and C_F,ie only to (4, 2).  The solver uses C_T,ei alone for the electrons (no drag toward the ion flow,
GX's β = 0 model; refused at β > 0) and no e–i collisions for the ions (velcro/kinetic.py).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf, eval_genlaguerre, eval_laguerre, gamma, gammainc, gammaincc, roots_legendre
from scipy.special import eval_legendre

from velcro import basis

SQRT_PI = np.sqrt(np.pi)
L_MAX, M_MAX = 24, 12


def _check_range(L: int, M: int) -> None:
    if L > L_MAX or M > M_MAX:
        raise ValueError(f"collision_model = 'landau' is built for L ≤ {L_MAX}, M ≤ {M_MAX} (got ({L}, {M})): the exact "
                         "quadrature has no accuracy limit, the cap bounds its construction time (≈ 100 s at (24, 12))")


# ------------------------------------------------------------------------------------------------ basis functions
def _hermite_derivs(L: int, v: np.ndarray):
    """ψ_ℓ(v), ψ_ℓ′, ψ_ℓ″ for ℓ < L on ``v`` → three (L, …) arrays (ψ_ℓ′ = √(2ℓ) ψ_{ℓ−1})."""
    psi = np.array([basis.psi(l, v) for l in range(L)])
    d1, d2 = np.zeros_like(psi), np.zeros_like(psi)
    for l in range(1, L):
        d1[l] = np.sqrt(2.0 * l) * psi[l - 1]
    for l in range(2, L):
        d2[l] = np.sqrt(2.0 * l) * np.sqrt(2.0 * (l - 1)) * psi[l - 2]
    return psi, d1, d2


def _laguerre_derivs(M: int, mu: np.ndarray):
    """L_m(μ), L_m′ = −L^{(1)}_{m−1}, L_m″ = L^{(2)}_{m−2} for m < M → three (M, …) arrays."""
    Lm = np.array([eval_laguerre(m, mu) for m in range(M)])
    d1 = np.array([-eval_genlaguerre(m - 1, 1, mu) if m >= 1 else np.zeros_like(mu) for m in range(M)])
    d2 = np.array([eval_genlaguerre(m - 2, 2, mu) if m >= 2 else np.zeros_like(mu) for m in range(M)])
    return Lm, d1, d2


def _hermite_poly_coefs(l: int) -> np.ndarray:
    """Power-series coefficients of ψ_ℓ(y) = H_ℓ(y)/√(2^ℓ ℓ!) (index = power of y)."""
    from numpy.polynomial import hermite as H
    c = H.herm2poly(np.eye(l + 1)[l]) * basis.hermite_norm(l)
    return c


def _laguerre_poly_coefs(m: int) -> np.ndarray:
    """Power-series coefficients of L_m(z)."""
    from numpy.polynomial import laguerre as La
    return La.lag2poly(np.eye(m + 1)[m])


# ------------------------------------------------------------------------------------------------ quadrature
def _grid(L: int, M: int, nv: int = 240, vmax: float = 9.0):
    """Gauss–Legendre nodes in v ∈ (0, vmax] and ξ ∈ (−1, 1) with the d³v weight 2π v² dv dξ."""
    xv, wv = roots_legendre(nv)
    v = 0.5 * vmax * (xv + 1.0)
    wv = 0.5 * vmax * wv
    nxi = L + 2 * M + 8
    xi, wxi = roots_legendre(nxi)
    V, XI = np.meshgrid(v, xi, indexing="ij")
    W = 2 * np.pi * (wv[:, None] * V ** 2) * wxi[None, :]
    return v, xi, V, XI, W


def _basis_on_grid(L: int, M: int, V, XI):
    """g = ψ_ℓ(v∥) L_m(μ) and its v- and ξ-derivatives, ∇²g and v·∇g on the (v, ξ) grid → dict of (L, M, nv, nxi)."""
    vpar, mu = V * XI, V ** 2 * (1.0 - XI ** 2)
    psi, dpsi, ddpsi = _hermite_derivs(L, vpar)
    Lm, dLm, ddLm = _laguerre_derivs(M, mu)
    g = psi[:, None] * Lm[None, :]
    # ∂_v g = ξ ψ′ L + 2v(1 − ξ²) ψ L′;  ∂_ξ g = v ψ′ L − 2v²ξ ψ L′
    g_v = XI * dpsi[:, None] * Lm[None, :] + 2 * V * (1 - XI ** 2) * psi[:, None] * dLm[None, :]
    g_xi = V * dpsi[:, None] * Lm[None, :] - 2 * V ** 2 * XI * psi[:, None] * dLm[None, :]
    # ∇²g = ψ″ L + ψ (4 L′ + 4 μ L″);  v·∇g = v∥ ψ′ L + 2μ ψ L′
    lap = ddpsi[:, None] * Lm[None, :] + psi[:, None] * (4 * dLm[None, :] + 4 * mu * ddLm[None, :])
    vdg = vpar * dpsi[:, None] * Lm[None, :] + 2 * mu * psi[:, None] * dLm[None, :]
    return dict(g=g, g_v=g_v, g_xi=g_xi, lap=lap, vdg=vdg)


# ------------------------------------------------------------------------------------------------ Maxwellian potentials
def maxwellian_potentials(v: np.ndarray, alpha: float):
    """H, H′, G′, G″ of the unit Maxwellian of thermal speed 1/α (in the test species' units) at |v|."""
    a = alpha
    x = a * v
    e = np.exp(-x ** 2)
    er = erf(x)
    H = er / v
    Hp = (2 * a * v * e / SQRT_PI - er) / v ** 2
    Gp = (1.0 - 1.0 / (2 * a ** 2 * v ** 2)) * er + e / (a * SQRT_PI * v)
    Gpp = er / (a ** 2 * v ** 3) - 2 * e / (a * SQRT_PI * v ** 2)
    return H, Hp, Gp, Gpp


# ------------------------------------------------------------------------------------------------ test-particle part
@lru_cache(maxsize=None)
def test_particle_matrix(L: int, M: int, alpha: float = 1.0, mass_ratio: float = 1.0, nv: int = 240, vmax: float = 9.0) -> np.ndarray:
    """C_T,ab in ν̂_ab units: species a's moments → species a's moments, the field species b a Maxwellian of thermal
    speed v_tb = v_ta/alpha and mass m_b = m_a/mass_ratio (``mass_ratio`` = m_a/m_b).  (LM × LM), index ℓ·M + m."""
    v, xi, V, XI, W = _grid(L, M, nv, vmax)
    B = _basis_on_grid(L, M, V, XI)
    H, Hp, Gp, Gpp = maxwellian_potentials(V, alpha)
    F0 = np.exp(-V ** 2) / np.pi ** 1.5
    # source f = g′ F0: f_v = (g′_v − 2v g′) F0, f_ξ = g′_ξ F0
    f_v = (B["g_v"] - 2 * V * B["g"]) * F0
    f_xi = B["g_xi"] * F0
    f = B["g"] * F0
    n = L * M
    gv = B["g_v"].reshape(n, -1); gxi = B["g_xi"].reshape(n, -1)
    fv = f_v.reshape(n, -1); fxi = f_xi.reshape(n, -1); ff = f.reshape(n, -1)
    w = W.reshape(-1)
    A = 0.5 * Gpp.reshape(-1)
    Bc = 0.5 * (Gp / V ** 3 * (1 - XI ** 2)).reshape(-1)
    Cc = mass_ratio * Hp.reshape(-1)
    C = -((gv * (w * A)) @ fv.T + (gxi * (w * Bc)) @ fxi.T - (gv * (w * Cc)) @ ff.T)
    C.setflags(write=False)                                      # cached: never hand out a writable shared array
    return C


# ------------------------------------------------------------------------------------------------ field-particle part
def _legendre_projection_table(lmax: int, amax: int, bmax: int):
    """Λ[l, a, b] = ((2l+1)/2) ∫ P_l(ξ) ξ^a (1 − ξ²)^b dξ (exact Gauss–Legendre)."""
    n = (lmax + amax + 2 * bmax) // 2 + 2
    x, w = roots_legendre(n)
    P = np.array([eval_legendre(l, x) for l in range(lmax + 1)])
    xa = np.array([x ** a for a in range(amax + 1)])
    ob = np.array([(1 - x ** 2) ** b for b in range(bmax + 1)])
    return np.einsum("lk,ak,bk,k,l->lab", P, xa, ob, w, 0.5 * (2 * np.arange(lmax + 1) + 1))


def _radial_integrals(k: int, a2: float, v: np.ndarray):
    """∫_0^v t^k e^{−a2 t²} dt and ∫_v^∞ (closed forms: incomplete gamma, s = (k+1)/2)."""
    s = 0.5 * (k + 1)
    pref = 0.5 * gamma(s) / a2 ** s
    x = a2 * v ** 2
    return pref * gammainc(s, x), pref * gammaincc(s, x)


def source_potentials(L: int, M: int, alpha: float, v: np.ndarray, mass_ratio: float):
    """Φ_{ℓ′m′}(v, l) for every source ψ_ℓ′ L_m′ F̂_b0 (unit density, thermal speed 1/alpha in a's units) and Legendre
    index l: Φ = v G′ − G + mass_ratio · H per harmonic → (L, M, lmax+1, nv)."""
    lmax = (L - 1) + 2 * (M - 1)
    Lam = _legendre_projection_table(lmax, L - 1, M - 1)
    a2 = alpha ** 2
    pref = alpha ** 3 / np.pi ** 1.5
    out = np.zeros((L, M, lmax + 1, len(v)))
    hc = [_hermite_poly_coefs(l) for l in range(L)]
    lc = [_laguerre_poly_coefs(m) for m in range(M)]
    kmax = (L - 1) + 2 * (M - 1)
    # radial integrals of t^k e^{−α²t²} for every power that appears, k = (a + 2b) + s with s ∈ {l+2, 1−l, l+4, 3−l}
    cache = {}

    def I(k):
        if k not in cache:
            cache[k] = _radial_integrals(k, a2, v)
        return cache[k]
    for l0 in range(L):
        for m0 in range(M):
            # f_l(v) = Σ_k q[l, k] v^k e^{−α²v²}:  q[l, a+2b] += pref c_a d_b α^{a+2b} Λ[l, a, b]
            q = np.zeros((lmax + 1, kmax + 1))
            for a, ca in enumerate(hc[l0]):
                if ca == 0.0:
                    continue
                for b, db in enumerate(lc[m0]):
                    if db == 0.0:
                        continue
                    q[:, a + 2 * b] += pref * ca * db * alpha ** (a + 2 * b) * Lam[:, a, b]
            q[np.arange(lmax + 1)[:, None] > np.arange(kmax + 1)[None, :]] = 0.0   # the harmonic l needs degree k ≥ l
            for l in range(lmax + 1):
                if not np.any(np.abs(q[l]) > 1e-14 * np.abs(q).max()):
                    continue
                c1 = 4 * np.pi / (2 * l + 1)
                H = np.zeros_like(v); G = np.zeros_like(v); Gp = np.zeros_like(v)
                for k in np.flatnonzero(np.abs(q[l]) > 1e-14 * np.abs(q).max()):
                    qk = q[l, k]
                    i1, _ = I(k + l + 2)            # ∫_0^v f v′^{l+2}
                    _, i2 = I(k + 1 - l)            # ∫_v^∞ f v′^{1−l}
                    i3, _ = I(k + l + 4)            # ∫_0^v f v′^{l+4}
                    _, i4 = I(k + 3 - l)            # ∫_v^∞ f v′^{3−l}
                    H += qk * c1 * (v ** (-l - 1) * i1 + v ** l * i2)
                    G += qk * c1 * ((v ** (-l - 1) * i3 + v ** (l + 2) * i2) / (2 * l + 3)
                                    - (v ** (1 - l) * i1 + v ** l * i4) / (2 * l - 1))
                    Gp += qk * c1 * ((-(l + 1) * v ** (-l - 2) * i3 + (l + 2) * v ** (l + 1) * i2) / (2 * l + 3)
                                     - ((1 - l) * v ** (-l) * i1 + l * v ** (l - 1) * i4) / (2 * l - 1))
                out[l0, m0, l] = v * Gp - G + mass_ratio * H
    return out


@lru_cache(maxsize=None)
def field_particle_matrix(L: int, M: int, alpha: float = 1.0, mass_ratio: float = 1.0, nv: int = 240, vmax: float = 9.0) -> np.ndarray:
    """C_F,ab in ν̂_ab units: species b's moments → species a's moments (a's Maxwellian responding to b's
    perturbation); ``alpha`` = v_ta/v_tb, ``mass_ratio`` = m_a/m_b.  (LM × LM), both sides index ℓ·M + m (the source
    index is species b's)."""
    v, xi, V, XI, W = _grid(L, M, nv, vmax)
    B = _basis_on_grid(L, M, V, XI)
    F0 = np.exp(-V ** 2) / np.pi ** 1.5
    div = (B["lap"] - 2 * B["vdg"]) * F0                        # ∇·(F0 ∇g) = F0 (∇²g − 2 v·∇g)
    Phi_l = source_potentials(L, M, alpha, v, mass_ratio)       # (L, M, lmax+1, nv)
    lmax = Phi_l.shape[2] - 1
    P = np.array([eval_legendre(l, xi) for l in range(lmax + 1)])   # (lmax+1, nxi)
    Phi = np.einsum("pqlv,lx->pqvx", Phi_l, P)                 # Φ_{source}(v, ξ)
    n = L * M
    C = -(div.reshape(n, -1) * W.reshape(-1)) @ Phi.reshape(n, -1).T
    C.setflags(write=False)
    return C


# ------------------------------------------------------------------------------------------------ species pairs
def like_species_matrix(L: int, M: int) -> np.ndarray:
    """C_aa = C_T,aa + C_F,aa in ν̂_aa units (dimensionless, the same for every species in its own units): the exact
    quadrature of :mod:`velcro.landau_exact`; read-only (cached — copy before writing)."""
    _check_range(L, M)
    from velcro.landau_exact import like_exact
    return like_exact(int(L), int(M))


def pair_matrices(L: int, M: int, mass_a: float, temp_a: float, mass_b: float, temp_b: float):
    """(C_T,ab, C_F,ab) in ν̂_ab units for species a colliding with b (masses and temperatures in common units):
    alpha = v_ta/v_tb = √(T_a m_b/(T_b m_a)), mass_ratio = m_a/m_b, θ = T_a/T_b; the exact quadrature of
    :mod:`velcro.landau_exact`, read-only (cached — copy before writing)."""
    _check_range(L, M)
    from velcro.landau_exact import pair_exact
    alpha = float(np.sqrt(temp_a * mass_b / (temp_b * mass_a)))
    return pair_exact(int(L), int(M), alpha, float(mass_a / mass_b), float(temp_a / temp_b))


def pair_matrices_rosenbluth(L: int, M: int, mass_a: float, temp_a: float, mass_b: float, temp_b: float):
    """:func:`pair_matrices` by the Rosenbluth-potential route (the independent cross-check; its field blocks are right
    only at low order, see the module docstring)."""
    alpha = float(np.sqrt(temp_a * mass_b / (temp_b * mass_a)))
    return test_particle_matrix(L, M, alpha, float(mass_a / mass_b)), field_particle_matrix(L, M, alpha, float(mass_a / mass_b))


def like_species_matrix_rosenbluth(L: int, M: int) -> np.ndarray:
    """:func:`like_species_matrix` by the Rosenbluth-potential route (accurate to 1e-13 up to (12, 6))."""
    return test_particle_matrix(L, M, 1.0, 1.0) + field_particle_matrix(L, M, 1.0, 1.0)


def lorentz_matrix(L: int, M: int, nv: int = 240, vmax: float = 9.0) -> np.ndarray:
    """The Lorentz pitch-angle operator (1/(2v³)) ∂_ξ(1 − ξ²)∂_ξ in ν̂ units, by quadrature (integrated by parts:
    −∫ (1 − ξ²)/(2v³) g_ξ f_ξ d³v): the m_e/m_i → 0 limit of C_T,ei."""
    v, xi, V, XI, W = _grid(L, M, nv, vmax)
    B = _basis_on_grid(L, M, V, XI)
    F0 = np.exp(-V ** 2) / np.pi ** 1.5
    n = L * M
    gxi = B["g_xi"].reshape(n, -1)
    fxi = (B["g_xi"] * F0).reshape(n, -1)
    w = (W * (1 - XI ** 2) / (2 * V ** 3)).reshape(-1)
    return -(gxi * w) @ fxi.T


def moment_functionals(L: int, M: int) -> np.ndarray:
    """Rows (3, LM): the density, parallel momentum (∫v̂∥ f = G₁₀/√2) and energy (∫v̂² f = G₂₀/√2 − G₀₁ + 3/2 G₀₀)
    functionals of the moment vector."""
    e = lambda l, m: np.eye(L * M)[l * M + m]
    rows = [e(0, 0)]
    rows.append(e(1, 0) / np.sqrt(2.0) if L > 1 else np.zeros(L * M))
    en = 1.5 * e(0, 0)
    if L > 2:
        en = en + e(2, 0) / np.sqrt(2.0)
    if M > 1:
        en = en - e(0, 1)
    rows.append(en)
    return np.array(rows)
