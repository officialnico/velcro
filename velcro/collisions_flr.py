"""Gyrokinetic finite-Larmor-radius (FLR) collision terms in the Hermite–Laguerre basis (``physics.collision_flr``).

Derivation and validation: the validation notes.  Conventions as :mod:`velcro.basis` (v̂ = v/v_t, v_t = √(2T/m),
μ̂ = v̂⊥², ψ_ℓ(v̂∥) L_m(μ̂) orthonormal under F₀) and :mod:`velcro.landau` (rates in ν̂ = Γ n/v_t³ for "landau", in ν for
"dougherty").  b = k⊥²ρ²/2 with ρ = v_t/Ω (``basis.gyro_coefficients``: 𝒥_m(b) = (b/2)^m e^{−b/2}/m!, the Laguerre
coefficients of J₀(√(2bμ̂))); q = k⊥ × b̂/Ω in v_t units has q² = 2b.

The gyrokinetic operator is the gyroaverage of the particle operator at fixed guiding centre (Abel et al. 2008 eq. 19):
C_k[h] = ⟨e^{iq·v} C[e^{−iq·v} h]⟩, so its Galerkin matrix is C_ij(b) = ∫ φ_i e^{iq·v} C[φ_j e^{−iq·v} F₀] d³v.

* **Test-particle parts, exact.**  A test-particle operator is ∂_i(D_ij ∂_j f + A_i f) with D, A functions of v alone;
  conjugating by e^{iq·v} gives (∂ − iq)·[D·(∂ − iq) f + A f], whose gyroaverage on a gyrophase-independent h is
  C_T[h] − ⟨q·D·q⟩ h (the terms odd in q average to zero).  So the gyroaveraged test part is **exactly affine in b**:
  C_T(b) = C_T(0) − b K with K_ij = ∫ φ_i φ_j F₀ w(v, ξ) d³v and ⟨q·D·q⟩ = b w (classical diffusion of guiding centres):
  - Dougherty and Lenard–Bernstein: D = ½ I, w = 1, K = I (GX's −ν b);
  - Landau: D = ½ ∂∂G_b (Rosenbluth's G of the field species' Maxwellian), w = ½[G″(1 − ξ²) + (G′/v)(1 + ξ²)]
    = ¼[ν_D(v)(v² + v∥²) + ν_∥(v) v⊥²]·(2/v²)… in Helander & Sigmar's speed functions ν_D = G′/v³, ν_∥ = G″/v²;
    e.g. K_(00)(00) = 4/(3√(2π)), K_(10)(10) = 6/(5√(2π)) for like species (closed forms).
* **Dougherty field part (restoring), exact.**  The linearised Dougherty restoring term ν F₀[2 v̂·δû + (4/3)(v̂² − 3/2)Θ]
  (δû = ∫v̂ f, Θ = ∫(v̂² − 3/2) f) gyroaverages to J₀-weighted parallel and energy terms and a J₁-weighted perpendicular
  one: 2 a∥a∥ᵀ + 2 a⊥a⊥ᵀ + (4/3) a_E a_Eᵀ with a∥ = v̂∥J₀, a⊥ = v̂⊥J₁(√(2bμ̂)), a_E = (v̂² − 3/2)J₀ projected on the basis
  (closed forms in 𝒥_m below).  With the test part this is GX's operator (device_funcs.cu rhs_linear /
  conservation_terms: −ν(b + 2l + m) h and the Jflr / √b·JflrB-weighted ū∥, ū⊥, T̄ terms) — the exact gyroaverage of
  the linearised Dougherty operator, hence symmetric and negative semi-definite at every b.
* **Landau field part (gyro-harmonic sectors).**  The exact field part at finite k⊥ involves Bessel functions of every
  order (Pan & Ernst 2019): with the Jacobi–Anger expansion e^{−iq·v} = Σₙ (−i)ⁿJₙ(qv⊥)e^{in(α−α_q)} and the rotation
  invariance of the Landau form, C_F(b) = Σₙ B_F[Jₙφ_i e^{−inα}, Jₙφ_j e^{inα}] exactly, n ∈ ℤ (±n equal).  Velcro keeps
  n = 0 and n = ±1, each on its own polynomial sector basis χ⁽ⁿ⁾_ℓp = ψ_ℓ v̂⊥ⁿe^{inα}L_p⁽ⁿ⁾(μ̂)/√((p+n)!/p!) with K₀ = M + 2,
  K₁ = M + 1 Laguerre functions: **C_F(b) ≈ 𝒬₀ᵀF⁽⁰⁾𝒬₀ + 2𝒬₁ᵀF⁽¹⁾𝒬₁**, F⁽ⁿ⁾ the exact field matrices on the sectors
  (velcro.landau_exact.field_sector_exact, exact quadrature) and 𝒬ₙ(b) the exact Bessel–Laguerre maps (finite sums of
  κ_k⁽ⁿ⁾(b) = (b/2)^{k+n/2}e^{−b/2}/(k+n)!, Hardy–Hille).  This carries the O(b) term exactly (|n| ≥ 2 enters at O(b²);
  J₀φ_j's O(b) needs L_M, inside K₀), contains the perpendicular-momentum restoring (v̂⊥e^{iα} is χ⁽¹⁾₀₀, so the density
  element is O(b²)), and b → 0 is C(0) exactly (𝒬₀(0) a selection, 𝒬₁(0) = 0; C_F taken as F⁽⁰⁾'s (L, M) block).  The
  first, registered J₀/J₁-dressed model (the n = 0 average on the retained basis and Sugama's J₁ momentum projection,
  ``landau_gk_model_j0j1``) failed its fidelity rule against the exact gyroaverage.

Conservation at finite k⊥ (Abel et al. 2008 eq. 25; Frei et al. 2021 eq. 3.5; Pan & Ernst 2019 §III): collisions conserve
particles, momentum and energy at the particle position, not at the guiding centre; the gyrocentre and particle-frame
(J₀-, J₀v∥-, J₁v⊥-, J₀v²-weighted) moments of C_k change by −ik·Γ_coll, the classical collisional fluxes, which vanish as
k⊥ → 0.  What the operators here conserve exactly is therefore the k⊥ = 0 part (the box average), as before.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf, gammainc, jv

from velcro import basis

SQRT2 = float(np.sqrt(2.0))
FLR_MODELS = ("dougherty", "lenard-bernstein", "landau")


# ------------------------------------------------------------------------------------------------ quadrature
@lru_cache(maxsize=None)
def _vmu_grid(nv: int, nmu: int):
    """Gauss–Hermite in v̂∥ (weights /√π) ⊗ Gauss–Laguerre in μ̂: ∫ F₀ f d³v = Σ w f (F₀ = π^{−3/2} e^{−v̂∥² − μ̂},
    d³v = π dv̂∥ dμ̂ after the gyroangle).  scipy's Golub–Welsch rules: numpy's laggauss overflows its weights beyond
    ~170 nodes (NaN at 192)."""
    from scipy.special import roots_hermite, roots_laguerre
    x, wx = roots_hermite(nv)
    mu, wmu = roots_laguerre(nmu)
    out = (np.asarray(x), np.asarray(wx) / np.sqrt(np.pi), np.asarray(mu), np.asarray(wmu))
    if not all(np.all(np.isfinite(a)) for a in out):
        raise FloatingPointError(f"non-finite Gauss rule (nv {nv}, nmu {nmu})")
    return out


def _basis_vmu(L: int, M: int, vpar: np.ndarray, mu: np.ndarray):
    """φ = ψ_ℓ(v̂∥) L_m(μ̂), ∂_∥φ and ∂_μφ on the product grid → three (L·M, nv, nmu) arrays, index ℓ·M + m."""
    psi = np.array([basis.psi(l, vpar) for l in range(L)])                                   # (L, nv)
    dpsi = np.zeros_like(psi)
    for l in range(1, L):
        dpsi[l] = np.sqrt(2.0 * l) * psi[l - 1]                                               # ψ_ℓ′ = √(2ℓ) ψ_{ℓ−1}
    from scipy.special import eval_genlaguerre, eval_laguerre
    lag = np.array([eval_laguerre(m, mu) for m in range(M)])                                 # (M, nmu)
    dlag = np.array([-eval_genlaguerre(m - 1, 1, mu) if m >= 1 else np.zeros_like(mu) for m in range(M)])
    n = L * M
    phi = np.einsum("lv,mu->lmvu", psi, lag).reshape(n, len(vpar), len(mu))
    dpar = np.einsum("lv,mu->lmvu", dpsi, lag).reshape(n, len(vpar), len(mu))
    dmu = np.einsum("lv,mu->lmvu", psi, dlag).reshape(n, len(vpar), len(mu))
    return phi, dpar, dmu


def test_part_speeds(v: np.ndarray, alpha: float = 1.0):
    """(D₁, D₂) = (G″, G′/v) of the field species' unit Maxwellian, thermal speed 1/alpha in the test species' units
    (the eigenvalues of ∂∂G along and across v), evaluated stably at small v through P(3/2, x²) = erf x − x erf′x
    (scipy's regularised lower incomplete gamma): G″ = P/(α²v³), G′/v = (erf x − P/(2x²))/v, x = αv.  In Helander &
    Sigmar's speed functions ν_∥ = G″/v² = 2G_ch(x)/v³… and ν_D = G′/v³ (Γ = 1, ν̂ units)."""
    v = np.asarray(v, dtype=float)
    x = alpha * v
    P = gammainc(1.5, x * x)
    D1 = P / (alpha ** 2 * v ** 3)
    D2 = (erf(x) - P / (2.0 * x * x)) / v
    return D1, D2


# ------------------------------------------------------------------------------------------------ test parts: K
def _speed_nodes(alpha: float, nv: int, vmax: float = 10.0):
    """Composite Gauss–Legendre nodes and weights in v ∈ (0, vmax]: one interval for α ≲ 2; for the e–i block (α ≫ 1)
    the field Maxwellian's speed 1/α is resolved by breaks at 1/α, 4/α and 16/α (D₂ = G′/v turns from ≈ (4/3√π)α to 1/v
    there, a feature the (v̂∥, μ̂) Gauss rules of F₀ cannot see)."""
    from numpy.polynomial.legendre import leggauss
    br = [0.0, vmax] if alpha <= 2.0 else sorted({0.0, 1.0 / alpha, 4.0 / alpha, min(1.0, 16.0 / alpha), vmax})
    x, w = leggauss(nv)
    v = np.concatenate([0.5 * (b - a) * (x + 1.0) + a for a, b in zip(br[:-1], br[1:])])
    wv = np.concatenate([0.5 * (b - a) * w for a, b in zip(br[:-1], br[1:])])
    return v, wv


@lru_cache(maxsize=None)
def classical_diffusion_matrix(L: int, M: int, alpha: float = 1.0, nv: int = 160) -> np.ndarray:
    """K of the Landau test part: K_ij = ∫ φ_i φ_j F₀ ½[D₁(1 − ξ²) + D₂(1 + ξ²)] d³v (ξ = v∥/v), so that the exact
    gyroaverage of C_T is C_T(0) − b K.  ``alpha`` = v_ta/v_tb of the field species (1 for like species; the e–i block
    has the ions' α = √(T_e m_i/(T_i m_e))).  Spherical quadrature: :func:`_speed_nodes` in v, Gauss–Legendre in ξ exact
    for the polynomial degree.  Symmetric positive definite, (LM × LM), read-only (cached)."""
    from numpy.polynomial.legendre import leggauss
    v, wv = _speed_nodes(float(alpha), nv)
    xi, wxi = leggauss(L + 2 * M + 4)
    V, XI = np.meshgrid(v, xi, indexing="ij")
    D1, D2 = test_part_speeds(V, alpha)
    wgt = 0.5 * (D1 * (1.0 - XI ** 2) + D2 * (1.0 + XI ** 2))
    W = 2.0 * np.pi * (wv[:, None] * V ** 2) * wxi[None, :] * np.exp(-V ** 2) / np.pi ** 1.5 * wgt
    vpar, mu = V * XI, V ** 2 * (1.0 - XI ** 2)
    P = np.array([basis.psi(l, vpar) * basis.laguerre(m, mu) for l in range(L) for m in range(M)]).reshape(L * M, -1)
    K = (P * W.reshape(-1)) @ P.T
    K = 0.5 * (K + K.T)
    K.setflags(write=False)
    return K


def classical_diffusion_closed_forms() -> dict:
    """Two elements of the like-species K in closed form: K_(00)(00) = (4/√π)(2/3)∫v e^{−v²} erf v dv
    = 4/(3√(2π)) and K_(10)(10) = 6/(5√(2π))."""
    return {(0, 0): 4.0 / (3.0 * np.sqrt(2 * np.pi)), (1, 0): 6.0 / (5.0 * np.sqrt(2 * np.pi))}


# ------------------------------------------------------------------------------------------------ Laguerre kernels
@lru_cache(maxsize=None)
def laguerre_triple(M: int) -> np.ndarray:
    """Λ[k, m, m′] = ∫ L_k L_m L_{m′} e^{−μ̂} dμ̂ for k ≤ 2M − 2 (zero beyond), exact Gauss–Laguerre → (2M − 1, M, M)."""
    K = max(2 * M - 1, 1)
    mu, w = basis.laguerre_quadrature(2 * M + 1)                                      # exact to degree 4M + 1 ≥ 4M − 4
    Lk = np.array([basis.laguerre(k, mu) for k in range(K)])
    Lm = Lk[:M]
    out = np.einsum("ki,mi,ni,i->kmn", Lk, Lm, Lm, w)
    out[np.abs(out) < 1e-13] = 0.0
    out[0] = np.eye(M)                                     # ∫ L_m L_m′ e^{−μ} = δ exactly: q(0) = I, so b = 0 adds nothing
    out.setflags(write=False)
    return out


def j0_galerkin(b: np.ndarray, M: int) -> np.ndarray:
    """q(b)_{mm′} = ∫ L_m L_{m′} J₀(√(2bμ̂)) e^{−μ̂} dμ̂ = Σ_k 𝒥_k(b) Λ[k, m, m′] (exact) → (M, M) + b.shape; q(0) = I."""
    b = np.asarray(b, dtype=float)
    lam = laguerre_triple(M)
    J = basis.gyro_coefficients(b, lam.shape[0])                                   # (K,) + b.shape
    return np.einsum("k...,kmn->mn...", J, lam)


def bessel_kernel(b: np.ndarray, n: int, K: int) -> np.ndarray:
    """κ_k⁽ⁿ⁾(b) = (b/2)^{k+n/2} e^{−b/2}/(k+n)!, k < K, the coefficients of J_n(√(2bμ̂)) = μ̂^{n/2} Σ_k κ_k L_k⁽ⁿ⁾(μ̂)
    (the Hardy–Hille generating function Σ_k L_k^{(α)}(x) t^k/Γ(k+α+1) = e^t (xt)^{−α/2} J_α(2√(xt)) at t = b/2); n = 0 gives
    𝒥_k(b).  Log space, b = 0 exact → (K,) + b.shape."""
    from scipy.special import gammaln
    b = np.asarray(b, dtype=float)
    k = np.arange(K, dtype=float).reshape((K,) + (1,) * b.ndim)
    pos = b > 0
    lb = np.log(np.where(pos, b, 1.0) / 2.0)
    out = np.exp((k + 0.5 * n) * lb - b / 2.0 - gammaln(k + n + 1))
    return np.where(pos, out, np.where((k == 0) & (n == 0), 1.0, 0.0))


@lru_cache(maxsize=None)
def _harmonic_triple(n: int, K: int, M: int) -> np.ndarray:
    """Λ⁽ⁿ⁾[k, p, m] = ∫ μ^n L_k⁽ⁿ⁾ L_p⁽ⁿ⁾ L_m e^{−μ} dμ / √((p+n)!/p!), k < K + M − 1, p < K, m < M (exact
    Gauss–Laguerre with weight μ^n e^{−μ}; zero for k > p + m) → (K + M − 1, K, M)."""
    from math import factorial
    from scipy.special import eval_genlaguerre, roots_genlaguerre
    nk = K + M - 1
    x, w = roots_genlaguerre(K + M + 2, n)
    Lk = np.array([eval_genlaguerre(k, n, x) for k in range(nk)])
    Lp = Lk[:K] / np.array([np.sqrt(factorial(p + n) / factorial(p)) for p in range(K)])[:, None]
    Lm = np.array([eval_genlaguerre(m, 0, x) for m in range(M)])
    out = np.einsum("ki,pi,mi,i->kpm", Lk, Lp, Lm, w)
    out[np.abs(out) < 1e-13 * np.abs(out).max()] = 0.0
    if n == 0:
        out[0] = np.eye(K, M)                            # ∫ L_p L_m e^{−μ} = δ exactly: 𝒬₀(0) is a selection, b = 0 exact
    out.setflags(write=False)
    return out


def bessel_galerkin(b: np.ndarray, n: int, M: int, K: int) -> np.ndarray:
    """𝒬ₙ(b)[p, m] = ⟨χ⁽ⁿ⁾_p, J_n(√(2bμ̂)) e^{inα} L_m⟩, the gyro-harmonic-n component of e^{−iq·v}φ_j on the sector basis
    ψ_ℓ v̂⊥ⁿ e^{inα} L_p⁽ⁿ⁾/√((p+n)!/p!) (the ℓ factor is the identity), p < K, m < M: Σ_k κ_k⁽ⁿ⁾(b) Λ⁽ⁿ⁾[k, p, m],
    a finite sum (k ≤ p + m) → (K, M) + b.shape.  𝒬₀(0) = [I; 0], 𝒬₁(0) = 0."""
    lam = _harmonic_triple(n, K, M)
    kap = bessel_kernel(b, n, lam.shape[0])
    return np.einsum("k...,kpm->pm...", kap, lam)


# ------------------------------------------------------------------------------------------------ Dougherty vectors
def dougherty_vectors(b: np.ndarray, M: int):
    """The gyroaveraged restoring functions' Laguerre coefficients (unsigned basis), each (M,) + b.shape:
    a∥ on ℓ = 1 (v̂∥J₀ = Σ (𝒥_m/√2) ψ₁L_m), a⊥ on ℓ = 0 (v̂⊥J₁(√(2bμ̂)) = √(b/2) Σ (𝒥_m − 𝒥_{m−1}) L_m, from
    √μ̂ J₁ = −√(2b) ∂_b J₀ and 𝒥_m′ = ½(𝒥_{m−1} − 𝒥_m)), a_E on ℓ = 0 ((μ̂ − 1)J₀: 2m𝒥_m − m𝒥_{m−1} − (m+1)𝒥_{m+1}) and
    on ℓ = 2 ((v̂∥² − ½)J₀ = Σ (𝒥_m/√2) ψ₂L_m).  At b = 0: a∥ = e₀/√2, a⊥ = 0, a_E0 = −e₁, a_E2 = e₀/√2 (the k⊥ = 0
    Dougherty restoring of velcro.collisions)."""
    b = np.asarray(b, dtype=float)
    J = basis.gyro_coefficients(b, M + 1)                                          # (M+1,) + b.shape
    Jm = J[:M]
    Jprev = np.concatenate([np.zeros_like(J[:1]), J[:M - 1]], axis=0)
    m = np.arange(M, dtype=float).reshape((M,) + (1,) * b.ndim)
    a_par = Jm / SQRT2
    a_perp = np.sqrt(0.5 * b) * (Jm - Jprev)
    a_E0 = 2 * m * Jm - m * Jprev - (m + 1) * J[1:M + 1]
    a_E2 = Jm / SQRT2
    return a_par, a_perp, a_E0, a_E2


def dougherty_gk_matrix(L: int, M: int, b: float, model: str = "dougherty") -> np.ndarray:
    """The gyroaveraged Dougherty (or Lenard–Bernstein) matrix at scalar b, (LM × LM), ν = 1: C(0) − b I + the J-dressed
    restoring minus its k⊥ = 0 value (the exact gyroaverage; GX's operator)."""
    from velcro.collisions import collision_matrix
    C = collision_matrix(model, L, M).astype(float).copy() - b * np.eye(L * M)
    if model == "lenard-bernstein":
        return C
    if model != "dougherty":
        raise ValueError(f"dougherty_gk_matrix: model {model!r}")
    return C + _dougherty_restoring(L, M, b) - _dougherty_restoring(L, M, 0.0)


def _dougherty_restoring(L: int, M: int, b: float) -> np.ndarray:
    a_par, a_perp, a_E0, a_E2 = (np.asarray(a, float) for a in dougherty_vectors(np.asarray(b, float), M))
    n = L * M
    R = np.zeros((n, n))
    e = lambda l, vec: np.concatenate([np.zeros(l * M), vec, np.zeros((L - l - 1) * M)]) if l < L else np.zeros(n)
    if L > 1:
        v = e(1, a_par)
        R += 2.0 * np.outer(v, v)
    v = e(0, a_perp)
    R += 2.0 * np.outer(v, v)
    v = e(0, a_E0) + (e(2, a_E2) if L > 2 else 0.0)
    R += (4.0 / 3.0) * np.outer(v, v)
    return R


# ------------------------------------------------------------------------------------------------ Landau: ρ(b), N⊥
@lru_cache(maxsize=None)
def _perp_tables(L: int, M: int, alpha: float = 1.0, nv: int = 96, nmu: int = 192):
    """(A1, A0, A2, N⊥, μ nodes, μ weights): ρ_i(b) = Σ_k w_k [J₁(a_k) A1[i,k] + q (J₀(a_k) A0[i,k] + (J₀ − J₂)(a_k) A2[i,k])]
    with a_k = √(2bμ_k), q = √(2b) (v̂∥ already integrated)."""
    x, wx, mu, wmu = _vmu_grid(nv, nmu)
    VP, MU = np.meshgrid(x, mu, indexing="ij")
    v2 = VP ** 2 + MU
    D1, D2 = test_part_speeds(np.sqrt(v2), alpha)
    phi, dpar, dmu = _basis_vmu(L, M, x, mu)
    sq = np.sqrt(MU)
    dperp = 2.0 * sq * dmu                                                          # ∂φ/∂v̂⊥ = 2√μ̂ ∂φ/∂μ̂
    S = D2 * dperp + (D1 - D2) * (sq / v2) * (VP * dpar + sq * dperp)
    A1 = 0.5 * np.einsum("nvu,v->nu", S, wx)
    A0 = 0.5 * np.einsum("nvu,v->nu", phi * D2, wx)
    A2 = 0.5 * np.einsum("nvu,v->nu", phi * (D1 - D2) * (MU / v2) * 0.5, wx)
    Nperp = float(0.5 * np.sum((D2 + (D1 - D2) * MU / (2.0 * v2)) * wx[:, None] * wmu[None, :]))
    return A1, A0, A2, Nperp, mu, wmu


def perp_restoring_vector(L: int, M: int, b: np.ndarray, alpha: float = 1.0, nmu: int = 192):
    """(ρ(b), N⊥): ρ (LM,) + b.shape, the gyroaveraged perpendicular-momentum restoring vector of the Landau model, and
    N⊥ = ⟨v_x, A v_x F₀⟩ (scalar); R⊥(b) = ρρᵀ/N⊥ (zero at b = 0)."""
    b = np.asarray(b, dtype=float)
    A1, A0, A2, Nperp, mu, wmu = _perp_tables(L, M, alpha, 96, nmu)
    bb = b.reshape(-1)
    a = np.sqrt(2.0 * bb[:, None] * mu[None, :])                                 # (N, nmu)
    q = np.sqrt(2.0 * bb)[:, None]
    J0, J1, J2 = jv(0, a), jv(1, a), jv(2, a)
    rho = (J1 * wmu) @ A1.T + q * ((J0 * wmu) @ A0.T + ((J0 - J2) * wmu) @ A2.T)  # (N, LM)
    return rho.T.reshape((L * M,) + b.shape), Nperp


def like_species_parts(L: int, M: int):
    """(C(0), C_T(0), C_F(0)) of the like-species Landau operator (velcro.landau_exact, read-only); C_F ≽ 0."""
    from velcro.landau import like_species_matrix
    from velcro.landau_exact import pair_exact
    C = like_species_matrix(L, M)
    CT, CF = pair_exact(int(L), int(M), 1.0, 1.0, 1.0)
    return C, CT, CF


LANDAU_SECTORS = (0, 1)        # the gyro-harmonics of the Landau field part that are kept (|n| ≥ 2 enter at O(b²))


def landau_sector_sizes(M: int) -> tuple:
    """Laguerre sizes of the n = 0 and n = 1 sector bases: K₀ = M + 2 (J₀φ_j's O(b) and part of its O(b²) beyond L_{M−1}),
    K₁ = M + 1; capped by the exact quadrature's range (velcro.landau.M_MAX)."""
    from velcro.landau import M_MAX
    K0, K1 = M + 2, M + 1
    if K0 > M_MAX:
        raise ValueError(f"physics.collision_flr with collision_model = 'landau' needs M ≤ {M_MAX - 2} (the n = 0 sector "
                         f"basis has M + 2 Laguerre functions; got M = {M})")
    return K0, K1


def landau_sectors(L: int, M: int):
    """(F⁽⁰⁾ on (L, K₀), F⁽¹⁾ on (L, K₁), C_F on (L, M)) — C_F taken as F⁽⁰⁾'s (L, M) block so that b = 0 is exact."""
    from velcro.landau_exact import field_sector_exact
    K0, K1 = landau_sector_sizes(M)
    F0 = field_sector_exact(int(L), int(K0), 0)
    F1 = field_sector_exact(int(L), int(K1), 1)
    sel = np.array([l * K0 + m for l in range(L) for m in range(M)])
    return F0, F1, np.asarray(F0)[np.ix_(sel, sel)]


def landau_gk_model(L: int, M: int, b: float) -> np.ndarray:
    """The Landau FLR operator at scalar b, (LM × LM) in ν̂ units: the exact test part C_T(0) − bK and
    the field part's gyro-harmonics n = 0 and n = ±1 on their sector bases, C(0) − bK + 𝒬₀ᵀF⁽⁰⁾𝒬₀ + 2𝒬₁ᵀF⁽¹⁾𝒬₁ − C_F."""
    C = like_species_parts(L, M)[0]
    K = classical_diffusion_matrix(L, M)
    F0, F1, CF = landau_sectors(L, M)
    K0, K1 = landau_sector_sizes(M)
    I = np.eye(L)
    Q0 = np.kron(I, bessel_galerkin(np.asarray(b, float), 0, M, K0))
    Q1 = np.kron(I, bessel_galerkin(np.asarray(b, float), 1, M, K1))
    return C - b * K + Q0.T @ F0 @ Q0 + 2.0 * Q1.T @ F1 @ Q1 - CF


def landau_gk_model_j0j1(L: int, M: int, b: float) -> np.ndarray:
    """The first (registered, F9-failed) model, kept for the record: C(0) − bK + [𝒬ᵀC_F𝒬 − C_F] + ρρᵀ/N⊥ — the n = 0
    average on the retained basis and Sugama's J₁ perpendicular-momentum projection only."""
    C, _, CF = like_species_parts(L, M)
    K = classical_diffusion_matrix(L, M)
    Q = np.kron(np.eye(L), j0_galerkin(np.asarray(b, float), M))
    rho, N = perp_restoring_vector(L, M, np.asarray(b, float))
    return C - b * K + (Q.T @ CF @ Q - CF) + np.outer(rho, rho) / N


def gk_matrix(model: str, L: int, M: int, b: float) -> np.ndarray:
    """The gyrokinetic collision matrix of ``model`` at scalar b (ν = 1; ν̂ for landau), (LM × LM)."""
    if model == "landau":
        return landau_gk_model(L, M, b)
    return dougherty_gk_matrix(L, M, b, model)


# ------------------------------------------------------------------------------------------------ run-time data
def check_config(cfg) -> None:
    """Refuse ``physics.collision_flr`` where its construction does not hold (instead of silently dropping physics)."""
    ph, nm = cfg.physics, cfg.numerics
    bad = []
    if ph.collision_model not in FLR_MODELS:
        bad.append(f"collision_model = {ph.collision_model!r} (no physical collisions to dress)")
    if ph.electrons != "kinetic" and ph.collisions_on != "h":
        bad.append("collisions_on = 'g' (the gyrokinetic operator acts on h)")
    if nm.krook_rate > 0 and nm.krook_width > 0:
        bad.append("Krook buffers (the FLR terms are applied in k_x: an x-periodic flux tube is needed)")
    if ph.Ti_profile or ph.n_profile:
        bad.append("T_i / n profiles (flux tube only: the symbols are taken at the reference surface)")
    from velcro.vlasov import exb_frame_kind
    if exb_frame_kind(cfg) == "shearing" and ph.exb_shear.gamma_E != 0.0:
        bad.append("physics.exb_shear in the shearing frame (b would follow k_x − θ_ky(t))")
    if bad:
        raise ValueError("physics.collision_flr does not support: " + "; ".join(bad))


def flr_data(model: str, L: int, M: int, b: np.ndarray) -> dict:
    """Per-point arrays for the right-hand side on a field b (e.g. (N_kx, N_ky, N_z)): ``b``; dougherty: ``vec``
    (4, M) + b.shape = (a∥, a⊥, a_E0, a_E2); landau: ``K``, ``CF`` (L, M, L, M), ``F0`` (L, K₀, L, K₀), ``F1``
    (L, K₁, L, K₁), ``q0`` (K₀, M) + b.shape, ``q1`` (K₁, M) + b.shape.  ``rad``: a bound on the spectral radius of the FLR increment
    max_b ‖C(b) − C(0)‖₂ over the field (for the CFL rate); ``rad0`` (dougherty / lenard-bernstein): ρ(C(0)), which the
    adiabatic CFL rate does not count without FLR (the Landau matrix's own radius is counted there as rad_coll)."""
    b = np.maximum(np.asarray(b, dtype=float), 0.0)
    bmax = float(b.max()) if b.size else 0.0
    out = dict(b=b)
    if model in ("dougherty", "lenard-bernstein"):
        from velcro.collisions import collision_matrix
        out["rad0"] = float(np.abs(np.linalg.eigvalsh(collision_matrix(model, L, M))).max())   # ρ(C(0)), symmetric
        if model == "dougherty":
            out["vec"] = np.stack(dougherty_vectors(b, M))
            v = np.stack(dougherty_vectors(np.linspace(0.0, max(bmax, 1e-12), 257), M))   # restoring norms on a b grid
            rest = 2 * np.sum(v[0] ** 2, 0) + 2 * np.sum(v[1] ** 2, 0) + (4 / 3) * (np.sum(v[2] ** 2, 0) + np.sum(v[3] ** 2, 0))
            out["rad"] = bmax + float(rest.max()) + 1.0 + 2.0          # + the k⊥ = 0 restoring removed (≤ 1 + 2)
        else:
            out["rad"] = bmax
        return out
    if model != "landau":
        raise ValueError(f"physics.collision_flr: no FLR terms for collision_model {model!r}")
    K = classical_diffusion_matrix(L, M)
    F0, F1, CF = landau_sectors(L, M)
    K0, K1 = landau_sector_sizes(M)
    out.update(K=np.asarray(K).reshape(L, M, L, M), CF=np.asarray(CF).reshape(L, M, L, M),
               F0=np.asarray(F0).reshape(L, K0, L, K0), F1=np.asarray(F1).reshape(L, K1, L, K1),
               q0=bessel_galerkin(b, 0, M, K0), q1=bessel_galerkin(b, 1, M, K1))
    # ‖𝒬ₙ‖ ≤ 1 (the sector components of a unit e^{−iq·v}φ), so ‖C(b) − C(0)‖ ≤ b‖K‖ + ‖F⁽⁰⁾‖ + 2‖F⁽¹⁾‖ + ‖C_F‖
    out["rad"] = (bmax * float(np.linalg.norm(K, 2)) + float(np.linalg.norm(F0, 2)) + 2.0 * float(np.linalg.norm(F1, 2))
                  + float(np.linalg.norm(CF, 2)))
    bad = [k for k, a in out.items() if not np.all(np.isfinite(np.asarray(a, dtype=float)))]
    if bad:                                                       # fail at build time, never inside a run
        raise FloatingPointError(f"physics.collision_flr: non-finite {bad}")
    return out


def landau_ei_flr_matrix(L: int, M: int, mass_e: float, temp_e: float, mass_i: float = 1.0, temp_i: float = 1.0):
    """K of the electrons' e–i test block (the ions' Maxwellian as the field species): the exact FLR of C_T,ei is
    −b_e K (velcro.kinetic's Landau e–i path has no field block)."""
    alpha = float(np.sqrt(temp_e * mass_i / (temp_i * mass_e)))
    return classical_diffusion_matrix(L, M, alpha)
