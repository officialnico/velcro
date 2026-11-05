"""An independent reference for the linearised Landau operator's matrix elements (tier-2 attack on velcro/landau.py).

Nothing is shared with velcro/landau.py except the definition of the basis (ψ_ℓ(v̂∥) orthonormal under e^{−v̂²}/√π,
L_m(μ̂) under e^{−μ̂}; F₀ = e^{−v̂²}/π^{3/2}).  velcro/landau.py uses the Rosenbluth form (closed-form Maxwellian
potentials for the test part, Legendre harmonics with incomplete-gamma radial integrals of monomials for the field part,
Gauss–Legendre on v ∈ (0, 9]).  Here the operator is the *Landau* form,

    C_ab(f_a, f_b) = (Γ_ab/2) ∂_i ∫ U_ij(v − v′) [f_b(v′) ∂_j f_a(v) − (m_a/m_b) f_a(v) ∂′_j f_b(v′)] d³v′,
    U_ij(u) = (δ_ij − û_iû_j)/u,

in its weak form.  With f_a = F_a χ (test part, f_b = F_b) and f_b = F_b χ_b (field part, f_a = F_a), θ = T_a/T_b,
α = v_ta/v_tb, (m_a/m_b) α² = θ and U·(v − v′) = 0:

    ⟨φ, C_T[F_a χ]⟩ = −½ ∬ F_a(v) F_b(v′) ∇φ(v)·U·[∇χ(v) − 2(1 − θ) v′ χ(v)]
    ⟨φ, C_F[F_b χ_b]⟩ = +½ ∬ F_a(v) F_b(v′) ∇φ(v)·U·[(m_a/m_b) ∇′χ_b(v′) + 2(1 − θ) v′ χ_b(v′)]

(in a's units, F_b of unit density and thermal speed 1/α, Γ_ab = 1: the ν̂_ab units of velcro/landau.py).  The
Gaussian F_a F_b′ = (α³/π³) e^{−(1+α²)V² − κu²} with u = v − v′, V = v − κu, κ = α²/(1+α²), so every integrand is a
polynomial times a Gaussian in (V, u) times U(u) ∝ 1/u:

* V by tensor Gauss–Hermite (exact: the integrand is a polynomial in V);
* |u| by Gauss–Laguerre in t = κu² (u² du/u e^{−κu²} = e^{−t} dt/2κ; after the V integral the integrand is even in u);
* the direction of u by Gauss–Legendre in cos θ_u, with the azimuth fixed (the V-integrated integrand is invariant under
  rotations about z because the basis is axisymmetric) — exact for polynomials.

No singular integrand is ever sampled (the 1/u is absorbed in the radial weight) and every rule is exact, so the result
is exact to rounding.  The rounding is benign: U is positive semi-definite and the weights are positive, so
Σ w |a·U·b| ≤ √(Σ w a·U·a)(Σ w b·U·b) — the summed magnitudes are bounded by the diagonal elements themselves.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln

SQRT_PI = np.sqrt(np.pi)


# ------------------------------------------------------------------------------------------------ 1-D tables
def psi_table(L: int, x):
    """ψ_ℓ(x), ψ_ℓ′(x) for ℓ < L by the orthonormal three-term recurrence → (L, …) each."""
    x = np.asarray(x, float)
    P = np.zeros((L,) + x.shape)
    P[0] = 1.0
    if L > 1:
        P[1] = np.sqrt(2.0) * x
    for l in range(1, L - 1):
        P[l + 1] = np.sqrt(2.0 / (l + 1)) * x * P[l] - np.sqrt(l / (l + 1)) * P[l - 1]
    D = np.zeros_like(P)
    for l in range(1, L):
        D[l] = np.sqrt(2.0 * l) * P[l - 1]
    return P, D


def lag_table(M: int, z, a: float = 0.0):
    """L^{(a)}_m(z) for m < M and its derivative −L^{(a+1)}_{m−1}(z) → (M, …) each."""
    z = np.asarray(z, float)

    def vals(MM, aa):
        P = np.zeros((MM,) + z.shape)
        if MM == 0:
            return P
        P[0] = 1.0
        if MM > 1:
            P[1] = 1.0 + aa - z
        for m in range(1, MM - 1):
            P[m + 1] = ((2 * m + 1 + aa - z) * P[m] - (m + aa) * P[m - 1]) / (m + 1)
        return P
    P = vals(M, a)
    D = np.zeros_like(P)
    if M > 1:
        D[1:] = -vals(M - 1, a + 1.0)
    return P, D


# ------------------------------------------------------------------------------------------------ bases on a product grid
def hl_eval(L: int, M: int):
    """The Hermite–Laguerre basis ψ_ℓ(v_z) L_m(v_x² + v_y²), index ℓ·M + m: (value, ∂_x, ∂_y, ∂_z) on the product grid
    v_z ⊗ v_x ⊗ v_y, each (L·M, n_z·n_x·n_y)."""
    def ev(vx, vy, vz):
        ps, dps = psi_table(L, vz)
        rho2 = vx[:, None] ** 2 + vy[None, :] ** 2
        lg, dlg = lag_table(M, rho2)
        n = L * M
        g = np.einsum("lk,mij->lmkij", ps, lg).reshape(n, -1)
        gx = np.einsum("lk,mij->lmkij", ps, 2 * vx[None, :, None] * dlg).reshape(n, -1)
        gy = np.einsum("lk,mij->lmkij", ps, 2 * vy[None, None, :] * dlg).reshape(n, -1)
        gz = np.einsum("lk,mij->lmkij", dps, lg).reshape(n, -1)
        return g, gx, gy, gz
    return ev


def sonine_eval(K: int):
    """The l = 1 Sonine basis φ_k = v_z L^{(3/2)}_k(v²), k ≤ K (value and gradient on the product grid)."""
    def ev(vx, vy, vz):
        v2 = vz[:, None, None] ** 2 + vx[None, :, None] ** 2 + vy[None, None, :] ** 2
        S, dS = lag_table(K + 1, v2, 1.5)
        Z = vz[:, None, None] * np.ones_like(v2)
        X = vx[None, :, None] * np.ones_like(v2)
        Y = vy[None, None, :] * np.ones_like(v2)
        g = (Z * S).reshape(K + 1, -1)
        gx = (Z * dS * 2 * X).reshape(K + 1, -1)
        gy = (Z * dS * 2 * Y).reshape(K + 1, -1)
        gz = (S + Z * dS * 2 * Z).reshape(K + 1, -1)
        return g, gx, gy, gz
    return ev


# ------------------------------------------------------------------------------------------------ the core
def landau_pair(evalf, n: int, deg_z: int, deg_xy: int, deg_tot: int, alpha: float = 1.0, mass_ratio: float = 1.0,
                theta: float = 1.0, evalf_b=None, extra: int = 2, full_only: bool = False):
    """(C_T, C_F) of species a on species b, ν̂_ab units, rows = test functions (a's basis), columns = sources (C_T:
    a's basis; C_F: b's basis evaluated at α v′).  ``deg_z`` / ``deg_xy`` bound each basis function's polynomial degree in
    v_z / in v_x (and v_y), ``deg_tot`` its total degree; the rules are chosen exact for the products (+ ``extra``).
    ``full_only`` (like species): return C_T + C_F from one product, −¼ ⟨(a − b)·U·(a − b)⟩."""
    evalf_b = evalf_b or evalf
    a2 = alpha ** 2
    kap = a2 / (1.0 + a2)
    th = 1.0 - theta
    # per-component degree of a source term: ≤ deg (+1 for the v′ factor when θ ≠ 1); of the product: twice that
    dz = deg_z + (1 if th != 0 else 0)
    dxy = deg_xy + (1 if th != 0 else 0)
    dt = deg_tot + (1 if th != 0 else 0)
    nz = dz + 1 + extra
    nxy = dxy + 1 + extra
    nt = dt // 2 + 2 + extra            # even polynomial of degree ≤ 2·dt in u → degree ≤ dt in t
    nc = dt + 2 + extra                 # polynomial of degree ≤ 2·dt + 2 in cos θ_u
    xz, wz = hermgauss(nz)
    xx, wx = hermgauss(nxy)
    s = 1.0 / np.sqrt(1.0 + a2)
    Vz, Vx, Vy = xz * s, xx * s, xx * s
    wV = (np.einsum("k,i,j->kij", wz, wx, wx) * s ** 3).reshape(-1)
    t, wt = laggauss(nt)
    c, wc = leggauss(nc)
    pref = alpha ** 3 / (2.0 * np.pi ** 3) * 2.0 * np.pi / (2.0 * kap)
    CT = np.zeros((n, n))
    CF = np.zeros((n, n))
    ones = np.ones((nz, nxy, nxy))
    for it in range(nt):
        u = np.sqrt(t[it] / kap)
        for ic in range(nc):
            cz = c[ic]
            sx = np.sqrt(max(1.0 - cz * cz, 0.0))
            ux, uz = u * sx, u * cz
            w = wt[it] * wc[ic] * wV
            g, gx, gy, gz = evalf(Vx + kap * ux, Vy, Vz + kap * uz)                 # test function side: v = V + κu
            pa1 = gx * cz - gz * sx                                                  # ∇φ·e₁, e₁ = (c, 0, −s)
            pa2 = gy                                                                 # ∇φ·e₂, e₂ = (0, 1, 0)
            vpx, vpz = Vx - (1 - kap) * ux, Vz - (1 - kap) * uz                      # v′ = V − (1 − κ)u
            h, hx, hy, hz = evalf_b(alpha * vpx, alpha * Vy, alpha * vpz)
            if full_only:                                                            # like species: Δ = a(v) − a(v′)
                d1 = pa1 - (hx * cz - hz * sx)
                d2 = pa2 - hy
                CT += -0.5 * pref * ((d1 * w) @ d1.T + (d2 * w) @ d2.T)
                continue
            if th != 0.0:
                ve1 = ((vpx[None, :, None] * cz) * ones - (vpz[:, None, None] * sx) * ones).reshape(-1)
                ve2 = (Vy[None, None, :] * ones).reshape(-1)
            sT1 = gx * cz - gz * sx
            sT2 = gy
            sF1 = mass_ratio * alpha * (hx * cz - hz * sx)
            sF2 = mass_ratio * alpha * hy
            if th != 0.0:
                sT1 = sT1 - 2 * th * ve1 * g
                sT2 = sT2 - 2 * th * ve2 * g
                sF1 = sF1 + 2 * th * ve1 * h
                sF2 = sF2 + 2 * th * ve2 * h
            pw1, pw2 = pa1 * w, pa2 * w
            CT -= pref * (pw1 @ sT1.T + pw2 @ sT2.T)
            CF += pref * (pw1 @ sF1.T + pw2 @ sF2.T)
    if full_only:
        return CT
    return CT, CF


def hl_degrees(L: int, M: int):
    return L - 1, 2 * (M - 1), (L - 1) + 2 * (M - 1)


def like_reference(L: int, M: int, extra: int = 2):
    """(C_T, C_F) of the like-species operator in Velcro's basis, ν̂ units."""
    dz, dxy, dt = hl_degrees(L, M)
    return landau_pair(hl_eval(L, M), L * M, dz, dxy, dt, extra=extra)


def like_reference_full(L: int, M: int, extra: int = 2):
    dz, dxy, dt = hl_degrees(L, M)
    return landau_pair(hl_eval(L, M), L * M, dz, dxy, dt, extra=extra, full_only=True)


def pair_reference(L: int, M: int, mass_a, temp_a, mass_b, temp_b, extra: int = 2):
    """(C_T,ab, C_F,ab) with velcro.landau.pair_matrices' arguments."""
    alpha = float(np.sqrt(temp_a * mass_b / (temp_b * mass_a)))
    dz, dxy, dt = hl_degrees(L, M)
    return landau_pair(hl_eval(L, M), L * M, dz, dxy, dt, alpha=alpha, mass_ratio=float(mass_a / mass_b),
                       theta=float(temp_a / temp_b), extra=extra)


def lorentz_reference(evalf, n: int, deg_tot: int, extra: int = 2):
    """−½∫F₀ ∇φ·U(v)·∇χ d³v (the m_e/m_i → 0 e–i operator, ν̂_ei units): Gauss–Laguerre in t = v², Gauss–Legendre in cos θ."""
    nt = deg_tot + 2 + extra
    nc = deg_tot + 2 + extra
    t, wt = laggauss(nt)
    c, wc = leggauss(nc)
    C = np.zeros((n, n))
    for it in range(nt):
        v = np.sqrt(t[it])
        for ic in range(nc):
            cz = c[ic]
            sx = np.sqrt(max(1.0 - cz * cz, 0.0))
            g, gx, gy, gz = evalf(np.array([v * sx]), np.array([0.0]), np.array([v * cz]))
            p1 = (gx * cz - gz * sx)[:, 0]
            p2 = gy[:, 0]
            w = wt[it] * wc[ic]
            C += -0.5 / np.pi ** 1.5 * 2 * np.pi * 0.5 * w * (np.outer(p1, p1) + np.outer(p2, p2))
    return C


# ------------------------------------------------------------------------------------------------ Sonine l = 1 problems
def sonine_gram(K: int) -> np.ndarray:
    """∫ φ_i φ_j F₀ d³v for φ_k = v_z L^{(3/2)}_k(v²): (2/(3√π)) Γ(k + 5/2)/k! δ_ij."""
    k = np.arange(K + 1)
    return np.diag(2.0 / (3.0 * SQRT_PI) * np.exp(gammaln(k + 2.5) - gammaln(k + 1)))


def sonine_lorentz(K: int) -> np.ndarray:
    """⟨φ_i, (1/2v³)∂_ξ(1−ξ²)∂_ξ φ_j F₀⟩ = −(2/(3√π)) ∫ L_i L_j e^{−t} dt (closed form by Gauss–Laguerre)."""
    t, w = laggauss(2 * K + 4)
    S, _ = lag_table(K + 1, t, 1.5)
    return -(2.0 / (3.0 * SQRT_PI)) * (S * w) @ S.T


def sonine_like(K: int, extra: int = 2) -> np.ndarray:
    """The like-species Landau operator (test + field) on the l = 1 Sonine basis, k ≤ K, ν̂ units."""
    deg = 2 * K + 1
    return landau_pair(sonine_eval(K), K + 1, deg, deg, deg, extra=extra, full_only=True)


TAU_E = 3 * SQRT_PI / 4             # ν̂_ee τ_e
TAU_I = 3 * SQRT_PI / (2 * np.sqrt(2.0))


def sonine_transport(Cee: np.ndarray, Cei: np.ndarray, K: int):
    """σ/τ_e, σ/σ_L, κ_e (zero current, Z = 1) and κ_i on the first K + 1 Sonine functions (the like-species ions without
    φ₀ = v_z, the momentum null direction)."""
    Gm = sonine_gram(K)
    A = (Cee + Cei)[:K + 1, :K + 1]
    dE = np.zeros(K + 1); dE[0] = 2 * Gm[0, 0]                     # ⟨φ_i, 2 v_z⟩
    dT = np.zeros(K + 1)
    if K >= 1:
        dT[1] = -Gm[1, 1]                                            # v_z(v² − 5/2) = −φ₁
    j = lambda c: c[0] * Gm[0, 0]                                     # ∫ v_z f
    fE = np.linalg.solve(A, -dE)
    out = dict(sigma=j(fE) / TAU_E)
    Al = Cei[:K + 1, :K + 1]
    out["sigma_L"] = j(np.linalg.solve(Al, -dE)) / TAU_E
    out["sigma_ratio"] = out["sigma"] / out["sigma_L"]
    if K >= 1:
        fT = np.linalg.solve(A, -dT)
        f = fT - j(fT) / j(fE) * fE
        out["kappa_e"] = 2 * (f @ dT) / TAU_E
        Ai = Cee[1:K + 1, 1:K + 1]
        fi = np.linalg.solve(Ai, -dT[1:])
        out["kappa_i"] = 2 * (fi @ dT[1:]) / TAU_I
    return out


# ------------------------------------------------------------------------------------------------ Velcro-basis transport
def hl_transport(Cee: np.ndarray, Cei: np.ndarray, L: int, M: int):
    """The constrained steady responses of tests/test_landau_operator.py (same definitions) for any (Cee, Cei)."""
    n = L * M
    e = lambda l, m: np.eye(n)[l * M + m]
    mom = np.array([e(0, 0), e(1, 0) / np.sqrt(2.0), 1.5 * e(0, 0) + e(2, 0) / np.sqrt(2.0) - e(0, 1)])

    def solve(C, drive, constraints):
        Q, _ = np.linalg.qr(np.asarray(constraints).T)
        P = np.eye(n) - Q @ Q.T
        A = P @ C @ P + (np.eye(n) - P)
        return P @ np.linalg.solve(A, P @ (-drive))
    e10 = e(1, 0)
    dE = np.sqrt(2.0) * e10
    j = lambda f: f @ e10 / np.sqrt(2.0)
    Ce = Cee + Cei
    out = {}
    fE = solve(Ce, dE, mom[[0, 2]])
    out["sigma"] = j(fE) / TAU_E
    out["sigma_L"] = j(solve(Cei, dE, mom[[0, 2]])) / TAU_E
    out["sigma_ratio"] = out["sigma"] / out["sigma_L"]
    v1 = e10 / np.sqrt(2)
    v3 = (np.sqrt(48) * e(3, 0) + 12 * e10 / np.sqrt(2)) / 8
    vmu = (e10 - e(1, 1)) / np.sqrt(2)
    dT = v3 + vmu - 2.5 * v1
    fT = solve(Ce, dT, mom[[0, 2]])
    f = fT - (v1 @ fT) / (v1 @ fE) * fE
    out["kappa_e"] = (f @ dT) * 2 / TAU_E
    fi = solve(Cee, dT, mom)
    out["kappa_i"] = (fi @ dT) * 2 / TAU_I
    return out
