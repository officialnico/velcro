"""The linearised Landau operator's matrix elements by exact quadrature of the Landau form (the production route of
:mod:`velcro.landau` since the tier-2 review; the review's independent reference, tests/attack/landau_ref.py).

In the weak form, with f_a = F_a χ (test part) or f_b = F_b χ_b (field part), θ = T_a/T_b, α = v_ta/v_tb, (m_a/m_b)α² = θ:

    ⟨φ, C_T[F_a χ]⟩   = −½ ∬ F_a(v) F_b(v′) ∇φ(v)·U(v − v′)·[∇χ(v) − 2(1 − θ) v′ χ(v)]
    ⟨φ, C_F[F_b χ_b]⟩ = +½ ∬ F_a(v) F_b(v′) ∇φ(v)·U(v − v′)·[(m_a/m_b) ∇′χ_b(v′) + 2(1 − θ) v′ χ_b(v′)]
    U_ij(u) = (δ_ij − û_iû_j)/u,

in species a's thermal units, F_b of unit density and thermal speed 1/α, Γ_ab = 1 (ν̂_ab units).  F_a F_b′ is a Gaussian in
the centre-of-mass and relative velocities V = v − κu, u = v − v′ (κ = α²/(1 + α²)), so every integrand is a polynomial in
V times a Gaussian, times U(u) ∝ 1/u: tensor Gauss–Hermite in V, Gauss–Laguerre in t = κu² (the 1/u enters the radial
weight: no singular sample), Gauss–Legendre in cos θ_u with the azimuth fixed (the basis is axisymmetric) — every rule exact
for the polynomial degrees involved, so the matrix is exact to rounding at any (L, M).  Against closed forms
(tests/attack/test_attack_landau_matrix.py): Hirshman–Sigmar's M⁰⁰ friction for four species pairs to 12–13 digits, the
Lorentz (1,0) element −4/(3√π) to 1e-15, symmetric and conserving n, u∥, E to 2e-16; Braginskii's two-polynomial κ_e
3.1616, κ_i 125/32 exactly.  The Rosenbluth-potential route of velcro.landau (monomial field part) agrees to 1e-13 up to
(12, 6); it loses precision beyond (field part 8.8e-6 of max|C| at (16, 8), wrong at (20, 10); cross field blocks wrong
from (8, 4)).  Cost on one CPU core: (16, 8) ≈ 14 s, (20, 10) ≈ 42 s, (24, 12) ≈ 100 s — cached per argument set.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss


def _psi_table(L: int, x):
    """ψ_ℓ(x) (orthonormal under e^{−x²}/√π) and ψ_ℓ′(x), ℓ < L, by the three-term recurrence → (L, …) each."""
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


def _lag_table(M: int, z, a: float = 0.0):
    """L^{(a)}_m(z), m < M, and its derivative −L^{(a+1)}_{m−1}(z) → (M, …) each."""
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


def _hl_eval(L: int, M: int):
    """ψ_ℓ(v_z) L_m(v_x² + v_y²), index ℓ·M + m: (value, ∂_x, ∂_y, ∂_z) on the product grid v_z ⊗ v_x ⊗ v_y."""
    def ev(vx, vy, vz):
        ps, dps = _psi_table(L, vz)
        rho2 = vx[:, None] ** 2 + vy[None, :] ** 2
        lg, dlg = _lag_table(M, rho2)
        n = L * M
        g = np.einsum("lk,mij->lmkij", ps, lg).reshape(n, -1)
        gx = np.einsum("lk,mij->lmkij", ps, 2 * vx[None, :, None] * dlg).reshape(n, -1)
        gy = np.einsum("lk,mij->lmkij", ps, 2 * vy[None, None, :] * dlg).reshape(n, -1)
        gz = np.einsum("lk,mij->lmkij", dps, lg).reshape(n, -1)
        return g, gx, gy, gz
    return ev


def landau_pair(evalf, n: int, deg_z: int, deg_xy: int, deg_tot: int, alpha: float = 1.0, mass_ratio: float = 1.0,
                theta: float = 1.0, evalf_b=None, extra: int = 2, full_only: bool = False):
    """(C_T, C_F) of species a on species b in ν̂_ab units — rows: test functions (a's basis); columns: sources (C_T: a's
    basis; C_F: b's basis, evaluated at α v′).  ``deg_z`` / ``deg_xy`` / ``deg_tot`` bound each basis function's degree in
    v_z, in v_x (and v_y) and in total; the rules are exact for the products (+ ``extra`` nodes).  ``full_only`` (like
    species, θ = 1, α = 1): C_T + C_F from one symmetric product, −¼⟨(∇φ(v) − ∇φ(v′))·U·(∇χ(v) − ∇χ(v′))⟩."""
    evalf_b = evalf_b or evalf
    a2 = alpha ** 2
    kap = a2 / (1.0 + a2)
    th = 1.0 - theta
    dz = deg_z + (1 if th != 0 else 0)
    dxy = deg_xy + (1 if th != 0 else 0)
    dt = deg_tot + (1 if th != 0 else 0)
    nz = dz + 1 + extra
    nxy = dxy + 1 + extra
    nt = dt // 2 + 2 + extra            # an even polynomial of degree ≤ 2·dt in u: degree ≤ dt in t
    nc = dt + 2 + extra                 # a polynomial of degree ≤ 2·dt + 2 in cos θ_u
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
            g, gx, gy, gz = evalf(Vx + kap * ux, Vy, Vz + kap * uz)                 # the test function at v = V + κu
            pa1 = gx * cz - gz * sx                                                  # ∇φ·e₁, e₁ = (c, 0, −s) ⊥ u
            pa2 = gy                                                                 # ∇φ·e₂, e₂ = (0, 1, 0)
            vpx, vpz = Vx - (1 - kap) * ux, Vz - (1 - kap) * uz                      # v′ = V − (1 − κ)u
            h, hx, hy, hz = evalf_b(alpha * vpx, alpha * Vy, alpha * vpz)
            if full_only:
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


def _degrees(L: int, M: int):
    return L - 1, 2 * (M - 1), (L - 1) + 2 * (M - 1)


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)              # cached: a caller writing in place would poison every later call
    return a


@lru_cache(maxsize=None)
def like_exact(L: int, M: int) -> np.ndarray:
    """C_aa = C_T,aa + C_F,aa in ν̂_aa units, (LM × LM), index ℓ·M + m; read-only (cached)."""
    dz, dxy, dt = _degrees(L, M)
    C = landau_pair(_hl_eval(L, M), L * M, dz, dxy, dt, full_only=True)
    return _frozen(0.5 * (C + C.T))      # symmetric by construction to rounding; made exactly symmetric


def _hl_eval_harmonic(L: int, K: int, n: int, sign: int):
    """χ_ℓp = ψ_ℓ(v_z) w^n L_p^{(n)}(v_x² + v_y²)/√((p+n)!/p!), w = v_x + i·sign·v_y (gyro-harmonic e^{i·sign·nα},
    orthonormal under F₀ with the complex conjugate), index ℓ·K + p: (value, ∂_x, ∂_y, ∂_z) on the product grid
    v_z ⊗ v_x ⊗ v_y, complex."""
    from math import factorial
    norm = np.array([1.0 / np.sqrt(factorial(p + n) / factorial(p)) for p in range(K)])

    def ev(vx, vy, vz):
        ps, dps = _psi_table(L, vz)
        X, Y = vx[:, None], vy[None, :]
        w = X + 1j * sign * Y
        wn = w ** n
        dwn = n * w ** (n - 1) if n else np.zeros_like(w)
        lg, dlg = _lag_table(K, X ** 2 + Y ** 2, float(n))
        lg, dlg = lg * norm[:, None, None], dlg * norm[:, None, None]
        f = wn[None] * lg
        fx = dwn[None] * lg + wn[None] * dlg * 2 * X[None]
        fy = 1j * sign * dwn[None] * lg + wn[None] * dlg * 2 * Y[None]
        m = L * K
        g = np.einsum("lk,mij->lmkij", ps, f).reshape(m, -1)
        gx = np.einsum("lk,mij->lmkij", ps, fx).reshape(m, -1)
        gy = np.einsum("lk,mij->lmkij", ps, fy).reshape(m, -1)
        gz = np.einsum("lk,mij->lmkij", dps, f).reshape(m, -1)
        return g, gx, gy, gz
    return ev


@lru_cache(maxsize=None)
def field_sector_exact(L: int, K: int, n: int, extra: int = 2) -> np.ndarray:
    """The like-species Landau field part on the gyro-harmonic-n sector, F⁽ⁿ⁾_ij = B_F[χ̄_i, χ_j] (χ of
    :func:`_hl_eval_harmonic`; B_F[a, c] = ∫ a C_F[c F₀] d³v, ν̂ units), (L·K × L·K), real symmetric positive
    semi-definite; n = 0 is pair_exact's C_F on (L, K).  The bilinear form is invariant under a common rotation of V and u
    about b̂ when the two functions carry opposite harmonics, so :func:`landau_pair`'s fixed-azimuth rule applies; every
    rule is exact for the polynomial degrees (deg_xy = n + 2(K − 1)).  Read-only (cached)."""
    evt = _hl_eval_harmonic(L, K, n, -1)                         # test functions χ̄ (e^{−inα})
    evs = _hl_eval_harmonic(L, K, n, +1)                         # sources χ (e^{+inα})
    dz, dxy = L - 1, n + 2 * (K - 1)
    dt = dz + dxy
    m = L * K
    kap, s = 0.5, 1.0 / np.sqrt(2.0)
    nz, nxy = dz + 1 + extra, dxy + 1 + extra
    nt, nc = dt // 2 + 2 + extra, dt + 2 + extra
    xz, wz = hermgauss(nz)
    xx, wx = hermgauss(nxy)
    Vz, Vx, Vy = xz * s, xx * s, xx * s
    wV = (np.einsum("k,i,j->kij", wz, wx, wx) * s ** 3).reshape(-1)
    t, wt = laggauss(nt)
    c, wc = leggauss(nc)
    pref = 1.0 / (2.0 * np.pi ** 3) * 2.0 * np.pi / (2.0 * kap)
    CF = np.zeros((m, m), dtype=complex)
    for it in range(nt):
        u = np.sqrt(t[it] / kap)
        for ic in range(nc):
            cz = c[ic]
            sx = np.sqrt(max(1.0 - cz * cz, 0.0))
            ux, uz = u * sx, u * cz
            w = wt[it] * wc[ic] * wV
            g, gx, gy, gz = evt(Vx + kap * ux, Vy, Vz + kap * uz)
            h, hx, hy, hz = evs(Vx - (1 - kap) * ux, Vy, Vz - (1 - kap) * uz)
            pa1, pa2 = gx * cz - gz * sx, gy
            sF1, sF2 = hx * cz - hz * sx, hy
            CF += pref * ((pa1 * w) @ sF1.T + (pa2 * w) @ sF2.T)
    if np.abs(CF.imag).max() > 1e-12 * np.abs(CF.real).max():
        raise FloatingPointError(f"field_sector_exact: imaginary part {np.abs(CF.imag).max():.1e}")
    return _frozen(0.5 * (CF.real + CF.real.T))


def like_exact_gk(L: int, M: int, b: float, extra_t: int = 40, extra_c: int = 40, part: str = "full") -> np.ndarray:
    """The exact gyroaverage of the like-species Landau operator at b = k⊥²ρ²/2 (velcro/collisions_flr.py; every Bessel
    order — Pan & Ernst 2019's operator): C_ij(b) = ∫ φ_i e^{iq·v} C[φ_j e^{−iq·v} F₀] d³v, q² = 2b, ν̂ units, (LM × LM).
    In the symmetric form of :func:`landau_pair` (``full_only``) the test function a = φ_i e^{iq·v} and the source
    c = φ_j e^{−iq·v} give (∇a − ∇′a′)·P⊥(u)·(∇c − ∇′c′) with ∇a = e^{iq·v}(∇φ_i + iqφ_i): the products at one point carry
    no phase (the test part; its q-terms are the classical diffusion), the cross products e^{±iq·u}, u = v − v′.  Rotating
    V and u together about b̂ (the basis is axisymmetric) moves q to the azimuth −β of a fixed u in the x–z plane; the β
    average is closed form: ⟨e^{iz cos β}⟩ = J₀(z), ⟨e^{iz cos β} cos β⟩ = iJ₁, ⟨e^{iz cos β} cos²β⟩ = (J₀ − J₂)/2,
    ⟨e^{iz cos β} sin²β⟩ = (J₀ + J₂)/2 (odd sin β terms vanish), z = q u sin θ_u.  The V rule stays exact (the phases
    depend on u only); the t and cos θ_u rules get ``extra_t`` / ``extra_c`` nodes beyond the polynomial-exact count
    (the Bessel factors are entire, so the error falls spectrally — tests compare two node counts).  ``part``: "full",
    "test" (the phase-free products: C_T(0) − b K exactly) or "field" (the cross products)."""
    dz, dxy, dt = _degrees(L, M)
    n = L * M
    evalf = _hl_eval(L, M)
    kap, s = 0.5, 1.0 / np.sqrt(2.0)
    nz, nxy = dz + 3, dxy + 3
    nt, nc = dt // 2 + 4 + extra_t, dt + 4 + extra_c
    xz, wz = hermgauss(nz)
    xx, wx = hermgauss(nxy)
    Vz, Vx, Vy = xz * s, xx * s, xx * s
    wV = (np.einsum("k,i,j->kij", wz, wx, wx) * s ** 3).reshape(-1)
    t, wt = laggauss(nt)
    c, wc = leggauss(nc)
    pref = 1.0 / (2.0 * np.pi ** 3) * 2.0 * np.pi / (2.0 * kap)
    q = float(np.sqrt(2.0 * b))
    from scipy.special import jv
    out = np.zeros((n, n))
    for it in range(nt):
        u = np.sqrt(t[it] / kap)
        for ic in range(nc):
            cz = c[ic]
            sx = np.sqrt(max(1.0 - cz * cz, 0.0))
            ux, uz = u * sx, u * cz
            w = wt[it] * wc[ic] * wV
            g, gx, gy, gz = evalf(Vx + kap * ux, Vy, Vz + kap * uz)                       # φ at v = V + u/2
            h, hx, hy, hz = evalf(Vx - (1 - kap) * ux, Vy, Vz - (1 - kap) * uz)          # φ at v′ = V − u/2
            g1, g2 = gx * cz - gz * sx, gy
            h1, h2 = hx * cz - hz * sx, hy
            acc = np.zeros((n, n))
            if part in ("full", "test"):
                acc += (g1 * w) @ g1.T + (g2 * w) @ g2.T + (h1 * w) @ h1.T + (h2 * w) @ h2.T
                if q:
                    acc += 0.5 * q * q * (cz * cz + 1.0) * ((g * w) @ g.T + (h * w) @ h.T)
            if part in ("full", "field"):
                z = q * u * sx
                J0, J1, J2 = (jv(k, z) for k in range(3)) if q else (1.0, 0.0, 0.0)
                X = J0 * ((g1 * w) @ h1.T + (g2 * w) @ h2.T)
                if q:
                    X += q * cz * J1 * ((g1 * w) @ h.T - (g * w) @ h1.T)
                    X += 0.5 * q * q * (cz * cz * (J0 - J2) + J0 + J2) * ((g * w) @ h.T)
                acc -= X + X.T                                                              # the two cross products
            out += -0.5 * pref * acc
    return 0.5 * (out + out.T)


@lru_cache(maxsize=None)
def pair_exact(L: int, M: int, alpha: float, mass_ratio: float, theta: float):
    """(C_T,ab, C_F,ab) in ν̂_ab units for species a colliding with b: ``alpha`` = v_ta/v_tb, ``mass_ratio`` = m_a/m_b,
    ``theta`` = T_a/T_b; read-only (cached)."""
    dz, dxy, dt = _degrees(L, M)
    CT, CF = landau_pair(_hl_eval(L, M), L * M, dz, dxy, dt, alpha=alpha, mass_ratio=mass_ratio, theta=theta)
    return _frozen(CT), _frozen(CF)
