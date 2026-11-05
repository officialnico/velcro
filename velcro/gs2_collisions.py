"""GS2's model collision operator (``physics.collision_model = "gs2"``) in the Hermite–Laguerre basis, at k⊥ = 0.

GS2 (collisions.fpp; Abel et al. 2008 PoP 15 122509, Barnes et al. 2009 PoP 16 072107) collides each species with
itself and the electrons with the ions through two split operators, each carrying its own conserving terms.  With v̂ the
speed in v_ts = √(2T_s/m_s) units (GS2's E = v̂²), G the Chandrasekhar function and ν̂ = vnewk_s (Helander & Sigmar's
ν̂_ss, Velcro's ``nu``):

* **Lorentz** (pitch-angle scattering): ``C_L[h] = ½ν_D L[h] + ν_D^ss v̂∥ F₀ U_L``, L = ∂_ξ(1 − ξ²)∂_ξ,
  ν_D = ν_D^ss + ν_D^ei (electrons) with ν_D^ss = ν̂ (erf v̂ − G)/v̂³ (GS2 ``vnew_D``, ``hee``) and ν_D^ei = Z_eff ν̂_ee/v̂³
  (GS2's ``zeff`` term: the infinitely massive ion limit; Velcro's e–i rate Z_i ν_e times :func:`ei_matrix`);
  U_L = ∫ ν_D^ss v̂∥ h / ∫ ν_D^ss v̂∥² F₀ (``init_lorentz_conserve``'s s0: like-species momentum restored, the e–i loss
  is friction on the ions at rest — GS2's drag toward the ion flow is on only at β > 0, refused here).
* **energy diffusion**: ``C_E[h] = (1/2v̂²)∂_v̂(ν_∥ v̂⁴ F₀ ∂_v̂(h/F₀)) + (ν_s − ν_D^ss) v̂∥ F₀ U_E + ν_E F₀ Q_E`` with
  ν_∥ = 2ν̂G/v̂³, ν_s = 4ν̂G/v̂ (``vnew_s``), U_E = ∫(ν_s − ν_D)v̂∥ h / ∫(ν_s − ν_D)v̂∥² F₀ (``bs0``) and
  ν_E = v̂²(ν_s(2 − 1/2v̂²) − 2ν_D) (``vnew_E``), Q_E = ∫ν_E h / ∫ v̂² ν_E F₀ (``bz0``).  The first term is exactly the
  energy-diffusion-plus-drag part of the like-species test-particle Landau operator; ν_E = −(1/v̂²F₀)∂_v̂(ν_∥v̂⁵F₀)
  identically, so the restoring terms make energy (and momentum, with ν_s − ν_D) conservation exact.
* GS2's ``collision_model``: 'default' = both; 'lorentz' = the Lorentz part (with the e–i scattering); 'ediffuse' = the
  energy-diffusion part alone (no pitch-angle scattering at all, e–i included).  ``const_v`` evaluates every frequency of the
  Lorentz part at v̂ = 1 (GS2 ``init_vnew`` local_energy = 1): only with 'lorentz', because GS2 keeps its energy-diffusion
  operator velocity-dependent under const_v while its restoring frequencies go constant (not reproduced).

Matrix elements ⟨ψ_ℓ L_m, C[ψ_ℓ′ L_m′ F₀]⟩ (the orthonormal basis of :mod:`velcro.basis`, F₀ = e^{−v̂²}/π^{3/2}): the
diffusive parts integrated by parts once (−½∫ν_D(1 − ξ²)g_ξ f_ξ F₀, −½∫ν_∥v̂² g_v f_v F₀ over d³v), the restoring terms
rank-one outer products of exact moment vectors — symmetric, negative semi-definite and conserving (particles, like-species
parallel momentum and energy) to quadrature rounding at any (L, M).  Quadrature: :mod:`velcro.landau`'s Gauss–Legendre
(v̂, ξ) grid, exact in ξ, with v̂ ∈ (0, vmax] sized to the basis (vmax = √(2L + 4M) + 7, nv = 6(L + 2M) ≥ 240).

Omitted, by GS2's own terms (this is the k⊥ = 0 operator): the gyrodiffusion −ν_D k⊥²v̂²(1 + ξ²)/4Ω² h and the J₀/J₁
weights of the restoring terms (GS2 carries them).  On the JET deck they are ≤ ν_D b_s with b_e ≈ 4·10⁻³ even in the
ballooning tails (k⊥ρ_i ≈ 4), i.e. < 10⁻³ of γ, and the ions' ν̂ is 10⁻⁴ of γ; ``physics.collision_flr`` refuses this model.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf

from velcro import landau as la

SQRT_PI = np.sqrt(np.pi)
TERMS = ("default", "lorentz", "ediffuse")


# ------------------------------------------------------------------------------------------------ GS2's frequencies
def chandrasekhar_G(x: np.ndarray) -> np.ndarray:
    """G(x) = (erf x − x erf′x)/2x² (GS2 ``hsg_func``); its series below x = 0.1 (the closed form cancels there)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 0.1
    xs = x[small]
    out[small] = (2 * xs / 3 - 2 * xs ** 3 / 5 + xs ** 5 / 7 - xs ** 7 / 27) / SQRT_PI
    xl = x[~small]
    out[~small] = (erf(xl) - xl * 2 * np.exp(-xl ** 2) / SQRT_PI) / (2 * xl ** 2)
    return out


def frequencies(x: np.ndarray, const_v: bool = False) -> dict:
    """GS2's like-species frequencies in ν̂ units at speed x = v̂ (``init_vnew``; const_v: every one at v̂ = 1), and the
    e–i pitch-angle frequency per unit ν̂_ei."""
    xe = np.ones_like(np.asarray(x, dtype=float)) if const_v else np.asarray(x, dtype=float)
    G = chandrasekhar_G(xe)
    nuD = (erf(xe) - G) / xe ** 3                 # GS2 vnew_D = vnewk hee / E^1.5, hee = erf − G
    nus = 4 * G / xe                              # vnew_s = 4 vnewk hsg / √E
    nupar = 2 * G / xe ** 3                       # the energy-diffusion coefficient (GS2 capg)
    nuE = xe ** 2 * (nus * (2 - 0.5 / xe ** 2) - 2 * nuD)    # vnew_E
    return dict(nuD=nuD, nus=nus, nupar=nupar, nuE=nuE, nuD_ei=1.0 / xe ** 3)


# ------------------------------------------------------------------------------------------------ quadrature
def _grid_size(L: int, M: int):
    vmax = max(9.0, float(np.sqrt(2.0 * L + 4.0 * M)) + 7.0)
    nv = max(240, 6 * (L + 2 * M))
    return nv, vmax


def _basis(L: int, M: int, V, XI):
    """g = ψ_ℓ(v̂∥) L_m(μ̂), ∂_v̂ g and ∂_ξ g on the (v̂, ξ) grid → three (L·M, P) arrays (P = nv·nξ)."""
    vpar, mu = V * XI, V ** 2 * (1.0 - XI ** 2)
    psi, dpsi, _ = la._hermite_derivs(L, vpar)
    Lm, dLm, _ = la._laguerre_derivs(M, mu)
    n, P = L * M, V.size
    g = (psi[:, None] * Lm[None, :]).reshape(n, P)
    g_v = (XI * dpsi[:, None] * Lm[None, :] + 2 * V * (1 - XI ** 2) * psi[:, None] * dLm[None, :]).reshape(n, P)
    g_xi = (V * dpsi[:, None] * Lm[None, :] - 2 * V ** 2 * XI * psi[:, None] * dLm[None, :]).reshape(n, P)
    return g, g_v, g_xi


@lru_cache(maxsize=16)
def _pieces(L: int, M: int, const_v: bool, nv: int, vmax: float) -> dict:
    """Every block of the operator on one grid (read-only arrays): the like-species Lorentz and energy-diffusion
    matrices, the e–i Lorentz matrix, and the restoring vectors with their normalisations."""
    _, _, V, XI, W = la._grid(L, M, nv, vmax)
    F0 = np.exp(-V ** 2) / np.pi ** 1.5
    fr = frequencies(V, const_v)
    g, g_v, g_xi = _basis(L, M, V, XI)
    w = (W * F0).reshape(-1)                      # d³v F₀
    vpar = (V * XI).reshape(-1)
    v2 = (V ** 2).reshape(-1)
    f = {k: a.reshape(-1) for k, a in fr.items()}
    lor = lambda nu: -0.5 * (g_xi * (w * nu * (1 - XI.reshape(-1) ** 2))) @ g_xi.T
    out = dict(
        lorentz=lor(f["nuD"]),
        lorentz_ei=lor(f["nuD_ei"]),
        ediffuse=-0.5 * (g_v * (w * f["nupar"] * v2)) @ g_v.T,
        uL=g @ (w * f["nuD"] * vpar), DL=float(np.sum(w * f["nuD"] * vpar ** 2)),
        uS=g @ (w * (f["nus"] - f["nuD"]) * vpar), DS=float(np.sum(w * (f["nus"] - f["nuD"]) * vpar ** 2)),
        uE=g @ (w * f["nuE"]), DE=float(np.sum(w * f["nuE"] * v2)),
    )
    for a in out.values():
        if isinstance(a, np.ndarray):
            a.setflags(write=False)
    return out


def _check(L: int, M: int, terms: str, const_v: bool):
    if terms not in TERMS:
        raise ValueError(f"gs2 collision terms {terms!r}: one of {TERMS}")
    if const_v and terms != "lorentz":
        raise ValueError("gs2 const_v is reproduced for terms = 'lorentz' only (GS2 keeps its energy-diffusion operator "
                         "velocity-dependent under const_v while its restoring frequencies go constant)")
    if L < 3 or M < 2:
        raise ValueError("collision_model = 'gs2' needs L ≥ 3 and M ≥ 2 (the energy moment)")


def like_matrix(L: int, M: int, terms: str = "default", const_v: bool = False, nv: int | None = None,
                vmax: float | None = None) -> np.ndarray:
    """GS2's like-species operator (ν̂ units, (L·M) × (L·M)): ∂_t G = ν̂ C G."""
    _check(L, M, terms, const_v)
    dn, dv = _grid_size(L, M)
    p = _pieces(L, M, bool(const_v), int(nv or dn), float(vmax or dv))
    C = np.zeros((L * M, L * M))
    if terms in ("default", "lorentz"):
        C = C + p["lorentz"] + np.outer(p["uL"], p["uL"]) / p["DL"]
    if terms in ("default", "ediffuse"):
        C = C + p["ediffuse"] + np.outer(p["uS"], p["uS"]) / p["DS"] + np.outer(p["uE"], p["uE"]) / p["DE"]
    return C


def ei_matrix(L: int, M: int, terms: str = "default", const_v: bool = False, nv: int | None = None,
              vmax: float | None = None) -> np.ndarray:
    """The electrons' e–i pitch-angle block per unit ν̂_ei = Z_i ν̂_ee: ½(1/v̂³)L (1 at const_v); zero for 'ediffuse'."""
    _check(L, M, terms, const_v)
    if terms == "ediffuse":
        return np.zeros((L * M, L * M))
    dn, dv = _grid_size(L, M)
    return np.array(_pieces(L, M, bool(const_v), int(nv or dn), float(vmax or dv))["lorentz_ei"])
