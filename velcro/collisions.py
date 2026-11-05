"""Collision operators in the Hermite–Laguerre basis (§3.1, §5.7).

Implemented models
------------------
* ``"dougherty"`` — linearised Dougherty (conserving Lenard–Bernstein) operator,
  diagonal in ``(ℓ, m)`` with eigenvalue ``−ν(ℓ + 2m)`` plus momentum- and
  energy-restoring terms.  Exact in this basis.
* ``"lenard-bernstein"`` — same without restoring terms (conserves particles only).
* ``"hyper"`` — no physical collisions (hyper-collisions only, see vlasov.py).
* ``"landau"`` — the linearised Landau (Coulomb) operator at k⊥ = 0, test- and field-particle parts, dense in
  ``(ℓ, m)`` (:mod:`velcro.landau`); ν is then ν̂_aa = Γ_aa n_a/v_ta³ (Helander & Sigmar), Braginskii's
  τ_i = (3√π/2√2)/ν̂_ii.  Validated to (16, 8) (tests/test_landau_operator.py).
* ``"gs2"`` — GS2's model operator at k⊥ = 0 (Lorentz + energy diffusion with its momentum / energy restoring terms;
  :mod:`velcro.gs2_collisions`), dense in ``(ℓ, m)``; ν is GS2's vnewk (ν̂_ss).  ``terms`` / ``const_v``: GS2's
  collision_model and const_v (tests/test_gs2_collisions.py).

These are the k⊥ = 0 matrices.  The gyrokinetic finite-Larmor-radius terms (classical diffusion −νb, the J₀/J₁-weighted
restoring) are added by ``physics.collision_flr`` (:mod:`velcro.collisions_flr`, applied in k_x by velcro/vlasov.py), and
the operator acts on the non-adiabatic h by default (``physics.collisions_on``).

The design document flags the collision model as load-bearing for the
velocity-space economy (§3.1 point 3), so the operator is a pluggable
``(ℓ, m)`` matrix and the convergence study of §5.7 is reported per model.
"""
from __future__ import annotations

import numpy as np


def collision_matrix(model: str, L: int, M: int, terms: str = "default", const_v: bool = False) -> np.ndarray:
    """Return the ``(L·M) × (L·M)`` matrix ``C`` such that ``∂_t G = ν C G`` (ν = 1); ``terms`` / ``const_v`` for
    ``"gs2"`` only."""
    l = np.arange(L)[:, None] * np.ones((1, M))
    m = np.ones((L, 1)) * np.arange(M)[None, :]
    C = np.diag(-(l + 2 * m).reshape(-1))
    if model == "hyper":
        return np.zeros((L * M, L * M))
    if model == "landau":
        from velcro.landau import like_species_matrix
        return like_species_matrix(L, M)
    if model == "gs2":
        from velcro.gs2_collisions import like_matrix
        return like_matrix(L, M, terms, const_v)
    if model == "lenard-bernstein":
        return C
    if model == "dougherty":
        if L > 1:
            # momentum restoring: +ν G_10 in the (1,0) equation
            i10 = 1 * M + 0
            C[i10, i10] += 1.0
        # energy restoring: δT/T = (2/3)[G_20/√2 − G_01];  adds 2ν δT/T [ (1/√2) e_20 − e_01 ]
        idx = []
        if L > 2:
            idx.append((2 * M + 0, 1.0 / np.sqrt(2.0)))
        if M > 1:
            idx.append((0 * M + 1, -1.0))
        for i, ci in idx:
            for j, cj in idx:
                C[i, j] += 2.0 * (2.0 / 3.0) * ci * cj
        return C
    raise ValueError(f"unknown collision model {model!r}")
