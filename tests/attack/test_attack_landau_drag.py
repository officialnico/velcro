"""The e←i field block C_F,ei: what it is, and how cheaply the missing drag toward the ion flow could be restored.

velcro/landau.py's monomial C_F,ei is wrong from (8, 4) on (L_mat_cross: 1.5, 1.4e2 and 3.0e3 × max|C_F| at (8,4), (12,6),
(16,8)), so kinetic.py leaves it out and the Landau e–i friction acts on u_e instead of u_e − u_i.  In the m_e/m_i → 0
limit the field part is the Lorentz drag toward the ion flow: C_ei[f_e] ≈ C_T,ei[f_e − 2v̂∥(u_i/v_te)F_e0], i.e.
C_F,ei ≈ −C_T,ei[:, (1,0)] (v̂t_i/v̂t_e) e₁₀ᵀ acting on the ions' moments (G_i,10 = √2 u_i/v_ti).  Here the exact C_F,ei
(the reference) is compared with that rank-1 model, which is exactly Galilean invariant by construction."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import landau_ref as ref  # noqa: E402

from velcro import landau as la  # noqa: E402

MR = 2.7e-4


@pytest.mark.parametrize("L,M,Te", [(8, 4, 1.0), (12, 6, 1.0), (16, 8, 1.0), (12, 6, 3.0)])
def test_rank_one_drag_model_of_the_ei_field_block(L, M, Te):
    RT, RF = ref.pair_reference(L, M, MR, Te, 1.0, 1.0)
    VT, _ = la.pair_matrices(L, M, MR, Te, 1.0, 1.0)
    vr = float(np.sqrt(1.0 / 1.0) / np.sqrt(Te / MR))                    # v̂t_i / v̂t_e
    i10 = 1 * M + 0
    model = np.zeros_like(RF)
    model[:, i10] = -VT[:, i10] * vr
    err = np.abs(model - RF).max() / np.abs(RF).max()
    col = np.abs(model[:, i10] - RF[:, i10]).max() / np.abs(RF[:, i10]).max()
    rest = np.abs(np.delete(RF, i10, axis=1)).max() / np.abs(RF).max()
    # Galilean invariance of the exact pair: electrons and ions co-moving (G_e,10 = √2U/v̂t_e, G_i,10 = √2U/v̂t_i)
    He = np.zeros(L * M); He[i10] = 1.0 / np.sqrt(Te / MR)
    Hi = np.zeros(L * M); Hi[i10] = 1.0
    gal_exact = np.abs(RT @ He + RF @ Hi).max() / np.abs(RT @ He).max()
    gal_model = np.abs(VT @ He + model @ Hi).max() / np.abs(VT @ He).max()
    gal_now = 1.0                                                          # landau_F = 0: the whole C_T,ei[He] remains
    print(f"\n({L},{M}) T_e/T_i {Te}: max|C_F,ei| {np.abs(RF).max():.4e}; rank-1 model error {err:.2e} of max (its column "
          f"{col:.2e}; all other ion moments ≤ {rest:.2e}); common-flow residual: exact pair {gal_exact:.1e}, "
          f"C_T + rank-1 drag {gal_model:.1e}, the code (no field block) {gal_now:.0f}")
    # (at T_e ≠ T_i the linearisation is about non-equilibrium Maxwellians: −U·∇C(F_e, F_i) ≠ 0, O(m_e/m_i)).  Job 2005: at
    # T_e = T_i the ion-flow column of the exact block IS the rank-1 drag (1e-14); the other ion moments (density,
    # temperature, heat flux → electrons: the O(m_e/m_i) energy exchange) reach 3.8–4.3 % of the drag element
    assert (gal_exact < 1e-10 or Te != 1.0) and gal_model < 1e-12 and err < 6e-2
    assert col < 1e-12 or Te != 1.0
