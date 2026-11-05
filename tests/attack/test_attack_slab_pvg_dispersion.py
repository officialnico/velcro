"""The parallel-velocity-gradient drive (``physics.exb_shear.u_prime``) against the local kinetic slab dispersion relation.

Velcro's slab equation for one Fourier mode e^{i(k_y y + n z)} (uniform B, adiabatic electrons, ``Cinv`` = 1, T̂_i = 1),
with ω_* (v̂, μ̂) = ½k_y [ω_n + ω_T (v̂² + μ̂ − 3/2) + 2v̂ ω_u] (the drive ∝ ∂_x ln F₀ of a Maxwellian shifted by u∥(x),
ω_u = −L_ref d(u∥/v_t)/dx) and k∥ = gpar · (the FD4 symbol of the z grid), gives
    (ω − k∥v̂) G = (k∥ v̂ + ω_*) J₀φ,   ∫ J₀ G F̂₀ = (1 − Γ₀(b) + 1/τ) φ,
hence the dispersion relation (ζ = ω/k∥, Z the plasma dispersion function, Γ_n(b) = I_n(b)e^{−b}, b = k⊥²ρ²/2):
    1 + 1/τ = −(1/k∥) [ (ω + ω_*n − 3/2 ω_*T) Γ₀ Z + ω_*T Γ₀ ζ(1 + ζZ) + ω_*T (Γ₀ + b(Γ₁ − Γ₀)) Z + ω_*u Γ₀ (1 + ζZ) ],
    ω_*n = ½k_y ω_n, ω_*T = ½k_y ω_T, ω_*u = k_y ω_u —
the classic local result (Catto, Rosenbluth & Liu, Phys. Fluids 16, 1719 (1973), unsheared field; the η_i slab mode
at ω_u = 0).  With u_prime = 0 it is the slab ITG, a check of the pre-existing operator; with ω_u ≠ 0 the n = +1 and
n = −1 parallel harmonics are no longer equivalent (n → −n is ω_u → −ω_u), so projecting Velcro's exact linear operator
onto each harmonic tests the sign of the PVG coupling, not only its magnitude.
"""
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from scipy.optimize import fsolve
from scipy.special import i0e, i1e, wofz

from velcro.config import VelcroConfig
from velcro.simulation import Simulation

LZ = 10.0                       # slab length in L_ref: gpar = 2π/10
NZ, L, M = 8, 32, 4
KY_INDEX = 3                    # k_y = 0.3 (L_y = 20π)


def _Z(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def _D(omega, kpar, ky, b, wn, wT, wu, tau):
    z = omega / kpar
    G0, G1 = i0e(b), i1e(b)
    sn, sT, su = 0.5 * ky * wn, 0.5 * ky * wT, ky * wu
    Zz = _Z(z)
    I = -((omega + sn - 1.5 * sT) * G0 * Zz + sT * G0 * z * (1 + z * Zz) + sT * (G0 + b * (G1 - G0)) * Zz
          + su * G0 * (1 + z * Zz)) / kpar
    return I - (1 + 1 / tau)


def _most_unstable_root(kpar, ky, b, wn, wT, wu, tau):
    f = lambda v: [np.real(_D(v[0] + 1j * v[1], kpar, ky, b, wn, wT, wu, tau)),
                   np.imag(_D(v[0] + 1j * v[1], kpar, ky, b, wn, wT, wu, tau))]
    best = None
    for wr in np.linspace(-6, 6, 25):
        for wi in (0.05, 0.3, 1.0, 2.0):
            s, info, ier, _ = fsolve(f, [wr, wi], full_output=True)
            if ier == 1 and abs(complex(*f(s))) < 1e-10 and s[1] > 0:
                if best is None or s[1] > best[1]:
                    best = s
    return None if best is None else complex(best[0], best[1])


def _sim(wn, wT, wu, tau=1.0):
    physics = dict(omega_n=dict(value=wn), omega_T=dict(value=wT), tau=tau, nu=0.0, nonlinear=False, Ti_profile=False,
                   n_profile=False)
    if wu:
        physics["exb_shear"] = dict(gamma_E=0.0, u_prime=wu)
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=4, Lx=2 * np.pi * 1000.0, Ny=10, Nz=NZ, L=L, M=M),
        geometry=dict(kind="slab", rho_star=1e-3, Lz_over_Lref=LZ),
        physics=physics,
        numerics=dict(hyper_x=0.0, hyper_z=0.0, hyper_v=1.0, krook_rate=0.0, krook_width=0.0, dt=0.01),
        init=dict(kind="noise", amplitude=1e-3, seed=0),
        run=dict(t_end=0.1, precision="fp64"))))


def _sector_omegas(sim, n, chunk=128):
    """ω = iλ of the linear operator restricted to the parallel harmonic e^{inz}: the uniform slab commutes with z
    translations, so the operator maps the harmonic's subspace to itself; it is applied to that subspace's basis
    (unit (ℓ, m, x) vectors times e^{inz}/√N_z), in chunks — no dense Jacobian of the whole k_y block."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    from velcro.vlasov import rhs
    op = dataclasses.replace(sim.op, nonlinear=False)
    shape = sim.G.shape
    Lh, Mh, Nx, Nky, Nz = shape
    z = np.asarray(sim.geom.z)
    u = np.exp(1j * n * z) / np.sqrt(Nz)                            # the harmonic, unit norm

    def f(c):                                                       # c: (Lh·Mh·Nx,) coefficients of the harmonic
        blk = jnp.asarray(c).reshape(Lh, Mh, Nx, 1) * jnp.asarray(u)[None, None, None, :]
        G = jnp.zeros(shape, dtype=sim.dtype).at[:, :, :, KY_INDEX, :].set(blk)
        phi = sim.fields.solve(G[0])
        out = rhs(G, phi, sim.fields.gyroaverage(phi), op)[:, :, :, KY_INDEX, :]
        return jnp.einsum("lmxz,z->lmx", out, jnp.conj(jnp.asarray(u))).reshape(-1)
    nb = Lh * Mh * Nx
    fv = jax.jit(jax.vmap(f))
    cols = []
    for i in range(0, nb, chunk):
        c = min(chunk, nb - i)
        E = np.zeros((c, nb), dtype=complex)
        E[np.arange(c), i + np.arange(c)] = 1.0
        cols.append(np.asarray(fv(jnp.asarray(E, dtype=sim.dtype))))
    An = np.concatenate(cols, axis=0).T
    return 1j * np.linalg.eigvals(An)


def _kpar(n):
    dz = 2 * np.pi / NZ
    return (2 * np.pi / LZ) * (8 * np.sin(n * dz) - np.sin(2 * n * dz)) / (6 * dz)


@pytest.mark.parametrize("wn,wT,wu", [(2.0, 20.0, 0.0), (2.0, 20.0, 5.0), (2.0, 20.0, -5.0), (0.0, 0.0, 12.0)])
def test_slab_eigenvalue_is_the_kinetic_dispersion_root(wn, wT, wu):
    sim = _sim(wn, wT, wu)
    ky = float(np.asarray(sim.ky)[KY_INDEX])
    b = 0.5 * ky ** 2
    checked = 0
    sectors = {n: _sector_omegas(sim, n) for n in (1, -1)}
    for n in (1, -1):
        kp = abs(_kpar(n))
        # n = −1 is n = +1 with v̂ → −v̂, i.e. with ω_u → −ω_u
        want = _most_unstable_root(kp, ky, b, wn, wT, wu if n == 1 else -wu, 1.0)
        om = sectors[n]
        got = om[np.argmax(om.imag)]
        print(f"ω_n {wn} ω_T {wT} ω_u {wu:+} n {n:+d}: analytic {want}  Velcro {got:.6f}")
        if want is None or want.imag < 0.02:
            continue                                                 # no clean unstable root in this sector
        assert abs(got - want) / abs(want) < 2e-3, (n, got, want)
        checked += 1
    assert checked >= 1
    if wu != 0.0:                                                    # the PVG term breaks the ±n symmetry
        assert abs(sectors[1].imag.max() - sectors[-1].imag.max()) > 1e-3
