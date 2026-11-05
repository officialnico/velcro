"""Equilibrium E×B flow shear ``physics.exb_shear`` (velcro/vlasov.py, fields.py, simulation.py).  KE_VALIDATION E8-0.

Truths used:
* off (absent, None, γ_E = u′ = 0) is the old code path: the same jaxpr, bitwise the same step; the adiabatic FFT field
  solver (the flux-tube limit, new for the adiabatic path) equals the dense one exactly in a slab and to the ρ* variation
  in the tube;
* the local term is −i k_y γ_E x G: with every other term off the right-hand side is that, and RK4 reproduces the exact
  solution G₀ e^{−i k_y γ_E x t}; the term is a phase rotation (Σ|G|² neutral) and enters the CFL rate;
* the shearing frame: with every other term off p is constant and the physical field e^{−iθ(t)x} p is the exact solution
  through a remap event (nothing dropped when the state has no band-edge content); with the full linear operator the frame
  right-hand side at θ = Δk_x on p equals e^{iΔk_x x} times the plain right-hand side on the periodic field
  e^{−iΔk_x x} p (the drift's ∂_x − iθ and the solver's k_x − θ are the physical operator); the frame's hyper_x is the
  stencil's symbol at the physical k_x − θ (E8-A4): the stencil at θ = 0 and at θ = Δk_x, the closed form between; with
  the Fourier bracket, the Fourier radial drift and x-dealiasing (E8-A6: numerics.bracket_dx / radial_drift = "fourier",
  x_dealias, new on the adiabatic path) the same frame identity holds to rounding, not to the FD4 stencil's 2e-3, the
  frame at θ = 0 is the plain operator, the 2/3 mask changes the identity only at O(ρ*) (the non-periodic coefficient
  profiles), and the remap drops the mode that leaves the band;
* the parallel-velocity-gradient drive equals the Hermite quadrature of 2v̂ ω_u Φ on the ℓ = 1 row;
* jax.jvp of the frame step equals finite differences.
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from velcro import basis
from velcro.config import VelcroConfig
from velcro.simulation import Simulation
from velcro.vlasov import exb_remap

ABSENT = object()


def _slab(exb=ABSENT, solver="dense", Nx=16, L=3, M=2, nonlinear=False, hyper_x=0.0):
    """Slab, no gradients / drifts / collisions / dissipation; on z-flat states (``_noise(zflat=True)``) streaming and E∥
    vanish too, so only the shear term can act."""
    physics = dict(omega_n=dict(value=0.0), omega_T=dict(value=0.0), tau=1.0, nu=0.0, nonlinear=nonlinear,
                   Ti_profile=False, n_profile=False)
    if exb is not ABSENT:
        physics["exb_shear"] = exb
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=20.0, Ny=10, Nz=4, L=L, M=M),
        geometry=dict(kind="slab", rho_star=1e-3),
        physics=physics,
        numerics=dict(hyper_x=hyper_x, hyper_z=0, hyper_v=0, krook_rate=0.0, krook_width=0.0, dt=0.01, field_solver=solver,
                      node_gyro="exact"),
        init=dict(kind="noise", amplitude=1.0, seed=3),
        run=dict(t_end=0.1, precision="fp64"))))


TUBE = dict(kind="s_alpha", rho_star=1e-5, r0=0.18, q0=1.4, shat=0.8, n0=1700)
LX_TUBE = 47.2689          # ŝ k_y,min L_x = 5


FOURIER_X = dict(bracket_dx="fourier", radial_drift="fourier")                      # E8-A6's Fourier operators
FOURIER_XD = dict(FOURIER_X, x_dealias=True)                                        # … with the 2/3 rule in k_x


def _tube(exb=ABSENT, solver="dense", Nx=16, Ny=7, Nz=8, L=4, M=2, nonlinear=False, integrator="rk4", krook=0.0,
          hyper_x=0.05, numerics=None):
    physics = dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), nu=0.01, nonlinear=nonlinear, Ti_profile=False,
                   n_profile=False)
    if exb is not ABSENT:
        physics["exb_shear"] = exb
    return Simulation(VelcroConfig.model_validate(dict(
        grid=dict(Nx=Nx, Lx=LX_TUBE, Ny=Ny, Nz=Nz, L=L, M=M),
        geometry=TUBE, physics=physics,
        numerics=dict(hyper_x=hyper_x, hyper_z=0.05, hyper_v=0.1, krook_rate=krook, krook_width=0.1 if krook else 0.0,
                      integrator=integrator, node_gyro="exact", dt=0.002, field_solver=solver, **(numerics or {})),
        init=dict(kind="noise", amplitude=1e-3, seed=5),
        run=dict(t_end=0.01, precision="fp64"))))


def _noise(sim, seed, scale=1.0, kmax=None, zflat=False):
    r = np.random.default_rng(seed)
    G = scale * (r.normal(size=sim.G.shape) + 1j * r.normal(size=sim.G.shape))
    G[:, :, :, 0, :] = G[:, :, :, 0, :].real
    if zflat:                                                  # no z structure: streaming and E∥ vanish in the slab
        G[...] = G[..., :1]
    if kmax is not None:                                       # keep |k_x| ≤ kmax modes only
        Gh = np.fft.fft(G, axis=2)
        m = np.abs(np.fft.fftfreq(G.shape[2], d=1.0 / G.shape[2]))
        Gh[:, :, m > kmax] = 0.0
        G = np.fft.ifft(Gh, axis=2)
    return jnp.asarray(G, dtype=sim.dtype)


# ------------------------------------------------------------------ Gate 0
def test_off_is_the_old_code_path_bitwise():
    sims = [_slab(), _slab(None), _slab(dict(gamma_E=0.0)), _slab(dict(gamma_E=0.0, frame="shearing", u_prime=0.0))]
    G = _noise(sims[0], 1)
    outs = [np.asarray(s._nsteps(G, 0.01, 5)) for s in sims]
    assert all(s.exb_theta is None and s.op.exb_x is None and s.op.pvg_T0 is None and not s.op.exb_frame for s in sims)
    assert all(s.op.kx_sym is None and s.op.x_mask is None and not s.op.drift_fourier and not s.op.bracket_fourier
               and not s.op.x_dealias for s in sims)                       # E8-A6's options are off by default
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
    j = [str(jax.make_jaxpr(lambda g: s._nsteps(g, 0.01, 2))(G)) for s in sims]
    assert j[1] == j[0] == j[2] == j[3]
    assert VelcroConfig().physics.exb_shear is None


@pytest.mark.parametrize("geometry", ["slab", "tube"])
def test_adiabatic_fft_field_solver_matches_the_dense_one(geometry):
    mk = _slab if geometry == "slab" else _tube
    dense, fft = mk(solver="dense"), mk(solver="fft")
    G = _noise(dense, 7)
    pd, Pd = dense._solve_fields(G)
    pf, Pf = fft._solve_fields(G)
    tol = 1e-11 if geometry == "slab" else 2e-3                # the tube keeps Velcro's O(ρ* x) radial variation
    assert np.allclose(np.asarray(pd), np.asarray(pf), atol=tol * np.abs(np.asarray(pd)).max())
    assert np.allclose(np.asarray(Pd), np.asarray(Pf), atol=tol * np.abs(np.asarray(Pd)).max())
    nd, nf = dense.fields.gyroaverage_nodes(pd), fft.fields.gyroaverage_nodes(pf)
    assert np.allclose(np.asarray(nd), np.asarray(nf), atol=tol * np.abs(np.asarray(nd)).max())
    if geometry == "slab":                                     # the dense gyro-density and the FFT one are the same operator
        rd = dense.fields.gyro_density(G[0])
        rf = fft.fields.gyro_density(G[0])
        assert np.allclose(np.asarray(rd), np.asarray(rf), atol=1e-12 * np.abs(np.asarray(rd)).max())


# ------------------------------------------------------------------ the local term
def test_local_term_is_the_sheared_flow_and_a_phase_rotation():
    gE = 0.7
    sim = _slab(dict(gamma_E=gE, frame="local"))
    assert sim.exb_theta is None and sim.op.exb_x is not None
    G = _noise(sim, 2, zflat=True)                             # z-flat: only the shear term acts (see _slab)
    ky = np.asarray(sim.ky)[None, :, None]
    x = np.asarray(sim.geom.x)[:, None, None]
    want = -1j * ky * gE * x * np.asarray(G)
    got = np.asarray(sim._rhs(G))
    assert np.allclose(got, want, atol=1e-13 * np.abs(want).max())
    dt, n = 0.01, 40
    Gn = np.asarray(sim._nsteps(G, dt, n))
    exact = np.asarray(G) * np.exp(-1j * ky * gE * x * (n * dt))
    assert np.allclose(Gn, exact, atol=1e-7 * np.abs(exact).max())             # RK4, (k_y γ_E x dt)⁴/120 per step
    w = np.where(np.arange(G.shape[3]) == 0, 1.0, 2.0)[None, None, None, :, None]
    assert np.sum(w * np.abs(Gn) ** 2) == pytest.approx(np.sum(w * np.abs(np.asarray(G)) ** 2), rel=1e-9)   # RK4 round-off
    # the CFL rate carries |γ_E x| k_y,max
    plain = _slab()
    r0, r1 = float(plain._rate(G)), float(sim._rate(G))
    assert r1 - r0 == pytest.approx(gE * np.abs(np.asarray(sim.geom.x)).max() * ky.max(), rel=1e-10)


# ------------------------------------------------------------------ the shearing frame
def test_frame_free_advection_is_exact_through_a_remap():
    gE = 0.7
    sim = _slab(dict(gamma_E=gE, frame="shearing"), solver="fft", Nx=32)
    assert sim.exb_theta is not None and sim.op.exb_frame and sim.op.exb_x is None
    G = _noise(sim, 4, kmax=6, zflat=True)                      # no content near the band edge: nothing is dropped
    ky = np.asarray(sim.ky)
    x = np.asarray(sim.geom.x)
    dkx = 2 * np.pi / sim.cfg.grid.Lx
    assert np.allclose(np.asarray(sim._rhs(G)), 0.0, atol=1e-14)   # p is constant in the frame
    dt = 0.01
    n = int(np.ceil(dkx / (ky[1] * gE * dt))) + 3                # through the first remap of the k_y,min row
    th0 = sim.exb_theta
    Gn, th = sim._nsteps(G, dt, n, th0)
    th = np.asarray(th)
    assert np.all(np.abs(th) <= 0.5 * dkx + 1e-12) and th[0] == 0.0
    t = n * dt
    assert np.any(np.abs(ky * gE * t) > 0.5 * dkx)                # a remap did happen
    # physical field: e^{−iθx} p must equal G₀ e^{−i k_y γ_E x t}; the frame's integer part is in the roll of the spectrum
    phys = np.asarray(Gn) * np.exp(-1j * th[None, None, None, :, None] * x[None, None, :, None, None])
    exact = np.asarray(G) * np.exp(-1j * ky[None, None, None, :, None] * gE * x[None, None, :, None, None] * t)
    assert np.allclose(phys, exact, atol=1e-11 * np.abs(exact).max())
    # the remap on its own is exact for a state without band-edge content
    G2, th2 = exb_remap(G, jnp.asarray(0.6 * dkx * (ky > 0), dtype=sim.rdtype), dkx, float(x[0]))
    phys2 = np.asarray(G2) * np.exp(-1j * np.asarray(th2)[None, None, None, :, None] * x[None, None, :, None, None])
    exact2 = np.asarray(G) * np.exp(-1j * 0.6 * dkx * (ky > 0)[None, None, None, :, None] * x[None, None, :, None, None])
    assert np.allclose(phys2, exact2, atol=1e-12 * np.abs(exact2).max())
    assert np.allclose(np.asarray(th2), -0.4 * dkx * (ky > 0))


def test_frame_rhs_is_the_physical_operator_on_the_sheared_field():
    """At θ = Δk_x the sheared field e^{−iΔk_x x} p is x-periodic, so the plain (no-shear) right-hand side on it is
    computable: the frame right-hand side on p must be e^{iΔk_x x} times it (the drift's ∂_x − iθ and the field solver's
    k_x − θ).  The FD4 stencil of the radial drift is not shift-covariant (its symbol is not additive in the wavenumber:
    the error is O(κ⁴θ)), so the state is kept at resolved modes (|k_x| ≤ 4 of 64: κΔx ≤ 0.39) and the band is 2e-3; at
    |k_x| ≤ 12 the same comparison gives 1.3e-2, the stencil's error, not the frame's."""
    plain = _tube(solver="fft", Nx=64, hyper_x=0.0)
    frame = _tube(dict(gamma_E=0.3, frame="shearing"), solver="fft", Nx=64, hyper_x=0.0)
    p = _noise(frame, 9, kmax=4)
    x = np.asarray(frame.geom.x)
    dkx = 2 * np.pi / frame.cfg.grid.Lx
    ky = np.asarray(frame.ky)
    theta = jnp.asarray(dkx * (ky > 0), dtype=frame.rdtype)
    E = np.exp(-1j * dkx * x)[None, None, :, None, None] * (ky > 0)[None, None, None, :, None] \
        + (ky == 0)[None, None, None, :, None]
    G = jnp.asarray(np.asarray(p) * E)
    want = np.asarray(plain._rhs(G)) / E
    got = np.asarray(frame.rhs_fn(p, theta))
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < 2e-3, err


def test_frame_fourier_bracket_and_drift_are_the_plain_fourier_operator_on_the_sheared_field():
    """E8-A6: with numerics.bracket_dx = radial_drift = "fourier" the comparison of
    test_frame_rhs_is_the_physical_operator_on_the_sheared_field holds to rounding — the Fourier ∂_x is shift-covariant
    (∂_x(e^{−iθx}p) = e^{−iθx}(∂_x − iθ)p exactly), the bracket's θ terms cancel between its two factors, and the drift
    carries −iθ.  Unlike the linear test, θ must be ∝ k_y here — θ_n = n Δk_x on the k_y row n — because the bracket's θ
    terms cancel only then (θ_ky1 k_y2 − k_y1 θ_ky2 = 0: the frame is the shear y′ = y − (γ_E t) x of the coordinates,
    under which the bracket is invariant); every row's sheared field e^{−i n Δk_x x} p_n is still periodic.  The state is
    kept at |k_x| ≤ 8 of 64 so that no product (≤ 16 on p, ≤ 20 on the sheared field) aliases; with the FD4 operators the
    same state gives the stencil's error (> 1e-3).  With x_dealias the same identity holds only to O(ρ*): the s-α tube's
    coefficient profiles are linear in x (not periodic), so they carry every k_x at O(ρ* L_x) and the mask truncates their
    products at band edges that differ by n modes between the two representations (5e-6 here, a property of the 2/3 rule
    on a ρ*-profiled box, not of the frame; 0 in a slab).  At θ = 0 the frame with the mask is the plain operator on any
    state, the plain right-hand side has no content beyond the band, and the remap with the mask drops the mode that
    leaves it."""
    plain = _tube(solver="fft", Nx=64, hyper_x=0.0, nonlinear=True, numerics=FOURIER_X)
    frame = _tube(dict(gamma_E=0.3, frame="shearing"), solver="fft", Nx=64, hyper_x=0.0, nonlinear=True, numerics=FOURIER_X)
    assert frame.op.drift_fourier and frame.op.bracket_fourier and not frame.op.x_dealias and frame.op.exb_frame
    assert plain.op.kx_bound == pytest.approx(31 * 2 * np.pi / LX_TUBE)
    p = _noise(frame, 9, kmax=8, scale=30.0)     # large enough for the bracket to matter; products ≤ 16 (p), ≤ 20 (sheared)
    x = np.asarray(frame.geom.x)
    dkx = 2 * np.pi / frame.cfg.grid.Lx
    ky = np.asarray(frame.ky)
    n = np.arange(len(ky))
    theta = jnp.asarray(dkx * n, dtype=frame.rdtype)
    E = np.exp(-1j * dkx * n[None, None, None, :, None] * x[None, None, :, None, None])
    G = jnp.asarray(np.asarray(p) * E)
    want = np.asarray(plain._rhs(G)) / E
    got = np.asarray(frame.rhs_fn(p, theta))
    scale = np.abs(want).max()
    assert np.abs(got - want).max() < 1e-12 * scale
    # the bracket did act, and the FD4 operators on the same state carry the stencil's error, not rounding
    lin = _tube(solver="fft", Nx=64, hyper_x=0.0, nonlinear=False, numerics=FOURIER_X)
    assert np.abs(np.asarray(lin._rhs(G)) / E - want).max() > 1e-3 * scale
    fd_plain = _tube(solver="fft", Nx=64, hyper_x=0.0, nonlinear=True)
    fd_frame = _tube(dict(gamma_E=0.3, frame="shearing"), solver="fft", Nx=64, hyper_x=0.0, nonlinear=True)
    fd_want = np.asarray(fd_plain._rhs(G)) / E
    assert np.abs(np.asarray(fd_frame.rhs_fn(p, theta)) - fd_want).max() > 1e-3 * np.abs(fd_want).max()
    # with the 2/3 rule: the identity to O(ρ*) on the same state (see the docstring), the mask's band in the CFL bound
    plain = _tube(solver="fft", Nx=64, hyper_x=0.0, nonlinear=True, numerics=FOURIER_XD)
    frame = _tube(dict(gamma_E=0.3, frame="shearing"), solver="fft", Nx=64, hyper_x=0.0, nonlinear=True, numerics=FOURIER_XD)
    assert frame.op.x_dealias and plain.op.kx_bound == pytest.approx(21 * 2 * np.pi / LX_TUBE)
    m = np.abs(np.fft.fftfreq(64, d=1.0 / 64))
    Fh = np.fft.fft(np.asarray(frame.G), axis=2)
    assert np.abs(Fh[:, :, m > 21]).max() < 1e-12 * np.abs(Fh).max()                      # the initial state is in the band
    want = np.asarray(plain._rhs(G)) / E
    assert np.abs(np.asarray(frame.rhs_fn(p, theta)) - want).max() < 1e-4 * np.abs(want).max()
    # θ = 0: the frame is the plain operator, on a full-band state; the right-hand side is inside the 2/3 band
    q = _noise(frame, 10, scale=30.0)
    zero = jnp.zeros(len(ky), dtype=frame.rdtype)
    r_plain = np.asarray(plain._rhs(q))
    assert np.allclose(np.asarray(frame.rhs_fn(q, zero)), r_plain, atol=1e-12 * np.abs(r_plain).max())
    assert np.abs(np.fft.fft(r_plain, axis=2)[:, :, m > 21]).max() < 1e-12 * np.abs(r_plain).max()
    # the remap with the mask: the mode rolled past |k_x| = 21 Δk_x is dropped (GX's g_shift), the rest moved exactly
    G2, th2 = exb_remap(q, jnp.asarray(0.6 * dkx * (ky > 0), dtype=frame.rdtype), dkx, float(x[0]), frame.op.x_mask)
    Gh2 = np.fft.fft(np.asarray(G2), axis=2)
    assert np.abs(Gh2[:, :, m > 21]).max() < 1e-12 * np.abs(Gh2).max()      # zero up to the FFT round trip
    G3, _ = exb_remap(q, jnp.asarray(0.6 * dkx * (ky > 0), dtype=frame.rdtype), dkx, float(x[0]))
    Gh3 = np.fft.fft(np.asarray(G3), axis=2)
    assert np.allclose(Gh2[:, :, m <= 21], Gh3[:, :, m <= 21], atol=1e-12 * np.abs(Gh3).max())
    assert np.abs(Gh3[:, :, m > 21]).max() > 1e-3 * np.abs(Gh3).max()      # without the mask that mode is kept


def test_frame_hyper_x_acts_on_the_sheared_k_x():
    """E8-A4: in the frame the radial hyperdissipation damps the physical field G = e^{−iθx} p at its own wavenumber
    k_x − θ_ky.  In the z-flat slab only hyper_x acts, so: at θ = 0 the frame's right-hand side is the plain stencil on p
    (to rounding); at θ = Δk_x (the sheared field e^{−iΔk_x x} p is x-periodic) it is e^{iΔk_x x} times the plain stencil
    on that field (to rounding); at a fractional θ it is the closed-form symbol 16 sin⁴((k_x − θ)Δx/2) on p's spectrum."""
    hx = 0.3
    plain = _slab(solver="fft", Nx=32, hyper_x=hx)
    frame = _slab(dict(gamma_E=0.7, frame="shearing"), solver="fft", Nx=32, hyper_x=hx)
    assert frame.exb_theta is not None and frame.op.exb_frame and frame.op.hyper_x == hx
    p = _noise(frame, 21, zflat=True)
    ky = np.asarray(frame.ky)
    x = np.asarray(frame.geom.x)
    dx = float(x[1] - x[0])
    dkx = 2 * np.pi / frame.cfg.grid.Lx
    scale = np.abs(np.asarray(plain._rhs(p))).max()
    assert scale > 0
    zero = jnp.zeros(len(ky), dtype=frame.rdtype)
    assert np.allclose(np.asarray(frame.rhs_fn(p, zero)), np.asarray(plain._rhs(p)), atol=1e-13 * scale)
    theta = jnp.asarray(dkx * (ky > 0), dtype=frame.rdtype)
    E = np.exp(-1j * dkx * x)[None, None, :, None, None] * (ky > 0)[None, None, None, :, None] \
        + (ky == 0)[None, None, None, :, None]
    want = np.asarray(plain._rhs(jnp.asarray(np.asarray(p) * E))) / E
    assert np.allclose(np.asarray(frame.rhs_fn(p, theta)), want, atol=1e-12 * scale)
    th = 0.37 * dkx * (ky > 0)
    kx = 2 * np.pi * np.fft.fftfreq(p.shape[2], d=dx)
    sym = 16 * np.sin(0.5 * dx * (kx[:, None] - th[None, :])) ** 4
    want2 = -hx * np.fft.ifft(sym[None, None, :, :, None] * np.fft.fft(np.asarray(p), axis=2), axis=2)
    assert np.allclose(np.asarray(frame.rhs_fn(p, jnp.asarray(th, dtype=frame.rdtype))), want2, atol=1e-12 * scale)


def test_pvg_drive_is_the_hermite_quadrature_of_2v_u_phi():
    u = 0.9
    on, off = _tube(dict(gamma_E=0.0, u_prime=u), L=5, M=3), _tube(L=5, M=3)
    assert on.exb_theta is None and on.op.pvg_T0 is not None and on.op.exb_x is None
    G = _noise(on, 11)
    phi, Phi = on._solve_fields(G)
    d = np.asarray(on._rhs(G)) - np.asarray(off._rhs(G))
    v, w = basis.hermite_quadrature(40)
    L = 5
    coef = np.array([np.sum(w * 2 * v * basis.psi(l, v) * basis.psi(0, v)) for l in range(L)])    # ⟨ψ_ℓ, 2v̂ ψ_0⟩
    assert coef[1] == pytest.approx(np.sqrt(2.0)) and np.allclose(np.delete(coef, 1), 0.0, atol=1e-13)
    want = -np.asarray(on.op.c_Cinv) * (1j * np.asarray(on.ky)[None, :, None]) * u * coef[:, None, None, None, None] * np.asarray(Phi)[None]
    assert np.allclose(d, want, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("integrator", ["rk4", "lawson"])
def test_jvp_of_the_frame_step_matches_finite_differences(integrator):
    sim = _tube(dict(gamma_E=0.4, frame="shearing"), solver="fft", integrator=integrator, nonlinear=True)
    G = _noise(sim, 13, scale=1e-2)
    D = _noise(sim, 14)
    th = sim.exb_theta + jnp.asarray(0.3 * sim.exb_dkx * (np.asarray(sim.ky) > 0), dtype=sim.rdtype)
    step = lambda g: sim._nsteps(g, 0.002, 3, th)[0]
    _, tang = jax.jvp(step, (G,), (D,))
    eps = 1e-6
    fd = (np.asarray(step(G + eps * D)) - np.asarray(step(G - eps * D))) / (2 * eps)
    assert np.allclose(np.asarray(tang), fd, atol=1e-7 * np.abs(fd).max())


def test_auto_frame_and_refusals():
    assert _tube(dict(gamma_E=0.2), solver="fft").exb_theta is not None                 # periodic box → shearing
    assert _tube(dict(gamma_E=0.2), solver="dense", krook=0.5).exb_theta is None          # buffers → local
    with pytest.raises(ValueError, match="fft"):
        _tube(dict(gamma_E=0.2, frame="shearing"), solver="dense")
    c = VelcroConfig.model_validate(dict(grid=dict(Nx=8, Lx=LX_TUBE, Ny=7, Nz=8, L=4, M=2), geometry=TUBE,
                                         physics=dict(exb_shear=dict(gamma_E=0.1), electrons="kinetic", Ti_profile=False,
                                                      n_profile=False), run=dict(precision="fp64")))
    with pytest.raises(ValueError, match="exb_shear"):
        Simulation(c)
    with pytest.raises(ValueError, match="x-periodic"):                  # E8-A6's options need a periodic box
        _tube(solver="dense", krook=0.5, numerics=FOURIER_XD)


def test_adiabatic_fft_field_solver_is_finite_in_fp32_and_its_j0_is_accurate():
    """The production runs are fp32: the solver's symbols at b = 0 and the exact-J0 node values must be finite (the
    float64 path's 1e-300 floor and bessel_jn's recurrence both fail in float32); the float32 J₀ against scipy to 1e-7."""
    from scipy.special import j0
    from velcro.fields import _j0
    x = np.linspace(0.0, 25.0, 501).astype(np.float32)
    assert np.abs(np.asarray(_j0(jnp.asarray(x))) - j0(x)).max() < 3e-6          # the rational approximation in float32 arithmetic
    assert np.abs(np.asarray(_j0(jnp.asarray(x.astype(np.float64)))) - j0(x.astype(np.float64))).max() < 1e-12
    jax.config.update("jax_enable_x64", False)
    try:
        physics = dict(omega_n=dict(value=2.2), omega_T=dict(value=6.9), nu=0.01, nonlinear=True, Ti_profile=False, n_profile=False)
        sim = Simulation(VelcroConfig.model_validate(dict(
            grid=dict(Nx=16, Lx=LX_TUBE, Ny=7, Nz=8, L=4, M=2), geometry=TUBE, physics=physics,
            numerics=dict(hyper_x=0.05, hyper_z=0.05, hyper_v=0.1, krook_rate=0.0, krook_width=0.0, integrator="lawson",
                          node_gyro="exact", field_solver="fft", cfl=1.1),
            init=dict(kind="noise", amplitude=1e-3, seed=5), run=dict(t_end=0.01, precision="fp32"))))
        G = sim.G
        phi, Phi = sim._solve_fields(G)
        assert np.all(np.isfinite(np.asarray(phi))) and np.all(np.isfinite(np.asarray(Phi)))
        assert np.all(np.isfinite(np.asarray(sim.fields.gyroaverage_nodes(phi))))
        assert np.isfinite(float(sim._rate(G))) and np.all(np.isfinite(np.asarray(sim._nsteps(G, 1e-3, 2))))
    finally:
        jax.config.update("jax_enable_x64", True)
