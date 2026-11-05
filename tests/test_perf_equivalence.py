"""The optimised operators must reproduce the frozen baseline (``tests/reference``) to roundoff.

Every kernel that was restructured for speed (field solver layout, dense-DFT binormal
transforms, tridiagonal streaming, diagonal hyper-collisions, ℓ = 0 drive terms, split
collision operator) is compared here against the original implementation on random inputs, on
several geometries and configurations, and on a saturated turbulent snapshot when one is
available.  A second group of tests checks structural properties (adjointness, gauge, linearity,
de-aliasing, reality) that do not depend on either implementation.
"""
from __future__ import annotations

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reference import ref_fields, ref_operators, ref_vlasov       # noqa: E402
from velcro import basis, fields, geometry, operators, vlasov            # noqa: E402
from velcro.config import VelcroConfig                                   # noqa: E402
from velcro.simulation import Simulation                                 # noqa: E402

RTOL = 1e-11


# ----------------------------------------------------------------------------- helpers
def _cfg(**over) -> VelcroConfig:
    """A small nonlinear s-α case; ``over`` are dotted overrides, e.g. physics.nu=0.1."""
    d = dict(
        grid=dict(Nx=16, Lx=60.0, Ny=12, Nz=8, L=6, M=3),
        geometry=dict(kind="s_alpha", rho_star=1 / 300, r0=0.18, q0=1.4, shat=0.8, n0=5),
        physics=dict(drive_gradB=True,   # the frozen reference right-hand side carries the −ω_B μ̂ term
                     omega_n=dict(kind="flattop", value=2.2, width=3.0, edge=8.0),
                     omega_T=dict(kind="flattop", value=6.9, width=3.0, edge=8.0),
                     tau=1.0, collision_model="dougherty", nu=0.02, nonlinear=True,
                     collisions_on="g"),  # the frozen reference applies C to G (the config default before 2026-10-03)
        # the frozen baseline in tests/reference evaluates Φ at the μ nodes by the truncated Laguerre sum
        numerics=dict(parallel_form="advective",   # as in the frozen reference
                      hyper_x=0.05, hyper_z=0.05, hyper_v=0.5, krook_rate=1.0, krook_width=0.12,
                      source_rate=0.05, cfl=0.4, node_gyro="laguerre"),
        init=dict(kind="noise", amplitude=1e-3, seed=0),
        run=dict(t_end=1.0, diag_every=0.5),
    )
    for k, v in over.items():
        sec, key = k.split(".")
        d[sec][key] = v
    return VelcroConfig.model_validate(d)


CONFIGS = {
    "s_alpha_nonlinear": {},
    "s_alpha_linear": {"physics.nonlinear": False},
    "slab_nonlinear": {"geometry.kind": "slab", "grid.Lx": 20.0},
    "lenard_bernstein": {"physics.collision_model": "lenard-bernstein", "physics.nu": 0.1},
    "hyper_collisions_only": {"physics.collision_model": "hyper", "physics.nu": 0.0},
    "hyper_ky_no_source": {"numerics.hyper_ky": 0.3, "numerics.source_rate": 0.0},
    "no_dissipation": {"numerics.hyper_x": 0.0, "numerics.hyper_z": 0.0, "numerics.hyper_v": 0.0,
                       "numerics.krook_rate": 0.0},
    "flat_profiles": {"physics.Ti_profile": False, "physics.n_profile": False, "physics.tau": 2.0},
    "odd_sizes": {"grid.Nx": 12, "grid.Ny": 10, "grid.Nz": 6, "grid.L": 5, "grid.M": 2},
    "single_laguerre": {"grid.M": 1, "grid.L": 4},
    "advective_bracket": {"numerics.bracket": "advective"},
    "advective_slab": {"numerics.bracket": "advective", "geometry.kind": "slab", "grid.Lx": 20.0},
}


def _random_state(sim, seed=0, real_ky0=True):
    rng = np.random.default_rng(seed)
    shape = sim.G.shape
    G = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if real_ky0:
        G[:, :, :, 0, :] = G[:, :, :, 0, :].real
    return jnp.asarray(G)


def _ref_solver(sim):
    g = sim.cfg.grid
    return ref_fields.build_field_solver(sim.geom, sim.ky, sim.prof["Ti"], sim.prof["Te"], g.M, g.Lx,
                                         dtype=sim.dtype)


def _ref_ops(sim):
    return ref_vlasov.build_operators(sim.geom, sim.prof, sim.cfg, sim.ky, sim.geom.twist_phase, sim.dtype)


def ref_bracket_skew(a, b, ky, dx, Ny):
    """The skew-symmetric bracket (the default since the blow-up fix) written with the frozen
    baseline transforms, so the optimised transforms are checked underneath the new form too."""
    nky = a.shape[-2]
    ddx, to_real_y, to_spec_y = ref_operators.ddx, ref_operators.to_real_y, ref_operators.to_spec_y
    ax, bx = ddx(a, dx), ddx(b, dx)
    ay, by = 1j * ky * a, 1j * ky * b
    r = lambda f: to_real_y(f, Ny)
    ra, rb = r(ax), r(ay)
    term_y = to_spec_y(ra * r(by), nky) + 1j * ky * to_spec_y(ra * r(b), nky)
    term_x = to_spec_y(rb * r(bx), nky) + ddx(to_spec_y(rb * r(b), nky), dx)
    return 0.5 * (term_y - term_x)


def _ref_bracket_for(sim_or_form):
    form = sim_or_form if isinstance(sim_or_form, str) else sim_or_form.cfg.numerics.bracket
    return {"advective": ref_operators.bracket, "skew": ref_bracket_skew}[form]


class _ref_bracket:
    """Context manager: make the frozen ``ref_vlasov.rhs`` use the bracket form of ``sim``."""
    def __init__(self, sim):
        self.form = sim.cfg.numerics.bracket

    def __enter__(self):
        self.saved = ref_vlasov.bracket
        ref_vlasov.bracket = _ref_bracket_for(self.form)

    def __exit__(self, *exc):
        ref_vlasov.bracket = self.saved


def _ref_rhs(sim, G):
    fs, op = _ref_solver(sim), _ref_ops(sim)
    phi = fs.solve(G[0]); Phi = fs.gyroaverage(phi)
    with _ref_bracket(sim):
        return ref_vlasov.rhs(G, phi, Phi, op)


def _close(a, b, rtol=RTOL, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    scale = max(np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale
    assert err < rtol, f"{what}: relative error {err:.2e}"


@pytest.fixture(scope="module")
def sims():
    return {name: Simulation(_cfg(**over)) for name, over in CONFIGS.items()}


# ----------------------------------------------------------------------------- field solver
@pytest.mark.parametrize("name", list(CONFIGS))
def test_field_solver_matches_reference(sims, name):
    sim = sims[name]
    fs, rf = sim.fields, _ref_solver(sim)
    for seed in (0, 1):
        G = _random_state(sim, seed, real_ky0=(seed == 0))
        phi_ref = rf.solve(G[0]); phi_new = fs.solve(G[0])
        _close(phi_new, phi_ref, what="solve")
        _close(fs.gyroaverage(phi_ref), rf.gyroaverage(phi_ref), what="gyroaverage")
        _close(fs.gyro_density(G[0]), rf.gyro_density(G[0]), what="gyro_density")
        assert fs.gyroaverage(phi_ref).shape == (sim.cfg.grid.M,) + tuple(phi_ref.shape)
        assert fs.gyro_density(G[0]).shape == tuple(phi_ref.shape)


def test_field_solver_under_jit_and_grad(sims):
    """The solver is used inside ``jax.jit``; make sure the new layout traces and differentiates."""
    sim = sims["s_alpha_nonlinear"]
    fs = sim.fields
    G0 = _random_state(sim)[0]
    f = jax.jit(lambda g: jnp.sum(jnp.abs(fs.gyroaverage(fs.solve(g))) ** 2))
    val = f(G0)
    assert np.isfinite(float(val))
    # forward-mode derivative along a random direction agrees with a finite difference
    dG = _random_state(sim, 1)[0]
    _, jvp = jax.jvp(f, (G0,), (dG,))
    h = 1e-6
    fd = (f(G0 + h * dG) - f(G0 - h * dG)) / (2 * h)
    assert abs(float(jvp) - float(fd)) < 1e-6 * abs(float(fd))


def test_gyroaverage_and_density_are_adjoint(sims):
    """⟨G₀, 𝒥 φ⟩ = ⟨𝒥† G₀, φ⟩ per (k_y, z); the operators are the same Hermitian matrices."""
    for name in ("s_alpha_nonlinear", "slab_nonlinear", "odd_sizes"):
        sim = sims[name]
        fs = sim.fields
        G0 = _random_state(sim, 3, real_ky0=False)[0]
        phi = _random_state(sim, 4, real_ky0=False)[0, 0]
        lhs = jnp.sum(jnp.conj(G0) * fs.gyroaverage(phi), axis=(0, 1))          # (Nky, Nz)
        rhs = jnp.sum(jnp.conj(fs.gyro_density(G0)) * phi, axis=0)              # (Nky, Nz)
        _close(lhs, rhs, rtol=1e-12, what=f"adjoint {name}")


def test_solve_is_linear_and_gauge_free(sims):
    sim = sims["s_alpha_nonlinear"]
    fs = sim.fields
    A, B = _random_state(sim, 5)[0], _random_state(sim, 6)[0]
    pa, pb, pab = fs.solve(A), fs.solve(B), fs.solve(0.3 * A - 1.7j * B)
    _close(pab, 0.3 * pa - 1.7j * pb, rtol=1e-12, what="linearity")
    assert float(jnp.abs(jnp.mean(pa[:, 0, :]))) < 1e-12 * float(jnp.max(jnp.abs(pa)))
    assert np.all(np.isfinite(np.asarray(pa)))


def test_gyroaverage_uniform_slab_symbol(sims):
    """In a uniform slab the operator must be 𝒥_m(b) on each Fourier mode (independent of layout)."""
    g = geometry.slab(32, 20.0, 12, Ly=2 * np.pi * 5, Nz=4)
    ky = operators.ky_grid(12, g.Ly)
    fs = fields.build_field_solver(g, ky, np.ones(32), np.ones(32), 4, 20.0)
    for n in (0, 2, 5):
        kx = 2 * np.pi * n / 20.0
        phi = np.exp(1j * kx * g.x)[:, None, None] * np.ones((1, len(ky), len(g.z)))
        b = 0.5 * (kx ** 2 + ky ** 2)
        expect = basis.gyro_coefficients(b, 4)[:, None, :, None] * phi[None]
        _close(fs.gyroaverage(jnp.asarray(phi)), expect, rtol=1e-10, what=f"symbol n={n}")


# ----------------------------------------------------------------------------- spatial operators
@pytest.mark.parametrize("Ny", [6, 10, 12, 24, 30, 46, 48])
def test_y_transforms_match_reference(Ny):
    nky = operators.n_ky(Ny)
    rng = np.random.default_rng(Ny)
    f = rng.normal(size=(3, 2, 5, nky, 4)) + 1j * rng.normal(size=(3, 2, 5, nky, 4))
    fj = jnp.asarray(f)
    r_new, r_ref = operators.to_real_y(fj, Ny), ref_operators.to_real_y(fj, Ny)
    assert r_new.shape == r_ref.shape == (3, 2, 5, Ny, 4)
    assert not jnp.iscomplexobj(r_new)
    _close(r_new, r_ref, rtol=1e-12, what="to_real_y")
    s_new, s_ref = operators.to_spec_y(r_ref, nky), ref_operators.to_spec_y(r_ref, nky)
    _close(s_new, s_ref, rtol=1e-12, what="to_spec_y")
    coef = jnp.asarray(rng.normal(size=(5, Ny, 4)))
    _close(operators.multiply_real_y(coef, fj, Ny), ref_operators.multiply_real_y(coef, fj, Ny),
           rtol=1e-12, what="multiply_real_y")
    # also under jit, with the transform matrices folded in as constants
    _close(jax.jit(lambda x: operators.to_spec_y(operators.to_real_y(x, Ny), nky))(fj),
           jax.jit(lambda x: ref_operators.to_spec_y(ref_operators.to_real_y(x, Ny), nky))(fj),
           rtol=1e-12, what="jit roundtrip")


def test_y_transforms_roundtrip_and_reality():
    """Spectral → real → spectral is the identity on the retained modes (k_y = 0 real part)."""
    for Ny in (12, 24, 48):
        nky = operators.n_ky(Ny)
        rng = np.random.default_rng(1)
        f = rng.normal(size=(2, nky, 3)) + 1j * rng.normal(size=(2, nky, 3))
        f[:, 0] = f[:, 0].real
        g = operators.to_spec_y(operators.to_real_y(jnp.asarray(f), Ny), nky)
        _close(g, f, rtol=1e-12, what=f"roundtrip Ny={Ny}")
        # real-space values of a single mode are cos/sin
        one = np.zeros((1, nky, 1), dtype=complex); one[0, 1, 0] = 1.0
        y = 2 * np.pi * np.arange(Ny) / Ny
        _close(operators.to_real_y(jnp.asarray(one), Ny)[0, :, 0], 2 * np.cos(y), rtol=1e-12, what="cos")
        one[0, 1, 0] = 1j
        _close(operators.to_real_y(jnp.asarray(one), Ny)[0, :, 0], -2 * np.sin(y), rtol=1e-12, what="sin")


def test_y_product_is_dealiased():
    """The product of two spectral fields through the real grid equals the exact truncated
    convolution of their (Hermitian-extended) coefficients: no aliasing with the 2/3 rule."""
    Ny = 24
    nky = operators.n_ky(Ny)
    rng = np.random.default_rng(2)
    a = rng.normal(size=nky) + 1j * rng.normal(size=nky); a[0] = a[0].real
    b = rng.normal(size=nky) + 1j * rng.normal(size=nky); b[0] = b[0].real
    K = nky - 1
    full = lambda c: np.concatenate([np.conj(c[:0:-1]), c])           # modes -K..K
    conv = np.convolve(full(a), full(b))                                # modes -2K..2K
    exact = conv[2 * K: 2 * K + nky]                                    # modes 0..K
    ar = operators.to_real_y(jnp.asarray(a)[None, :, None], Ny)
    br = operators.to_real_y(jnp.asarray(b)[None, :, None], Ny)
    prod = operators.to_spec_y(ar * br, nky)[0, :, 0]
    _close(prod, exact, rtol=1e-12, what="dealiased product")


def test_stencils_match_reference():
    rng = np.random.default_rng(3)
    f = jnp.asarray(rng.normal(size=(4, 2, 12, 5, 8)) + 1j * rng.normal(size=(4, 2, 12, 5, 8)))
    phase = jnp.asarray(np.exp(-1j * rng.normal(size=(12, 5, 1))))
    for name, new, ref in (
        ("ddx", operators.ddx(f, 0.37), ref_operators.ddx(f, 0.37)),
        ("hyper_x", operators.hyper_x(f), ref_operators.hyper_x(f)),
        ("ddz", operators.ddz(f, 0.21, phase), ref_operators.ddz(f, 0.21, phase)),
        ("hyper_z", operators.hyper_z(f, phase), ref_operators.hyper_z(f, phase)),
        ("z_ghosts", operators.z_ghosts(f, phase, 2), ref_operators.z_ghosts(f, phase, 2)),
    ):
        _close(new, ref, rtol=1e-13, what=name)


def test_brackets_match_reference():
    """Both forms against the frozen transforms: the advective form against the baseline
    ``bracket``, the skew form against the same formula written with the baseline primitives."""
    Ny, Nz, Lx, Ly = 24, 3, 10.0, 12.0
    nky = operators.n_ky(Ny)
    ky = jnp.asarray(2 * np.pi * np.arange(nky) / Ly)[None, :, None]
    w = np.where(np.arange(nky) == 0, 1.0, 2.0)[None, None, None, :, None]
    rng = np.random.default_rng(4)
    for Nx in (16, 20):
        a = jnp.asarray(rng.normal(size=(1, 3, Nx, nky, Nz)) + 1j * rng.normal(size=(1, 3, Nx, nky, Nz)))
        b = jnp.asarray(rng.normal(size=(5, 3, Nx, nky, Nz)) + 1j * rng.normal(size=(5, 3, Nx, nky, Nz)))
        adv = operators.bracket_advective(a, b, ky, Lx / Nx, Ny)
        _close(adv, ref_operators.bracket(a, b, ky, Lx / Nx, Ny), rtol=1e-12, what=f"advective Nx={Nx}")
        ba = operators.bracket_advective(b, a, ky, Lx / Nx, Ny)
        assert np.max(np.abs(np.asarray(adv + ba))) < 1e-12 * np.max(np.abs(np.asarray(adv)))
        skew = operators.bracket(a, b, ky, Lx / Nx, Ny)
        _close(skew, ref_bracket_skew(a, b, ky, Lx / Nx, Ny), rtol=1e-12, what=f"skew Nx={Nx}")
        assert operators.BRACKETS["skew"] is operators.bracket and operators.BRACKETS["advective"] is operators.bracket_advective
        # the skew form's defining property, with the optimised transforms: Σ b {a, b} = 0 for grid-scale fields
        inj = np.sum(np.real(np.conj(np.asarray(b)) * np.asarray(skew)) * w)
        scale = np.sqrt(np.sum(np.abs(np.asarray(b)) ** 2 * w) * np.sum(np.abs(np.asarray(skew)) ** 2 * w))
        assert abs(inj) / scale < 1e-13
        ab = adv
        # real bilinearity (the real-space product is only real-linear in the spectral coefficients)
        c = 0.7
        for br in (operators.bracket, operators.bracket_advective):
            ab = br(a, b, ky, Lx / Nx, Ny)
            _close(br(a, c * b, ky, Lx / Nx, Ny), c * ab, rtol=1e-12, what="bilinear")
            a2 = jnp.asarray(rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape))
            _close(br(a + a2, b, ky, Lx / Nx, Ny), ab + br(a2, b, ky, Lx / Nx, Ny), rtol=1e-12, what="additive")


# ----------------------------------------------------------------------------- Vlasov rhs
@pytest.mark.parametrize("name", list(CONFIGS))
def test_rhs_matches_reference(sims, name):
    sim = sims[name]
    for seed in (0, 1):
        G = _random_state(sim, seed, real_ky0=(seed == 0))
        _close(sim._rhs(G), _ref_rhs(sim, G), what=f"rhs {name} seed={seed}")


@pytest.mark.parametrize("name", list(CONFIGS))
def test_rhs_pieces_match_reference(sims, name):
    """Each term family in isolation: switch the others off through the operator pytree so a
    compensating error between two terms cannot hide."""
    import equinox as eqx
    sim = sims[name]
    op, rop = sim.op, _ref_ops(sim)
    fs = sim.fields
    G = _random_state(sim, 7)
    phi = fs.solve(G[0]); Phi = fs.gyroaverage(phi)
    zero = lambda a: jnp.zeros_like(a)

    def keep(o, names):
        """Zero every coefficient field except ``names`` (static flags are left alone)."""
        fields_ = ["c_stream", "c_mirror", "c_Kx", "c_Ky", "c_Cinv", "c_omegaB", "omega_n", "omega_T", "nu", "krook"]
        for f in fields_:
            if f not in names:
                o = eqx.tree_at(lambda o_, f=f: getattr(o_, f), o, zero(getattr(o, f)))
        return o

    for names in (["c_stream"], ["c_mirror"], ["c_Kx", "c_Ky"], ["c_omegaB", "omega_n", "omega_T"],
                  ["c_Cinv"], ["nu"], ["krook"], []):
        new = vlasov.rhs(G, phi, Phi, keep(op, names))
        with _ref_bracket(sim):
            ref = ref_vlasov.rhs(G, phi, Phi, keep(rop, names))
        _close(new, ref, what=f"{name} terms={names or 'dissipation+source only'}")


def test_rhs_matches_reference_on_saturated_snapshot():
    # results are not versioned (*.npz is git-ignored): a saturated snapshot, if one has been saved here
    candidates = [ROOT / "results" / "reference.npz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("no saturated snapshot available")
    d = np.load(path, allow_pickle=True)
    snap = d["snapshots"][-1]
    L, M, Nx, Nky, Nz = snap.shape
    # the comparison configuration (_cfg: what the frozen reference implements) on the snapshot's grid.  It was
    # a production case that follows the defaults (drive_gradB off and the skew parallel form
    # since 2026-09-21, collisions on h since 10-03), so this test failed wherever the snapshot exists (15 % on the Mac
    # run of 2026-10-04; the cards have no snapshot and skip it).
    sim = Simulation(_cfg(**{"grid.Nx": Nx, "grid.Ny": 3 * (Nky - 1) + 1, "grid.Nz": Nz, "grid.L": L, "grid.M": M}))
    G = jnp.asarray(snap)
    _close(sim._rhs(G), _ref_rhs(sim, G), what="rhs on saturated turbulence")
    dt = sim.dt_from_cfl(G)
    Gn = sim._nsteps(G, dt, 3)
    assert np.all(np.isfinite(np.asarray(Gn)))


def test_cfl_rates_match_reference(sims):
    for name, sim in sims.items():
        G = _random_state(sim, 8)
        fs = sim.fields
        phi = fs.solve(G[0]); Phi = fs.gyroaverage(phi)
        L, M = sim.cfg.grid.L, sim.cfg.grid.M
        new = vlasov.cfl_rates(G, phi, Phi, sim.op, L, M)
        ref = ref_vlasov.cfl_rates(G, phi, Phi, _ref_ops(sim), L, M)
        assert abs(float(new) - float(ref)) < 1e-11 * abs(float(ref)), name


def test_operator_pytree_keeps_reference_arrays(sims):
    """Every array the baseline operator carried is still there with identical values."""
    for name, sim in sims.items():
        op, rop = sim.op, _ref_ops(sim)
        for f in ("V", "mirror", "energy", "drive_T", "mu", "coll", "hypv", "lag_to_nodes", "nodes_to_lag",
                  "c_stream", "c_mirror", "c_Kx", "c_Ky", "c_Cinv", "c_omegaB", "Ti", "inv_Ti", "omega_n",
                  "omega_T", "nu", "krook", "ky", "phase", "fsw"):
            _close(getattr(op, f), getattr(rop, f), rtol=1e-14, what=f"{name}.{f}")
        for f in ("kymax", "dx", "dz", "Ny", "axisymmetric", "hyper_x", "hyper_z", "hyper_ky", "source_rate", "nonlinear"):
            assert getattr(op, f) == getattr(rop, f), (name, f)


def test_rhs_is_jit_and_vmap_friendly(sims):
    sim = sims["s_alpha_nonlinear"]
    G = _random_state(sim, 9)
    f = jax.jit(sim._rhs)
    _close(f(G), sim._rhs(G), rtol=1e-14, what="jit")
    Gs = jnp.stack([G, 0.5 * G])
    out = jax.vmap(sim._rhs)(Gs)
    _close(out[1], sim._rhs(0.5 * G), rtol=1e-13, what="vmap")


# ----------------------------------------------------------------------------- time advance
def test_rk4_steps_match_reference_rk4(sims):
    for name in ("s_alpha_nonlinear", "slab_nonlinear", "odd_sizes", "advective_bracket"):
        sim = sims[name]
        G = 1e-2 * _random_state(sim, 10)
        dt = sim.dt_from_cfl(G)
        rf, rop = _ref_solver(sim), _ref_ops(sim)

        def f(G):
            phi = rf.solve(G[0]); return ref_vlasov.rhs(G, phi, rf.gyroaverage(phi), rop)

        Gr = G
        with _ref_bracket(sim):
            for _ in range(4):
                k1 = f(Gr); k2 = f(Gr + 0.5 * dt * k1); k3 = f(Gr + 0.5 * dt * k2); k4 = f(Gr + dt * k3)
                Gr = Gr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        _close(sim._nsteps(G, dt, 4), Gr, rtol=1e-10, what=f"4 RK4 steps {name}")


def test_short_run_is_deterministic_and_finite():
    a = Simulation(_cfg()).run(verbose=False)
    b = Simulation(_cfg()).run(verbose=False)
    assert np.all(np.isfinite(a.phi2)) and len(a.t) == 3
    assert np.array_equal(np.asarray(a.phi2), np.asarray(b.phi2))
    assert np.array_equal(np.asarray(a.Q), np.asarray(b.Q))


def test_diagnostics_match_reference_formulas(sims):
    """Diagnostics are computed from the new field solve; they must agree with the same formulas
    evaluated on the reference solve."""
    sim = sims["s_alpha_nonlinear"]
    G = _random_state(sim, 11)
    d = sim._diagnostics(G)
    rf = _ref_solver(sim)
    phi = rf.solve(G[0])
    w = np.where(np.arange(phi.shape[1]) == 0, 1.0, 2.0)
    _close(d["phi2"], np.sum(np.abs(np.asarray(phi)) ** 2 * w[None, :, None]), rtol=1e-12, what="phi2")
    _close(d["phi_ky"], np.sum(np.abs(np.asarray(phi)) ** 2, axis=(0, 2)), rtol=1e-12, what="phi_ky")
    _close(d["W"], np.sum(np.abs(np.asarray(G)) ** 2 * w[None, None, None, :, None]), rtol=1e-12, what="W")


def test_fp32_path_runs_and_tracks_fp64():
    sim64 = Simulation(_cfg())
    G = _random_state(sim64, 12)
    r64 = np.asarray(sim64._rhs(G))
    sim32 = Simulation(_cfg(**{"run.precision": "fp32"}))
    assert sim32.G.dtype == jnp.complex64
    r32 = np.asarray(sim32._rhs(jnp.asarray(G, dtype=jnp.complex64)))
    assert r32.dtype == np.complex64
    assert np.all(np.isfinite(r32))
    assert np.max(np.abs(r32 - r64)) < 1e-4 * np.max(np.abs(r64))
    jax.config.update("jax_enable_x64", True)     # Simulation(fp32) flips the global flag; restore for later tests
