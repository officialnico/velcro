"""An independent reference for velcro/miller.py: the Miller local equilibrium built by brute force from the
Grad–Shafranov equation — no Bishop integrals (a_s, b_s, c_s), no Miller eq. (21), no dB/dr formula, no aprime.

Construction (all in GX's normalisation, a = 1, F = R_geo, μ0 p = β/2 so dp/dρ = β'/2):

* the Miller family R(ρ, t) = R0(ρ) + ρ cos(t + arcsin δ(ρ) sin t), Z = κ(ρ) ρ sin t with R0' = −|shift|, κ' = akappri,
  δ' = tripri (optionally with extra second-order terms ``c2`` — the physics must not depend on them: a gauge test of
  this reference); first derivatives by complex step, the Hessian of ρ(R, Z) by 4th-order differences of those;
* ψ near the surface: ψ(ρ, t) = ψ1 (ρ − ρc) + ½ g(t) (ρ − ρc)², with ψ1 from q on the central surface and g(t) from
  the Grad–Shafranov equation at ρc:  Δ*ψ = −R² p_ψ − F F_ψ  ⇒  g |∇ρ|² = −R² p_ψ − F F_ψ − ψ1 Δ*ρ;
* q(ψ) on the neighbouring true flux surfaces ψ = ±δψ (q = −(F/2π) ∮ J_ρt / (R ∂_ρψ) dt for B = F∇φ + ∇φ×∇ψ), and
  F_ψ fixed so that ŝ = (ρ/q) dq/dρ is the requested one (dq/dψ by Richardson-extrapolated central differences);
* the straight-field-line angle θ_f(ψ, t) on each true surface, ∂θ_f/∂ψ|_t by central differences, so that
  α = φ − q(ψ) θ_f and ∇α are explicit vectors; ∇B from the Hessian of ψ, the curvature b·∇b from the explicit field;
* the GS2/GX arrays from their definitions (x = (q/ρ) ψ, y = ψ1 α):  gds2 = ψ1² |∇α|², gds21 = q_ρ ψ1 ∇α·∇ψ,
  gds22 = q_ρ² |∇ψ|², gbdrift = 2 ψ1 (B×∇B)·∇α / B³, gbdrift0 = 2 q_ρ (B×∇B)·∇ψ / B³, cvdrift = 2 ψ1 (b×κ)·∇α / B,
  grho = |∇ρ|, gradpar = b·∇θ_f  (the un-halved eik convention of velcro.miller.miller_fine).

Numpy only.  Nothing here imports velcro.
"""
from __future__ import annotations

import numpy as np

CS = 1e-30          # complex step


def cumint(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """∫_0^t f on the uniform periodic grid t_j = 2πj/N (rfft; Nyquist dropped): periodic part + mean·t."""
    N = len(f)
    fh = np.fft.rfft(f)
    k = np.arange(len(fh), dtype=float)
    gh = np.zeros_like(fh)
    gh[1:] = fh[1:] / (1j * k[1:])
    if N % 2 == 0:
        gh[-1] = 0.0
    G = np.fft.irfft(gh, n=N)
    return G - G[0] + fh[0].real / N * t


def trig_eval(f: np.ndarray, tq: np.ndarray, deriv: int = 0) -> np.ndarray:
    """The trigonometric interpolant of the periodic samples f (on t_j = 2πj/N) at the points tq (or its derivative)."""
    N = len(f)
    fh = np.fft.rfft(f) / N
    k = np.arange(len(fh), dtype=float)
    w = np.full(len(fh), 2.0)
    w[0] = 1.0
    if N % 2 == 0:
        w[-1] = 1.0
    ph = np.exp(1j * np.outer(np.asarray(tq, float), k))
    c = fh * w * (1j * k) ** deriv
    return np.real(ph @ c)


def _d4(fn, h):
    return (fn(-2 * h) - 8 * fn(-h) + 8 * fn(h) - fn(2 * h)) / (12 * h)


def _rich(fn, h):
    """d/dx at 0 by central differences at h and h/2, Richardson-extrapolated."""
    d1 = (fn(h) - fn(-h)) / (2 * h)
    d2 = (fn(h / 2) - fn(-h / 2)) / h
    return (4 * d2 - d1) / 3


class Family:
    def __init__(self, rhoc, Rmaj, shift=0.0, akappa=1.0, akappri=0.0, tri=0.0, tripri=0.0, c2=(0.0, 0.0, 0.0)):
        self.rhoc, self.Rmaj, self.shift = float(rhoc), float(Rmaj), float(shift)
        self.ak, self.akp, self.tri, self.trip = float(akappa), float(akappri), float(tri), float(tripri)
        self.c2 = tuple(float(c) for c in c2)

    def RZ(self, rho, t):
        d = rho - self.rhoc
        R0 = self.Rmaj - abs(self.shift) * d + self.c2[0] * d * d
        kap = self.ak + self.akp * d + self.c2[1] * d * d
        dl = self.tri + self.trip * d + self.c2[2] * d * d
        return R0 + rho * np.cos(t + np.arcsin(dl) * np.sin(t)), kap * rho * np.sin(t)

    def d1(self, rho, t):
        t = np.asarray(t, float)
        rho = np.asarray(rho, float) * np.ones_like(t)
        Rr, Zr = self.RZ(rho + 1j * CS, t + 0j)
        Rt, Zt = self.RZ(rho + 0j, t + 1j * CS)
        return Rr.imag / CS, Rt.imag / CS, Zr.imag / CS, Zt.imag / CS

    def inv(self, rho, t):
        """ρ_R, ρ_Z, t_R, t_Z and J = R_ρ Z_t − R_t Z_ρ at (ρ, t)."""
        Rr, Rt, Zr, Zt = self.d1(rho, t)
        J = Rr * Zt - Rt * Zr
        return Zt / J, -Rt / J, -Zr / J, Rr / J, J

    def hess_rho(self, rho, t):
        hr, ht = 2e-3 * self.rhoc, 2e-3
        rR, rZ, tR, tZ, _ = self.inv(rho, t)
        dRr = _d4(lambda e: self.inv(rho + e, t)[0], hr)
        dRt = _d4(lambda e: self.inv(rho, t + e)[0], ht)
        dZr = _d4(lambda e: self.inv(rho + e, t)[1], hr)
        dZt = _d4(lambda e: self.inv(rho, t + e)[1], ht)
        return dict(RR=dRr * rR + dRt * tR, RZ=dRr * rZ + dRt * tZ, ZR=dZr * rR + dZt * tR, ZZ=dZr * rZ + dZt * tZ)


def _triple(a, b, c):
    """(a × b)·c for vectors given as (R, φ, Z) component tuples (right-handed)."""
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return x * c[0] + y * c[1] + z * c[2]


class LocalEquilibrium:
    """The reference on t_j = 2πj/N.  ``Fpsi``: impose F' instead of solving it from ŝ (then ŝ_true is reported)."""

    def __init__(self, rhoc, Rmaj, R_geo, q, shat, shift=0.0, akappa=1.0, akappri=0.0, tri=0.0, tripri=0.0,
                 betaprim=0.0, N=1024, c2=(0.0, 0.0, 0.0), Fpsi=None, dpsi_rel=1e-3):
        fam = Family(rhoc, Rmaj, shift, akappa, akappri, tri, tripri, c2)
        self.fam, self.N = fam, int(N)
        t = 2 * np.pi * np.arange(self.N) / self.N
        self.t = t
        R, Z = fam.RZ(rhoc, t)
        rR, rZ, tR, tZ, J = fam.inv(rhoc, t)
        if np.any(J <= 0) or np.any(R <= 0):
            raise ValueError("reference: J ≤ 0 or R ≤ 0 on the surface")
        F0 = float(R_geo)
        psi1 = -F0 * np.mean(J / R) / q                       # q = −F ⟨J/(R ψ1)⟩
        ppsi = 0.5 * betaprim / psi1
        H = fam.hess_rho(rhoc, t)
        dstar = H["RR"] + H["ZZ"] - rR / R
        grho2 = rR ** 2 + rZ ** 2
        sgn = np.sign(psi1)

        def g_of(Fp):
            return (-R ** 2 * ppsi - F0 * Fp - psi1 * dstar) / grho2

        def surface(dpsi, Fp):
            g = g_of(Fp)
            if dpsi == 0.0:
                eta = np.zeros_like(t)
            else:
                eta = 2 * dpsi / (psi1 + sgn * np.sqrt(psi1 ** 2 + 2 * g * dpsi))
            Rr, Rt, Zr, Zt = fam.d1(rhoc + eta, t)
            Rs, _ = fam.RZ(rhoc + eta, t)
            w = (Rr * Zt - Rt * Zr) / (Rs * (psi1 + g * eta))      # J_ψt / R
            qs = -(F0 + Fp * dpsi) * np.mean(w)
            return qs, cumint(w, t) / np.mean(w)

        h = dpsi_rel * abs(psi1) * rhoc
        q1_target = shat * q / (rhoc * psi1)                         # dq/dψ = (ŝ q/ρ)/ψ1
        q1_0 = _rich(lambda d: surface(d, 0.0)[0], h)
        q1_1 = _rich(lambda d: surface(d, 1.0)[0], h)
        Fpsi_solved = (q1_target - q1_0) / (q1_1 - q1_0)
        self.Fpsi_solved = float(Fpsi_solved)
        Fp = Fpsi_solved if Fpsi is None else float(Fpsi)
        q1 = _rich(lambda d: surface(d, Fp)[0], h)
        self.Fpsi, self.q1 = float(Fp), float(q1)
        self.shat_true = float(rhoc * psi1 * q1 / q)
        q0, thf = surface(0.0, Fp)
        self.q_check = float(q0)
        dthf_dpsi = _rich(lambda d: surface(d, Fp)[1], h)
        w0 = J / (R * psi1)
        dthf_dt = w0 / np.mean(w0)
        g0 = g_of(Fp)

        # gradients on the central surface (cylindrical R, φ, Z components)
        psiR, psiZ = psi1 * rR, psi1 * rZ
        thR = dthf_dpsi * psiR + dthf_dt * tR
        thZ = dthf_dpsi * psiZ + dthf_dt * tZ
        pRR = psi1 * H["RR"] + g0 * rR * rR
        pRZ = psi1 * 0.5 * (H["RZ"] + H["ZR"]) + g0 * rR * rZ
        pZZ = psi1 * H["ZZ"] + g0 * rZ * rZ
        BR, Bphi, BZ = psiZ / R, F0 / R, -psiR / R
        B = np.sqrt(BR ** 2 + Bphi ** 2 + BZ ** 2)
        dBR = (pRZ / R - psiZ / R ** 2, pZZ / R)
        dBphi = (Fp * psiR / R - F0 / R ** 2, Fp * psiZ / R)
        dBZ = (-pRR / R + psiR / R ** 2, -pRZ / R)
        BgR = (BR * dBR[0] + Bphi * dBphi[0] + BZ * dBZ[0]) / B
        BgZ = (BR * dBR[1] + Bphi * dBphi[1] + BZ * dBZ[1]) / B
        bR, bphi, bZ = BR / B, Bphi / B, BZ / B
        db = {k: tuple(dc[i] / B - bc * (BgR if i == 0 else BgZ) / B for i in range(2))
              for k, dc, bc in (("R", dBR, bR), ("phi", dBphi, bphi), ("Z", dBZ, bZ))}
        kR = bR * db["R"][0] + bZ * db["R"][1] - bphi ** 2 / R
        kphi = bR * db["phi"][0] + bZ * db["phi"][1] + bphi * bR / R
        kZ = bR * db["Z"][0] + bZ * db["Z"][1]
        self.ing = dict(R=R, Z=Z, psiR=psiR, psiZ=psiZ, thR=thR, thZ=thZ, BR=BR, Bphi=Bphi, BZ=BZ, B=B, BgR=BgR,
                        BgZ=BgZ, kR=kR, kphi=kphi, kZ=kZ, grho=np.sqrt(grho2), dthf_dt=dthf_dt, dthf_dpsi=dthf_dpsi,
                        thf_minus_t=thf - t)
        self.scal = dict(psi1=float(psi1), q=float(q), q1=float(q1), qrho=float(q1 * psi1), F0=F0, Fpsi=float(Fp),
                         ppsi=float(ppsi), rhoc=float(rhoc))
        self.theta_f = thf
        self.arrays = self.arrays_from(self.ing, thf)
        # the equal-arc angle of this reference: dθ_eq/dt = gradpar_eq (dθ_f/dt)/gradpar
        gp = self.arrays["gradpar"]
        self.gradpar_eq = float(2 * np.pi / (2 * np.pi * np.mean(dthf_dt / gp)))
        self.theta_eq = self.gradpar_eq * cumint(dthf_dt / gp, t)
        self.dpsidrho, self.kxfac = float(psi1), float(abs(q * psi1 / rhoc))

    def arrays_from(self, ing: dict, thf: np.ndarray) -> dict:
        s = self.scal
        psi1, q, q1, qrho = s["psi1"], s["q"], s["q1"], s["qrho"]
        R = ing["R"]
        aR = -q1 * thf * ing["psiR"] - q * ing["thR"]
        aZ = -q1 * thf * ing["psiZ"] - q * ing["thZ"]
        aphi = 1.0 / R
        Bv = (ing["BR"], ing["Bphi"], ing["BZ"])
        gB = (ing["BgR"], 0.0 * R, ing["BgZ"])
        ga = (aR, aphi, aZ)
        gpsi = (ing["psiR"], 0.0 * R, ing["psiZ"])
        B = ing["B"]
        bv = tuple(c / B for c in Bv)
        kv = (ing["kR"], ing["kphi"], ing["kZ"])
        return dict(
            bmag=B, grho=ing["grho"],
            gradpar=(ing["BR"] * ing["thR"] + ing["BZ"] * ing["thZ"]) / B,
            gds2=psi1 ** 2 * (aR ** 2 + aZ ** 2 + aphi ** 2),
            gds21=qrho * psi1 * (aR * ing["psiR"] + aZ * ing["psiZ"]),
            gds22=qrho ** 2 * (ing["psiR"] ** 2 + ing["psiZ"] ** 2),
            gbdrift=2 * psi1 * _triple(Bv, gB, ga) / B ** 3,
            gbdrift0=2 * qrho * _triple(Bv, gB, gpsi) / B ** 3,
            cvdrift=2 * psi1 * _triple(bv, kv, ga) / B,
            B_dot_grad_alpha=Bv[0] * aR + Bv[1] * aphi + Bv[2] * aZ,
            bgrad=(ing["BR"] * ing["BgR"] + ing["BZ"] * ing["BgZ"]) / B ** 2,        # b·∇ ln B (GX's bgrad, exact)
        )

    def at_theta_eq(self, z: np.ndarray) -> dict:
        """The arrays at the equal-arc angles z ∈ [−π, π) (θ_f continued from −π to π, as GX's grid): t from
        θ_eq(t) = z mod 2π by Newton on the trigonometric interpolant, every periodic ingredient interpolated."""
        z = np.asarray(z, float)
        zz = np.mod(z, 2 * np.pi)
        per = self.theta_eq - self.t                                 # periodic
        tq = zz.copy()
        for _ in range(60):
            fval = tq + trig_eval(per, tq) - zz
            tq = tq - fval / (1.0 + trig_eval(per, tq, 1))
        ing = {k: trig_eval(v, tq) for k, v in self.ing.items() if isinstance(v, np.ndarray)}
        thf = tq + ing["thf_minus_t"]
        thf = np.where(z < 0, thf - 2 * np.pi, thf)
        out = self.arrays_from(ing, thf)
        out["t"] = tq
        out["theta_f"] = thf
        return out
