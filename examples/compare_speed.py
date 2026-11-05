"""Run GKX on a case for a fixed wall budget, then Velcro on the same case for the same budget, and compare them.

Each code runs in its own process, one after the other on the same GPU (XLA_PYTHON_CLIENT_PREALLOCATE=false for both),
for at most --budget wall seconds (default 180) counted from process start, start-up included. Each advances its own
deck in chunks of simulated time and records the wall clock after every chunk. Reported per code:

* start-up: wall seconds from process start until the code is stepping at its steady rate, which covers imports, set-up
  and compilation. It is the end of the first chunk minus one steady chunk.
* speed: wall seconds per a/v_ti over the chunks after the first (median).
* reach: the simulated time it got to inside the budget.
* a chart, compare_speed.png: progress (simulated time against wall time) and the heat fluxes against time.
* accuracy: the ion (and, with kinetic electrons, electron) heat flux, averaged over the window both codes reached
  (from --window-start, default 20 a/v_ti, to the smaller of their final times), in GX/GKX gyro-Bohm units, with block
  standard errors, the difference and its z-score. The codes start from different noise, so turbulent fluxes agree only
  in the mean: |z| <~ 2 is what agreement looks like. Short windows give large errors.

    python examples/compare_speed.py --pair ke              # examples/gkx_ke: gkx_cyclone.toml, velcro_cyclone.toml
    python examples/compare_speed.py --pair adiabatic
    python examples/compare_speed.py --pair coll_em          # collisions + delta-B_par, examples/gkx_ke/*_coll_em.toml
    python examples/compare_speed.py --gkx my.toml --velcro my_velcro.toml --budget 300

Needs GKX 2.8.1 (`import gkx`) and Velcro (`import velcro`) in the Python that runs this script. Velcro's chunks are
separate Simulation.run calls, so an adaptive config re-estimates its step at every chunk.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T_START = time.perf_counter()
HERE = Path(__file__).resolve().parent
PAIRS = {
    "ke": (HERE / "gkx_ke" / "gkx_cyclone.toml", HERE / "gkx_ke" / "velcro_cyclone.toml"),
    "adiabatic": (HERE / "gkx_adiabatic" / "gkx_cyclone_adiabatic.toml", HERE / "gkx_adiabatic" / "velcro_cyclone_adiabatic.toml"),
    "coll_em": (HERE / "gkx_ke" / "gkx_cyclone_coll_em.toml", HERE / "gkx_ke" / "velcro_cyclone_coll_em.toml"),
}
TIME_FACTOR = 1.9642   # t[a/v_ti] = TIME_FACTOR * t[R/v_t] for R/a = 2.77778, the examples' geometry
Q_FACTOR = 0.3666      # Q[GX gyro-Bohm] = Q_FACTOR * Q[Velcro] there: 2 sqrt(2) (a/R)^2 in s-alpha


def elapsed() -> float:
    return time.perf_counter() - T_START


def emit(event: dict) -> None:
    """One line per finished chunk on stdout, for a live display (examples/race.py) to read."""
    print("RACE " + json.dumps(event), flush=True)


# --------------------------------------------------------------------------------------------------------- workers
def worker_gkx(deck: str, budget: float, chunk: float) -> dict:
    """GKX 2.8.1 through its API: gkx.load, gkx.prepare for one chunk, then the prepared backend chunk after chunk."""
    from dataclasses import replace

    import numpy as np
    t_imp = elapsed()
    import gkx
    import jax
    t_imp = elapsed() - t_imp
    case = gkx.load(deck)
    tm = case.time
    fixed = bool(tm.fixed_dt) and tm.dt is not None
    if fixed:
        n = max(1, int(round(chunk / float(tm.dt))))
        h, steps = n * float(tm.dt), n
        tkw = dict(t_max=h, sample_stride=n, diagnostics_stride=n, run_to="t_max")
    else:                                    # an adaptive stepper: each chunk integrates to t_max = chunk
        h = chunk
        steps = int(max(200, 3 * chunk / max(float(tm.dt or 0.01), 1e-3)))
        tkw = dict(t_max=h, sample_stride=1, diagnostics_stride=1, run_to="t_max")
    case = case.replace(time=replace(tm, **tkw))
    kw = dict(ky_target=float(case.run.ky)) if getattr(case.run, "ky", None) is not None else {}
    prep = gkx.prepare(case, steps=steps, Nl=int(case.run.Nl), Nm=int(case.run.Nm), **kw)
    be = prep._backend
    G = be.initial_state
    t_off, rows, ts, qi, qe = 0.0, [], [], [], []
    while True:
        w0 = elapsed()
        _, d, G, _ = be.run(G)
        jax.block_until_ready(G)
        w1 = elapsed()
        tt = t_off + np.asarray(d.t, dtype=float)
        q = np.atleast_2d(np.asarray(d.heat_flux_species_t, dtype=float))
        dt_reached = float(np.asarray(d.t, dtype=float)[-1]) if len(np.atleast_1d(d.t)) else h
        t_off += dt_reached if not fixed else h
        ts += tt.tolist()
        qi += q[:, 0].tolist()
        qe += (q[:, 1] if q.shape[1] > 1 else np.full(len(q), np.nan)).tolist()
        rows.append(dict(wall_end=w1, wall=w1 - w0, t_end=t_off))
        emit(dict(kind="lap", wall_end=w1, t_end=t_off, t=tt.tolist(), qi=q[:, 0].tolist(),
                  qe=(q[:, 1] if q.shape[1] > 1 else np.full(len(q), np.nan)).tolist()))
        if not np.all(np.isfinite(q)):
            return dict(code="GKX", rows=rows, t=ts, Qi=qi, Qe=qe, import_s=t_imp, stop="non-finite")
        if elapsed() + (w1 - w0) > budget:
            return dict(code="GKX", rows=rows, t=ts, Qi=qi, Qe=qe, import_s=t_imp, stop="budget")


def worker_velcro(config: str, budget: float, chunk: float, time_factor: float, q_factor: float) -> dict:
    """Velcro through its API: Simulation(config), then Simulation.run to the next chunk end, chunk after chunk."""
    import numpy as np
    t_imp = elapsed()
    import jax
    from velcro.config import VelcroConfig
    from velcro.simulation import Simulation
    t_imp = elapsed() - t_imp
    sim = Simulation(VelcroConfig.from_file(config))
    hv = chunk / time_factor                 # the chunk in Velcro's time unit
    rows, t_v, k0 = [], 0.0, 0
    while True:
        w0 = elapsed()
        t_v += hv
        sim.run(t_end=t_v, verbose=False)
        jax.block_until_ready(sim.G)
        w1 = elapsed()
        rows.append(dict(wall_end=w1, wall=w1 - w0, t_end=time_factor * float(sim.t)))
        dg = sim.diag.asdict()
        new_t = time_factor * np.asarray(dg["t"][k0:], dtype=float)
        new_qi = q_factor * np.asarray(dg["Q"][k0:], dtype=float)
        new_qe = (q_factor * np.asarray(dg["Qe"][k0:], dtype=float) if "Qe" in dg and len(dg["Qe"]) > k0
                  else np.full(len(new_t), np.nan))
        k0 = len(dg["t"])
        emit(dict(kind="lap", wall_end=w1, t_end=time_factor * float(sim.t), t=new_t.tolist(), qi=new_qi.tolist(),
                  qe=new_qe.tolist()))
        q_last = dg.get("Q", [np.nan])[-1]
        if not np.isfinite(float(q_last)):
            break
        if elapsed() + (w1 - w0) > budget:
            break
    dg = sim.diag.asdict()
    t = time_factor * np.asarray(dg["t"], dtype=float)
    qi = q_factor * np.asarray(dg["Q"], dtype=float)
    qe = q_factor * np.asarray(dg["Qe"], dtype=float) if "Qe" in dg and len(dg["Qe"]) else np.full(len(t), np.nan)
    _, keep = np.unique(t, return_index=True)          # each run() call records its start again: drop the repeats
    return dict(code="Velcro", rows=rows, t=t[keep].tolist(), Qi=qi[keep].tolist(), Qe=qe[keep].tolist(), import_s=t_imp,
                stop="budget" if np.isfinite(qi[-1]) else "non-finite")


# ---------------------------------------------------------------------------------------------------------- report
def summarise(r: dict) -> dict:
    import numpy as np
    rows = r["rows"]
    if not rows:
        return dict(startup_s=None, s_per_a_vti=None, t_reached=0.0)
    spans = [rows[0]["t_end"]] + [b["t_end"] - a["t_end"] for a, b in zip(rows[:-1], rows[1:])]
    per = [row["wall"] / s for row, s in zip(rows[1:], spans[1:]) if s > 0]
    speed = float(np.median(per)) if per else None
    steady_first = speed * spans[0] if speed else 0.0
    return dict(startup_s=rows[0]["wall_end"] - steady_first, s_per_a_vti=speed, t_reached=rows[-1]["t_end"],
                chunks=len(rows), stop=r.get("stop"), import_s=r.get("import_s"))


def window_stats(t, q, t0, t1):
    """Mean and block standard error of q over [t0, t1]: equal-time blocks of about 5 a/v_ti, at least 3."""
    import numpy as np
    t, q = np.asarray(t, float), np.asarray(q, float)
    w = (t >= t0) & (t <= t1) & np.isfinite(q)
    if w.sum() < 6 or t1 <= t0:
        return None
    nb = max(3, int((t1 - t0) // 5))
    edges = np.linspace(t0, t1, nb + 1)
    means = [q[(t >= a) & (t < b) & w].mean() for a, b in zip(edges[:-1], edges[1:]) if ((t >= a) & (t < b) & w).any()]
    if len(means) < 3:
        return None
    return dict(mean=float(np.mean(q[w])), err=float(np.std(means, ddof=1) / np.sqrt(len(means))), blocks=len(means))


COLORS = {"GKX": "#2563eb", "Velcro": "#ea580c"}   # checked for colour-blind separation (worst adjacent dE 31)


def plot(results: dict, acc: dict, window, path: Path) -> Path | None:
    """compare_speed.png: progress (simulated time against wall time), then Q_i and (when present) Q_e against time with
    the comparison window shaded and the window means dashed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is not installed: no chart (pip install matplotlib)")
        return None
    has_qe = any(np.isfinite(np.asarray(r["Qe"], float)).any() for r in results.values())
    n = 3 if has_qe else 2
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0), constrained_layout=True)
    ax = axes[0]
    for code, r in results.items():
        s = r["summary"]
        w = [0.0, s["startup_s"]] + [row["wall_end"] for row in r["rows"]]      # flat during start-up: no progress
        t = [0.0, 0.0] + [row["t_end"] for row in r["rows"]]
        ax.plot(w, t, color=COLORS.get(code), lw=2, marker="o", ms=3, label=f"{code}: start-up {s['startup_s']:.0f} s, "
                f"{s['s_per_a_vti']:.3g} s per a/v_ti")
        ax.axvline(s["startup_s"], color=COLORS.get(code), lw=1, ls=":")
    ax.set_xlabel("wall time, s")
    ax.set_ylabel("simulated time t, a/v_ti")
    ax.set_title("Progress (dotted: start-up)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ax, key, label in zip(axes[1:], ("Qi", "Qe"), ("Q_i", "Q_e")):
        for code, r in results.items():
            t, q = np.asarray(r["t"], float), np.asarray(r[key], float)
            if np.isfinite(q).any():
                ax.plot(t, q, color=COLORS.get(code), lw=1.5, label=code)
                st = acc.get(label, {}).get(code)
                if st:
                    ax.hlines(st["mean"], window[0], window[1], color=COLORS.get(code), lw=1.5, ls="--")
        if window[1] > window[0]:
            ax.axvspan(window[0], window[1], color="0.5", alpha=0.12, lw=0)
        d = acc.get(label, {}).get("velcro_minus_gkx")
        ax.set_title(f"{label}, GX gyro-Bohm units" + (f"  (Velcro - GKX {100 * d['rel']:+.1f} %, z {d['z']:+.2f})" if d else ""),
                     fontsize=10)
        ax.set_xlabel("t, a/v_ti")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", choices=sorted(PAIRS), default="ke")
    ap.add_argument("--gkx", type=Path, help="a GKX TOML deck (replaces the pair's)")
    ap.add_argument("--velcro", type=Path, help="a Velcro TOML/JSON config (replaces the pair's)")
    ap.add_argument("--budget", type=float, default=180.0, help="wall seconds per code, start-up included")
    ap.add_argument("--chunk", type=float, default=2.8, help="simulated time per timed chunk, a/v_ti")
    ap.add_argument("--window-start", type=float, default=20.0, help="start of the flux window, a/v_ti")
    ap.add_argument("--time-factor", type=float, default=TIME_FACTOR, help="t[a/v_ti] per t[Velcro unit]")
    ap.add_argument("--q-factor", type=float, default=Q_FACTOR, help="Q[GX units] per Q[Velcro unit]")
    ap.add_argument("--skip-gkx", action="store_true")
    ap.add_argument("--skip-velcro", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="where to write compare_speed.json (default: a new temp folder)")
    ap.add_argument("--gkx-python", default=sys.executable, help="the Python that runs GKX (its own environment)")
    ap.add_argument("--velcro-python", default=sys.executable, help="the Python that runs Velcro")
    ap.add_argument("--worker", choices=("gkx", "velcro"), help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    gkx_deck, velcro_cfg = (a.gkx or PAIRS[a.pair][0]).resolve(), (a.velcro or PAIRS[a.pair][1]).resolve()
    if a.worker:                                        # inside a child process: run one code, write its JSON
        r = (worker_gkx(str(gkx_deck), a.budget, a.chunk) if a.worker == "gkx"
             else worker_velcro(str(velcro_cfg), a.budget, a.chunk, a.time_factor, a.q_factor))
        a.worker_out.write_text(json.dumps(r))
        return
    out = (a.out or Path(tempfile.mkdtemp(prefix="compare_speed_"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip() if os.path.exists("/usr/bin/nvidia-smi") or shutil_which("nvidia-smi") else "?"
    print(f"GPU: {gpu or '?'}\nGKX deck {gkx_deck}\nVelcro config {velcro_cfg}\nbudget {a.budget:g} s per code, chunks of "
          f"{a.chunk:g} a/v_ti", flush=True)
    results = {}
    for code, skip in (("gkx", a.skip_gkx), ("velcro", a.skip_velcro)):
        if skip:
            continue
        jf = out / f"{code}.json"
        print(f"\n{code}: running for up to {a.budget:g} s ...", flush=True)
        py = a.gkx_python if code == "gkx" else a.velcro_python
        cmd = [py, str(Path(__file__).resolve()), "--worker", code, "--worker-out", str(jf), "--budget", str(a.budget),
               "--chunk", str(a.chunk), "--time-factor", str(a.time_factor), "--q-factor", str(a.q_factor),
               "--gkx", str(gkx_deck), "--velcro", str(velcro_cfg)]
        w0 = time.perf_counter()
        with open(out / f"{code}.log", "w") as log:
            p = subprocess.run(cmd, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"), stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - w0
        if p.returncode != 0 or not jf.exists():
            print(f"  {code} failed (exit {p.returncode}); see {out / (code + '.log')}")
            continue
        r = json.loads(jf.read_text())
        r["summary"] = summarise(r)
        r["summary"]["process_wall_s"] = wall
        results[r["code"]] = r
        s = r["summary"]
        print(f"  start-up {s['startup_s']:.1f} s, {s['s_per_a_vti']:.4g} s per a/v_ti, reached t = {s['t_reached']:.1f} a/v_ti "
              f"in {wall:.0f} s ({s['stop']})", flush=True)
    print(f"\n{'code':8s} {'start-up, s':>12s} {'s per a/v_ti':>13s} {'reached t':>10s}")
    for code, r in results.items():
        s = r["summary"]
        print(f"{code:8s} {s['startup_s']:12.1f} {s['s_per_a_vti']:13.4g} {s['t_reached']:10.1f}")
    acc = {}
    if len(results) == 2:
        g, v = results["GKX"]["summary"], results["Velcro"]["summary"]
        if g["s_per_a_vti"] and v["s_per_a_vti"]:
            print(f"\nspeed, GKX / Velcro wall per simulated time: {g['s_per_a_vti'] / v['s_per_a_vti']:.3f} (> 1: Velcro faster)")
            print(f"start-up, GKX / Velcro: {g['startup_s'] / v['startup_s']:.2f}")
        t_end = min(g["t_reached"], v["t_reached"])
        print(f"\naccuracy: heat flux over t in [{a.window_start:g}, {t_end:.1f}] a/v_ti, GX gyro-Bohm units (mean +- block s.e.)")
        for key, label in (("Qi", "Q_i"), ("Qe", "Q_e")):
            vals = {}
            for code in ("GKX", "Velcro"):
                st = window_stats(results[code]["t"], results[code][key], a.window_start, t_end)
                if st:
                    vals[code] = st
                    print(f"  {label} {code:7s} {st['mean']:10.3f} +- {st['err']:.3f}  ({st['blocks']} blocks)")
            if len(vals) == 2:
                gg, vv = vals["GKX"], vals["Velcro"]
                diff, sig = vv["mean"] - gg["mean"], (gg["err"] ** 2 + vv["err"] ** 2) ** 0.5
                z = diff / sig if sig > 0 else float("nan")
                print(f"  {label} Velcro - GKX {diff:+.3f} ({100 * diff / gg['mean']:+.1f} %), z = {z:+.2f}")
                vals["velcro_minus_gkx"] = dict(diff=diff, rel=diff / gg["mean"], z=z)
            if vals:
                acc[label] = vals
        if not acc:
            print("  the common window is too short for a flux comparison (raise --budget or lower --window-start)")
    if results:
        win = (a.window_start, min(r["summary"]["t_reached"] for r in results.values()))
        png = plot(results, acc, win, out / "compare_speed.png")
        if png:
            print(f"chart: {png}")
    summary = dict(gpu=gpu, gkx_deck=str(gkx_deck), velcro_config=str(velcro_cfg), budget_s=a.budget, chunk=a.chunk,
                   codes={c: r["summary"] for c, r in results.items()}, accuracy=acc)
    (out / "compare_speed.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'compare_speed.json'}")


def shutil_which(name):
    import shutil
    return shutil.which(name)


if __name__ == "__main__":
    main()
