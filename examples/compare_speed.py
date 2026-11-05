"""Run GKX on a case, then Velcro on the same case, one after the other on the same GPU, and measure their speed.

Speed is wall seconds per unit of simulated time (a/v_ti), measured the same way for both codes. Each code runs twice,
to t1 and to t2 (a/v_ti), and

    speed = (wall(t2) - wall(t1)) / (t2 - t1)

so start-up, compilation and output writing cancel. Only one code runs at a time. Both get
XLA_PYTHON_CLIENT_PREALLOCATE=false. GKX gets a copy of its deck with t_max = t and run_to = "t_max"; Velcro gets --t-end
in its own unit (R/v_t = t / TIME_FACTOR, 1.9642 on these Cyclone cases, R/a = 2.77778).

    python examples/compare_speed.py --pair ke                    # examples/gkx_ke: gkx_cyclone.toml, velcro_cyclone.toml
    python examples/compare_speed.py --pair adiabatic --t1 50 --t2 150
    python examples/compare_speed.py --gkx my_gkx.toml --velcro my_velcro.toml --time-factor 1.9642

Needs `gkx` (GKX's command line) and Velcro (`python -m velcro`) installed in the environment that runs this script.
Writes compare_speed.json next to the working directory's runs (in --workdir).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = {
    "ke": (HERE / "gkx_ke" / "gkx_cyclone.toml", HERE / "gkx_ke" / "velcro_cyclone.toml"),
    "adiabatic": (HERE / "gkx_adiabatic" / "gkx_cyclone_adiabatic.toml", HERE / "gkx_adiabatic" / "velcro_cyclone_adiabatic.toml"),
}
TIME_FACTOR = 1.9642   # t[a/v_ti] = TIME_FACTOR * t[R/v_t] for R/a = 2.77778, the examples' geometry


def gkx_deck_at(src: Path, t: float, dst: Path) -> Path:
    """A copy of the GKX deck that runs to t: t_max = t, run_to = "t_max" in [time]."""
    lines = src.read_text().splitlines(keepends=True)
    i = next((k for k, line in enumerate(lines) if line.strip() == "[time]"), None)
    if i is None:
        raise SystemExit(f"{src}: no [time] section")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].lstrip().startswith("[")), len(lines))
    section = [line for line in lines[i + 1:j] if not re.match(r"\s*(t_max|run_to)\s*=", line)]
    dst.write_text("".join(lines[:i + 1] + [f"t_max = {t!r}\n", 'run_to = "t_max"\n'] + section + lines[j:]))
    return dst


def timed(cmd, cwd: Path, log: Path) -> float:
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed (exit {p.returncode}); see {log}")
    return wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", choices=sorted(PAIRS), default="ke")
    ap.add_argument("--gkx", type=Path, help="a GKX TOML deck (replaces the pair's)")
    ap.add_argument("--velcro", type=Path, help="a Velcro TOML/JSON config (replaces the pair's)")
    ap.add_argument("--t1", type=float, default=20.0, help="the shorter run, a/v_ti")
    ap.add_argument("--t2", type=float, default=60.0, help="the longer run, a/v_ti")
    ap.add_argument("--time-factor", type=float, default=TIME_FACTOR, help="t[a/v_ti] per t[Velcro unit]")
    ap.add_argument("--skip-gkx", action="store_true")
    ap.add_argument("--skip-velcro", action="store_true")
    ap.add_argument("--workdir", type=Path, default=None, help="where the runs write (default: a new temporary folder)")
    a = ap.parse_args()
    if not a.t2 > a.t1 > 0:
        raise SystemExit("need t2 > t1 > 0")
    gkx_deck, velcro_cfg = (a.gkx or PAIRS[a.pair][0]).resolve(), (a.velcro or PAIRS[a.pair][1]).resolve()
    work = (a.workdir or Path(tempfile.mkdtemp(prefix="compare_speed_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu = shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                                         capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu or 'unknown'}\nruns in {work}\nGKX deck {gkx_deck}\nVelcro config {velcro_cfg}\n"
          f"t1 = {a.t1}, t2 = {a.t2} a/v_ti (Velcro: {a.t1 / a.time_factor:.4g}, {a.t2 / a.time_factor:.4g} R/v_t)", flush=True)
    rows = []
    if not a.skip_gkx:
        if not shutil.which("gkx"):
            raise SystemExit("`gkx` is not on PATH (pip install gkx, or pass --skip-gkx)")
        walls = []
        for t in (a.t1, a.t2):
            d = work / f"gkx_t{t:g}"
            d.mkdir(exist_ok=True)
            deck = gkx_deck_at(gkx_deck, t, d / "case.toml")
            print(f"GKX    to t = {t:g} a/v_ti ...", flush=True)
            walls.append(timed(["gkx", str(deck)], d, d / "run.log"))
            print(f"       {walls[-1]:.1f} s", flush=True)
        rows.append(dict(code="GKX", wall_t1=walls[0], wall_t2=walls[1], s_per_a_vti=(walls[1] - walls[0]) / (a.t2 - a.t1)))
    if not a.skip_velcro:
        walls = []
        for t in (a.t1, a.t2):
            d = work / f"velcro_t{t:g}"
            d.mkdir(exist_ok=True)
            print(f"Velcro to t = {t:g} a/v_ti ...", flush=True)
            walls.append(timed([sys.executable, "-m", "velcro", "run", str(velcro_cfg), "--t-end", f"{t / a.time_factor:.10g}",
                                "--quiet"], d, d / "run.log"))
            print(f"       {walls[-1]:.1f} s", flush=True)
        rows.append(dict(code="Velcro", wall_t1=walls[0], wall_t2=walls[1], s_per_a_vti=(walls[1] - walls[0]) / (a.t2 - a.t1)))
    print(f"\n{'code':8s} {'wall to t1, s':>14s} {'wall to t2, s':>14s} {'s per a/v_ti':>13s}")
    for r in rows:
        print(f"{r['code']:8s} {r['wall_t1']:14.1f} {r['wall_t2']:14.1f} {r['s_per_a_vti']:13.4g}")
    if len(rows) == 2 and rows[1]["s_per_a_vti"] > 0:
        print(f"\nGKX / Velcro, wall per simulated time: {rows[0]['s_per_a_vti'] / rows[1]['s_per_a_vti']:.3f}"
              f"  (> 1: Velcro faster)")
    out = dict(gpu=gpu, gkx_deck=str(gkx_deck), velcro_config=str(velcro_cfg), t1=a.t1, t2=a.t2, time_factor=a.time_factor,
               rows=rows)
    (work / "compare_speed.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {work / 'compare_speed.json'}")


if __name__ == "__main__":
    main()
