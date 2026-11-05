"""GYRO GRAND PRIX: race GKX and Velcro on a matched case, one after the other, with a live terminal display.

Pick a track (one of GKX's examples with a Velcro twin in this folder). GKX runs first, then Velcro, each alone on the
GPU in its own process for the same wall budget (default 3 minutes, start-up included). While a code runs, its car
moves along its lane by simulated time and the charts update after every chunk:
- the warm-up clock (start-up: imports, set-up, compilation);
- the pace (wall seconds per a/v_ti);
- the distance (simulated time reached);
- Q_i against t, and distance against wall time.

At the finish: the podium (start-up, pace, distance), the heat-flux comparison over the window both codes reached,
compare_speed.png and race.json.

    pip install rich "plotext<6" matplotlib   # the display (plotext 5: 6 changed its API); matplotlib for the saved chart
    python examples/race.py                   # menu
    python examples/race.py --track 1 --budget 180
    python examples/race.py --demo            # a 30-second synthetic race to try the display (no GKX/Velcro, no GPU)

The engine is examples/compare_speed.py: the same workers, the same measurements.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_speed as cs  # noqa: E402

try:
    import plotext as plx
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import IntPrompt, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("the race display needs rich and plotext 5:  pip install rich \"plotext<6\"   "
             "(or run examples/compare_speed.py for plain output)")
if not hasattr(plx, "plot"):
    sys.exit(f"plotext {getattr(plx, '__version__', '?')} has a different API: pip install \"plotext<6\"")

TRACKS = [
    dict(key="ke", name="Cyclone, kinetic electrons", origin="GKX examples/05_kinetic_electrons", note="imex-ars3 vs IMEX ARK3"),
    dict(key="adiabatic", name="Cyclone, adiabatic electrons", origin="GKX benchmarks/cases/cyclone_nonlinear_t400",
         note="rk4-pi vs Lawson"),
    dict(key="coll_em", name="Cyclone, collisions + δB∥", origin="GKX examples/05 + Dougherty, β 1e-3",
         note="imex-ars3 vs IMEX ARK3"),
]
CARS = {"GKX": dict(color=cs.COLORS["GKX"], rgb=(37, 99, 235), number="27"),
        "Velcro": dict(color=cs.COLORS["Velcro"], rgb=(234, 88, 12), number="5")}
CHECKER = "[black on white]▚▞[/][white on black]▚▞[/]" * 3


class Lane:
    def __init__(self, code: str):
        self.code, self.status, self.t0 = code, "waiting", None
        self.rows, self.t, self.qi, self.qe = [], [], [], []
        self.error = ""

    @property
    def warm(self) -> float:
        if self.t0 is None:
            return 0.0
        return (self.rows[0]["wall_end"] if self.rows else time.perf_counter() - self.t0)

    def add_lap(self, wall_end: float, t_end: float) -> None:
        prev = self.rows[-1]["wall_end"] if self.rows else 0.0
        self.rows.append(dict(wall_end=wall_end, wall=wall_end - prev, t_end=t_end))

    def startup(self):
        """Start-up as the podium measures it: the first lap's end minus one lap at the steady pace."""
        if not self.rows:
            return None
        p = self.pace()
        return self.rows[0]["wall_end"] - (p * self.rows[0]["t_end"] if p else 0.0)

    @property
    def dist(self) -> float:
        return self.rows[-1]["t_end"] if self.rows else 0.0

    def pace(self):
        if len(self.rows) < 2:
            return None
        per = [(b["wall_end"] - a["wall_end"]) / (b["t_end"] - a["t_end"]) for a, b in zip(self.rows[:-1], self.rows[1:])
               if b["t_end"] > a["t_end"]]
        return sorted(per)[len(per) // 2] if per else None

    def result(self) -> dict:
        r = dict(code=self.code, rows=self.rows, t=self.t, Qi=self.qi, Qe=self.qe, stop=self.status)
        r["summary"] = cs.summarise(r)
        return r


def run_lane(lane: Lane, args, gkx_deck: Path, velcro_cfg: Path, out: Path) -> None:
    """Run one code's worker process and feed its laps into the lane."""
    lane.status, lane.t0 = "warming up", time.perf_counter()
    if args.demo:
        return demo_lane(lane, args)
    jf = out / f"{lane.code.lower()}.json"
    cmd = [sys.executable, str(HERE / "compare_speed.py"), "--worker", lane.code.lower(), "--worker-out", str(jf),
           "--budget", str(args.budget), "--chunk", str(args.chunk), "--gkx", str(gkx_deck), "--velcro", str(velcro_cfg)]
    with open(out / f"{lane.code.lower()}.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                             env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
        for line in p.stdout:
            if line.startswith("RACE "):
                ev = json.loads(line[5:])
                lane.add_lap(ev["wall_end"], ev["t_end"])
                lane.t += ev["t"]
                lane.qi += ev["qi"]
                lane.qe += ev["qe"]
                lane.status = "racing"
            else:
                log.write(line)
        p.wait()
    if p.returncode != 0:
        lane.status, lane.error = "crashed", f"exit {p.returncode}, see {out / (lane.code.lower() + '.log')}"
    else:
        lane.status = "finished"


def demo_lane(lane: Lane, args) -> None:
    """A synthetic race (DEMO): made-up start-up, pace and fluxes, to try the display without a GPU."""
    rnd = random.Random(7 if lane.code == "GKX" else 11)
    startup, pace = (8.0, 0.20) if lane.code == "GKX" else (2.0, 0.13)
    budget, h = args.budget, args.chunk
    time.sleep(startup)
    t = 0.0
    while time.perf_counter() - lane.t0 + pace * h <= budget:
        time.sleep(pace * h)
        t0, t = t, t + h
        ts = [t0 + h * (k + 1) / 4 for k in range(4)]
        base = [240 * (1 - math.exp(-x / 8)) for x in ts]
        lane.t += ts
        lane.qi += [b + rnd.gauss(0, 25) for b in base]
        lane.qe += [1.2 * b + rnd.gauss(0, 25) for b in base]
        lane.add_lap(time.perf_counter() - lane.t0, t)
        lane.status = "racing"
    lane.status = "finished"


# ----------------------------------------------------------------------------------------------------------- display
def track_line(lane: Lane, scale: float, width: int) -> Text:
    car = CARS[lane.code]
    pos = 0 if scale <= 0 else min(width - 1, int((width - 1) * lane.dist / scale))
    road = Text("▕", style="grey50")
    road.append("═" * pos, style=car["color"])
    road.append("🏎" if lane.status != "crashed" else "💥")
    road.append("·" * (width - 1 - pos), style="grey35")
    road.append("▏🏁")
    return road


def lanes_table(lanes, scale: float, width: int) -> Table:
    tb = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False, pad_edge=False)
    tb.add_column("car", no_wrap=True)
    tb.add_column("lane", ratio=1, no_wrap=True)
    tb.add_column("warm-up", justify="right")
    tb.add_column("pace s / a/v_ti", justify="right")
    tb.add_column("distance", justify="right")
    tb.add_column("status", justify="left")
    icons = {"waiting": "⏳ on the grid", "warming up": "🔧 warming up", "racing": "🔥 racing", "finished": "🏁 finished",
             "crashed": "💥 crashed"}
    for ln in lanes:
        car = CARS[ln.code]
        p = ln.pace()
        tb.add_row(Text(f"#{car['number']} {ln.code}", style=f"bold {car['color']}"), track_line(ln, scale, width),
                   (f"{ln.startup():5.1f} s" if ln.rows and ln.startup() is not None else f"{ln.warm:5.1f} s…") if ln.t0 else "–", f"{p:.3g}" if p else "–", f"{ln.dist:6.1f}",
                   icons.get(ln.status, ln.status))
    return tb


def chart(series, title: str, xlabel: str, width: int, height: int) -> Text:
    plx.clear_figure()
    plx.plotsize(width, height)
    plx.theme("clear")
    plx.title(title)
    plx.xlabel(xlabel)
    for name, x, y, rgb in series:
        if len(x) > 1:
            plx.plot(x, y, label=name, color=rgb, marker="braille")
    return Text.from_ansi(plx.build())


def screen(lanes, track, args, banner: str):
    console_w = Console().width
    scale = max([ln.dist for ln in lanes] + [10.0]) * 1.08
    head = Text.assemble((" GYRO GRAND PRIX ", "bold black on yellow"), "  ", (track["name"], "bold"),
                         ("  ·  " + track["origin"], "grey62"), ("  ·  " + f"{args.budget:g} s per car", "grey62"))
    if args.demo:
        head.append("   DEMO: synthetic data", style="bold red")
    lw = max(20, console_w - 70)
    q_series = [(ln.code, ln.t, [q if q == q else 0.0 for q in ln.qi], CARS[ln.code]["rgb"]) for ln in lanes]
    d_series = []
    for ln in lanes:
        if ln.t0 is None:
            continue
        su = ln.startup() if ln.rows else ln.warm                  # flat while warming up: no distance yet
        d_series.append((ln.code, [0.0, su] + [r["wall_end"] for r in ln.rows], [0.0, 0.0] + [r["t_end"] for r in ln.rows],
                         CARS[ln.code]["rgb"]))
    cw, ch = max(30, console_w // 2 - 4), 14
    charts = Table.grid(expand=True)
    charts.add_column(ratio=1)
    charts.add_column(ratio=1)
    charts.add_row(chart(q_series, "Q_i (GX gyro-Bohm)", "t, a/v_ti", cw, ch),
                   chart(d_series, "distance vs wall clock (flat = warm-up)", "wall, s", cw, ch))
    return Group(Panel(head, box=box.HEAVY, border_style="yellow"),
                 Panel(lanes_table(lanes, scale, lw), title=CHECKER + " LANES " + CHECKER, border_style="grey50"),
                 Panel(charts, title="telemetry", border_style="grey50"),
                 Align.center(Text(banner, style="bold yellow")))


def lights(console: Console) -> None:
    for k in range(1, 6):
        console.print(Align.center(Text("● " * k + "○ " * (5 - k), style="bold red")), end="\r")
        time.sleep(0.45)
    console.print(Align.center(Text("   LIGHTS OUT — GO!   ", style="bold black on green")))


def podium(console: Console, lanes, args, out: Path) -> None:
    res = {ln.code: ln.result() for ln in lanes if ln.rows}
    tb = Table(title="🏆  PODIUM", box=box.DOUBLE_EDGE, title_style="bold yellow")
    for c in ("car", "warm-up (start-up), s", "pace, s per a/v_ti", "distance, a/v_ti", "finish"):
        tb.add_column(c, justify="right" if c != "car" else "left")
    order = sorted(res.values(), key=lambda r: (r["summary"]["s_per_a_vti"] or 1e9))
    for k, r in enumerate(order):
        s = r["summary"]
        tb.add_row(Text(("🥇 " if k == 0 else "🥈 ") + r["code"], style=f"bold {CARS[r['code']]['color']}"),
                   f"{s['startup_s']:.1f}" if s["startup_s"] is not None else "–",
                   f"{s['s_per_a_vti']:.4g}" if s["s_per_a_vti"] else "–", f"{s['t_reached']:.1f}", r["stop"])
    console.print(tb)
    acc = {}
    if len(res) == 2:
        g, v = res["GKX"]["summary"], res["Velcro"]["summary"]
        if g["s_per_a_vti"] and v["s_per_a_vti"]:
            ratio = g["s_per_a_vti"] / v["s_per_a_vti"]
            fast = "Velcro" if ratio > 1 else "GKX"
            su = (f";  warm-up: GKX {g['startup_s']:.1f} s vs Velcro {v['startup_s']:.1f} s"
                  if g["startup_s"] is not None and v["startup_s"] is not None else "")
            console.print(Text(f"pace: {fast} is {max(ratio, 1 / ratio):.2f}x faster per simulated time{su}", style="bold"))
        t_end = min(g["t_reached"], v["t_reached"])
        console.print(Text(f"\naccuracy: heat flux over t in [{args.window_start:g}, {t_end:.1f}] a/v_ti, GX gyro-Bohm units",
                           style="bold"))
        for key, label in (("Qi", "Q_i"), ("Qe", "Q_e")):
            vals = {}
            for code in ("GKX", "Velcro"):
                st = cs.window_stats(res[code]["t"], res[code][key], args.window_start, t_end)
                if st:
                    vals[code] = st
            if len(vals) == 2:
                gg, vv = vals["GKX"], vals["Velcro"]
                diff, sig = vv["mean"] - gg["mean"], (gg["err"] ** 2 + vv["err"] ** 2) ** 0.5
                z = diff / sig if sig > 0 else float("nan")
                vals["velcro_minus_gkx"] = dict(diff=diff, rel=diff / gg["mean"], z=z)
                verdict = "agree within the noise" if abs(z) <= 2 else "differ"
                console.print(f"  {label}: GKX {gg['mean']:.2f} ± {gg['err']:.2f}, Velcro {vv['mean']:.2f} ± {vv['err']:.2f}"
                              f"  → {100 * diff / gg['mean']:+.1f} %, z {z:+.2f} ({verdict})")
                acc[label] = vals
        if not acc:
            console.print("  the common window is too short for a flux comparison (raise --budget or lower --window-start)")
        png = cs.plot(res, acc, (args.window_start, t_end), out / "compare_speed.png")
        if png:
            console.print(f"\nchart: {png}")
    (out / "race.json").write_text(json.dumps(dict(demo=bool(args.demo), budget_s=args.budget,
                                                   codes={c: r["summary"] for c, r in res.items()}, accuracy=acc), indent=1))
    console.print(f"results: {out / 'race.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--track", type=int, choices=range(1, len(TRACKS) + 1), help="skip the menu")
    ap.add_argument("--budget", type=float, default=None, help="wall seconds per car, start-up included (default 180)")
    ap.add_argument("--chunk", type=float, default=2.8, help="simulated time per lap, a/v_ti")
    ap.add_argument("--window-start", type=float, default=20.0, help="start of the flux window, a/v_ti")
    ap.add_argument("--demo", action="store_true", help="a synthetic race to try the display (no GKX/Velcro/GPU)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    console = Console()
    if args.track is None:
        menu = Table(title="🏁  PICK A TRACK", box=box.ROUNDED, title_style="bold yellow")
        menu.add_column("#", justify="right")
        menu.add_column("track")
        menu.add_column("from")
        menu.add_column("steppers (GKX vs Velcro)")
        for k, tr in enumerate(TRACKS, 1):
            menu.add_row(str(k), tr["name"], tr["origin"], tr["note"])
        console.print(menu)
        args.track = int(Prompt.ask("track", choices=[str(k) for k in range(1, len(TRACKS) + 1)], default="1"))
        if args.budget is None:
            args.budget = float(IntPrompt.ask("seconds per car", default=30 if args.demo else 180))
    if args.budget is None:
        args.budget = 30.0 if args.demo else 180.0
    track = TRACKS[args.track - 1]
    gkx_deck, velcro_cfg = cs.PAIRS[track["key"]]
    out = (args.out or Path.cwd() / f"race_{track['key']}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    lanes = [Lane("GKX"), Lane("Velcro")]
    lights(console)
    with Live(screen(lanes, track, args, ""), console=console, refresh_per_second=4, transient=False) as live:
        for k, lane in enumerate(lanes):
            banner = f"{lane.code} on track: lap 1 is engine warm-up (imports, set-up, compiling)"
            th = threading.Thread(target=run_lane, args=(lane, args, gkx_deck, velcro_cfg, out), daemon=True)
            th.start()
            while th.is_alive():
                live.update(screen(lanes, track, args, banner))
                time.sleep(0.25)
            live.update(screen(lanes, track, args, f"{lane.code} {lane.status}" + (f": {lane.error}" if lane.error else "")))
            time.sleep(0.5)
        live.update(screen(lanes, track, args, "🏁 CHEQUERED FLAG"))
    podium(console, lanes, args, out)


if __name__ == "__main__":
    main()
