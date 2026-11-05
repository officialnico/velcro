"""Command-line entry point:  python -m velcro run config.toml  |  python -m velcro info"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="velcro", description="Velcro global gyrokinetic solver")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a simulation from a TOML/JSON config")
    r.add_argument("config")
    r.add_argument("--t-end", type=float, default=None)
    r.add_argument("--output", default=None)
    r.add_argument("--quiet", action="store_true")
    sub.add_parser("info", help="print JAX backend and version information")
    args = p.parse_args(argv)
    if args.cmd == "info":
        import jax, velcro
        print(f"velcro {velcro.__version__}  jax {jax.__version__}  devices {jax.devices()}")
        return 0
    from velcro.config import VelcroConfig
    from velcro.simulation import Simulation
    cfg = VelcroConfig.from_file(args.config)
    if args.output:
        cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"output": args.output})})
    sim = Simulation(cfg)
    sim.run(t_end=args.t_end, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
