#!/usr/bin/env python
"""CLI entry point: run BioVenture Monte Carlo simulation and save outputs.

Usage
-----
    python scripts/run_simulation.py [options]

Examples
--------
    # Default config paths, 10 000 iterations:
    python scripts/run_simulation.py --n-iterations 10000

    # Custom seed, verbose, write both report formats:
    python scripts/run_simulation.py --seed 99 --verbose --report --markdown

    # Override output directory:
    python scripts/run_simulation.py --output-dir results/run_01
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_simulation",
        description="Run BioVenture Monte Carlo simulation and save outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-config", default="configs/base.yaml",
        help="Path to base.yaml",
    )
    p.add_argument(
        "--priors-config", default="configs/priors.yaml",
        help="Path to priors.yaml",
    )
    p.add_argument(
        "--correlations-config", default="configs/correlations.yaml",
        help="Path to correlations.yaml",
    )
    p.add_argument(
        "--processes-config", default="configs/processes.yaml",
        help="Path to processes.yaml",
    )
    p.add_argument(
        "--calibration-config", default="configs/calibration.yaml",
        help="Path to calibration.yaml",
    )
    p.add_argument(
        "--output-dir", default="outputs",
        help="Directory for all output files",
    )
    p.add_argument(
        "--n-iterations", type=int, default=None,
        help="Override n_iterations from base.yaml",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override master_seed from base.yaml",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Write a plain-text report to <output-dir>/report.txt",
    )
    p.add_argument(
        "--markdown", action="store_true",
        help="Write a Markdown report to <output-dir>/report.md",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-iteration progress at 10% milestones",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Deferred imports keep --help instant and isolate import errors.
    from bioventure.config import BioVentureConfig
    from bioventure.io import (
        generate_markdown_report,
        generate_text_report,
        save_recorder,
        save_report,
        save_summary,
    )
    from bioventure.simulation import Aggregator, SimulationEngine

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #
    print("Loading configuration...")
    cfg = BioVentureConfig.from_yaml_files(
        base_path=args.base_config,
        priors_path=args.priors_config,
        correlations_path=args.correlations_config,
        processes_path=args.processes_config,
        calibration_path=args.calibration_config,
    )

    if args.n_iterations is not None:
        cfg = replace(cfg, n_iterations=args.n_iterations)
    if args.seed is not None:
        cfg = replace(cfg, master_seed=args.seed)

    n = cfg.n_iterations
    print(f"  n_iterations : {n}")
    print(f"  master_seed  : {cfg.master_seed}")

    # ------------------------------------------------------------------ #
    # Engine                                                               #
    # ------------------------------------------------------------------ #
    engine = SimulationEngine(cfg)

    milestone = max(1, n // 10)

    def _progress(i: int) -> None:
        if args.verbose and (i + 1) % milestone == 0:
            print(f"  [{100 * (i + 1) / n:5.1f}%]  iteration {i + 1:,}/{n:,}")

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #
    print("Running simulation...")
    t0 = time.perf_counter()
    recorder = engine.run(callback=_progress)
    elapsed = time.perf_counter() - t0
    print(
        f"  Completed {n:,} iterations in {elapsed:.2f} s "
        f"({n / elapsed:,.0f} iter/s)"
    )

    # ------------------------------------------------------------------ #
    # Aggregate                                                            #
    # ------------------------------------------------------------------ #
    print("Aggregating results...")
    result = Aggregator().summarise(recorder)

    fin = result.financial
    print(
        f"  MOIC  median={fin.get('moic_median', float('nan')):.3f}  "
        f"p5={fin.get('moic_p5', float('nan')):.3f}  "
        f"p95={fin.get('moic_p95', float('nan')):.3f}"
    )
    print(
        f"  IRR   median={fin.get('irr_median', float('nan')):.1%}  "
        f"PoL={fin.get('probability_of_loss', float('nan')):.1%}"
    )

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {out}/...")

    rec_path = save_recorder(recorder, out / "recorder")
    print(f"  recorder  → {rec_path.name}")

    sum_path = save_summary(result.to_flat_dict(), out / "summary")
    print(f"  summary   → {sum_path.name}")

    meta = {
        "n_iterations": n,
        "master_seed": cfg.master_seed,
        "elapsed_s": f"{elapsed:.2f}",
    }

    if args.report:
        txt = generate_text_report(result, metadata=meta)
        rpt_path = save_report(txt, out / "report.txt")
        print(f"  report    → {rpt_path.name}")

    if args.markdown:
        md = generate_markdown_report(result, metadata=meta)
        md_path = save_report(md, out / "report.md")
        print(f"  markdown  → {md_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    