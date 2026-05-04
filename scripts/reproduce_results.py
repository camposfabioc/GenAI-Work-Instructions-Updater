"""One-command reproduction of all project results.

Modes
-----
    python scripts/reproduce_results.py              # eval-only (default)
    python scripts/reproduce_results.py --full        # re-run pipeline + eval
    python scripts/reproduce_results.py --smoke       # 2-WI wiring check

Eval-only (default):
    Assumes results/improved/WI-*.json already exist.
    Runs eval, validators, semantic eval, generates comparison doc + figures.
    Cost: ~$0.10 (entailment judge only). Time: ~2 min.

Full:
    Re-runs the improved pipeline on all 30 WIs, then runs eval-only.
    Cost: ~$0.25. Time: ~10 min. Results will vary slightly between runs.

Smoke:
    Re-runs the improved pipeline on 2 WIs, runs eval on those 2 only.
    No results doc or figures. Pure wiring check.
    Cost: ~$0.02. Time: ~1 min.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_IMPROVED_DIR = _RESULTS_DIR / "improved"
_BASELINE_DIR = _RESULTS_DIR / "baseline"
_FIGURES_DIR = _RESULTS_DIR / "figures"


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess, abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=_SRC_DIR)
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def _check_results_exist(results_dir: Path) -> None:
    wi_files = list(results_dir.glob("WI-*.json"))
    if not wi_files:
        print(
            f"Error: No WI-*.json files found in {results_dir}\n"
            f"Run with --full to generate pipeline results first, or\n"
            f"ensure the pre-computed results are present.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(wi_files)} WI result files in {results_dir}")


def step_run_pipeline(smoke: bool = False) -> None:
    cmd = [sys.executable, str(_SRC_DIR / "run_pipeline.py"), "improved"]
    if smoke:
        cmd.append("--smoke")
    _run(cmd, "Running improved pipeline" + (" (smoke)" if smoke else ""))


def step_eval(skip_m2: bool = False) -> None:
    cmd = [sys.executable, str(_SRC_DIR / "eval.py"), "improved"]
    if skip_m2:
        cmd.append("--skip-m2")
    _run(cmd, "Running eval harness")


def step_validators() -> None:
    cmd = [sys.executable, str(_SRC_DIR / "validators.py"), str(_IMPROVED_DIR)]
    _run(cmd, "Running validators")


def step_semantic_eval() -> None:
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "eval_semantic.py"),
        str(_IMPROVED_DIR),
    ]
    _run(cmd, "Running semantic eval")


def step_generate_figures() -> None:
    cmd = [sys.executable, str(_SCRIPTS_DIR / "generate_figures.py")]
    _run(cmd, "Generating figures")


def step_generate_results_doc() -> None:
    cmd = [sys.executable, str(_SCRIPTS_DIR / "generate_results_doc.py")]
    _run(cmd, "Generating baseline_vs_improved.md")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce all project results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="Re-run pipeline on all 30 WIs before eval (~$0.25, ~10 min)",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Wiring check: 2 WIs, eval only, no doc/figures (~$0.02, ~1 min)",
    )
    ap.add_argument(
        "--skip-m2",
        action="store_true",
        help="Skip M2 entailment judge (saves ~$0.10, faster)",
    )
    args = ap.parse_args()

    print(f"Project root: {_PROJECT_ROOT}")
    print(f"Mode: {'full' if args.full else 'smoke' if args.smoke else 'eval-only'}")

    if args.smoke:
        # Smoke: run pipeline on 2 WIs, eval without M2, no doc/figures
        step_run_pipeline(smoke=True)
        step_eval(skip_m2=True)
        step_validators()
        step_semantic_eval()
        print("\n" + "="*60)
        print("  SMOKE CHECK PASSED")
        print("="*60)
        return

    if args.full:
        # Full: re-run pipeline, then fall through to eval-only
        step_run_pipeline(smoke=False)

    # Eval-only (default) — or second half of --full
    _check_results_exist(_IMPROVED_DIR)

    step_eval(skip_m2=args.skip_m2)
    step_validators()
    step_semantic_eval()
    step_generate_figures()
    step_generate_results_doc()

    print("\n" + "="*60)
    print("  ALL STEPS COMPLETE")
    print("="*60)
    print(f"\n  Results doc: {_RESULTS_DIR / 'baseline_vs_improved.md'}")
    print(f"  Figures:     {_FIGURES_DIR}/")
    print(f"  Metrics:     {_IMPROVED_DIR / 'metrics.json'}")
    print(f"  Validators:  {_IMPROVED_DIR / 'validation_results.json'}")
    print(f"  Semantic:    {_IMPROVED_DIR / 'semantic_eval.json'}")


if __name__ == "__main__":
    main()
