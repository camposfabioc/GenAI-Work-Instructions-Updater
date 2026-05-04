"""Orchestrate a pipeline across Work Instructions.

Reads WI markdown files, runs the chosen pipeline, and saves structured
outputs to ``results/<pipeline_name>/``.

Best-effort error handling: a failure on one WI logs and skips; three
consecutive failures abort the run (per Day 2 design decision 4.3).

Usage
-----
    python src/run_pipeline.py baseline --smoke      # WI-001 + WI-023
    python src/run_pipeline.py baseline              # all 30 WIs
    python src/run_pipeline.py improved --smoke
    python src/run_pipeline.py improved
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pipelines
from llm import CHEAP_MODEL, SEED, get_call_log, reset_call_log
from retriever import ClauseRetriever, build_change_map
from schemas import Standard

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_WI_DIR = _DATA_DIR / "work_instructions"
_V1_JSON_PATH = _DATA_DIR / "standards" / "acme_qs_v1.json"
_V2_MD_PATH = _DATA_DIR / "standards" / "acme_qs_v2.md"
_V2_JSON_PATH = _DATA_DIR / "standards" / "acme_qs_v2.json"
_GLOSSARY_PATH = _DATA_DIR / "glossary.json"
_RESULTS_ROOT = _REPO_ROOT / "results"

_SMOKE_WIS = ["WI-001", "WI-002"]
_MAX_CONSECUTIVE_FAILURES = 3


def _all_wi_ids() -> list[str]:
    """Return sorted list of all WI IDs present on disk."""
    ids = sorted(p.stem for p in _WI_DIR.glob("WI-*.md"))
    if not ids:
        raise RuntimeError(f"No WI files found in {_WI_DIR}")
    return ids


def _load_wi_markdown(wi_id: str) -> str:
    path = _WI_DIR / f"{wi_id}.md"
    if not path.exists():
        raise FileNotFoundError(f"WI not found: {path}")
    return path.read_text(encoding="utf-8")


def _prompt_hash(pipeline_name: str) -> str:
    """SHA256 of the pipeline's system prompt — for run reproducibility tracking."""
    if pipeline_name == "baseline":
        prompt = pipelines._BASELINE_SYSTEM_PROMPT
    elif pipeline_name == "improved":
        prompt = pipelines._IMPROVED_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", choices=["baseline", "improved"])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run only on smoke set: {_SMOKE_WIS}",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output dir (defaults to results/<pipeline>/)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (_RESULTS_ROOT / args.pipeline)
    out_dir.mkdir(parents=True, exist_ok=True)

    wi_ids = _SMOKE_WIS if args.smoke else _all_wi_ids()

    print(f"Pipeline: {args.pipeline}")
    print(f"Output dir: {out_dir}")
    print(f"WIs to process: {len(wi_ids)} {'(smoke)' if args.smoke else ''}")
    print()

    v2_md = _V2_MD_PATH.read_text(encoding="utf-8")

    # Load shared resources for improved pipeline
    v1 = retriever_obj = change_map = glossary = None
    if args.pipeline == "improved":
        print("Loading v1 and v2 standards...", file=sys.stderr)
        v1 = Standard.model_validate(_load_json(_V1_JSON_PATH))
        v2 = Standard.model_validate(_load_json(_V2_JSON_PATH))
        glossary = _load_json(_GLOSSARY_PATH)
        print("Building change map...", file=sys.stderr)
        change_map = build_change_map(v1, v2)
        print(
            f"  changed={len(change_map.changed)}  "
            f"unchanged={len(change_map.unchanged)}  "
            f"v1_only={len(change_map.v1_only)}  "
            f"v2_only={len(change_map.v2_only)}",
            file=sys.stderr,
        )
        print("Building retriever index...", file=sys.stderr)
        retriever_obj = ClauseRetriever(v2)
        print("Ready.\n", file=sys.stderr)

    reset_call_log()
    failures: list[dict] = []
    consecutive_failures = 0
    n_succeeded = 0
    started_at = datetime.now(timezone.utc)

    for wi_id in wi_ids:
        try:
            wi_md = _load_wi_markdown(wi_id)

            if args.pipeline == "baseline":
                result = pipelines.baseline(wi_md, v2_md)
            elif args.pipeline == "improved":
                result = pipelines.improved(
                    wi_md, v1, v2, glossary, retriever_obj, change_map
                )
            else:
                raise ValueError(f"Unknown pipeline: {args.pipeline}")

            (out_dir / f"{wi_id}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            n_succeeded += 1
            consecutive_failures = 0
            print(f"  ✓ {wi_id} — {len(result.edits)} proposals")

        except Exception as exc:
            consecutive_failures += 1
            failures.append(
                {
                    "wi_id": wi_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  ✗ {wi_id} — {type(exc).__name__}: {exc}", file=sys.stderr)

            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print(
                    f"\nAborting: {_MAX_CONSECUTIVE_FAILURES} consecutive failures.",
                    file=sys.stderr,
                )
                _save_failures(out_dir, failures)
                sys.exit(1)

    finished_at = datetime.now(timezone.utc)

    # Save sidecar artifacts
    _save_failures(out_dir, failures)
    _save_call_log(out_dir)
    _save_metadata(
        out_dir,
        pipeline_name=args.pipeline,
        n_processed=n_succeeded,
        n_failed=len(failures),
        started_at=started_at,
        finished_at=finished_at,
        smoke=args.smoke,
    )

    print()
    print(f"Done: {n_succeeded} succeeded, {len(failures)} failed.")

    print()
    print(f"Done: {n_succeeded} succeeded, {len(failures)} failed.")


def _save_failures(out_dir: Path, failures: list[dict]) -> None:
    (out_dir / "_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )


def _save_call_log(out_dir: Path) -> None:
    log = get_call_log()
    (out_dir / "_call_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )


def _save_metadata(
    out_dir: Path,
    *,
    pipeline_name: str,
    n_processed: int,
    n_failed: int,
    started_at: datetime,
    finished_at: datetime,
    smoke: bool,
) -> None:
    meta = {
        "pipeline": pipeline_name,
        "model": CHEAP_MODEL,
        "seed": SEED,
        "smoke": smoke,
        "n_wis_processed": n_processed,
        "n_wis_failed": n_failed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": (finished_at - started_at).total_seconds(),
        "prompt_hash": _prompt_hash(pipeline_name),
    }
    (out_dir / "_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
