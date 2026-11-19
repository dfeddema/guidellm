#!/usr/bin/env python3
"""Run a guidellm SPEED-Bench benchmark once per category of a split.

guidellm aggregates each run per scheduling strategy, not per dataset category,
so a per-category breakdown is produced by running one benchmark per category
and labeling each run. This helper drives that sweep: it loads the category list
for a SPEED-Bench config, invokes ``guidellm run`` for each (tagging the run with
``--label category=<name>``), and writes each run's artifacts into its own
directory so the results stay separable.

Example::

    python contrib/speed_bench_per_category.py \\
        --config qualitative \\
        --target http://localhost:8000/v1 \\
        --model my-model \\
        --tokenizer Qwen/Qwen3-8B \\
        --max-requests 100

Run a subset of categories with ``--categories coding math``. Any extra args
after ``--`` are forwarded verbatim to each ``guidellm run`` invocation, e.g.::

    python contrib/speed_bench_per_category.py --config qualitative ... -- \\
        --profile kind=concurrent,streams=4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from guidellm.schemas.data.deserializers.speed_bench import (
    SPEED_BENCH_CONFIGS,
    categories_for_config,
)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=SPEED_BENCH_CONFIGS,
        help="SPEED-Bench config (split) to sweep.",
    )
    parser.add_argument("--target", required=True, help="OpenAI-compatible endpoint.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer id for token counting.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Subset of categories to run (default: all categories in the config).",
    )
    parser.add_argument(
        "--profile",
        default="synchronous",
        help="guidellm profile kind spec (default: synchronous).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=100,
        help="max_requests constraint per category (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        default="speed_bench_runs",
        help="Directory to hold one subdirectory of artifacts per category.",
    )
    parser.add_argument(
        "--guidellm",
        default="guidellm",
        help="guidellm executable to invoke (default: 'guidellm' on PATH).",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    # argparse leaves a leading "--" in extras; drop it so forwarded flags are clean.
    forwarded = [a for a in extra if a != "--"]

    categories = args.categories or list(categories_for_config(args.config))
    if not categories:
        print(f"No known categories for config {args.config!r}.", file=sys.stderr)
        return 2

    out_root = Path(args.output_dir)
    failures: list[str] = []

    for category in categories:
        run_dir = out_root / f"{args.config}_{category}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.guidellm,
            "run",
            "--backend",
            f"kind=openai_http,target={args.target},model={args.model}",
            "--data",
            f"kind=speed_bench,config={args.config},category={category}",
            "--profile",
            f"kind={args.profile}" if "=" not in args.profile else args.profile,
            "--constraint",
            f"kind=max_requests,count={args.max_requests}",
            "--tokenizer",
            f"kind=huggingface_auto,model={args.tokenizer}",
            "--label",
            f"category={category}",
            "--label",
            f"speed_bench_config={args.config}",
            *forwarded,
        ]
        print(f"\n=== SPEED-Bench {args.config} / {category} -> {run_dir} ===")
        # Run with cwd=run_dir so each run's default artifacts stay separated.
        result = subprocess.run(cmd, cwd=run_dir, check=False)  # noqa: S603
        if result.returncode != 0:
            failures.append(category)
            print(
                f"category {category!r} exited with code {result.returncode}",
                file=sys.stderr,
            )

    print(
        f"\nCompleted {len(categories) - len(failures)}/{len(categories)} categories."
    )
    if failures:
        print(f"Failed categories: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
