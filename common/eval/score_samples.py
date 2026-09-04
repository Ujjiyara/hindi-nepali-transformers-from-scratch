"""Score generated continuations that were produced somewhere without sacrebleu.

The evaluation runs on a Kaggle GPU, whose image has no sacrebleu and whose
session may have no internet. Generation is the expensive half and text
scoring is the cheap half, so the two are separable: the GPU writes every
continuation into ``samples.json`` and this fills the overlap metrics back into
``results.json`` afterwards, on any machine.

Run from the repository root::

    python -m common.eval.score_samples --run-dir Hindi/runs/base/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common.eval.text_metrics import score_generations

log = logging.getLogger("score")


def score_run(run_dir: Path) -> dict:
    """Read ``samples.json``, score every temperature, update ``results.json``."""
    samples = json.loads((run_dir / "samples.json").read_text(encoding="utf-8"))
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    scored = []
    for block in samples["samples"]:
        metrics = score_generations(
            block["hypotheses"], block["references"], block["temperature"]
        )
        scored.append(metrics.as_dict())

    results["generation"] = scored
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("scored %d temperatures into %s", len(scored), results_path)
    return results


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Score saved generations.")
    parser.add_argument("--run-dir", required=True, type=Path, action="append",
                        help="an eval directory holding samples.json (repeatable)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    for directory in args.run_dir:
        score_run(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
