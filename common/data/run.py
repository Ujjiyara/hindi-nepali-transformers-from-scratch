"""Pack the Phase 1 splits into the token arrays Phase 2 trains on.

Run from the repository root, once per language::

    python -m common.data.run pack --data-root Hindi/data --lang hi \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model

    python -m common.data.run pack --data-root Nepali/data --lang ne \
        --tokenizer Nepali/tokenizer/ne_unigram_16000.model

Each language is packed with its own tokenizer and writes to its own directory.
Nothing is shared, and the sidecar records which tokenizer produced which file
so a mismatched pair is caught before training rather than after.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sentencepiece as spm

from common.data.pack import pack_split, write_sidecar

log = logging.getLogger("data")

SPLITS = ("train", "validation", "test")


def cmd_pack(args: argparse.Namespace) -> int:
    """pack: encode every split of one language into a flat uint16 array."""
    data_root: Path = args.data_root
    out_dir: Path = args.out_dir or data_root / "tokens"

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    log.info("tokenizer %s, vocabulary %d", args.tokenizer, sp.get_piece_size())

    stats = []
    for split in SPLITS:
        split_path = data_root / "processed" / f"{split}.jsonl.gz"
        if not split_path.exists():
            log.error("missing %s -- run common.pipeline.run first", split_path)
            return 1
        stats.append(
            pack_split(
                split_path,
                sp,
                out_dir / f"{split}.bin",
                num_threads=args.threads,
                limit=args.limit,
            )
        )

    sidecar = write_sidecar(out_dir, args.lang, args.tokenizer,
                            sp.get_piece_size(), stats)
    log.info("wrote %s", sidecar)
    for entry in stats:
        log.info("  %-10s %9d documents  %12d tokens", entry["split"],
                 entry["documents"], entry["tokens"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Token array construction.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="encode splits into flat uint16 arrays")
    pack.add_argument("--data-root", required=True, type=Path,
                      help="e.g. Hindi/data")
    pack.add_argument("--lang", required=True, help="language code, e.g. hi")
    pack.add_argument("--tokenizer", required=True, type=Path,
                      help="this language's SentencePiece .model file")
    pack.add_argument("--out-dir", type=Path, default=None,
                      help="default <data-root>/tokens")
    pack.add_argument("--threads", type=int, default=4)
    pack.add_argument("--limit", type=int, default=None,
                      help="stop after N documents (smoke tests only)")
    pack.set_defaults(func=cmd_pack)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
