"""Kaggle batch job: evaluate both pretrained checkpoints on GPU.

Attached inputs:
  lma-phase2-code    the implementation, configs, tokenizers and test text
  lma-phase2-tokens  the packed token arrays
  lma-phase-2-pretraining  the training run's output, for its best.pt files

Scoring the full validation and test splits of a 24M model is hours of work on
a CPU and minutes on a GPU, which is why this runs here rather than locally.
The overlap metrics (BLEU, chrF, ROUGE-L) are *not* computed here: sacrebleu is
not in the Kaggle image and the session has no internet, so every generated
continuation is written to samples.json and scored afterwards by
``common.eval.score_samples``.
"""

import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

WORK = pathlib.Path("/kaggle/working")
REPO = WORK / "repo"
INPUT = pathlib.Path("/kaggle/input")

LANGS = ("Hindi", "Nepali")
PREFIX = {"Hindi": "hi", "Nepali": "ne"}
ROLE = {"Hindi": "Model H", "Nepali": "Model L"}


def find_input(*markers: str) -> pathlib.Path:
    """Locate an attached input by a file it is known to contain.

    Kaggle's mount layout varies -- an attachment can appear at
    /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug> -- and a slug
    can be derived from a title rather than the requested id. Searching for a
    marker is more reliable than hard-coding a path, and it fails with a
    listing of what actually got attached.
    """
    for marker in markers:
        depth = len(pathlib.PurePath(marker).parts)
        for hit in sorted(INPUT.glob(f"**/{marker}")):
            return hit.parents[depth - 1]
    listing = {}
    for path in sorted(INPUT.rglob("*"))[:200] if INPUT.exists() else []:
        if path.is_dir():
            listing[str(path.relative_to(INPUT))] = sorted(
                q.name for q in path.iterdir()
            )[:6]
    raise FileNotFoundError(f"none of {markers} found; /kaggle/input holds {listing}")


def unpack_code() -> None:
    """Put the implementation into the working directory."""
    REPO.mkdir(parents=True, exist_ok=True)
    source = find_input("code.tar.gz", "common/model/gpt.py")
    tarball = source / "code.tar.gz"
    if tarball.exists():
        with tarfile.open(tarball) as archive:
            try:
                archive.extractall(REPO, filter="data")
            except TypeError:
                archive.extractall(REPO)
    else:
        shutil.copytree(source, REPO, dirs_exist_ok=True)
    print("code:", sorted(q.name for q in REPO.iterdir()), flush=True)


def link_tokens(tokens_dir: pathlib.Path) -> None:
    """Point each language's token directory at the read-only dataset."""
    for lang in LANGS:
        directory = REPO / lang / "data" / "tokens"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.bin", "validation.bin", "test.bin", "_stats_pack.json"):
            target = directory / name
            if not target.exists():
                target.symlink_to(tokens_dir / f"{PREFIX[lang]}_{name.lstrip('_')}")
            assert target.resolve().exists(), f"missing {target}"


def main() -> int:
    print("/kaggle/input:", sorted(str(q.relative_to(INPUT))
                                   for q in INPUT.glob("*/*/*"))[:12], flush=True)
    unpack_code()
    link_tokens(find_input("hi_train.bin"))
    checkpoints = find_input("checkpoints/Hindi/best.pt") / "checkpoints"
    print("checkpoints:", checkpoints, flush=True)

    import torch
    count = torch.cuda.device_count()
    print("torch", torch.__version__, "| GPUs:", count,
          [torch.cuda.get_device_name(i) for i in range(count)], flush=True)

    started = time.time()
    codes = {}
    for lang in LANGS:
        out_dir = WORK / "eval" / lang
        command = [
            sys.executable, "-u", "-m", "common.eval.run",
            "--checkpoint", str(checkpoints / lang / "best.pt"),
            "--tokens", f"{lang}/data/tokens",
            "--tokenizer", f"{lang}/tokenizer/{PREFIX[lang]}_unigram_16000.model",
            "--processed", f"{lang}/data/processed",
            "--label", f"{lang} ({ROLE[lang]})",
            "--out-dir", str(out_dir),
            "--figures", str(WORK / "figures"),
            "--batch-size", "8",
            "--prompts", "64",
        ]
        print(f"\n=== {lang} ===\n{' '.join(command)}", flush=True)
        codes[lang] = subprocess.run(command, cwd=REPO).returncode
        print(f"{lang}: exit {codes[lang]}", flush=True)

    print(f"finished in {(time.time() - started) / 60:.1f} min: {codes}", flush=True)

    # Drop the symlinks; they point into /kaggle/input and would be dead in the
    # saved output.
    for lang in LANGS:
        for link in (REPO / lang / "data" / "tokens").glob("*"):
            if link.is_symlink():
                link.unlink()
    shutil.rmtree(REPO / "common", ignore_errors=True)

    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
