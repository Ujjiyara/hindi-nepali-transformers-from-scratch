"""Kaggle batch job: pretrain Model H and Model L, one per GPU.

Attached inputs:
  lma-phase2-code    code.tar.gz -- the repository's implementation and configs
  lma-phase2-tokens  hi_*.bin / ne_*.bin -- the packed token arrays
  lma-phase2-train   (optional) a previous run's output, for resuming

Everything written to /kaggle/working is saved as this run's output, so the
checkpoints and metrics survive the session ending. Training imports only
torch, numpy and yaml, all preinstalled, so the kernel needs no internet.
"""

import os
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


def find_input(*markers: str) -> pathlib.Path:
    """Locate an attached dataset by a file it is known to contain.

    Kaggle's mount layout is not stable across images -- a dataset can appear
    at /kaggle/input/<slug> or, as here, at
    /kaggle/input/datasets/<owner>/<slug> -- and the slug is not always the id
    that was requested, since it can be derived from the title instead.
    Searching for a marker file is therefore more reliable than hard-coding a
    path, and it fails with a listing of what actually got attached rather than
    a bare FileNotFoundError.

    Returns the directory that the marker's *relative path* hangs off, so a
    marker of "common/model/gpt.py" returns the repository root, not the
    directory holding gpt.py.
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
    raise FileNotFoundError(
        f"no attached dataset contains any of {markers}; /kaggle/input holds "
        f"{listing}"
    )


def unpack_code() -> None:
    """Put the implementation into the working directory.

    Kaggle expands an uploaded archive when it builds the dataset, so the code
    normally arrives already extracted; the tarball branch is kept for the case
    where it does not. Copied rather than used in place because /kaggle/input
    is read-only and Python writes __pycache__ beside the sources.
    """
    REPO.mkdir(parents=True, exist_ok=True)
    source = find_input("code.tar.gz", "common/model/gpt.py")
    tarball = source / "code.tar.gz"
    if tarball.exists():
        with tarfile.open(tarball) as archive:
            # filter= was added in 3.12 and is the default there; guard so the
            # same script runs on whatever Python the Kaggle image ships.
            try:
                archive.extractall(REPO, filter="data")
            except TypeError:
                archive.extractall(REPO)
    else:
        shutil.copytree(source, REPO, dirs_exist_ok=True)
    print("code:", sorted(q.name for q in REPO.iterdir()), flush=True)


def link_tokens(tokens_dir: pathlib.Path) -> None:
    """Point each language's token directory at the read-only dataset.

    Symlinks rather than copies: the arrays are 1.7GB together and the working
    directory is capped, and numpy memory-maps them fine through a link.
    """
    for lang in LANGS:
        directory = REPO / lang / "data" / "tokens"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.bin", "validation.bin", "test.bin", "_stats_pack.json"):
            source = tokens_dir / f"{PREFIX[lang]}_{name.lstrip('_')}"
            target = directory / name
            if not target.exists():
                target.symlink_to(source)
            assert target.resolve().exists(), f"missing {source}"
        print(f"{lang}: tokens linked", flush=True)


def restore_checkpoints() -> None:
    """Copy a previous run's checkpoints in, so this session resumes."""
    previous = None
    for candidate in sorted(INPUT.glob("**/checkpoints")):
        if candidate.is_dir():
            previous = candidate.parent
            break
    if previous is None:
        print("no previous run attached; starting from scratch", flush=True)
        return
    for lang in LANGS:
        source = previous / "checkpoints" / lang
        if not source.exists():
            continue
        target = WORK / "checkpoints" / lang
        target.mkdir(parents=True, exist_ok=True)
        for checkpoint in source.glob("*.pt"):
            shutil.copy2(checkpoint, target / checkpoint.name)
        print(f"{lang}: restored {len(list(target.glob('*.pt')))} checkpoints",
              flush=True)


def launch(lang: str, device: str) -> tuple[subprocess.Popen, pathlib.Path]:
    """Start one language's training run, logging to its own file."""
    log_path = WORK / f"{lang}.log"
    command = [
        sys.executable, "-u", "-m", "common.train.run",
        "--model-config", f"{lang}/configs/model.yaml",
        "--train-config", f"{lang}/configs/train.yaml",
        "--tokens", f"{lang}/data/tokens",
        "--checkpoint-dir", str(WORK / "checkpoints" / lang),
        "--run-dir", str(WORK / "runs" / lang),
        "--device", device,
    ]
    print(f"{lang}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command, cwd=REPO, stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    return process, log_path


def main() -> int:
    print("/kaggle/input:", sorted(str(q.relative_to(INPUT))
                                   for q in INPUT.glob("*/*/*"))[:12]
          if INPUT.exists() else "MISSING", flush=True)
    unpack_code()
    tokens_dir = find_input("hi_train.bin")
    print("tokens dataset:", tokens_dir, flush=True)
    link_tokens(tokens_dir)
    restore_checkpoints()

    import torch

    count = torch.cuda.device_count()
    print("torch", torch.__version__, "| GPUs:", count,
          [torch.cuda.get_device_name(i) for i in range(count)], flush=True)

    # Two GPUs runs the models side by side. One GPU runs them in sequence
    # rather than sharing: two 24M models plus their activations would fit,
    # but they would contend for bandwidth and neither would finish sooner.
    # Two GPUs runs the models side by side. One GPU runs them in sequence
    # rather than sharing: two 24M models plus their activations would fit,
    # but they would contend for bandwidth and neither would finish sooner.
    parallel = count >= 2
    devices = {lang: (f"cuda:{i}" if parallel else "cuda:0" if count else "cpu")
               for i, lang in enumerate(LANGS)}
    started = time.time()
    codes = {}

    if parallel:
        running = [(lang, *launch(lang, devices[lang])) for lang in LANGS]
        offsets = {lang: 0 for lang in LANGS}
        # Echo each log's new lines with a prefix, so the kernel log shows both
        # runs interleaved but attributable.
        while any(process.poll() is None for _, process, _ in running):
            time.sleep(60)
            for lang, _, log_path in running:
                if not log_path.exists():
                    continue
                text = log_path.read_text(errors="ignore")
                fresh, offsets[lang] = text[offsets[lang]:], len(text)
                for line in fresh.strip().splitlines()[-4:]:
                    print(f"[{lang}] {line}", flush=True)
        codes = {lang: process.returncode for lang, process, _ in running}
    else:
        for lang in LANGS:
            process, log_path = launch(lang, devices[lang])
            process.wait()
            codes[lang] = process.returncode

    print(f"finished in {(time.time() - started) / 3600:.2f} h: {codes}", flush=True)

    for lang in LANGS:
        for line in (WORK / f"{lang}.log").read_text(errors="ignore").splitlines()[-8:]:
            print(f"[{lang}] {line}", flush=True)

    # The symlinks point into /kaggle/input and would be dead in the saved
    # output, so drop them; the real arrays stay in their own dataset.
    for lang in LANGS:
        for link in (REPO / lang / "data" / "tokens").glob("*"):
            if link.is_symlink():
                link.unlink()

    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
