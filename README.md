[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages


Language Models and Agents — Monsoon 2026 — Individual Project

Two fully independent decoder-only Transformer language models, trained from
scratch:

| | Language | Tier | Script |
|---|---|---|---|
| **Model H** | Hindi (`hi`) | higher-resource, non-English Indian | Devanagari |
| **Model L** | Nepali (`ne`) | lower-resource (from the permitted list) | Devanagari |

The two models share **no** data, tokenizer, vocabulary, or weights. Only
implementation code is shared; every artifact is per-language and lives under
`Hindi/` or `Nepali/`.

**Phase 1 report:** [`report/phase1.md`](report/phase1.md)  
**Phase 2 report:** [`report/phase2.md`](report/phase2.md)  
**Phase 3 report:** [`report/phase3.md`](report/phase3.md)

### Phase 1 result

| | Documents | Characters | Training tokens | Manual (of tokens) | Vocabulary |
|---|---|---|---|---|---|
| **Model H** (Hindi) | 870,914 | 2,062,310,348 | **523,670,863** | 29.5% | 16,000 |
| **Model L** (Nepali) | 883,371 | 1,702,940,920 | **357,473,575** | 22.9% | 16,000 |

Both clear the ≥20% manual floor, which is measured on tokens after
deduplication. Model H meets the ~500M training-token target; Model L reaches
357M, and `report/phase1.md` §4.5 reports the exact count and why it stops
there. Every number here is generated from measured statistics files, not
typed by hand.

### Phase 2 result

Both models are the same ~24.1M-parameter decoder-only Transformer, implemented
from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and `nn.Dropout` only — no
`nn.Transformer*`, no HuggingFace model classes, no pre-built attention block.

| Setting | Value |
|---|---|
| Layers / heads / model dim | 10 / 6 / 384 (head dim 64) |
| Feed-forward width | 1536 (4x) |
| Context length | 512 (learned absolute positions — a hard ceiling) |
| Vocabulary | 16,000 per model, never shared |
| Trainable parameters | **24,086,016** (embedding tied with the output head) |
| Planned steps (one epoch) | 15,981 (Model H), 10,909 (Model L) |

Correctness is asserted rather than assumed: `tests/` checks that the batched
head reshape matches per-head column slicing, that attention rows sum to one
with an exactly-zero upper triangle, that changing token *t+1* leaves the
logits at *t* **bitwise** identical, that no gradient path runs from later
positions to output *t*, and that a checkpoint resume reproduces an
uninterrupted run's losses exactly.

```bash
python -m pytest tests/ -q
```

---

## Google Drive

The train/validation/test splits for both languages, and (from Phase 2)
checkpoints:

**https://drive.google.com/drive/folders/1Psq9GORzHs6PE-EFO3y0S2oVQrf8nl4l?usp=sharing**

Sharing is set to "anyone with the link", so the folder opens without
requesting access; verified from a signed-out browser. Layout:

```
LMA_Individual_project_data/
  Hindi/
    processed/          train.jsonl.gz, validation.jsonl.gz, test.jsonl.gz,
                        _stats_pipeline.json, _token_counts.json
    tokens/             train.bin, validation.bin, test.bin, _stats_pack.json
    checkpoints/        best.pt, step_*.pt          (Phase 2)
    checkpoints/finetune/
                        best.pt, step_*.pt          (Phase 3, reasoning finetune)
    reasoning/          train, validation, test, test_long .jsonl  (Phase 3)
  Nepali/
    processed/          same five files
    tokens/             same four files
    checkpoints/        best.pt, step_*.pt          (Phase 2)
    checkpoints/finetune/
                        best.pt, step_*.pt          (Phase 3, reasoning finetune)
    reasoning/          same four files             (Phase 3)
```

The finetuned checkpoints are in exactly the same format as the pretraining
ones (weights, AdamW state, schedule, step, data position, RNG, geometry), plus
an `extra` record naming the stage and the pretrained checkpoint they started
from, so an interrupted finetune resumes the same way. The reasoning `.jsonl`
files are also fully reproducible from the committed generator and lexicons
(step 12 below); the Drive copy is for convenience.

The `_stats_*.json` sidecars are tiny and are included deliberately: every
figure in the reports is generated from them, so any number can be checked
against the file it came from.

Tokenizer `.model` and `.vocab` files are **committed to this repository**
under `Hindi/tokenizer/` and `Nepali/tokenizer/` — 3.5MB and 3.8MB, small
enough for git — so they do not need to be fetched from Drive. Corpora, token
arrays and checkpoints are Drive-only: they are far too large for git. Each
checkpoint is ~276MB and holds model weights, AdamW moment estimates, the
schedule, the fp16 loss scale, the step, the data position and the RNG state,
which is what makes a run resumable rather than merely restartable.

**Training run logs are in the repository, not on Drive**: each run's
`config.json` and `metrics.jsonl` live under `Hindi/runs/<name>/` and
`Nepali/runs/<name>/`, because the brief grades what is on the branch and a
loss curve stored only on Drive is not.

---

## Repository layout

```
common/                     shared implementation (no data)
  scrape/                   sitemap / RSS / crawl discovery, polite fetching, extraction
  wiki/                     streaming Wikimedia XML dump reader + wikitext cleaner
  downloaded/               streaming CC-100 and Sangraha ingesters
  pipeline/                 normalise -> dedup -> split
  tokenizer/                SentencePiece training, evaluation, vocab sweep,
                            exact token counting, plots
  model/                    config, causal self-attention, block, GPT
  data/                     corpus -> flat uint16 token array, batch sampler
  train/                    LR schedule, resumable checkpoints, pretraining loop
  eval/                     perplexity/BPB, generation, BLEU/chrF/ROUGE-L,
                            attention statistics, figures
  reasoning/                synthetic comparative-reasoning data generator (Phase 3)
  finetune/                 answer-only finetune loop, reasoning evaluation,
                            pretrained-vs-finetuned attention comparison, figures
  report/                   generates report/phase{1,2}.md from measured stats

Hindi/                      Model H — self-contained data and artifacts
  configs/                  scrape_sites.yaml, tokenizer.yaml, model.yaml, train.yaml,
                            reasoning.yaml (lexicon + templates), finetune.yaml
  data/{raw,interim,processed,urls,logs}/     (git-ignored, see Drive)
  data/tokens/              packed .bin token arrays (git-ignored, see Drive)
  tokenizer/                *.model, *.vocab, sweep.json
  checkpoints/              (git-ignored, see Drive)
  runs/<name>/              config.json, metrics.jsonl, eval/results.json (committed)
  runs/finetune/            dataset_stats.json, config.json, metrics.jsonl, train.log,
                            eval/{pretrained,finetuned}.json, per-example predictions,
                            eval/attention.{json,npz} (committed)
Nepali/                     Model L — same structure, no shared files

tests/                      correctness tests for attention, masking, resume, metrics
kaggle/                     batch training and evaluation kernels + their metadata
report/
  phase1.md                 Phase 1 report
  phase2.md                 Phase 2 report
  figures/                  all plots
```

`Hindi/data/`, `Nepali/data/` and both `checkpoints/` are git-ignored. Every
stage writes a `_stats.json` sidecar next to its output, and every run writes
`config.json` and `metrics.jsonl` into its run directory; the reports are
generated from those files, so no figure or table depends on a checkpoint being
present.

---

## Reproduction

```bash
pip install -r requirements.txt
```

All commands are run from the repository root.

### 1. Wikimedia dumps

```bash
# Hindi
mkdir -p Hindi/data/raw/wikipedia && cd Hindi/data/raw/wikipedia
for w in hiwiki hiwikisource hiwikibooks hiwikiquote hiwikiversity; do
  wget -c "https://dumps.wikimedia.org/$w/latest/$w-latest-pages-articles.xml.bz2"
done && cd -

# Nepali  (note: 'newiki' is Nepali; 'newwiki' with two w's is Newari)
mkdir -p Nepali/data/raw/wikipedia && cd Nepali/data/raw/wikipedia
for w in newiki newikibooks newiktionary; do
  wget -c "https://dumps.wikimedia.org/$w/latest/$w-latest-pages-articles.xml.bz2"
done && cd -
```

Nepali has no Wikisource, Wikiquote or Wikinews dump; `newikibooks` and
`newiktionary` are the only sister projects that exist for it.

Extract them. Wikipedia/Wikibooks/Wikiquote use the article namespace; Wikisource
needs `--namespaces 250`, because its namespace-0 entries are transclusion shells
(`<pages index="book.pdf" ... />`) containing no literal text — the works
themselves are stored one proofread page per entry.

```bash
python -m common.wiki.run --dump Hindi/data/raw/wikipedia/hiwiki-latest-pages-articles.xml.bz2 \
  --out Hindi/data/interim/wikipedia/hiwiki.jsonl.gz \
  --lang hi --site hi.wikipedia.org --source hiwiki --workers 4

python -m common.wiki.run --dump Hindi/data/raw/wikipedia/hiwikisource-latest-pages-articles.xml.bz2 \
  --out Hindi/data/interim/wikipedia/hiwikisource.jsonl.gz \
  --lang hi --site hi.wikisource.org --source hiwikisource --namespaces 250 --workers 4

python -m common.wiki.run --dump Nepali/data/raw/wikipedia/newiki-latest-pages-articles.xml.bz2 \
  --out Nepali/data/interim/wikipedia/newiki.jsonl.gz \
  --lang ne --site ne.wikipedia.org --source newiki --workers 4
```

### 2. Scraped news

`triage` first — it reports whether each site is reachable, permitted by
`robots.txt`, and actually serves server-rendered Devanagari. Sites that render
via JavaScript, or serve legacy non-Unicode fonts, return almost no Devanagari to
an HTTP client and should be dropped rather than crawled.

```bash
python -m common.scrape.run triage   --config Nepali/configs/scrape_sites.yaml
python -m common.scrape.run discover --config Nepali/configs/scrape_sites.yaml --site onlinekhabar
python -m common.scrape.run fetch    --config Nepali/configs/scrape_sites.yaml --site onlinekhabar
python -m common.scrape.run extract  --config Nepali/configs/scrape_sites.yaml --site onlinekhabar
```

Omitting `--site` runs every site in the config, one thread per site, which is
how the overnight collection was actually done: a single site cannot be fetched
faster without being impolite, so throughput comes from fetching several at
once.

Same four steps with `Hindi/configs/scrape_sites.yaml`. `fetch` is resumable: it
keeps an on-disk set of fetched URLs, so an interrupted run continues where it
stopped. `--limit N` caps a run for testing.

`discover` is interruptible: sending it SIGINT makes it stop crawling and write
the URLs it has found so far, and re-running it merges with the existing URL
file rather than replacing it. A crawl over a large news site runs for hours,
so how long to let it run is a decision worth making while it runs.

`extract` may be run against a fetch that is still in progress — it works from
the stored raw bytes and tolerates the partially written last manifest line —
so a long scrape can be turned into usable text without waiting for it to end.

The scraper honours `robots.txt` and any declared `Crawl-delay`, identifies
itself in the User-Agent, and rate-limits per site. Requests are spaced
start-to-start, so a declared delay is honoured as the minimum interval between
requests that it is defined to be.

### 3. Downloaded corpora (CC-100 and Sangraha)

Two downloaded sources are used. CC-100 is streamed and decompressed on the
fly, so the multi-gigabyte archive is never stored; output is sharded so an
interrupted run keeps its completed shards.

```bash
python -m common.downloaded.cc100 --url https://data.statmt.org/cc-100/hi.txt.xz \
  --out Hindi/data/interim/downloaded/cc100_hi.jsonl.gz --lang hi --target-chars 1250000000

python -m common.downloaded.cc100 --url https://data.statmt.org/cc-100/ne.txt.xz \
  --out Nepali/data/interim/downloaded/cc100_ne.jsonl.gz --lang ne --target-chars 600000000
```

Sangraha (AI4Bharat, CC-BY-4.0) supplies the second half. Its **verified**
split is scraped from human-verified sites plus text OCR'd from Indic-language
PDFs and transcribed from video and podcasts — registers CC-100, being raw
Common Crawl, does not contain.

```bash
python -m common.downloaded.sangraha --lang hi --subset verified \
  --out Hindi/data/interim/downloaded/sangraha_hi.jsonl.gz --target-chars 400000000

python -m common.downloaded.sangraha --lang ne --subset verified \
  --out Nepali/data/interim/downloaded/sangraha_ne.jsonl.gz --target-chars 900000000
```

The **synthetic** split is deliberately not used. It is English Wikipedia
machine-translated into Indic languages plus romanised transliterations of
that, so training a Nepali model on it would corrupt the thing being measured,
and the romanised half fails the Devanagari filter anyway. It is also 10,588M
of Sangraha's 12,896M Nepali tokens, so picking subsets by size alone would
have gone badly wrong.

Shards are read smallest-first and deleted as they are consumed, so peak disk
stays near one shard rather than the 17GB the full Nepali verified split
occupies.

### 4. Build the corpora

Normalise, deduplicate and split. Reads everything under the language's
`interim/` tree.

```bash
python -m common.pipeline.run --data-root Hindi/data  --lang hi
python -m common.pipeline.run --data-root Nepali/data --lang ne
```

Each run exits nonzero if manually collected text is under 20% of the corpus,
measured *after* deduplication — the share is not knowable before then, because
dedup decides which copy of a near-duplicate survives and the downloaded
corpora overlap Wikipedia heavily.

This guard counts **characters**, because at this point no tokenizer exists to
count tokens with. The rule itself is stated in tokens, and the authoritative
check is step 6 below, which reports the token split. The two are close but not
identical — manual text tokenises denser here, so its token share runs slightly
above its character share, making the character guard the conservative one.

If it trips, either collect more manual text or drop downloaded shards, e.g.:

```bash
python -m common.pipeline.run --data-root Nepali/data --lang ne \
  --exclude cc100_ne.part0011 cc100_ne.part0010
```

### 5. Tokenizers

Trained on the **training split only**, so no held-out text influences the
vocabulary.

```bash
python -m common.tokenizer.run sweep --data-root Hindi/data  --lang hi \
  --out-dir Hindi/tokenizer  --vocab-sizes 8000,16000,32000
python -m common.tokenizer.run sweep --data-root Nepali/data --lang ne \
  --out-dir Nepali/tokenizer --vocab-sizes 8000,16000,32000

python -m common.tokenizer.run plot \
  --sweep "Hindi (hi)=Hindi/tokenizer/sweep.json" "Nepali (ne)=Nepali/tokenizer/sweep.json" \
  --out-dir report/figures
```

Record the chosen vocabulary size in each language's `configs/tokenizer.yaml`;
the next step reads it from there.

### 6. Exact token counts

**Run this before the report.** The pipeline can only estimate tokens as
characters / 3.5, because it runs before the tokenizer it would need exists.
That estimate is out by 10% for Hindi and 33% for Nepali, so the report prefers
measured counts and falls back to the estimate — labelled as one — when they
are absent.

This encodes every document with the tokenizer that language selected and
writes `processed/_token_counts.json`, including the manual-versus-downloaded
split **in tokens**, which is the form the ≥20% rule is stated in.

```bash
python -m common.tokenizer.count --data-root Hindi/data  --lang hi
python -m common.tokenizer.count --data-root Nepali/data --lang ne
```

### 7. Report

```bash
python -m common.report.phase1 --lang hi=Hindi --lang ne=Nepali --out report/phase1.md
```

### 8. Pack the corpora into token arrays

Encodes every document once into a flat `uint16` array — documents
concatenated, separated by the tokenizer's end-of-sequence id — so training
never re-tokenizes and never pads. A training example is a slice: take
`block_size + 1` ids, use `[:-1]` as input and `[1:]` as target.

```bash
python -m common.data.run pack --data-root Hindi/data  --lang hi \
    --tokenizer Hindi/tokenizer/hi_unigram_16000.model
python -m common.data.run pack --data-root Nepali/data --lang ne \
    --tokenizer Nepali/tokenizer/ne_unigram_16000.model
```

| | Train tokens | Validation | Test | Size |
|---|---|---|---|---|
| Hindi | 524,524,486 | 5,246,084 | 5,324,491 | 1001 MB |
| Nepali | 358,339,370 | 3,685,994 | 3,574,013 | 684 MB |

These exceed the Phase 1 counts by exactly one token per document — the
end-of-sequence separator.

### 9. Pretrain

Run once per language; the two models never share a process, a device, or a
weight.

```bash
python -m common.train.run \
    --model-config Hindi/configs/model.yaml \
    --train-config Hindi/configs/train.yaml \
    --tokens Hindi/data/tokens \
    --checkpoint-dir Hindi/checkpoints \
    --run-dir Hindi/runs/base
```

Re-running the same command resumes from the newest checkpoint. Use
`--stop-at-step N` to end a session early **without** reshaping the
learning-rate curve — that is the flag for a capped Kaggle session;
`--max-steps` shortens the whole run and does reshape it.

**On Kaggle (2x T4), both models in parallel, one per GPU:**

```python
import subprocess
procs = [subprocess.Popen(
    ["python", "-m", "common.train.run",
     "--model-config", f"{L}/configs/model.yaml",
     "--train-config", f"{L}/configs/train.yaml",
     "--tokens", f"{L}/data/tokens",
     "--checkpoint-dir", f"/kaggle/working/checkpoints/{L}",
     "--run-dir", f"/kaggle/working/runs/{L}",
     "--device", dev],
    stdout=open(f"/kaggle/working/{L}.log", "w"), stderr=subprocess.STDOUT)
    for L, dev in [("Hindi", "cuda:0"), ("Nepali", "cuda:1")]]
for p in procs:
    p.wait()
```

Launch with **Save & Run All (Commit)** so the run survives the browser
closing. Checkpoints are ~276 MB each; `keep_checkpoints` bounds how many
remain on disk.

#### Running it on Kaggle headlessly

`kaggle/train_kernel.py` is the batch job that was actually run, and
`kaggle/kernel-metadata.json` its configuration. It attaches the token arrays
and this code as private datasets, trains both models (in parallel when two
GPUs are present, in sequence otherwise), and resumes automatically if a
previous run's output is attached as a third source.

```bash
kaggle datasets create -p <staging-dir> --keep-tabular   # tokens, then code
kaggle kernels push -p kaggle/
kaggle kernels status sakshamgoyal69/lma-phase-2-pretraining
kaggle kernels output sakshamgoyal69/lma-phase-2-pretraining -p kaggle-output/
```

Three things about Kaggle that the script has to absorb rather than assume.

**Mount paths.** Datasets appear at `/kaggle/input/datasets/<owner>/<slug>/`,
not `/kaggle/input/<slug>/`, and an uploaded archive is expanded when the
dataset is built, so the code arrives as a directory tree rather than the
tarball that was uploaded. `find_input()` locates each dataset by a marker file
it contains instead of by path, and fails with a listing of what was actually
attached.

**The accelerator must be named exactly.** `machine_shape` accepts only
`NvidiaTeslaT4`, `NvidiaTeslaP100` or `Tpu1VmV38`; anything else is silently
normalised to a generic `Gpu` with no error, which allocates a P100.

**A P100 is unusable here.** Kaggle's preinstalled PyTorch carries no compiled
kernels for Pascal (sm_60), so every CUDA call fails with `no kernel image is
available for execution on the device`. The T4 is Turing (sm_75) and works.
This is also why `resolve_dtype` decides bfloat16 from the compute capability
rather than from `torch.cuda.is_bf16_supported()`, which returns True on both
of those cards and would trade the T4's fp16 tensor cores for software-emulated
bfloat16.

### 10. Evaluate

```bash
python -m common.eval.run \
    --checkpoint Hindi/checkpoints/best.pt \
    --tokens Hindi/data/tokens \
    --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
    --processed Hindi/data/processed \
    --label "Hindi (Model H)" \
    --out-dir Hindi/runs/base/eval
```

Writes `results.json` (perplexity, bits-per-byte, BLEU/chrF/ROUGE-L at greedy
and T=0.5/1.0/1.5, diversity diagnostics, per-head attention statistics) and
`samples.json`, plus attention heatmaps into `report/figures/`. The checkpoint
carries its own geometry, so an evaluation cannot silently run against the
wrong architecture.

### 11. Phase 2 figures and report

```bash
python -m common.report.phase2 \
    --run hi=Hindi/runs/base --run ne=Nepali/runs/base \
    --out report/phase2.md
```

### 12. Generate the reasoning datasets (Phase 3)

```bash
python -m common.reasoning.generate \
    --config Hindi/configs/reasoning.yaml \
    --out Hindi/data/reasoning \
    --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
    --stats Hindi/runs/finetune/dataset_stats.json
```

and the same with `Nepali/` and `ne_unigram_16000.model`. Each language's
`reasoning.yaml` holds its own entity names (with grammatical gender), four
attributes (height, age, weight, price) and sentence templates written from
that language's grammar; the generator only fills slots, so every label is
known by construction. Output is deterministic from the seed in the config.

| Split | Examples | Names | Chains |
|---|---|---|---|
| `train` | 20,000 | 60% of names (Hindi 138 people + 34 objects, Nepali 118 + 35) | 3–4 entities |
| `validation` | 1,000 | disjoint 20% | 3–4 entities |
| `test` | 2,000 | disjoint 20% | 3–4 entities |
| `test_long` | 1,000 | test names | 5 entities (unseen length) |

Four task families in equal proportion: `numeric_pair` (two stated values;
greater / smaller / equal), `numeric_triple` (largest / smallest of three),
`chain_superlative` (only pairwise comparisons such as "A is taller than B",
stated in shuffled order and randomly with the "greater" or "smaller"
adjective; find the extreme) and `chain_relation` (yes/no about two entities,
80% of them not compared directly). Train–test leakage is prevented by
splitting the entity names before generation and verified on every run
(`dataset_stats.json` → `leakage`, all zero); `tests/test_reasoning_data.py`
re-derives every label from the prompt text.

### 13. Reasoning finetune

```bash
python -m common.finetune.run \
    --init-checkpoint Hindi/checkpoints/best.pt \
    --train-config Hindi/configs/finetune.yaml \
    --data Hindi/data/reasoning \
    --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
    --checkpoint-dir Hindi/checkpoints/finetune \
    --run-dir Hindi/runs/finetune \
    --threads 11
```

and the same for Nepali. Each model starts from **its own** pretrained
`best.pt` (weights only; fresh AdamW state), keeps its own tokenizer unchanged,
and is trained with cross-entropy on the answer tokens and EOS only — prompt
targets are masked to -1. Hyperparameters are in `<Language>/configs/finetune.yaml`
and are copied into `runs/finetune/config.json`: batch 32, 3,000 steps (4.8
epochs), AdamW (β = 0.9/0.95, weight decay 0.1 on matrices), peak LR 3e-4 with
100 warmup steps and cosine decay to 3e-5, gradient clip 1.0, dropout 0.1, and
the tied token embedding / output head **frozen**. Validation (512 held-out-name
examples) runs every 100 steps, and `best.pt` is the checkpoint with the highest
validation exact match. Re-running the same command resumes from the newest
`step_*.pt`.

Why frozen embeddings and 3,000 steps: the first attempt, with a trainable
embedding and 48 training names, memorised those names as an answer vocabulary
and answered held-out questions with names absent from the prompt. The short
runs that established this, and the fix, are kept with their logs in
[`Hindi/runs/finetune_ablations/`](Hindi/runs/finetune_ablations/README.md).

These runs were done on CPU (Intel Core Ultra 9 185H, both languages in
parallel with 11 threads each, ~1.9 s/step): sequences are ~34 tokens, so a
GPU is not needed. On a GPU, drop `--threads`.

### 14. Reasoning evaluation

```bash
for stage in pretrained finetuned; do
  ckpt=Hindi/checkpoints/best.pt
  [ $stage = finetuned ] && ckpt=Hindi/checkpoints/finetune/best.pt
  python -m common.finetune.evaluate \
      --checkpoint $ckpt \
      --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
      --data Hindi/data/reasoning \
      --label "Hindi (Model H), $stage" \
      --out Hindi/runs/finetune/eval/$stage.json
done
```

Scores both checkpoints on the identical `test` and `test_long` examples:
**exact match** (greedy decode to EOS equals the answer), **prefix match**
(output starts with the answer as a whole word) and **choice accuracy** (the
candidate answer with the highest log-probability after the prompt; the fair
metric for a pretrained model that never learned the answer format), overall
and by task, attribute and hop count. Per-example predictions are written to
`<stage>_predictions.jsonl` for error analysis.

### 15. Attention before and after finetuning

```bash
python -m common.finetune.attention \
    --pretrained Hindi/checkpoints/best.pt \
    --finetuned Hindi/checkpoints/finetune/best.pt \
    --tokenizer Hindi/tokenizer/hi_unigram_16000.model \
    --data Hindi/data/reasoning \
    --tokens Hindi/data/tokens \
    --label "Hindi (Model H)" \
    --out Hindi/runs/finetune/eval/attention.json
```

Reruns the Phase 2 entropy and distance statistics for both checkpoints on
400 reasoning test prompts and on held-out corpus text, and measures where the
final prompt token (where the answer is predicted) attends: entity mentions,
the correct entity, the question, and the first-token sink.

### 16. Phase 3 figures

```bash
python -m common.finetune.figures \
    --run Hindi/runs/finetune --run Nepali/runs/finetune --out report/figures
```

Draws the finetuning curves, accuracy overview and breakdowns by task and hop
count, pretrained-vs-finetuned heatmaps for layers 0 and 9 of each model,
per-head attention change, layer profiles and answer-position attention —
all from the committed JSON, with no checkpoint needed.

---

## Notes on constraints

- No pretrained models and no pretrained tokenizers are used. SentencePiece
  supplies the training algorithm only; every vocabulary here is fitted on this
  project's own text.
- `trafilatura` and `beautifulsoup4` are used for HTML article extraction. They
  are general-purpose parsing libraries, not pretrained models or corpora.
- Manual vs. downloaded provenance is recorded on every document
  (`collection_method`) and the manual fraction is reported **after**
  deduplication, since dedup can otherwise change it silently.
- The Transformer is built from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and
  `nn.Dropout` only. Positional embeddings, multi-head attention and causal
  masking are implemented in `common/model/`; no `nn.Transformer*` module, no
  HuggingFace model class, and no fused attention kernel such as
  `F.scaled_dot_product_attention` is used, since that is a pre-built attention
  block.
- `sacrebleu` supplies BLEU and chrF. ROUGE-L is implemented in
  `common/eval/text_metrics.py` rather than taken from the `rouge-score`
  package, whose tokenizer applies `[^a-z0-9]+ -> " "` and therefore deletes
  every Devanagari character, silently scoring every pair as zero.
- Model H and Model L share implementation code and nothing else: separate
  corpora, tokenizers, vocabularies, token arrays, weights, optimiser states
  and checkpoints. Neither is ever initialised from the other.
- Phase 3 keeps that separation: separate reasoning lexicons and datasets
  (seeded with the language, so they are not translations of one another),
  and each finetune starts from its own language's pretrained checkpoint with
  its own tokenizer left unchanged. No existing reasoning benchmark is used;
  all finetuning data is generated by `common/reasoning/generate.py`.
