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
  Nepali/
    processed/          same five files
```

The two `_stats_*.json` sidecars are tiny and are included deliberately: every
figure in `report/phase1.md` is generated from them, so any number in the
report can be checked against the file it came from.

Tokenizer `.model` and `.vocab` files are **committed to this repository**
under `Hindi/tokenizer/` and `Nepali/tokenizer/` — 3.5MB and 3.8MB, small
enough for git — so they do not need to be fetched from Drive. The corpora are
Drive-only: they are far too large for git.

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
  report/                   generates report/phase1.md from measured stats

Hindi/                      Model H — self-contained data and artifacts
  configs/scrape_sites.yaml
  data/{raw,interim,processed,urls,logs}/     (git-ignored, see Drive)
  tokenizer/                *.model, *.vocab, sweep.json
Nepali/                     Model L — same structure, no shared files

report/
  phase1.md                 Phase 1 report
  figures/                  all plots
```

`data/` is git-ignored. Every pipeline stage writes a `_stats.json` sidecar next
to its output; the report is generated from those files.

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
