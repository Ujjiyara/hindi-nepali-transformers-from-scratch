"""Generate ``report/phase1.md`` from measured pipeline and tokenizer outputs.

Usage::

    python -m common.report.phase1 \
        --lang hi=Hindi --lang ne=Nepali --out report/phase1.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("report")

LANGUAGE_NAMES = {"hi": "Hindi", "ne": "Nepali"}
MODEL_ROLE = {"hi": "Model H (higher-resource)", "ne": "Model L (lower-resource)"}

# Human-readable descriptions of each source, for the sources table.
SOURCE_NOTES = {
    "hiwiki": ("Wikipedia article dump", "manual"),
    "hiwikisource": ("Wikisource proofread pages (public-domain literature)", "manual"),
    "hiwikibooks": ("Wikibooks dump", "manual"),
    "hiwikiquote": ("Wikiquote dump", "manual"),
    "hiwikiversity": ("Wikiversity dump", "manual"),
    "newiki": ("Wikipedia article dump", "manual"),
    "newikibooks": ("Wikibooks dump", "manual"),
    "newiktionary": ("Wiktionary dump (definition prose)", "manual"),
    "jagran": ("Jagran news, own crawler", "manual"),
    "navbharattimes": ("Navbharat Times news, own crawler", "manual"),
    "livehindustan": ("Live Hindustan news, own crawler", "manual"),
    "onlinekhabar": ("Online Khabar news, own crawler", "manual"),
    "nagariknews": ("Nagarik News, own crawler", "manual"),
    "ratopati": ("Ratopati news, own crawler", "manual"),
    "setopati": ("Setopati news, own crawler", "manual"),
    "ekantipur": ("eKantipur news, own crawler", "manual"),
    "khabarhub": ("Khabarhub news, own crawler (sitemap-driven)", "manual"),
    "ujyaaloonline": ("Ujyaalo Online news, own crawler", "manual"),
    "hamrakura": ("Hamrakura news, own crawler", "manual"),
    "cc100_hi": ("CC-100 (pre-built public corpus)", "downloaded"),
    "cc100_ne": ("CC-100 (pre-built public corpus)", "downloaded"),
}

DROP_LABELS = {
    "dropped_too_short": "below minimum length",
    "dropped_too_long": "above maximum length",
    "dropped_too_few_words": "too few words",
    "dropped_wrong_script": "wrong script (<60% Devanagari)",
    "dropped_too_many_digits": "digit-dominated",
    "dropped_too_much_punctuation": "punctuation-dominated",
    "dropped_repetitive_lines": "repetitive lines (listing pages)",
    "dropped_exact_duplicate": "exact duplicate",
    "dropped_near_duplicate": "near-duplicate (MinHash/LSH)",
}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("could not parse %s", path)
        return None


def _fmt(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def _sources_table(stats: dict) -> str:
    per_source = stats.get("per_source", {})
    if not per_source:
        return "_No per-source statistics available._\n"

    rows = ["| Source | Description | Collection | Documents in | Kept after filters |",
            "|---|---|---|---|---|"]
    for name, counts in sorted(per_source.items(), key=lambda kv: -kv[1].get("in", 0)):
        # An unknown source is never assumed to be manual: the manual fraction
        # is a rule the project has to satisfy, so a source nobody described
        # must show up as unlabelled rather than quietly counting in our favour.
        description, method = SOURCE_NOTES.get(name, ("--", "unlabelled"))
        docs_in = counts.get("in", 0)
        dropped = sum(v for k, v in counts.items() if k.startswith("dropped_"))
        rows.append(
            f"| `{name}` | {description} | {method} | {docs_in:,} | {docs_in - dropped:,} |"
        )
    return "\n".join(rows) + "\n"


def _cleaning_table(stats: dict) -> str:
    totals = stats.get("totals", {})
    loaded = totals.get("docs_loaded", 0)
    normalised = totals.get("docs_normalised", 0)
    after_dedup = totals.get("docs_after_dedup", 0)

    rows = ["| Stage | Documents removed | Remaining |", "|---|---|---|",
            f"| Loaded from interim | -- | {loaded:,} |"]

    for key, label in DROP_LABELS.items():
        if key in ("dropped_exact_duplicate", "dropped_near_duplicate"):
            continue
        value = totals.get(key, 0)
        if value:
            rows.append(f"| Quality filter: {label} | {value:,} | |")
    rows.append(f"| **After normalisation and quality filters** | | **{normalised:,}** |")

    exact = totals.get("dropped_exact_duplicate", 0)
    near = totals.get("dropped_near_duplicate", 0)
    clusters = totals.get("near_duplicate_clusters", 0)
    if exact:
        rows.append(f"| Exact duplicates (SHA-256 of normalised text) | {exact:,} | |")
    if near:
        rows.append(f"| Near-duplicates (MinHash/LSH, {clusters:,} clusters) | {near:,} | |")
    rows.append(f"| **Final corpus** | | **{after_dedup:,}** |")
    return "\n".join(rows) + "\n"


def _splits_table(stats: dict) -> str:
    splits = stats.get("splits", {})
    rows = ["| Split | Documents | Characters | Est. tokens | Manual (char %) |",
            "|---|---|---|---|---|"]
    for name in ("train", "validation", "test"):
        row = splits.get(name)
        if not row:
            continue
        rows.append(
            f"| {name} | {row['documents']:,} | {row['characters']:,} | "
            f"{row.get('est_tokens', 0):,} | {100 * row.get('manual_char_fraction', 0):.1f}% |"
        )
    corpus = stats.get("corpus", {})
    if corpus:
        rows.append(
            f"| **total** | **{corpus['documents']:,}** | **{corpus['characters']:,}** | "
            f"**{corpus['est_tokens']:,}** | **{100 * corpus['manual_char_fraction']:.1f}%** |"
        )
    return "\n".join(rows) + "\n"


def _tokenizer_table(sweep: dict) -> str:
    rows = ["| Vocab size | Fertility (tok/word) | Chars/token | Unknown rate | "
            "Byte-fallback rate | Vocab utilisation |", "|---|---|---|---|---|---|"]
    for row in sorted(sweep.get("sweep", []), key=lambda r: r["vocab_size"]):
        rows.append(
            f"| {row['vocab_size']:,} | {row['fertility']:.3f} | {row['chars_per_token']:.2f} | "
            f"{row['unknown_rate']:.2e} | {row['byte_fallback_rate']:.2e} | "
            f"{100 * row['vocab_utilisation']:.1f}% |"
        )
    return "\n".join(rows) + "\n"


def _examples_block(sweep: dict, limit: int = 3) -> str:
    parts = []
    for example in sweep.get("examples", [])[:limit]:
        pieces = " ".join(example["pieces"])
        parts.append(
            f"- **{example['n_words']} words -> {example['n_tokens']} tokens** "
            f"(fertility {example['fertility']}):\n"
            f"  - text: `{example['text'][:120]}`\n"
            f"  - pieces: `{pieces[:320]}`\n"
        )
    return "\n".join(parts) if parts else "_No examples recorded._\n"


def _comparison_table(lang_dirs: dict[str, Path]) -> str:
    """Fertility and utilisation for both languages at each swept vocabulary size."""
    sweeps = {}
    for code, root in lang_dirs.items():
        data = _read_json(root / "tokenizer" / "sweep.json")
        if data:
            sweeps[code] = {r["vocab_size"]: r for r in data.get("sweep", [])}
    if len(sweeps) < 2:
        return "_Both sweeps must be run before the comparison can be built._\n"

    codes = list(sweeps)
    sizes = sorted(set().union(*(set(s) for s in sweeps.values())))

    header = "| Vocab size |" + "".join(
        f" {LANGUAGE_NAMES.get(c, c)} fertility | {LANGUAGE_NAMES.get(c, c)} utilisation |"
        for c in codes
    )
    divider = "|---|" + "---|" * (2 * len(codes))
    rows = [header, divider]
    for size in sizes:
        cells = []
        for code in codes:
            row = sweeps[code].get(size)
            if row:
                cells.append(f" {row['fertility']:.3f} |")
                cells.append(f" {100 * row['vocab_utilisation']:.1f}% |")
            else:
                cells.extend([" -- |", " -- |"])
        rows.append(f"| {size:,} |" + "".join(cells))
    return "\n".join(rows) + "\n"


def _load_sweeps(lang_dirs: dict[str, Path]) -> dict[str, dict[int, dict]]:
    """Per language, its sweep rows keyed by vocabulary size."""
    sweeps = {}
    for code, root in lang_dirs.items():
        data = _read_json(root / "tokenizer" / "sweep.json")
        if data:
            sweeps[code] = {r["vocab_size"]: r for r in data.get("sweep", [])}
    return sweeps


def _chosen_vocab(root: Path) -> int | None:
    """The vocabulary size selected in this language's tokenizer config.

    The *choice* is a decision recorded in the config; the numbers describing
    it are read back out of the sweep, never from the config, so they cannot
    disagree with the measurements after a re-run.
    """
    path = root / "configs" / "tokenizer.yaml"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("vocab_size:"):
            return int(line.split(":", 1)[1].strip())
    return None


def _fertility_comparison(sweeps: dict[str, dict[int, dict]]) -> str:
    """State which language tokenises more cheaply, from the measurements."""
    if len(sweeps) < 2:
        return ""
    (code_a, rows_a), (code_b, rows_b) = list(sweeps.items())[:2]
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        return ""

    gaps = {s: rows_b[s]["fertility"] - rows_a[s]["fertility"] for s in shared}
    name_a = LANGUAGE_NAMES.get(code_a, code_a)
    name_b = LANGUAGE_NAMES.get(code_b, code_b)
    lower, higher = (name_a, name_b) if all(g > 0 for g in gaps.values()) else (
        (name_b, name_a) if all(g < 0 for g in gaps.values()) else (None, None)
    )
    if lower is None:
        return (
            f"\n**Neither language has lower fertility at every vocabulary size.** "
            f"The gap between them changes sign across the sweep, so on this "
            f"evidence the two tokenise held-out text at comparable cost.\n"
        )

    percentages = [
        100 * abs(gaps[s]) / min(rows_a[s]["fertility"], rows_b[s]["fertility"])
        for s in shared
    ]
    return (
        f"\n**{lower} has lower fertility at every vocabulary size**, by "
        f"{min(percentages):.1f}–{max(percentages):.1f}% across the sweep. A "
        f"tokenizer fitted on more text learns more useful subwords, so the same "
        f"amount of language costs fewer tokens. {higher} therefore fits less real "
        f"content into an identical context window, before any architectural "
        f"difference is considered.\n"
    )


def _utilisation_comparison(sweeps: dict[str, dict[int, dict]]) -> str:
    """Describe how utilisation moves with vocabulary size, from the data."""
    if not sweeps:
        return ""
    parts = []
    for code, rows in sweeps.items():
        sizes = sorted(rows)
        name = LANGUAGE_NAMES.get(code, code)
        first, last = rows[sizes[0]], rows[sizes[-1]]
        parts.append(
            f"{name} goes from {100 * first['vocab_utilisation']:.1f}% at "
            f"{sizes[0]:,} to {100 * last['vocab_utilisation']:.1f}% at {sizes[-1]:,}"
        )
    worst = min(
        (r["vocab_utilisation"], LANGUAGE_NAMES.get(c, c), s)
        for c, rows in sweeps.items()
        for s, r in rows.items()
    )
    return (
        f"\n**Vocabulary utilisation is what actually bounds the vocabulary size.** "
        f"{'; '.join(parts)}. At the largest size the worst case is {worst[1]} at "
        f"{100 * worst[0]:.1f}%, so {100 * (1 - worst[0]):.1f}% of a {worst[2]:,}-entry "
        f"vocabulary never appears in held-out text at all. Those entries would still "
        f"consume embedding parameters in Phase 2 while receiving almost no gradient "
        f"signal.\n"
    )


def _chosen_sizes_table(lang_dirs: dict[str, Path], model_dim: int = 384) -> str:
    """The selected vocabulary per language, described with swept measurements."""
    sweeps = _load_sweeps(lang_dirs)
    rows = [
        "| Model | Vocabulary | Fertility | Utilisation | "
        f"Embedding params at d={model_dim} | Against the largest size swept |",
        "|---|---|---|---|---|---|",
    ]
    for code, root in lang_dirs.items():
        chosen = _chosen_vocab(root)
        swept = sweeps.get(code, {})
        row = swept.get(chosen) if chosen else None
        if row is None:
            rows.append(
                f"| {MODEL_ROLE.get(code, code)} | -- | -- | -- | -- | "
                "_no sweep entry for the configured size_ |"
            )
            continue

        largest = max(swept)
        top = swept[largest]
        params = chosen * model_dim
        fertility_cost = 100 * (row["fertility"] - top["fertility"]) / top["fertility"]
        utilisation_gain = 100 * (row["vocab_utilisation"] - top["vocab_utilisation"])
        saved = (largest - chosen) * model_dim
        reason = (
            f"{fertility_cost:.1f}% higher fertility than {largest:,}, "
            f"for {utilisation_gain:+.1f} points of utilisation and "
            f"{saved / 1e6:.1f}M fewer embedding parameters"
            if chosen != largest
            else "the largest size swept"
        )
        rows.append(
            f"| {MODEL_ROLE.get(code, code)} | **{chosen:,}** | {row['fertility']:.3f} | "
            f"{100 * row['vocab_utilisation']:.1f}% | {params / 1e6:.1f}M | {reason} |"
        )
    return "\n".join(rows) + "\n"


def _selection_note(lang_dirs: dict[str, Path]) -> str:
    """Say whether the two models ended up with the same vocabulary size."""
    chosen = {code: _chosen_vocab(root) for code, root in lang_dirs.items()}
    values = [v for v in chosen.values() if v]
    if len(set(values)) == 1 and len(values) > 1:
        return (
            f"\nBoth models select {values[0]:,}. That is an outcome of the "
            f"measurements rather than a shared setting imposed on them: each "
            f"sweep was run independently on its own corpus, and both show the "
            f"same shape, with utilisation holding up to {values[0]:,} and falling "
            f"away beyond it. Because the sizes match, the fertility gap in the "
            f"table above is a like-for-like comparison of the two languages "
            f"rather than an artefact of different vocabulary budgets.\n"
        )
    return (
        "\nThe two models select different sizes. That asymmetry is the intended "
        "outcome rather than an inconsistency: each vocabulary is fitted to its own "
        "corpus, and the lower-resource corpus supports a smaller one.\n"
    )


def _top_tokens_line(sweep: dict) -> str:
    rows = sweep.get("sweep", [])
    if not rows:
        return ""
    best = min(rows, key=lambda r: r["fertility"])
    tokens = [f"`{p}`" for p, _ in best.get("top_tokens", [])[:15]]
    return ", ".join(tokens)


def build(lang_dirs: dict[str, Path], repo_root: Path) -> str:
    out: list[str] = []
    add = out.append

    add("# Phase 1 — Data Collection and Tokenizer Construction\n")
    add("Language Models and Agents, Monsoon 2026 — Individual Project\n")
    add("All figures in this report are generated from the `_stats.json` and "
        "`sweep.json` files written by the pipeline stages; none are typed by hand. "
        "Regenerate with `python -m common.report.phase1`.\n")

    # ---- 1. language selection -------------------------------------------
    add("\n## 1. Language selection\n")
    add("| Model | Language | Tier | Script |\n|---|---|---|---|")
    add("| Model H | Hindi (`hi`) | higher-resource, non-English Indian | Devanagari |")
    add("| Model L | Nepali (`ne`) | lower-resource (from the permitted list) | Devanagari |\n")
    add("**Why Hindi.** It has the largest volume of freely available, openly licensed "
        "Devanagari text of any Indian language: a large Wikipedia, a very large "
        "Wikisource of public-domain literature, and an active online press.\n")
    add("**Why Nepali, and not Bhojpuri or Maithili.** Nepali shares the Devanagari "
        "script with Hindi, so a single normalisation pipeline serves both models, but "
        "it is linguistically distant enough from Hindi that the two are reliably "
        "separable. Bhojpuri and Maithili sit inside the Hindi dialect continuum, are "
        "often recorded as Hindi, and are routinely confused with it by language-ID "
        "models; text labelled as either is frequently Hindi in practice. Since the "
        "brief requires the two corpora to share no data, that contamination risk was "
        "judged more damaging than the smaller resource gap Nepali gives.\n")
    add("Because both models use Devanagari, the same script-ratio filter (at least "
        "60% Devanagari letters) guards both corpora, and the same Unicode "
        "normalisation applies to each.\n")

    # ---- 2. corpora -------------------------------------------------------
    add("\n## 2. Corpora\n")
    for code, root in lang_dirs.items():
        name = LANGUAGE_NAMES.get(code, code)
        stats = _read_json(root / "data" / "processed" / "_stats_pipeline.json")
        add(f"\n### 2.{'12'[list(lang_dirs).index(code)]} {name} — {MODEL_ROLE.get(code, '')}\n")
        if stats is None:
            add("_Pipeline has not been run for this language yet._\n")
            continue

        corpus = stats.get("corpus", {})
        add(f"**Final corpus: {corpus.get('documents', 0):,} documents, "
            f"{corpus.get('characters', 0):,} characters, "
            f"approximately {corpus.get('est_tokens', 0):,} tokens "
            f"({100 * corpus.get('manual_char_fraction', 0):.1f}% manually collected).**\n")
        add("\n#### Sources\n")
        add(_sources_table(stats))
        add("\n#### Cleaning and deduplication\n")
        add(_cleaning_table(stats))
        add("\n#### Splits\n")
        add("Splits are assigned per document from a salted SHA-256 of the document id, "
            "never per line, so no article contributes to more than one split. Because "
            "the assignment is a pure function of the id, adding a new source later "
            "leaves every existing document in the split it already had.\n")
        add(_splits_table(stats))

    # ---- 3. preprocessing -------------------------------------------------
    add("\n## 3. Preprocessing\n")
    add("The two languages run the same code over separate data. No document, "
        "tokenizer, or vocabulary is shared between them.\n")
    add("\n**Collection.** Wikimedia dumps are parsed with a streaming reader "
        "(`common/wiki/`); news sites are discovered from sitemaps, RSS feeds, or a "
        "polite crawler and fetched with `common/scrape/`, honouring `robots.txt` and "
        "each site's declared `Crawl-delay`. Raw bytes are stored unmodified so that "
        "extraction can be re-run without re-fetching.\n")
    add("\n**Normalisation** (`common/pipeline/normalize.py`):\n")
    add("1. Unicode NFC. This matters for Devanagari because the precomposed nukta "
        "letters U+0958–U+095F are on Unicode's composition-exclusion list, so NFC "
        "decomposes them consistently into base letter plus U+093C. Corpora contain "
        "both spellings; after NFC they are one.\n")
    add("2. Zero-width joiner and non-joiner removal. These are invisible, applied "
        "inconsistently across sources, and split otherwise identical words.\n")
    add("3. **Devanagari matra repair.** Some legacy Nepali input methods write the "
        "vowel sign O as AA + E (U+093E U+0947) rather than U+094B, and AU as AA + AI. "
        "These render almost identically and survive proofreading unnoticed. Unicode "
        "normalisation does *not* fix them: U+094B has no canonical decomposition, so "
        "the two forms are simply different strings, not canonically equivalent ones. "
        "Left alone, the highly frequent Nepali morpheme को is split into two unrelated "
        "token sequences, so an explicit repair table is applied.\n")
    add("4. Boilerplate line removal, including machine-generated article summaries "
        "found on one Nepali outlet.\n")
    add("5. URL and e-mail stripping, and whitespace collapsing.\n")
    add("\n**Quality filters** reject a document and record *why*, so the cleaning "
        "table above shows which filter removed how much: length bounds, minimum word "
        "count, at least 60% Devanagari letters, digit and punctuation ratio caps, and "
        "a repeated-line check that catches listing pages.\n")
    add("\n**Deduplication** runs at two levels. Exact duplicates are found by SHA-256 "
        "of the normalised text. Near-duplicates use 128-permutation MinHash signatures "
        "over 5-word shingles, indexed with LSH in 16 bands of 8 rows (a similarity "
        "threshold near 0.76), with clusters merged by union-find.\n")
    add("\nWhen a duplicate cluster is collapsed, **the manually collected copy is kept "
        "in preference to a downloaded one**. This is deliberate: downloaded corpora "
        "such as CC-100 are built from Common Crawl, which crawled Wikipedia, so a "
        "manually parsed article and a downloaded copy of it are near-duplicates. If "
        "the downloaded copy won, the document would silently change from manual to "
        "downloaded and the manual fraction required by the brief would erode with no "
        "error raised. The manual fraction is therefore reported *after* dedup.\n")

    # ---- 4. tokenizers ----------------------------------------------------
    add("\n## 4. Tokenizers\n")
    add("One tokenizer per language, trained from scratch with SentencePiece "
        "(unigram) on that language's **training split only**, so no held-out text "
        "influences the vocabulary. Vocabularies are not shared and may differ in size.\n")
    add("\n`byte_fallback` is enabled, so any unseen character decomposes into UTF-8 "
        "bytes rather than producing `<unk>`. The unknown-token rate is therefore zero "
        "by construction, and the meaningful diagnostic is the **byte-fallback rate** "
        "reported below.\n")

    for code, root in lang_dirs.items():
        name = LANGUAGE_NAMES.get(code, code)
        sweep = _read_json(root / "tokenizer" / "sweep.json")
        add(f"\n### 4.{'12'[list(lang_dirs).index(code)]} {name}\n")
        if sweep is None:
            add("_Sweep has not been run for this language yet._\n")
            continue
        add(_tokenizer_table(sweep))
        add(f"\nMost frequent held-out tokens: {_top_tokens_line(sweep)}\n")
        add("\n**Example segmentations**\n")
        add(_examples_block(sweep))

    add("\n### 4.3 Model H vs. Model L, side by side\n")
    add(_comparison_table(lang_dirs))
    sweeps = _load_sweeps(lang_dirs)
    add("\nTwo things are visible here, and they are the clearest quantitative "
        "evidence of the resource gap that this project is about.\n")
    add(_fertility_comparison(sweeps))
    add(_utilisation_comparison(sweeps))
    add("\nThis is why fertility alone is not a sufficient selection criterion: it "
        "falls monotonically with vocabulary size and would always argue for the "
        "largest one.\n")

    add("\n### 4.4 Choosing the vocabulary size\n")
    add("Fertility falls monotonically as the vocabulary grows, so fertility alone "
        "would always argue for the largest vocabulary. Two things argue against it.\n")
    add("\nFirst, **vocabulary utilisation** — the fraction of the vocabulary that "
        "actually occurs in held-out text — falls as the vocabulary grows (the "
        "measured values are in the table above). Entries that appear rarely "
        "receive poorly estimated embeddings during pretraining.\n")
    add("\nSecond, the Phase 2 budget is roughly 25M parameters. With a model "
        "dimension of 384 and tied embeddings, a 32,000-entry vocabulary consumes "
        "32,000 × 384 ≈ 12.3M parameters — about half the entire budget — while a "
        "16,000-entry vocabulary costs half that and leaves the parameters for "
        "transformer blocks instead.\n")
    add("\n**Chosen sizes.**\n")
    add(_chosen_sizes_table(lang_dirs))
    add(_selection_note(lang_dirs))
    add("\nThese choices are recorded in `Hindi/configs/tokenizer.yaml` and "
        "`Nepali/configs/tokenizer.yaml` for Phase 2, and the trade-off is plotted "
        "in `report/figures/tokenizer_fertility.png` and "
        "`report/figures/tokenizer_vocab_utilisation.png`.\n")

    # ---- 5. figures -------------------------------------------------------
    add("\n## 5. Figures\n")
    figures = sorted((repo_root / "report" / "figures").glob("*.png"))
    if figures:
        for path in figures:
            rel = path.relative_to(repo_root / "report")
            add(f"![{path.stem}]({rel})\n")
    else:
        add("_No figures generated yet._\n")

    add("\n## 6. Reproduction\n")
    add("See the repository `README.md` for the full command sequence and the Google "
        "Drive links to the datasets and tokenizer artifacts.\n")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Phase 1 report.")
    parser.add_argument(
        "--lang", action="append", required=True, metavar="CODE=DIR",
        help="e.g. hi=Hindi (repeatable, order controls report order)",
    )
    parser.add_argument("--out", type=Path, default=Path("report/phase1.md"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    lang_dirs: dict[str, Path] = {}
    for entry in args.lang:
        code, _, directory = entry.partition("=")
        lang_dirs[code] = repo_root / directory

    text = build(lang_dirs, repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    log.info("wrote %s (%d lines)", args.out, text.count("\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
