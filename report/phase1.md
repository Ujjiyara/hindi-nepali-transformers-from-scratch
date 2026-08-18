# Phase 1 — Data Collection and Tokenizer Construction

Language Models and Agents, Monsoon 2026 — Individual Project

All figures in this report are generated from the `_stats.json` and `sweep.json` files written by the pipeline stages; none are typed by hand. Regenerate with `python -m common.report.phase1`.


## 1. Language selection

| Model | Language | Tier | Script |
|---|---|---|---|
| Model H | Hindi (`hi`) | higher-resource, non-English Indian | Devanagari |
| Model L | Nepali (`ne`) | lower-resource (from the permitted list) | Devanagari |

**Why Hindi.** It has the largest volume of freely available, openly licensed Devanagari text of any Indian language: a large Wikipedia, a very large Wikisource of public-domain literature, and an active online press.

**Why Nepali, and not Bhojpuri or Maithili.** Nepali shares the Devanagari script with Hindi, so a single normalisation pipeline serves both models, but it is linguistically distant enough from Hindi that the two are reliably separable. Bhojpuri and Maithili sit inside the Hindi dialect continuum, are often recorded as Hindi, and are routinely confused with it by language-ID models; text labelled as either is frequently Hindi in practice. Since the brief requires the two corpora to share no data, that contamination risk was judged more damaging than the smaller resource gap Nepali gives.

Because both models use Devanagari, the same script-ratio filter (at least 60% Devanagari letters) guards both corpora, and the same Unicode normalisation applies to each.


## 2. Corpora


### 2.1 Hindi — Model H (higher-resource)

**Final corpus: 676,405 documents, 1,641,897,345 characters, approximately 469,113,527 tokens (32.8% manually collected).**


#### Sources

| Source | Description | Collection | Documents in | Kept after filters |
|---|---|---|---|---|
| `cc100_hi` | CC-100 (pre-built public corpus) | downloaded | 487,850 | 487,247 |
| `hiwikisource` | Wikisource proofread pages (public-domain literature) | manual | 160,558 | 156,997 |
| `hiwiki` | Wikipedia article dump | manual | 87,043 | 86,747 |
| `livehindustan` | Live Hindustan news, own crawler | manual | 8,311 | 8,309 |
| `navbharattimes` | Navbharat Times news, own crawler | manual | 4,499 | 4,495 |
| `jagran` | Jagran news, own crawler | manual | 2,789 | 2,787 |
| `hiwikibooks` | Wikibooks dump | manual | 2,075 | 2,067 |
| `hiwikiquote` | Wikiquote dump | manual | 1,210 | 1,209 |
| `hiwikiversity` | Wikiversity dump | manual | 225 | 225 |


#### Cleaning and deduplication

| Stage | Documents removed | Remaining |
|---|---|---|
| Loaded from interim | -- | 754,560 |
| Quality filter: below minimum length | 2 | |
| Quality filter: too few words | 62 | |
| Quality filter: digit-dominated | 283 | |
| Quality filter: punctuation-dominated | 4,117 | |
| Quality filter: repetitive lines (listing pages) | 13 | |
| **After normalisation and quality filters** | | **750,083** |
| Exact duplicates (SHA-256 of normalised text) | 24,100 | |
| Near-duplicates (MinHash/LSH, 13,533 clusters) | 49,578 | |
| **Final corpus** | | **676,405** |


#### Splits

Splits are assigned per document from a salted SHA-256 of the document id, never per line, so no article contributes to more than one split. Because the assignment is a pure function of the id, adding a new source later leaves every existing document in the split it already had.

| Split | Documents | Characters | Est. tokens | Manual (char %) |
|---|---|---|---|---|
| train | 662,939 | 1,609,029,687 | 459,722,767 | 32.7% |
| validation | 6,644 | 16,200,487 | 4,628,710 | 32.5% |
| test | 6,822 | 16,667,171 | 4,762,048 | 35.1% |
| **total** | **676,405** | **1,641,897,345** | **469,113,527** | **32.8%** |


### 2.2 Nepali — Model L (lower-resource)

**Final corpus: 352,722 documents, 671,969,987 characters, approximately 191,991,424 tokens (18.6% manually collected).**


#### Sources

| Source | Description | Collection | Documents in | Kept after filters |
|---|---|---|---|---|
| `cc100_ne` | CC-100 (pre-built public corpus) | downloaded | 346,720 | 346,272 |
| `onlinekhabar` | Online Khabar news, own crawler | manual | 25,274 | 25,273 |
| `newiki` | Wikipedia article dump | manual | 21,878 | 21,870 |
| `nagariknews` | Nagarik News, own crawler | manual | 2,971 | 2,951 |
| `khabarhub` | Khabarhub news, own crawler (sitemap-driven) | manual | 1,432 | 1,431 |
| `ratopati` | Ratopati news, own crawler | manual | 1,104 | 1,102 |
| `hamrakura` | Hamrakura news, own crawler | manual | 942 | 940 |
| `ekantipur` | eKantipur news, own crawler | manual | 937 | 937 |
| `newiktionary` | Wiktionary dump (definition prose) | manual | 857 | 845 |
| `setopati` | Setopati news, own crawler | manual | 820 | 818 |
| `ujyaaloonline` | Ujyaalo Online news, own crawler | manual | 535 | 535 |
| `newikibooks` | Wikibooks dump | manual | 168 | 168 |


#### Cleaning and deduplication

| Stage | Documents removed | Remaining |
|---|---|---|
| Loaded from interim | -- | 403,638 |
| Quality filter: below minimum length | 5 | |
| Quality filter: too few words | 19 | |
| Quality filter: digit-dominated | 30 | |
| Quality filter: punctuation-dominated | 439 | |
| Quality filter: repetitive lines (listing pages) | 3 | |
| **After normalisation and quality filters** | | **403,142** |
| Exact duplicates (SHA-256 of normalised text) | 11,578 | |
| Near-duplicates (MinHash/LSH, 15,674 clusters) | 38,842 | |
| **Final corpus** | | **352,722** |


#### Splits

Splits are assigned per document from a salted SHA-256 of the document id, never per line, so no article contributes to more than one split. Because the assignment is a pure function of the id, adding a new source later leaves every existing document in the split it already had.

| Split | Documents | Characters | Est. tokens | Manual (char %) |
|---|---|---|---|---|
| train | 345,613 | 658,312,412 | 188,089,260 | 18.6% |
| validation | 3,603 | 6,959,859 | 1,988,531 | 20.7% |
| test | 3,506 | 6,697,716 | 1,913,633 | 18.0% |
| **total** | **352,722** | **671,969,987** | **191,991,424** | **18.6%** |


## 3. Preprocessing

The two languages run the same code over separate data. No document, tokenizer, or vocabulary is shared between them.


**Collection.** Wikimedia dumps are parsed with a streaming reader (`common/wiki/`); news sites are discovered from sitemaps, RSS feeds, or a polite crawler and fetched with `common/scrape/`, honouring `robots.txt` and each site's declared `Crawl-delay`. Raw bytes are stored unmodified so that extraction can be re-run without re-fetching.


**Normalisation** (`common/pipeline/normalize.py`):

1. Unicode NFC. This matters for Devanagari because the precomposed nukta letters U+0958–U+095F are on Unicode's composition-exclusion list, so NFC decomposes them consistently into base letter plus U+093C. Corpora contain both spellings; after NFC they are one.

2. Zero-width joiner and non-joiner removal. These are invisible, applied inconsistently across sources, and split otherwise identical words.

3. **Devanagari matra repair.** Some legacy Nepali input methods write the vowel sign O as AA + E (U+093E U+0947) rather than U+094B, and AU as AA + AI. These render almost identically and survive proofreading unnoticed. Unicode normalisation does *not* fix them: U+094B has no canonical decomposition, so the two forms are simply different strings, not canonically equivalent ones. Left alone, the highly frequent Nepali morpheme को is split into two unrelated token sequences, so an explicit repair table is applied.

4. Boilerplate line removal, including machine-generated article summaries found on one Nepali outlet.

5. URL and e-mail stripping, and whitespace collapsing.


**Quality filters** reject a document and record *why*, so the cleaning table above shows which filter removed how much: length bounds, minimum word count, at least 60% Devanagari letters, digit and punctuation ratio caps, and a repeated-line check that catches listing pages.


**Deduplication** runs at two levels. Exact duplicates are found by SHA-256 of the normalised text. Near-duplicates use 128-permutation MinHash signatures over 5-word shingles, indexed with LSH in 16 bands of 8 rows (a similarity threshold near 0.76), with clusters merged by union-find.


When a duplicate cluster is collapsed, **the manually collected copy is kept in preference to a downloaded one**. This is deliberate: downloaded corpora such as CC-100 are built from Common Crawl, which crawled Wikipedia, so a manually parsed article and a downloaded copy of it are near-duplicates. If the downloaded copy won, the document would silently change from manual to downloaded and the manual fraction required by the brief would erode with no error raised. The manual fraction is therefore reported *after* dedup.


## 4. Tokenizers

One tokenizer per language, trained from scratch with SentencePiece (unigram) on that language's **training split only**, so no held-out text influences the vocabulary. Vocabularies are not shared and may differ in size.


`byte_fallback` is enabled, so any unseen character decomposes into UTF-8 bytes rather than producing `<unk>`. The unknown-token rate is therefore zero by construction, and the meaningful diagnostic is the **byte-fallback rate** reported below.


### 4.1 Hindi

| Vocab size | Fertility (tok/word) | Chars/token | Unknown rate | Byte-fallback rate | Vocab utilisation |
|---|---|---|---|---|---|
| 8,000 | 1.403 | 3.62 | 0.00e+00 | 3.59e-03 | 97.6% |
| 16,000 | 1.280 | 3.96 | 0.00e+00 | 3.93e-03 | 97.7% |
| 32,000 | 1.208 | 4.20 | 0.00e+00 | 4.17e-03 | 90.8% |


Most frequent held-out tokens: `▁के`, `,`, `▁में`, `▁है`, `।`, `▁की`, `▁से`, `▁को`, `▁का`, `▁और`, `.`, `▁पर`, `▁हैं`, `▁ने`, `-`


**Example segmentations**

- **16 words -> 33 tokens** (fertility 2.062):
  - text: `REAL NEWS THAT INDIAN MEDIA DON'T DARE TO PUBLISH: बंगारू लक्ष्मण को चार साल की जेल`
  - pieces: `▁RE AL ▁NEWS ▁ TH AT ▁INDIA N ▁M ED IA ▁D ON ' T ▁ DA RE ▁TO ▁P UB L ISH : ▁बंग ा रू ▁लक्ष्मण ▁को ▁चार ▁साल ▁की ▁जेल`

- **15 words -> 17 tokens** (fertility 1.133):
  - text: `Home धर्म व्रत कथा - विधि बुधवार व्रत करने की विधि, कथा, उद्देश्य एवं फल`
  - pieces: `▁Home ▁धर्म ▁व्रत ▁कथा ▁- ▁विधि ▁बुधवार ▁व्रत ▁करने ▁की ▁विधि , ▁कथा , ▁उद्देश्य ▁एवं ▁फल`

- **8 words -> 9 tokens** (fertility 1.125):
  - text: `इस प्रश्न का उत्तर यहां दिया गया है:`
  - pieces: `▁इस ▁प्रश्न ▁का ▁उत्तर ▁यहां ▁दिया ▁गया ▁है :`


### 4.2 Nepali

| Vocab size | Fertility (tok/word) | Chars/token | Unknown rate | Byte-fallback rate | Vocab utilisation |
|---|---|---|---|---|---|
| 8,000 | 1.518 | 4.22 | 0.00e+00 | 3.10e-03 | 97.7% |
| 16,000 | 1.360 | 4.71 | 0.00e+00 | 3.46e-03 | 97.6% |
| 32,000 | 1.261 | 5.08 | 0.00e+00 | 3.73e-03 | 89.9% |


Most frequent held-out tokens: `▁।`, `,`, `▁र`, `▁छ`, `को`, `।`, `मा`, `ले`, `का`, `▁पनि`, `▁'`, `▁भएको`, `लाई`, `▁लागि`, `▁छन्`


**Example segmentations**

- **10 words -> 11 tokens** (fertility 1.1):
  - text: `आज सुनचाँदीको बजार भाऊ कति छ ? - Everest Dainik`
  - pieces: `▁आज ▁सुनचाँदीको ▁बजार ▁भ ाऊ ▁कति ▁छ ▁? ▁- ▁Everest ▁Dainik`

- **31 words -> 34 tokens** (fertility 1.097):
  - text: `स्मार्टफोनमा सानो सानो समस्याले पनि कहिलेकाही हामीलाई दुख दिईरहेको हुन्छ । कहिलेकाही मोबाइलमा केहि समस्या आउन सक्छ त्यस्`
  - pieces: `▁स्मार्टफोन मा ▁सानो ▁सानो ▁समस्याले ▁पनि ▁कहिलेकाही ▁हामीलाई ▁दुख ▁दिई रहेको ▁हुन्छ ▁। ▁कहिलेकाही ▁मोबाइलमा ▁केहि ▁समस्या ▁आउन ▁सक्छ ▁त्यस्तो ▁अवस्थामा ▁मोबाइल ▁कुनै ▁पसलमा ▁बनाउन ▁ला नु ▁बन्दा ▁पनि ▁घरमै ▁बसेर ▁बनाउने ▁प्रयास ▁गर्न`

- **32 words -> 44 tokens** (fertility 1.375):
  - text: `कैलालीको घोडाघोडी नगरपालिका- १ सुखडमा शिक्षा तथा प्राकृतीक संरक्षण बिकास मञ्च सुखडको आयोजनामा यहि पुस २८ गते देखि माघ ५ `
  - pieces: `▁कैलालीको ▁घोडाघोडी ▁नगरपालिका - ▁१ ▁सुख ड मा ▁शिक्षा ▁तथा ▁प्राकृत ीक ▁संरक्षण ▁बिकास ▁मञ्च ▁सुख ड को ▁आयोजनामा ▁यहि ▁पुस ▁२८ ▁गते ▁देखि ▁माघ ▁५ ▁गते ▁सम्म ▁हु ुु ने ▁बृहत् ▁घोडा घा डी ▁साँ स्कृत ीक ▁महोत्सवको ▁तयारी`


### 4.3 Model H vs. Model L, side by side

| Vocab size | Hindi fertility | Hindi utilisation | Nepali fertility | Nepali utilisation |
|---|---|---|---|---|
| 8,000 | 1.403 | 97.6% | 1.518 | 97.7% |
| 16,000 | 1.280 | 97.7% | 1.360 | 97.6% |
| 32,000 | 1.208 | 90.8% | 1.261 | 89.9% |


Two things are visible here, and they are the clearest quantitative evidence of the resource gap that this project is about.


**Hindi has lower fertility at every vocabulary size**, by 4.3–8.2% across the sweep. A tokenizer fitted on more text learns more useful subwords, so the same amount of language costs fewer tokens. Nepali therefore fits less real content into an identical context window, before any architectural difference is considered.


**Vocabulary utilisation is what actually bounds the vocabulary size.** Hindi goes from 97.6% at 8,000 to 90.8% at 32,000; Nepali goes from 97.7% at 8,000 to 89.9% at 32,000. At the largest size the worst case is Nepali at 89.9%, so 10.1% of a 32,000-entry vocabulary never appears in held-out text at all. Those entries would still consume embedding parameters in Phase 2 while receiving almost no gradient signal.


This is why fertility alone is not a sufficient selection criterion: it falls monotonically with vocabulary size and would always argue for the largest one.


### 4.4 Choosing the vocabulary size

Fertility falls monotonically as the vocabulary grows, so fertility alone would always argue for the largest vocabulary. Two things argue against it.


First, **vocabulary utilisation** — the fraction of the vocabulary that actually occurs in held-out text — falls as the vocabulary grows (the measured values are in the table above). Entries that appear rarely receive poorly estimated embeddings during pretraining.


Second, the Phase 2 budget is roughly 25M parameters. With a model dimension of 384 and tied embeddings, a 32,000-entry vocabulary consumes 32,000 × 384 ≈ 12.3M parameters — about half the entire budget — while a 16,000-entry vocabulary costs half that and leaves the parameters for transformer blocks instead.


**Chosen sizes.**

| Model | Vocabulary | Fertility | Utilisation | Embedding params at d=384 | Against the largest size swept |
|---|---|---|---|---|---|
| Model H (higher-resource) | **16,000** | 1.280 | 97.7% | 6.1M | 6.0% higher fertility than 32,000, for +6.8 points of utilisation and 6.1M fewer embedding parameters |
| Model L (lower-resource) | **8,000** | 1.518 | 97.7% | 3.1M | 20.4% higher fertility than 32,000, for +7.8 points of utilisation and 9.2M fewer embedding parameters |


The two models select different sizes. That asymmetry is the intended outcome rather than an inconsistency: each vocabulary is fitted to its own corpus, and the lower-resource corpus supports a smaller one.


These choices are recorded in `Hindi/configs/tokenizer.yaml` and `Nepali/configs/tokenizer.yaml` for Phase 2, and the trade-off is plotted in `report/figures/tokenizer_fertility.png` and `report/figures/tokenizer_vocab_utilisation.png`.


## 5. Figures

![tokenizer_chars_per_token](figures/tokenizer_chars_per_token.png)

![tokenizer_fertility](figures/tokenizer_fertility.png)

![tokenizer_top_tokens](figures/tokenizer_top_tokens.png)

![tokenizer_vocab_utilisation](figures/tokenizer_vocab_utilisation.png)


## 6. Reproduction

See the repository `README.md` for the full command sequence and the Google Drive links to the datasets and tokenizer artifacts.
