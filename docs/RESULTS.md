# Results in full

The short version is in the [README](../README.md). Tables are generated from `results/`; a test checks every number, rank and
example quoted here.

**Words used here.** *BM25* = classic keyword search. *Dense* = a neural model that compares meanings. *Hybrid* = both lists merged. *Gold
passage* = the passage that answers the question. *Recall@10* (R@10) = how often the gold passage is among the first ten. *MRR@10* = the
average of 1/rank of the gold passage, counted as 0 below rank ten. *Holm* = a correction for testing several hypotheses at once. *Pool* = the
set of passages that is searched.

## What the results show

- **Stemming and compound splitting help BM25 a lot:** +10.5 points (70.8% to 81.4%). Stemming gives +6.3, splitting another +4.2. Both are larger than I predicted.
- **The help survives inside a hybrid, in smaller form, in the headline pool:** +3.5 points (88.9% to 92.4%). This was the main question. I had predicted at most +1. With the pool as shipped it is +2.7 points and does not pass the corrected test.
- **The hybrid does not beat the dense model alone.** Both reach 92.4%; 45 questions are won and 45 lost. I had predicted a gain. On this data a small off-the-shelf dense model is as good as the best hybrid, and adding *plain* BM25 to it makes it worse (88.9%).
- **LLM query rewriting is not shown to help.** +0.8 points, to 93.2% (10 questions won, 2 lost) does not pass the test after correction for six comparisons. On the development set it had looked better (12 won, none lost).

**The number fixed in the protocol,** for the system with the best development score (the dense model alone): +21.6 points Recall@10 over plain BM25 on 1,025 German questions (92.4% vs. 70.8% on GermanDPR test, headline pool; linguistic preprocessing adds +3.5 points inside the RRF hybrid). For the full pipeline with LLM rewriting it is +22.3 points, but its last 0.8 points come from the step whose hypothesis was not supported.

Four of six hypotheses were supported, two were not. All six are in [the table](#the-six-hypotheses) with the predictions I wrote beforehand.

A second view of the same data, stricter for BM25: with GermanDPR's pool exactly as shipped (see [the two pools](#the-two-pools)), the +3.5 points inside the hybrid shrink to +2.7 and no longer pass the corrected test, and the dense model alone is ahead of the hybrid on MRR@10. The verdicts above use the pool that was named as the headline before the test run.

All differences are computed before rounding, so they can differ by 0.1 from the rounded percentages.

## The six hypotheses

Written on 2026-11-11, before any system other than plain BM25 existed ([`HYPOTHESES.md`](../HYPOTHESES.md)). "Supported" means: the difference is positive and its p value, corrected with Holm's method across the six tests, is below 0.05. Values are differences in Recall@10 on the test set, in percentage points.

<!-- BEGIN GENERATED: test-hypotheses -->
|  | Claim | Predicted (points) | Result in points [95 % CI] | Verdict | Corrected p | Won / lost | Prediction |
|---|---|---|---|---|---:|---:|---|
| **H1** | Stems or lemmas help BM25 | +1 to +3 | **+6.3** [+4.1, +8.7] | **supported** | < 0.001 | 92 / 27 | larger than predicted |
| **H2** | Compound splitting helps on top | 0 to +2 | **+4.2** [+2.1, +6.4] | **supported** | < 0.001 | 68 / 25 | larger than predicted |
| **H3** | The preprocessing gain survives next to a dense model | 0 to +1 (shrinks) | **+3.5** [+1.5, +5.8] | **supported** | 0.005 | 59 / 23 | larger than predicted |
| **H4a** | The hybrid beats the dense model alone | a gain | **+0.0** [−2.3, +2.1] | **not supported** | 1.000 | 45 / 45 | no gain found |
| **H4b** | The hybrid beats the best lexical system alone | a gain | **+11.0** [+8.5, +13.6] | **supported** | < 0.001 | 119 / 6 | as predicted |
| **H5** | LLM rewriting of unsure queries helps | about 0 | **+0.8** [+0.1, +1.5] | **not supported** | 0.106 | 10 / 2 | as predicted |
<!-- END GENERATED: test-hypotheses -->

p values come from a randomisation test with 10,000 draws, so nothing smaller than 0.0001 can be measured; such values are written as "< 0.001". Three of my six predictions were too cautious (H1, H2, H3), one was wrong (H4a), two held (H4b, H5).

## All systems on the test set

1,025 questions on 192 gold passages. The interval is a 95% bootstrap interval that resamples gold passages, not questions, because several questions share one passage.

<!-- BEGIN GENERATED: test-systems -->
| System | R@1 | R@5 | R@10 [95 % CI] | R@100 | MRR@10 | as shipped: R@10 | as shipped: MRR@10 |
|---|---:|---:|---|---:|---:|---:|---:|
| BM25 on plain words (baseline) | 0.410 | 0.631 | **0.708** [0.668, 0.747] | 0.887 | 0.502 | 0.670 | 0.328 |
| BM25 on stems | 0.445 | 0.685 | **0.772** [0.734, 0.808] | 0.939 | 0.548 | 0.732 | 0.399 |
| BM25 on stems + compound parts, whole word kept | 0.466 | 0.714 | **0.805** [0.769, 0.840] | 0.952 | 0.576 | 0.780 | 0.429 |
| BM25 on stems, compounds replaced by their parts | 0.483 | 0.732 | **0.814** [0.778, 0.848] | 0.958 | 0.590 | 0.784 | 0.463 |
| Dense model alone (multilingual-e5-small) | 0.594 | 0.862 | **0.924** [0.904, 0.942] | 0.989 | 0.709 | 0.916 | 0.641 |
| Hybrid: plain BM25 + dense | 0.528 | 0.811 | **0.889** [0.861, 0.914] | 0.995 | 0.652 | 0.874 | 0.525 |
| Hybrid: stemmed BM25 + dense | 0.583 | 0.839 | **0.912** [0.888, 0.934] | 0.996 | 0.692 | 0.899 | 0.573 |
| Hybrid: stems + parts (whole kept) + dense | 0.588 | 0.855 | **0.921** [0.900, 0.941] | 0.995 | 0.704 | 0.908 | 0.603 |
| Hybrid: stems + parts (replaced) + dense | 0.604 | 0.861 | **0.924** [0.902, 0.944] | 0.995 | 0.714 | 0.901 | 0.607 |
| Hybrid + LLM rewriting of unsure queries | 0.606 | 0.865 | **0.932** [0.911, 0.951] | 0.995 | 0.718 | 0.907 | 0.610 |
<!-- END GENERATED: test-systems -->

### The two pools

GermanDPR ships three "hard negatives" with every question: passages that deepset found with BM25 and that do not contain the answer. If they stay in the pool, they are a trap built specifically for BM25. On the development questions, one of a question's own hard negatives outranks its gold passage for 76% of the questions under plain BM25, and it is the top hit for 73%.

- **Headline pool** (`own_hard_negatives_dropped`): each question's own hard negatives are removed from that question's ranking. They stay in the pool as ordinary distractors for every other question.
- **As shipped** (`as_shipped`): nothing is removed. Reported beside the headline for every system.

The choice of the headline pool was fixed before any comparison system was built. It favours the lexical systems compared with the as-shipped pool, so both are shown. The same six tests on the as-shipped pool:

<!-- BEGIN GENERATED: test-hypotheses-as-shipped -->
|  | Claim | Result in points [95 % CI] | Verdict | Corrected p | Won / lost |
|---|---|---|---|---:|---:|
| **H1** | Stems or lemmas help BM25 | **+6.1** [+3.9, +8.5] | **supported** | < 0.001 | 101 / 38 |
| **H2** | Compound splitting helps on top | **+5.3** [+3.2, +7.5] | **supported** | < 0.001 | 83 / 29 |
| **H3** | The preprocessing gain survives next to a dense model | **+2.7** [+0.4, +5.1] | **not supported** | 0.076 | 62 / 34 |
| **H4a** | The hybrid beats the dense model alone | **−1.5** [−4.1, +0.9] | **not supported** | 0.289 | 43 / 58 |
| **H4b** | The hybrid beats the best lexical system alone | **+11.7** [+9.2, +14.3] | **supported** | < 0.001 | 127 / 7 |
| **H5** | LLM rewriting of unsure queries helps | **+0.6** [+0.0, +1.2] | **not supported** | 0.211 | 7 / 1 |
<!-- END GENERATED: test-hypotheses-as-shipped -->

H1, H2 and H4b hold in both pools. H3 is positive in both but passes the corrected test only in the headline pool. On the as-shipped pool the dense model alone is also better than the hybrid on MRR@10 (0.641 vs. 0.607, p = 0.009).

## Development set

1,000 questions sampled from the training split (seed 13), on 778 gold passages. Every choice was made here: stems rather than lemmas, which splitter, how parts enter the index, the routing rule for the LLM. Sorted by Recall@10. Two rows, `bm25_parts` and `hybrid_parts_dense`, were added after the test run for the demo on the project's web page; they were never run on the test set.

<!-- BEGIN GENERATED: dev-systems -->
| System | R@1 | R@5 | R@10 [95 % CI] | R@100 | MRR@10 | as shipped: R@10 | as shipped: MRR@10 |
|---|---:|---:|---|---:|---:|---:|---:|
| BM25 on plain words (baseline) | 0.437 | 0.660 | **0.741** [0.714, 0.769] | 0.903 | 0.532 | 0.693 | 0.351 |
| BM25 on lemmas | 0.443 | 0.708 | **0.791** [0.765, 0.816] | 0.947 | 0.559 | 0.749 | 0.419 |
| BM25 on plain words + compound parts | 0.452 | 0.702 | **0.793** [0.766, 0.819] | 0.946 | 0.561 | 0.752 | 0.392 |
| BM25 on plain words, compounds replaced by their parts (added after the test run, dev only) | 0.453 | 0.702 | **0.803** [0.776, 0.830] | 0.940 | 0.566 | 0.761 | 0.401 |
| BM25 on stems | 0.478 | 0.719 | **0.808** [0.783, 0.832] | 0.956 | 0.583 | 0.762 | 0.438 |
| BM25 on lemmas + compound parts, whole word kept | 0.468 | 0.734 | **0.821** [0.796, 0.845] | 0.971 | 0.583 | 0.782 | 0.445 |
| BM25 on stems + HanTa compound parts | 0.500 | 0.754 | **0.843** [0.820, 0.866] | 0.968 | 0.611 | 0.800 | 0.470 |
| BM25 on stems + compound parts, whole word kept | 0.498 | 0.763 | **0.847** [0.823, 0.870] | 0.976 | 0.611 | 0.798 | 0.473 |
| BM25 on stems, compounds replaced by their parts | 0.496 | 0.763 | **0.850** [0.828, 0.872] | 0.974 | 0.610 | 0.802 | 0.481 |
| Hybrid: plain BM25 + dense | 0.593 | 0.813 | **0.888** [0.867, 0.908] | 0.998 | 0.690 | 0.875 | 0.581 |
| Hybrid: plain words with parts (replaced) + dense (added after the test run, dev only) | 0.610 | 0.851 | **0.910** [0.891, 0.929] | 0.997 | 0.715 | 0.897 | 0.618 |
| Hybrid: stemmed BM25 + dense | 0.611 | 0.864 | **0.918** [0.900, 0.935] | 0.998 | 0.721 | 0.909 | 0.618 |
| Hybrid: stems + parts (replaced) + dense | 0.628 | 0.889 | **0.936** [0.920, 0.951] | 0.997 | 0.741 | 0.927 | 0.652 |
| Hybrid: stems + parts (whole kept) + dense | 0.631 | 0.889 | **0.938** [0.922, 0.953] | 0.998 | 0.744 | 0.928 | 0.652 |
| Hybrid + LLM rewriting of unsure queries | 0.636 | 0.898 | **0.948** [0.934, 0.961] | 0.997 | 0.751 | 0.933 | 0.661 |
| Hybrid (whole kept) + LLM rewriting of unsure queries | 0.638 | 0.897 | **0.948** [0.934, 0.961] | 0.998 | 0.751 | 0.932 | 0.658 |
| Dense model alone (multilingual-e5-small) | 0.652 | 0.904 | **0.949** [0.935, 0.962] | 0.987 | 0.761 | 0.942 | 0.706 |
<!-- END GENERATED: dev-systems -->

On the development set the dense model alone was ahead of the best hybrid (0.949 vs. 0.936). The three best lexical systems (0.843, 0.847, 0.850) are too close to tell apart; `bm25_parts_stem` was taken as "best lexical" by the rule written beforehand (highest development Recall@10).

## How the pieces work

**Splitter.** [`splitter.py`](../src/compound_search/splitter.py) follows Koehn & Knight (2003): a word may be cut into known words, and the cut with the highest geometric mean of part frequencies wins against the uncut word. So a frequent, fixed compound stays whole and a rare one is split. Additions for German: linking elements that are only allowed after the sounds they really follow (*Bundes|gericht*, *Sonnen|schein*), a dropped final *-e* (*Schul|buch*), frequencies pooled over inflected forms, and parts that must be nouns. The word list comes from the passage pool itself; no outside lexicon and no labels are used.

**Dense model.** `intfloat/multilingual-e5-small`, as published, no fine-tuning. According to its technical report, GermanDPR and GermanQuAD are not among its supervised training sets.

**Hybrid.** Reciprocal rank fusion (k = 60) of the top 100 of BM25 and the top 100 of the dense model. No weights, nothing tuned.

**LLM rewriting.** A query goes to the local model `gemma4:e2b-it-qat` only if BM25 and the dense model disagree on the top passage and no passage is ranked near the top by both. The rule looks at rankings and scores only, never at gold passages or answers. It sent 5.3% of the development questions to the LLM (53 of 1,000, headline pool) and 6.5% of the test questions (67 of 1,025, headline pool; 56 with the pool as shipped; 83 under at least one of the two). The model is asked for four to six search terms. The question with these terms added is searched again, and all four rankings are fused.

### How good is the splitter by itself?

100 capitalised words drawn at random from development questions, half of them 12 letters or longer. I labelled each word with its cuts by hand, on a page that shows the word and its question and nothing a splitter did: [`labels/compounds_labels.csv`](../labels/compounds_labels.csv). The page is [`labels/label_compounds.html`](../labels/label_compounds.html).

<!-- BEGIN GENERATED: splitter -->
| Splitter | Precision | Recall | F1 | Split correctly | Wrong cuts | Compound missed | Split a non-compound | Left whole correctly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Own splitter (`LexiconSplitter`) | 0.84 | 0.77 | 0.80 | 37 | 2 | 9 | 5 | 47 |
| HanTa (`HantaSplitter`) | 0.95 | 0.77 | 0.85 | 37 | 1 | 10 | 1 | 51 |
<!-- END GENERATED: splitter -->

With 100 words, F1 0.80 against 0.85 is too small a difference to rank the two. Both get 37 of the 48 words that I cut exactly right. They fail differently: HanTa split one word that I left whole (*Sound|track*), my splitter split five (*Hop|era*, *Tom|mie*, *Marsh|all*, *Sound|track*, *Griechen|land*), four of them names. *Soundtrack* is a loanword, although I had labelled its type as "name".

**These labels were revised once, after the scores were known.** In my first pass I had also cut at a prefix, a suffix or an ending in eight words: *Ver|schiebung*, *Juden|tum*, *Wissen|schaften*, *Preußisch|e* and four more. The labelling page's own instruction counts a derived word as "not a compound", so these cuts were removed by one rule: a cut stays only where a word meets a word. Every changed row says so in its note column. The first pass is kept in [`labels/compounds_labels_first_pass.csv`](../labels/compounds_labels_first_pass.csv); on it the scores are F1 0.73 and 0.77, because neither splitter looks for such cuts.

An earlier set of labels made by an AI is kept in [`labels/compounds_labels_ai.csv`](../labels/compounds_labels_ai.csv). It agrees with the revised labels on the cuts of 94 of the 100 words (86 with the first pass). Scored against the AI labels, both splitters reach F1 0.86 ([`results/splitter_eval_ai_labels.json`](../results/splitter_eval_ai_labels.json)).

On speed, splitting the pool's 216,350 distinct capitalised words took my splitter 31 seconds, including building its word list, and HanTa 149 seconds (one run each on the same laptop, [`results/splitter_speed.json`](../results/splitter_speed.json)).

## What the errors show

50 development questions, sampled by category, with ranks under every system: [`results/error_analysis_50.md`](../results/error_analysis_50.md). Counts are out of the 1,000 development questions, headline pool.

- **Splitting wins (56 questions).** In 53 of them, splitting gives the question and its gold passage an index term they did not share before. In 39 the question itself contains a word that is cut. *Einwohnerdichte* against *Einwohnerzahl* and *Bevölkerungsdichte*: rank 43 with plain words, 50 with stems, 1 with split compounds. A simple noun in the question also finds compounds in the passage: *Befehl* against *Befehlskette* and *Oberbefehlshaber*, from not in the top 100 to rank 3.
- **Splitting losses (28 questions).** In 16 of them the question contains a word that the splitter cuts, and several of these words are names. *Straßburg* becomes *Straße* + *Burg*, so "Welches Unternehmen betreibt die Straßenbahn in Straßburg?" falls from rank 3 to 12: the question now also matches every passage that mentions a street or a castle. "Wer ist Gabe Newell?" falls from rank 1 to 12 once *Newell* is cut into *new* + *ell* (on stems; on plain words the same cut does no harm). *Deutschland*, *Estlands* and *Ostpreußen* are cut too. The system that was frozen replaces a compound by its parts, so the whole name is no longer in the index to rescue such cases. In the other 12 questions no word of the question is cut; there the ranking changes because the passages are indexed differently.
- **Why preprocessing still matters in the hybrid (59 questions).** Rank fusion rewards passages that both lists contain. In 50 of these 59 questions plain BM25 does not have the gold passage in its first 100, so that passage has only its dense rank, and passages that appear in both lists overtake it. With stems and compound parts, BM25 ranks the gold passage higher in all 59.
- **LLM rewriting wins (12 questions).** Some are real repairs: the misspelt *Verfassungskommitee* is corrected to *Verfassungskomitee* (rank 12 to 4), and "Wie groß ist North Carolina?" gains the word *Fläche* (11 to 3). In one case the model simply knows the answer: for "Welcher Fluss fließt durch Philadelphia?" it adds *Delaware River* (13 to 3). And some are luck: for "Was versteht man unter NES?" the model invents "Netzwerk-Energie-System", and the rank still improves from 12 to 7, probably because the expansion repeats *NES*.
- **Missed by everything (29 questions).** In 7 of them another passage in the top ten contains the gold answer string, so the system may have found a right passage that GermanDPR did not label. The other 22 are real misses.

## Limits

- **One dataset.** GermanDPR questions were written by annotators who had the passage in front of them. Real search queries are shorter and messier. I make no claim beyond this dataset.
- **A small pool.** 26,372 passages. In a pool of millions, exact word matches matter more, and the balance between BM25 and a dense model may differ.
- **One gold passage per question.** Other passages that also answer the question count as misses.
- **192 gold passages in the test set.** Many questions share a passage, so the real sample is smaller than 1,025. The intervals account for this.
- **The headline pool helps BM25.** See [the two pools](#the-two-pools). The main result (H3) passes the corrected test in the headline pool only.
- **One dense model, one LLM, one prompt.** Both models are small. A larger LLM or a tuned fusion weight could change H4a and H5. Fusion weights were fixed in advance on purpose.
- **The dense model has probably read German Wikipedia.** Its pre-training used Wikipedia text. It was not trained on GermanDPR questions, but the passages may not be new to it.
- **The splitter's word list comes from the whole passage pool,** including passages that are gold for test questions. It uses word counts only, as a search engine would on its own index.
- **The routing rule for the LLM was tuned on the development set.** Its effect there (+1.2 points, 12 won, 0 lost) shrank on the test set (+0.8, 10 won, 2 lost).
- **The compound labels are one person's,** on 100 words, and were revised once after the scores were known. The splitter scores move with the labels: F1 0.73 and 0.77 on the first pass, 0.80 and 0.85 after the revision, 0.86 and 0.86 on an AI's labels.
- **The hypotheses were fixed in a local file,** not in a public registry, and this folder has no version history from that day. The file dates are the only record.

## Protocol

Full text in [`HYPOTHESES.md`](../HYPOTHESES.md).

- **Development and test are separate.** All choices on 1,000 training questions. The 1,025 test questions were searched once, after the configuration was frozen. Later the unchanged systems were run on them twice more, only to check that the results repeat; both repeats gave identical reports.
- **Fixed, never tuned:** BM25 with k1 = 1.2 and b = 0.75; the dense model as published; rank fusion with k = 60 over the top 100; the LLM at temperature 0 with a fixed seed.
- **Metrics:** Recall@10 (main) and MRR@10.
- **Statistics:** 95% percentile bootstrap with 10,000 resamples of gold passages; paired sign-flip randomisation test on per-passage differences with 10,000 draws, two-sided; Holm correction across the six Recall@10 tests.
- **Failed hypotheses are reported** with the same weight as supported ones.

## Reproduce in full

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps

# four Parquet files, pinned commits, checked by SHA-256
.venv/bin/python -I -m compound_search.download --raw data/raw

.venv/bin/python -I -m pytest -q   # the unit tests, a few seconds
./reproduce.sh                     # 37 minutes on an M2 with 8 GB
```

Notes:

- The first dense run encodes all passages and stores them under `data/cache/`. In the clean-copy check it took 28 minutes, on a laptop that was doing other work at the same time; later runs read the stored embeddings.
- The LLM's answers for all routed queries are stored in `data/cache/rewrites_*.json` and are part of the repository, so the rewriting step repeats exactly without Ollama. To generate them again, delete that file and have `ollama run gemma4:e2b-it-qat` available.
- `reproduce.sh` also re-runs the frozen systems on the test questions. That is a repeat of the one test run, not a second attempt: do not change a system and run it again.
- `./check_fresh_copy.sh /path/to/new/folder` does all of the above in a clean copy and compares every result file with the saved ones. On 2026-11-12 that gave 33 identical files, with a new environment, a fresh download and freshly computed embeddings.
- Re-running the development systems from scratch gives the saved ranks for all 1,000 questions and all 15 systems of the study (checked on 2026-11-12).

Single steps:

```bash
.venv/bin/python -I -m compound_search.stage0 --raw data/raw --out results/stage0_baseline.json
.venv/bin/python -I -m compound_search.dev --raw data/raw --out results/dev bm25_plain bm25_stem bm25_parts_stem
.venv/bin/python -I -m compound_search.error_analysis --raw data/raw --dev-results results/dev --cache data/cache \
    --out-json results/error_analysis_50.json --out-md results/error_analysis_50.md
.venv/bin/python -I -m compound_search.labels score --raw data/raw --labels labels/compounds_labels.csv \
    --out results/splitter_eval.json
.venv/bin/python -I -m compound_search.tables --results results   # rewrite the tables
```

## Files

- [`HYPOTHESES.md`](../HYPOTHESES.md): hypotheses, test rules, frozen configuration, dated notes.
- `src/compound_search/`
  - `data.py`: loading, passage pool, development split. `download.py`: fetches and checks the data.
  - `text.py`: tokenisers, stems, lemmas. `bm25.py`: the index. `splitter.py`: both compound splitters.
  - `dense.py`: the dense encoder and index. `rewrite.py`: routing rule and LLM rewriting.
  - `runs.py`, `metrics.py`, `stats.py`, `report.py`, `pipeline.py`: pools, rank fusion, scores, tests, and the workbench that builds any system by name.
  - `stage0.py`, `dev.py`, `test_eval.py`, `error_analysis.py`: the runners for each stage.
  - `labels.py`, `label_page.py`: the 100-word sheet, the labelling page and the splitter score.
  - `tables.py`, `figure.py`: write the tables of README.md and docs/RESULTS.md and the chart from `results/`.
  - `residual_check.py`, `splitter_speed.py`: two small measurements quoted in the text.
  - `compare_results.py`: compares two `results/` folders, to check a reproduction.
  - `export_site.py`: the data file for the project's web page.
- `tests/`: one file per module, plus `test_readme_examples.py`, which looks up every rank, count and example quoted in README.md and docs/RESULTS.md in `results/`.
- `results/`: one JSON report per system for the development set (17) and the test set (10), `test_summary.json`, the error analysis, the splitter score and speed, the check of the missed questions.
- `labels/`: the 100-word sheet, the labelling page, my labels (`compounds_labels.csv`), their first pass (`compounds_labels_first_pass.csv`) and the earlier AI labels (`compounds_labels_ai.csv`).
- `reproduce.sh`: every stage in order. `check_fresh_copy.sh`: the same in a clean copy, compared with the saved results.
- `LICENSE`: MIT for the code. The data is not part of this repository; GermanDPR is CC BY 4.0.
