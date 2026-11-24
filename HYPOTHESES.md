# Hypotheses and test protocol

Written on 2026-11-11, after Stage 0 (plain BM25 on the dev set) and before any other
system was built. The test questions had not been searched at this point.

Sections 1 to 5 are fixed from here on. Section 6 is filled in when the config is frozen,
just before the single test run. Later changes are added at the bottom with a date; nothing is deleted.

## 1. Setup

- **Passages:** all GermanDPR positives and hard negatives, train and test, deduplicated (26,372).
- **Test queries:** the 1,025 GermanDPR test questions. Searched once, after the freeze.
- **Dev queries:** 1,000 train questions, random sample with seed 13. All choices are made here.
- **Gold:** the positive passage of the question.
- **Headline pool:** "own hard negatives dropped". Each question's own three hard negatives
  were mined by deepset with BM25, so they are removed from that question's ranking.
  They remain distractors for all other questions.
- **Second pool:** "as shipped", reported beside the headline for every system.

## 2. Metrics and statistics

- **Main metric:** Recall@10. **Second metric:** MRR@10.
- **Intervals:** 95 % percentile bootstrap, 10,000 resamples, resampling gold passages
  (not questions), because several questions share one gold passage.
- **Paired test against a comparison system:** sign-flip randomisation test on the
  per-passage differences, 10,000 draws, two-sided.
- **Multiple tests:** the six Recall@10 tests below form one family, corrected with Holm.
  A hypothesis counts as **supported** if the difference is positive and the corrected p is below 0.05.
  Otherwise it is **not supported**. MRR@10 results are reported with intervals and uncorrected p values, as secondary.
- All tests use the headline pool on the test set.

## 3. Hypotheses

Each line names the comparison and my prediction, written before seeing any dev result for these systems.

| ID | Claim | Comparison (Recall@10) | Prediction |
|---|---|---|---|
| H1 | Stems or lemmas help BM25 | BM25 + stems or lemmas vs. plain BM25 | small gain, +1 to +3 points |
| H2 | Compound splitting helps on top | BM25 + splitting vs. the same system without splitting | small gain, 0 to +2 points |
| H3 | The preprocessing gain survives next to a dense model | hybrid(best lexical, dense) vs. hybrid(plain BM25, dense) | shrinks, 0 to +1 point |
| H4a | The hybrid beats the dense model alone | hybrid(best lexical, dense) vs. dense | gain |
| H4b | The hybrid beats the best lexical system alone | hybrid(best lexical, dense) vs. best lexical | gain |
| H5 | LLM rewriting of unsure queries helps | hybrid + rewriting vs. the same hybrid | about 0; the local model is small |

H3 is the main question of the study. Dense vs. plain BM25 is reported but is not a hypothesis.

## 4. What may be chosen on the dev set

- Stems or lemmas for H1, and the tool for each.
- The compound splitter (an open tool or the project's own), how its parts enter the index
  (added next to the whole word, or replacing it), and whether it is combined with stems or lemmas.
- "Best lexical" = the lexical system with the highest dev Recall@10 on the headline pool. Ties go to the simpler system.
- The score rule that sends a query to the LLM, its threshold, and the rewriting prompt.

## 5. What is fixed and never tuned

- BM25: k1 = 1.2, b = 0.75 (Lucene defaults) for every lexical system.
- Dense model: multilingual-e5-small, used as published, no fine-tuning.
- Hybrid: reciprocal rank fusion with k = 60 over the top 100 of each system.
- LLM: `gemma4:e2b-it-qat` in Ollama, temperature 0. The routing rule may use retrieval
  scores and rankings only, never gold passages or gold answers.
- The headline number: X = Recall@10 of the final system minus Recall@10 of plain BM25, on the
  test set, headline pool. The final system is the one with the best dev Recall@10 at the freeze.
  The number is reported whatever its size or sign. The share of it that comes from
  linguistic preprocessing is the H3 difference, stated next to it.

## 6. Frozen config

Frozen on 2026-11-12 after Stages 0–5 on the dev set (1,000 train questions, seed 13) and before any test question was searched:

- **Baseline (`bm25_plain`):** Plain lowercase Unicode word tokens (`\w+`), BM25 $k_1 = 1.2, b = 0.75$ (dev Recall@10 = 0.741 on `own_hard_negatives_dropped`).
- **H1 system (`bm25_stem`):** Snowball German stemmer (`snowballstemmer`), chosen over `simplemma` lemmas (`bm25_lemma`) because dev Recall@10 was 0.808 vs. 0.791.
  - **H1 test comparison:** `bm25_stem` vs. `bm25_plain`.
- **H2 system & "Best lexical" (`bm25_parts_stem`):** Snowball German stems combined with the project's corpus-trained `LexiconSplitter` (`src/compound_search/splitter.py`, learned from passage-pool word frequencies only) replacing noun compounds with their constituent parts (`keep_whole=False`). Chosen because it achieved the highest lexical dev Recall@10 on the headline pool (0.850 vs. 0.847 for `bm25_split_stem` and 0.843 for `bm25_hanta_stem`).
  - **H2 test comparison:** `bm25_parts_stem` vs. `bm25_stem`.
- **Dense model (`dense`):** `intfloat/multilingual-e5-small` used as published without fine-tuning (`passage: ` and `query: ` prefixes, masked mean pooling, L2 normalisation; dev Recall@10 = 0.949).
- **Control hybrid (`hybrid_plain_dense`):** Unweighted Reciprocal Rank Fusion ($k = 60$ over top 100) of `bm25_plain` and `dense` (dev Recall@10 = 0.888).
- **Best-lexical hybrid (`hybrid_parts_stem_dense`):** Unweighted Reciprocal Rank Fusion ($k = 60$ over top 100) of `bm25_parts_stem` and `dense` (dev Recall@10 = 0.936).
  - **H3 test comparison:** `hybrid_parts_stem_dense` vs. `hybrid_plain_dense`.
  - **H4a test comparison:** `hybrid_parts_stem_dense` vs. `dense`.
  - **H4b test comparison:** `hybrid_parts_stem_dense` vs. `bm25_parts_stem`.
- **H5 rewritten hybrid (`hybrid_parts_stem_dense_rewrite`):** Routes a query to local `gemma4:e2b-it-qat` in Ollama (`temperature = 0.0`, `seed = 13`, `num_predict = 48`, `think = False`, using `PROMPT_TEMPLATE` in `src/compound_search/rewrite.py`) if and only if `bm25_parts_stem` and `dense` disagree on the rank-1 passage AND the top-1 fused RRF score is strictly below $1/61 + 1/70 = 0.030679$ (`DEFAULT_MAX_RRF_TOP1`). For routed queries, the expanded query is searched on `bm25_parts_stem` and `dense` and fused with the original lexical and dense rankings via RRF ($k = 60$, top 100); non-routed queries keep their exact `hybrid_parts_stem_dense` ranking (dev Recall@10 = 0.948).
  - **H5 test comparison:** `hybrid_parts_stem_dense_rewrite` vs. `hybrid_parts_stem_dense`.
- **Final system for the headline number:** By the rule in Section 5 ("the one with the best dev Recall@10 at the freeze"), standalone `dense` (0.949) and `hybrid_parts_stem_dense_rewrite` (0.948, diff -0.001, $p = 1.0000$) are within 1 query on dev; `dense` is the strict numerical winner on dev (0.949) and `hybrid_parts_stem_dense_rewrite` is the best hybrid/full-pipeline system (0.948). Both differences over `bm25_plain` are reported alongside the H3 difference.


## 7. Notes added after the test run (2026-11-12, later the same day)

Nothing above this line was changed. These notes record what a re-check of the finished study found. The test questions were not searched again.

1. **Which system the headline number is about.** By the rule in Section 5, the final system is the one with the best dev Recall@10 at the freeze. That is `dense` (0.949). The protocol's sentence is therefore the dense one: +21.6 points over plain BM25. The full-pipeline sentence (+22.3 points) is reported beside it, as Section 6 announced. Its extra 0.8 points are the H5 difference, which was not supported, so the README does not lead with it.
2. **The as-shipped pool.** Section 2 fixes the headline pool for all six tests, and the verdicts follow it. For the record, the same tests on the as-shipped pool: H1 +6.1, H2 +5.3 and H4b +11.7 points are supported; H3 is +2.7 points with Holm p = 0.076 and does not pass; H4a is −1.5 points; H5 is +0.6 points. The README shows both tables.
3. **How many queries went to the LLM.** The rule of Section 6 routed 53 of the 1,000 dev questions (5.3 %) in the headline pool and 46 in the as-shipped pool. On the test set, 83 of 1,025 questions were routed in at least one of the two pools (counted from the stored LLM answers). Reports now record the routed query ids.
4. **Stage 0 report.** `stage0.py` computed how often a question's own hard negatives outrank its gold passage and then discarded the numbers. Fixed, and `results/stage0_baseline.json` was written again on the dev and train questions: 75.6 % of dev questions, top hit for 73.2 %. The dev scores in it are unchanged.
5. **Reproduction.** All 15 dev systems were rebuilt from scratch. Every one gives the saved gold rank for all 1,000 questions in both pools.
6. **Splitter score.** The 100-word sheet was labelled by an AI, without sight of any splitter's output: `labels/compounds_labels_ai.csv`. Both splitters reach F1 0.86. Labels by a person would replace this.
7. **Corrections to the write-up.** Three statements in the first README and notes were not backed by the data and were replaced: the examples for splitting losses (they were not in the sample), two examples for LLM wins (those queries did not change), and the claim that unlabelled passages dominate the remaining misses (7 of 29 by the answer-string check).
8. **What this file can prove.** It was a local file without version control, so its dates rest on file timestamps only.
9. **Two systems added for the website.** `bm25_parts` (plain words, compounds replaced by their parts) and `hybrid_parts_dense` (the same, fused with the dense model) were built on 2026-11-12 so that every switch combination in the web demo is a real run and "split compounds" means the same with and without stems. They were run on the dev set only (Recall@10 0.803 and 0.910) and are not part of any hypothesis.
10. **Repeat of the test run, as a check.** The frozen systems were run on the test questions a second time, unchanged, to see whether the saved numbers can be reproduced. All ten test reports and `test_summary.json` came out identical. No system was changed before or after, and no choice was made from it. The one thing the repeat added is a record of which test questions the routing rule sent to the LLM, which the code did not write at the time of the first run: 67 of 1,025 (6.5 %) in the headline pool, 56 in the as-shipped pool, 83 in at least one of the two. That record was added to `results/test/hybrid_parts_stem_dense_rewrite.json`; the scores in the file are those of the first run.
11. **Splitter speed.** Earlier texts compared the run time of two whole dev runs ("about 5 seconds against about 150") and one version said "28 times faster". Neither was a measurement of the splitters. Measured on the same words, each starting cold (`results/splitter_speed.json`): 31 seconds for the project's splitter, including the word count it starts from, and 149 seconds for HanTa.
12. **Error analysis.** `results/error_analysis_50.json` was written again with counts over all dev questions of each category; the 50 sampled questions are the same. The sentence that splitting losses are "mostly names" was not backed by a count and was replaced: in 16 of the 28 lost questions the question contains a word that the splitter cuts, and several of those words are names; in the other 12 no word of the question is cut.
13. **Two sentences made exact.** The opening example said a keyword search "sees no shared word" between the *Einwohnerdichte* question and its passage; they share the city's name, and the text now says so. The LLM is asked for four to six search terms; it returned four to six in 131 of the 164 stored answers.
14. **Reproduction in a clean copy.** `./check_fresh_copy.sh` copied the project without its environment, data and stored embeddings, installed the pinned packages, downloaded the data, ran `./reproduce.sh` and compared all 33 JSON files under `results/` with the saved ones (run times ignored). All 33 were identical. This was the second unchanged repeat of the test run (the first is note 10). Section 1 says the test questions are "searched once"; that describes the run that produced the results. Both repeats used the frozen systems and changed nothing.
15. **The author's own labels for the 100 compounds (2026-11-12, evening).** The author labelled the sheet on the labelling page, without sight of any splitter's output: `labels/compounds_labels.csv`. These labels now carry the splitter score in the write-up (`results/splitter_eval.json`): own splitter precision 0.82, recall 0.65, F1 0.73; HanTa 0.92, 0.65, 0.77. The AI labels of note 6 are kept for comparison. The two sets agree on the cuts of 86 of the 100 words (`results/label_agreement.json`); in 10 of the other 14 the author cut a word that the AI left whole, among them cuts at a prefix or suffix (*Ver|schiebung*, *Juden|tum*), which neither splitter looks for. Nothing in any retrieval system depends on these labels, so no retrieval result changed.
16. **The labels of note 15 were revised once, after the scores were known (2026-11-12, evening).** The first pass had cuts at a prefix, a suffix or an ending in eight words. They were then changed at the author's request by one rule: a cut stays only where a word meets a word. Changed: *Ver|schiebung*, *Juden|tum*, *Juden|tum|s*, *Wissen|schaften*, *Infra|rot*, *Zusammen|kunft* (now uncut, "not a compound"), *Preußisch|e* (now uncut, still "name"), *Holz|fest|igkeit* (now *Holz|festigkeit*). Left as labelled, because both sides are words: *Vor|schläge*, *Mit|gliedern*, *Sieben|jährigen*. The basis is the labelling page's own instruction, written before any labelling, which counts a derived word as "not a compound". Because the revision came after the scores, both are reported: first pass (`labels/compounds_labels_first_pass.csv`, `results/splitter_eval_first_pass.json`) own splitter F1 0.73, HanTa 0.77; revised (`labels/compounds_labels.csv`, `results/splitter_eval.json`) own splitter precision 0.84, recall 0.77, F1 0.80; HanTa 0.95, 0.77, 0.85. Every changed row carries a note in the file. The revised labels agree with the AI labels on the cuts of 94 of 100 words (first pass: 86). The numbers of note 15 describe the first pass.
17. **Wording only (2026-11-24).** For publication, what this file called a summary sentence is now called the headline number, here and in the code. No number, test or result changed.
