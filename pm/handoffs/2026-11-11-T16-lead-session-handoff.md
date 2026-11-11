# 2026-11-11 — Lead session handoff (T14 → T16 wrap-up)

Read with `pm/README.md`, `pm/STATUS.md`, `pm/research/T16-wrapup-plan.md` (the agreed plan),
`pm/research/T15-guided-story.md` (the story), `pm/research/T14-agreement-gate.md` (results write-up),
`pm/DECISIONS.md` (last ~15 rows are today's) and `pm/KNOWLEDGE.md` K36–K48. This note holds only what those
files do not.

## Where the project is
- **Method (frozen):** the agreement gate, always on. α(x) = α·[argmax c ≠ argmax z], no parameters
  (`configs/method/cal_tda_agree.yaml`). The baseline is TDA. The headline protocol is P2 (α = 2, β = 5 everywhere).
  P1 goes in the appendix.
- **Story** (agreed with Idan): self-vote → unsure vs confident CLIP → agreement vs disagreement (lemmas)
  → method → where it helps (relationship, no threshold) → results. Design and diagnosis come from **val**; **test**
  is results only; test/dev versions of the diagnosis figures are labelled confirmations. The paper is a
  **course report + presentation**, framed **diagnosis first**: the mechanism is the novel part.
- **Results so far (P2):**
  - test, 10 seeds: gate 56.66 / 5.51 vs TDA 56.75 / 10.96;
  - dev, 4 sets: 76.78 / 3.27 vs 76.79 / 7.35;
  - ImageNet-val: 68.22 / 1.93 vs 68.28 / 6.25 (K46);
  - read-before-write TDA (no self-vote) removes about 40% of TDA's excess ECE (K47);
  - the negative cache helps under the gate (K48);
  - Fig. 3 analogue (K44).
- **Prior art** (`related-work.md`, third pass): ZAEC (arXiv 2608.05945) is the closest. It is a post-hoc, online,
  label-free entropy-matching temperature, re-implemented exactly on our traces
  (`results/analysis/t16/zaec_preliminary.csv`): test 5.95 vs the gate's 5.51; gate + ZAEC 5.27. Show it as a
  comparison row (a calibrator on top of TDA, not an adaptation method) plus gate + ZAEC. Also cite
  Malach & Shalev-Shwartz 2017 (the disagreement principle) and MARC (margin no-flip).

## Uncommitted work in the tree (commit first)
- `online_method/cal_tda.py`: `read_before_write` option, plus `configs/method/tda_read_first.yaml`, with tests in
  `test_cal_tda_trace.py`.
- `online_tta.py`: ECALP and DMN-weak are now scored. ECALP's output is already a distribution; DMN-weak's
  is a sum of two distributions, row-normalized. Tests in `test_online_tta_wiring.py` and `test_feature_replay.py`.
- `analysis/story.py`: tested module, renamed from `t15_story_figures.py`, with `tests/test_story.py`. Writes
  `analysis/figures/story_*.png` (val, test, dev) and `results/analysis/story/*.csv`.
- `results/analysis/t16/*.csv` (ZAEC, read-first, negative-cache ablation, ImageNet records), `T16-wrapup-plan.md`,
  and the related-work, KNOWLEDGE and DECISIONS updates.
- The full suite last passed at 357/16 skipped, before the read-first and scoring changes. The touched test files
  pass. **Run the full suite before committing.**

## Running in the background when this was written
- DMN-weak on the 5 test sets, 3 seeds → `replay_results/t16_baselines/`.
- ECALP on the 5 test sets, 3 seeds (about 4 h, O(n²) label propagation) → same folder.
- **Fresh-set Colab run: DONE.** `../features_2026-11-11_new.zip` (commit 0abbf37, T4); all 9 sets (stl10,
  cifar10, voc2007, cifar100, gtsrb, cars, imagenet_r, imagenet_a, imagenet_sketch) are **already unzipped into
  `features/`**. `pytest_gpu.txt` in the zip has exit code 1: check it is the known autocast-emulation deviation
  (K-row in DECISIONS, 18 fp16 ULP). ImageNet-val features (`features/imagenet/`) are also in place.

## Next, in order
1. **Commit** (ask Idan), then the **comparison table** on the 5 test sets (`results/analysis/t16/comparison.csv`,
   tested script):
   - CLIP;
   - TDA;
   - **gate**;
   - read-first TDA;
   - TDA + T (T fitted on val: `analysis/t14_results.py fit_temperature`);
   - TDA + ZAEC;
   - gate + ZAEC;
   - DMN-weak;
   - ECALP.

   Turn the ZAEC function (scratch version in the plan's numbers) into a tested function in `analysis/`. TDA must be
   the baseline for all paired Δ.
2. **Fresh sets**, when the zip arrives:
   - replay CLIP, TDA (full trace) and the gate under P2, 3 seeds (`scripts/replay_dev.sh` handles dev-style
     sets);
   - report per-dataset results and ΔECE vs TDA;
   - redo `story.py` `fig_step5`, the "where it helps" ρ, over every dataset (fresh included).
3. **Figures audit (val/test separation)**:
   - main-text diagnosis figures from val (`story_*_val.png`);
   - results from test;
   - Fig. 3 analogue: replace with the gated version (CLIP / +neg / gated pos / gate / TDA,
     `negative_cache_ablation_gated.csv`), designed on val and shown on test;
   - reliability diagrams: test.
4. **Course report** (`report/`, LaTeX), **about 4 pages, professor-level, not simplistic** (T16 plan §10 overrides §1):
   restructure diagnosis-first. Every number from tracked
   CSVs. Proofs in the appendix. Earlier attempts in one table. P1 appendix.
5. **Deck v3** (`presentation/build_pptx.py`), **20 slides at most** (backup slides after them are fine), per T16 plan §10.
   The TDA section uses **TDA paper Fig. 1 and/or Fig. 2** (arXiv 2403.18293 source), cited; reuse v2's TDA slides
   where they fit. The interim review deck
   is the claude.ai artifact "Agreement-gated TDA".
6. A reviewer agent on the analysis, report and deck (RULES 13), then STATUS and the handoff.

## Gotchas
- Windows: Python is `C:\Users\User\miniconda3\envs\caltta\python.exe`; set `PYTHONUTF8=1`. A shell hook
  sometimes creates empty files named after fragments of code containing `>` (e.g. `0.6).double().mean())`).
  Check they are empty, then delete them; never stage them.
- `git add` specific files only. The untracked Colab leftovers (`colab_run.json`, `datasets_verify_*.json`,
  `download_datasets.log`, `pytest_gpu.txt`) stay out.
- The replay of 1000-class sets is slow on CPU (ImageNet: about 30–60 min per seed per method).
- Commit only when Idan says so. He pushes; Claude cannot push to `macwave12`.
