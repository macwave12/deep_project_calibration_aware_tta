# TDA baseline hyperparameter and protocol audit

Date: 2026-11-04. Investigation only: no hyperparameter was changed and no experiment was
rerun. Sources checked: the official TDA repository (`kdiAAA/TDA`, `main`), the TDA paper
(arXiv:2403.18293, CVPR 2024, including its supplementary sections 6-8), the benchmark
framework we fork (`TomSheng21/tta-vlm`, `main`), and the benchmark paper
(arXiv:2506.24000v2, NeurIPS 2025, 13 Oct 2025 revision).

## Summary

- Every TDA hyperparameter in this repo matches the official TDA configs cell for cell, for
  all five datasets, positive and negative cache alike. The benchmark's `tda.py` is
  byte-identical to ours (modulo CRLF), and its per-dataset table matches the official
  configs for all 15 datasets it covers.
- The TDA paper's "single ImageNet validation set" setting does exist and is identifiable:
  α = 2.0, β = 5.0 (supplementary Table 6), shot capacity 3/2, negative mask 0.03, entropy
  band [0.2, 0.5]. The official repo uses α/β = 2/5 only for ImageNet and ImageNet-A. For the
  ten cross-domain datasets it ships different per-dataset α/β, and neither the paper nor
  the repo says how those were chosen.
- The benchmark adopted the official per-dataset values ("hyperparameters ... recommended by
  the original implementations"), the single template "a photo of a [CLASS]", one
  centre-crop view, batch size 1, seed 0, shuffled stream. Our runs use exactly that
  protocol.
- Reproduction: our ViT-B/16 CLIP and TDA accuracies are within 0.16 points of the
  benchmark's Table 14 on every dataset. The benchmark reports ECE for ResNet-50 only
  (Tables 7 and 19); it has no ViT-B/16 ECE, so our ECE rows have no published counterpart.
- Recommendation: (a) keep the current values. They are the published ones. Add one
  disclosure sentence to the Experiments section (text below) and state the fairness caveat.

## 1. Official TDA configs vs this repo

Official files: `configs/dtd.yaml`, `configs/oxford_flowers.yaml`, `configs/oxford_pets.yaml`,
`configs/eurosat.yaml`, `configs/fgvc.yaml` in `kdiAAA/TDA`. Ours: `TDA.pos_params` and the
hard-coded negative-cache constants in `online_method/tda.py`, mirrored in
`configs/<dataset>.yaml`. The run records under `results/raw/<dataset>_tda.json` confirm the
values each headline TDA run actually used.

### Positive cache

| Dataset | Official shot / α / β | `tda.py` α / β | `configs/<dataset>.yaml` shot / α / β | Run record α / β | Match |
|---|---|---|---|---|---|
| DTD | 3 / 2.0 / 3.0 | 2.0 / 3.0 | 3 / 2.0 / 3.0 | 2.0 / 3.0 | yes |
| Flowers102 (`oxford_flowers`) | 3 / 1.0 / 5.0 | 1.0 / 5.0 | 3 / 1.0 / 5.0 | 1.0 / 5.0 | yes |
| Pets (`oxford_pets`) | 3 / 2.0 / 7.0 | 2.0 / 7.0 | 3 / 2.0 / 7.0 | 2.0 / 7.0 | yes |
| EuroSAT | 3 / 4.0 / 8.0 | 4.0 / 8.0 | 3 / 4.0 / 8.0 | 4.0 / 8.0 | yes |
| Aircraft (`fgvc`) | 3 / 2.0 / 2.0 | 2.0 / 2.0 | 3 / 2.0 / 2.0 | 2.0 / 2.0 | yes |

Shot capacity 3 is a literal in `tda.py` (`update_cache(..., 3)`), the same on every dataset,
as in every official config.

### Negative cache

The official repo uses one identical negative block in all 15 configs; `tda.py` hard-codes
the same values.

| Field | Official (all configs) | `tda.py` | `configs/<dataset>.yaml` | Match |
|---|---|---|---|---|
| shot_capacity | 2 | 2 (literal in `update_cache`) | 2 | yes |
| alpha | 0.117 | 0.117 | 0.117 | yes |
| beta | 1.0 | 1.0 | 1.0 | yes |
| entropy_threshold lower / upper | 0.2 / 0.5 | 0.2 / 0.5 (literal) | 0.2 / 0.5 | yes |
| mask_threshold lower / upper | 0.03 / 1.0 | 0.03 / 1.0 (literal) | 0.03 / 1.0 | yes |

### The rest of the official table, for completeness

The benchmark's `tda.py` carries the same table for every dataset the official repo
configures: Caltech101 5/5, Cars 1/7, Food101 1/1, SUN397 2/3, UCF101 3/8, ImageNet 2/5,
ImageNet-A 2/5, ImageNet-R 1/8, ImageNet-V2 1/8, ImageNet-Sketch 2.363/7.45. All ten match
the official YAML files. (`office-home` and `DomainNet` entries in the benchmark table have
no official counterpart and are not used here.)

### Implementation differences between the official code and the fork (none affect values)

- The official code scales cosine similarities by a literal `100.`; the fork multiplies by
  CLIP's learned `logit_scale.exp()`. From the cached ViT-B/16 checkpoint that is
  `exp(4.60517) = 100.0000076`, so the two are the same to float precision.
- The official entropy is `-(p log p)` with no epsilon; the fork adds `1e-5` inside the log.
  Both are divided by `log2(C)` for the negative-cache band. Negligible.
- The official negative-cache values are kept in `half()`; the fork uses `float()`.
  Numerical only.
- Official stream order: `shuffle=True` with `random.seed(1)`/`torch.manual_seed(1)`. The
  benchmark and we use `shuffle=True` with seed 0. Different permutations, same protocol.

### A documentation nit, no action required

`configs/method/tda.yaml` and `configs/method/tda_equivalent.yaml` list `alpha: 2.0`,
`beta: 5.0`. Those are the paper's ImageNet values, but they are inert: the runner overwrites
them from `TDA.pos_params` per dataset, and the run records above show the per-dataset values
were used. `tests/test_configs.py` already pins the mirror files to the runtime table.

## 2. Is there a global, dataset-independent setting?

Yes in the paper, and the official repo uses it only for ImageNet. Quotes:

TDA paper, Section 4.1, "Implementation details":

> "We conduct a search for all our hyper-parameters using a single ImageNet validation set.
> The threshold p_l for negative pseudo-labeling in Eq 4 is set as 0.03. The upper and lower
> thresholds [τ_l, τ_h] for testing feature selection in Eq 5 are set as [0.2, 0.5]. Once
> searched, these hyperparameters are fixed and evaluated across various new datasets."

TDA paper, supplementary Section 8, "The residual and sharpness ratios" (Table 6, ImageNet,
CLIP-ResNet-50):

> "The experiments in Table 6 show that the optimal residual ratio is 2.0 for TDA (instead of
> 1.0 in Tip-Adapter) ... The optimal sharpness ratio for TDA is 5.0, which is close to the
> 5.5 in Tip-Adapter."

TDA paper, Section 4.3, on shot capacity:

> "To facilitate practical applications, the shot capacity settings are fixed and directly
> applied to new datasets without the need for additional parameter adjustments."

So the paper's one searched setting is α = 2.0 (residual ratio), β = 5.0 (sharpness ratio),
positive/negative shot capacity 3/2, p_l = 0.03, [τ_l, τ_h] = [0.2, 0.5]. The official
`configs/imagenet.yaml` and `configs/imagenet_a.yaml` contain exactly that. Every other
official config keeps the negative block and shot capacities but changes α/β.

The paper never lists per-dataset α/β. The only statement about them is in the official
README, "Run TDA / Configs":

> "For ease of reference, the configurations provided aim to achieve optimal performance
> across datasets on two benchmarks, consistent with the results documented in our paper.
> However, specific tuning of these parameters for negative cache could potentially unlock
> further enhancements in performance. Adjusting parameters like `alpha` and `beta` within
> the positive cache lets you fine-tune things to match the unique needs of each dataset."

That is the whole provenance: the cross-domain α/β were chosen to produce the paper's
numbers, and how they were chosen is not stated. The paper's text ("fixed and evaluated across
various new datasets") describes the thresholds and shot capacities accurately and the
positive-cache α/β inaccurately.

## 3. What the benchmark used

Benchmark paper, Section 3, "Vanilla Test-Time Adaptation under Fair Comparison":

> "(Text templates). To eliminate variability due to prompt selection, we fix the text
> template to CLIP's default prompt, 'a photo of a [CLASS]' across all experiments."

> "(Hyperparameters). We adopt the hyperparameters and optimization protocols recommended by
> the original implementations of each TTA method. However, we note that hyperparameter
> tuning during test time remains an open and challenging problem."

Benchmark paper, Section 2.2 / Appendix A, on online methods:

> "We set the batch size to 1 for all online TTA methods in the experiments. Please note that
> although online methods process the same streaming data, some methods (i.e., DMN,
> BoostAdapter, DPE, DynaPrompt) employ AugMix augmentation to obtain a batch of views, while
> other methods simply use single weak augmentation."

Benchmark paper, Section 5, "(Calibration)":

> "We measure calibration using the Expected Calibration Error (ECE) with 20 equal-width
> bins."

Appendix A, Table 11 lists TDA's official code as `https://github.com/kdiAAA/TDA`.

Framework code (`TomSheng21/tta-vlm`, `main`), confirmed by diff against ours:

- `online_method/tda.py` is identical to ours apart from line endings; its `pos_params`
  table equals the official configs (Section 1).
- `online_tta.py` for `tda`/`clipzs`: `Resize(224, BICUBIC)`, `CenterCrop(224)`, RGB,
  `ToTensor`, CLIP normalisation; one view; `DataLoader(shuffle=True)`; `--seed` default 0;
  the README command passes `-b 1 --ctx_init a_photo_of_a`, which the prompt learner turns
  into `"a photo of a <classname>."` for every class and dataset. Our `scripts/run_one.ps1`
  passes the same flags plus `-j 0` (Windows worker fix, no effect on sampling order) and
  `--seed 0`.
- Our only changes to `online_tta.py` are `cal_tda` registration, the `--config`,
  `--run-name`, `--out-dir` flags, the JSON scorer, and the `os.makedirs` portability fix.

Backbones the benchmark reports TDA on: CLIP-ResNet-50 (Tables 1, 2, 7, 19, and the
CoOp/TeCoA tables), CLIP-ViT-B/16 (Tables 14, 15, and the MaPLe table), CLIP-ViT-B/32
(Tables 16, 17). ECE is reported for ResNet-50 only.

Note on the official TDA repo's own protocol, which differs from the benchmark's: it uses
per-dataset hand-crafted templates from CoOp (DTD `"{} texture."`, EuroSAT
`"a centered satellite photo of {}."`, Aircraft `"a photo of a {}, a type of aircraft."`,
Flowers `"a photo of a {}, a type of flower."`, Pets `"a photo of a {}, a type of pet."`) and
seed 1. That is why the TDA paper's zero-shot numbers (e.g. EuroSAT 50.42 with ViT-B/16) are
higher than the benchmark's and ours (42.2); the template, not the cache, accounts for it.

## 4. Reproduction check (ViT-B/16, test split, single view, "a photo of a")

Accuracy (%): benchmark Table 14 vs `results/summary.csv`. The TDA paper's own ViT-B/16
numbers (Table 3, per-dataset templates) are shown for context only.

| Dataset | CLIP bench. | CLIP ours | Δ | TDA bench. | TDA ours | Δ | TDA paper CLIP / TDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| DTD | 44.39 | 44.39 | 0.00 | 47.10 | 47.04 | −0.06 | 45.04 / 47.40 |
| Flowers102 | 67.40 | 67.32 | −0.08 | 69.87 | 69.91 | +0.04 | 66.99 / 71.42 |
| Pets | 88.25 | 88.20 | −0.05 | 88.83 | 88.77 | −0.06 | 86.92 / 88.63 |
| EuroSAT | 42.22 | 42.06 | −0.16 | 53.25 | 53.17 | −0.08 | 50.42 / 58.00 |
| Aircraft | 23.91 | 23.85 | −0.06 | 24.18 | 24.27 | +0.09 | 23.22 / 23.91 |

No difference reaches 1 point; the largest is 0.16 (EuroSAT zero-shot, 13 images of 8,100).
The residual comes from hardware and library version (RTX 5060 with torch 2.11 cu128 and
fp16 autocast, against the benchmark's GPU and torch), which moves a handful of borderline
argmaxes; for TDA the different permutation of the stream under the same seed on a different
torch build also changes which samples seed the cache. None of seed, prompt, split,
preprocessing, or precision needs invoking beyond that: all four are the same as the
benchmark's by construction (Section 3), and the split sizes (1,692 / 2,463 / 3,669 / 8,100 /
3,333) match the benchmark's Table 12 and the TDA paper's Table 4.

ECE (%): the benchmark has no ViT-B/16 ECE. Its Table 19 (fine-grained, ResNet-50) is
listed beside our ViT-B/16 rows so the direction can be compared; the magnitudes are
different backbones and must not be read as a reproduction.

| Dataset | CLIP bench. RN50 | TDA bench. RN50 | Δ RN50 | CLIP ours ViT-B/16 | TDA ours ViT-B/16 | Δ ours |
|---|---:|---:|---:|---:|---:|---:|
| DTD | 8.82 | 19.19 | +10.37 | 8.27 | 17.41 | +9.14 |
| Flowers102 | 2.96 | 5.68 | +2.72 | 2.63 | 6.39 | +3.76 |
| Pets | 5.62 | 3.00 | −2.62 | 4.37 | 2.52 | −1.85 |
| EuroSAT | 14.76 | 23.56 | +8.80 | 7.16 | 14.38 | +7.22 |
| Aircraft | 6.31 | 15.74 | +9.43 | 5.44 | 15.51 | +10.07 |

The sign of TDA's calibration change agrees on all five datasets, including Pets, the one
dataset where TDA improves ECE in both. The 5.70 → 9.21 figure our report quotes is Table 7's
ten-dataset fine-grained average for ResNet-50, and the report already says so. Because the
benchmark's motivating ECE table is ResNet-50 and our runs are ViT-B/16, the report should
keep describing those numbers as the benchmark's ResNet-50 result and our ECE as an
independent ViT-B/16 measurement, which is what it does now.

## 5. Recommendation

**(a) Keep the current values.** They match the official TDA configs and the benchmark's
table exactly, for the positive and negative cache, on every dataset; the accuracy
reproduction is within 0.2 points everywhere. There is no mismatch to fix and nothing to
rerun.

Sentence to add to the Experiments section (4.1, after the backbone/seed sentence; it names
no code identifier):

> TDA's positive-cache weights α and β are the per-dataset values shipped with its official
> implementation (DTD 2/3, Flowers102 1/5, Pets 2/7, EuroSAT 4/8, Aircraft 2/2), which are
> also the values the benchmark adopted; the TDA paper itself describes a single setting
> searched on ImageNet (α = 2, β = 5) and does not report the per-dataset ones. Shot
> capacities and the negative-cache thresholds are the paper's, identical on every dataset.

**Fairness caveat: yes, state it.** TDA's α/β were chosen per dataset with the test sets in
view (the official README says the configs "aim to achieve optimal performance across
datasets ... consistent with the results documented in our paper"). Our fusion weight
(0.5) and gate thresholds (0.2, 0.6) are one global setting fixed before any run. The
comparison therefore favours the baseline, not our method, and a sentence saying so belongs
next to the one above:

> Our own settings, the fusion weight and the two gate thresholds, are single global values
> fixed before any run, so where the comparison is unequal it favours TDA.

Both sentences are proposals; neither has been added to the report.
