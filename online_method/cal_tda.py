"""Calibration-aware TDA: three training-free changes to the TDA cache loop.

1. Admission by top-2 margin instead of entropy, so the cache stops rewarding
   samples the model is merely loud about.
2. Fusion in probability space instead of adding logits, so the cache can change
   *which* class wins without inflating *how confident* the answer is.
3. A final softmax temperature chosen by label-free leave-one-out estimation.

Each change is independently switchable from config, which is what makes the
ablation study free. This module holds the pure operations; the runner that
plugs them into the upstream loop lives below (see docs/UPSTREAM_SEAMS.md).

Note for anyone reading a `cal_tda` run's `save_results.pt`: its `output` column
holds **probabilities**, whereas `tda`/`clipzs` write raw logits there. Nothing
in the file marks the difference, so check the method before comparing them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

import torch
from torch.nn import functional as F

from online_method.tda import TDA, Entropy, compute_cache_logits, update_cache
from utils.loo_temperature import default_grid, loo_accuracy, search_temperature


def margin_score(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 score. Large margin = a decisive, unambiguous prediction."""
    if logits.shape[-1] < 2:
        raise ValueError("margin needs at least two classes")
    top2 = logits.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def should_admit(logits: torch.Tensor, threshold: float) -> bool:
    """Admit a sample to the cache only when its top-2 margin clears `threshold`."""
    return bool(margin_score(logits.reshape(-1)) > threshold)


def probabilistic_fusion(
    clip_logits: torch.Tensor,
    cache_logits: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Weighted merge of two *distributions*, returning probabilities.

    Unlike `clip_logits + alpha * cache_logits`, a convex combination is bounded
    by its most confident input, so agreement between CLIP and the cache cannot
    manufacture confidence that neither source had.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    clip_probs = clip_logits.softmax(dim=-1)
    cache_probs = cache_logits.softmax(dim=-1)
    return (1.0 - weight) * clip_probs + weight * cache_probs


@dataclass
class CalTdaConfig:
    """Every knob of our method. Each `use_*` flag isolates one contribution,
    which is exactly what the ablation table toggles."""

    use_margin_admission: bool = True
    margin_threshold: float = 0.1
    use_prob_fusion: bool = True
    fusion_weight: float = 0.5
    use_loo_temperature: bool = True

    # Inherited from TDA; kept so `cal_tda` with all flags off == `tda`.
    entropy_threshold: float = 0.5
    alpha: float = 2.0
    beta: float = 5.0
    shot_capacity: int = 3

    def as_dict(self) -> dict:
        return asdict(self)

    def inert_fields(self) -> list[str]:
        """Fields that could not have influenced a `CalTDA` run under this config.

        A record that lists an inert knob beside live ones invites a reader of
        the ablation table to conclude the knob produced a number when it had no
        effect at all.
        """
        inert = [
            # `admit_to_cache`'s entropy branch is unreachable from `CalTDA`:
            # upstream admits to the positive cache unconditionally, so the
            # runner bypasses the fallback to stay equivalent with flags off.
            # `entropy_threshold` is therefore inert in *every* cal_tda run.
            'entropy_threshold',
        ]
        if self.use_prob_fusion:
            inert.append('alpha')  # the fusion branch never reads it
        else:
            inert.append('fusion_weight')  # the additive branch never reads it
        if not self.use_margin_admission:
            inert.append('margin_threshold')
        return sorted(inert)

    def as_record_dict(self) -> dict:
        """`as_dict()` with inert knobs removed and named under `inert`.

        Removed rather than zeroed so no downstream table can average or plot a
        value that never touched the run; named so their absence reads as
        deliberate rather than as a missing field.
        """
        inert = self.inert_fields()
        data = {k: v for k, v in self.as_dict().items() if k not in inert}
        data['inert'] = inert
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CalTdaConfig":
        """Build a config from a plain mapping (e.g. a parsed YAML file).

        Unknown keys are a hard error: a typo'd flag name in a config file would
        otherwise silently leave the contribution at its default, and the whole
        ablation table would then measure the wrong thing.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown CalTdaConfig field(s): {', '.join(unknown)}; "
                f"known fields are {', '.join(sorted(known))}"
            )
        return cls(**data)


def pre_temperature_logits(
    clip_logits: torch.Tensor,
    cache_logits: torch.Tensor,
    config: CalTdaConfig,
) -> torch.Tensor:
    """The logit-space quantity that `finalize_probs` divides by the temperature.

    Kept separate so the runner can collect the *same* scores the temperature
    will later be applied to; `test_pre_temperature_logits_match_finalize`
    pins the two definitions together.
    """
    if config.use_prob_fusion:
        fused = probabilistic_fusion(clip_logits, cache_logits, config.fusion_weight)
        return fused.clamp_min(1e-12).log()
    return clip_logits + config.alpha * cache_logits


def finalize_probs(
    clip_logits: torch.Tensor,
    cache_logits: torch.Tensor,
    config: CalTdaConfig,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Combine CLIP and cache evidence into final probabilities.

    With every flag off this is exactly upstream TDA: softmax(clip + alpha*cache).
    """
    if config.use_prob_fusion:
        probs = probabilistic_fusion(clip_logits, cache_logits, config.fusion_weight)
        if config.use_loo_temperature and temperature != 1.0:
            # Re-enter logit space to apply temperature, then renormalize.
            probs = (probs.clamp_min(1e-12).log() / temperature).softmax(dim=-1)
        return probs

    logits = clip_logits + config.alpha * cache_logits
    if config.use_loo_temperature:
        logits = logits / temperature
    return logits.softmax(dim=-1)


def admit_to_cache(clip_logits: torch.Tensor, config: CalTdaConfig) -> bool:
    """Cache-admission decision: margin-based when enabled, else TDA's entropy rule."""
    if config.use_margin_admission:
        return should_admit(clip_logits, config.margin_threshold)

    probs = clip_logits.reshape(-1).softmax(dim=-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    return entropy < config.entropy_threshold


class CalTDA(TDA):
    """TDA with margin admission, probabilistic fusion and LOO temperature.

    Subclasses upstream `TDA` rather than copying it, so with every `use_*` flag
    off the only behaviour left is upstream's: `softmax(clip + alpha*pos -
    neg_alpha*neg)`, same caches, same per-dataset alpha/beta. Any difference
    the ablation measures therefore comes from the overrides alone.

    Ground-truth discipline: nothing in this class reads, receives or compares a
    label. It returns a probability vector per sample as `return_dict['output']`;
    the driver pairs those with labels and calls the scorer exactly once.
    """

    def __init__(self, model, device, config=None):
        super().__init__(model, device)
        self.config = config or CalTdaConfig()
        self._score_history = []
        self.temperature = 1.0
        self.temperatures = []
        self._n_estimated = 0
        self._n_admit_offered = 0
        self._n_admit_admitted = 0

    def prepare_model_and_optimization(self, args):
        super().prepare_model_and_optimization(args)

        # Ruled decision: where the per-dataset TDA table and the method config
        # disagree on alpha/beta, the dataset's TDA-tuned values win, and they
        # are applied identically to `tda` and `cal_tda` so the comparison is
        # fair. `CalTdaConfig`'s own alpha/beta are only defaults for unit tests
        # that construct it directly. Writing them back onto the config means
        # the values recorded in the run's `hyperparams` are the ones used.
        self.config.alpha = self.pos_alpha
        self.config.beta = self.pos_beta

        self._score_history = []
        self.temperature = 1.0
        self.temperatures = []
        self._n_estimated = 0
        self._n_admit_offered = 0
        self._n_admit_admitted = 0

    def temperature_summary(self):
        """What contribution #3 actually did, for the run record.

        `search_temperature` grid-searches a bounded range and returns a
        *boundary* value silently when the target confidence is unreachable, so
        a run that saturated at the grid edge on every sample would otherwise be
        indistinguishable from one that found a healthy interior optimum. The
        boundary fractions below are what tells the two apart.

        `grid_min` is the identity temperature (1.0): `default_grid()` is
        smoothing-only by design (see `utils/loo_temperature`), so a high
        `frac_at_grid_min` does not mean the search saturated at some sharpening
        extreme -- it means the label-free estimate wanted to sharpen the
        distribution further and was correctly held at the no-op temperature
        instead. That is the smooth-only constraint doing its job, not a
        failure to find an interior optimum. `frac_at_grid_max` keeps its
        original reading: saturation at the softening extreme.
        """
        if not self.config.use_loo_temperature or not self.temperatures:
            return None
        grid = default_grid()
        low, high = grid[0], grid[-1]
        values = self.temperatures
        n = len(values)
        at_low = sum(1 for t in values if t == low)
        at_high = sum(1 for t in values if t == high)
        return {
            'n_samples': n,
            # Samples where the search actually ran; the rest defaulted to 1.0
            # because no scores had been collected yet (the first sample).
            'n_estimated': self._n_estimated,
            'mean': sum(values) / n,
            'min': min(values),
            'max': max(values),
            'grid_min': low,
            'grid_max': high,
            'frac_at_grid_min': at_low / n,
            'frac_at_grid_max': at_high / n,
            'frac_at_grid_boundary': (at_low + at_high) / n,
        }

    def admission_summary(self):
        """What contribution #1 actually did, for the run record.

        `margin_threshold` is a single global constant applied to every
        dataset (deliberately not tuned per dataset, which would be test-set
        tuning). Logit margins are not guaranteed to share a scale across a
        10-class and a 100-class dataset, so this is the only signal that
        would reveal the gate admitting almost everything on one dataset and
        almost nothing on another.

        When `use_margin_admission` is False, upstream's behaviour is
        *unconditional* admission (see `adaptation_process`), so
        `admission_rate` legitimately reads 1.0 in that case -- that is
        correct, not a bug: it documents that the margin gate did not run
        rather than hiding the flag behind a missing summary.
        """
        if self._n_admit_offered == 0:
            return None
        return {
            'n_offered': self._n_admit_offered,
            'n_admitted': self._n_admit_admitted,
            'admission_rate': self._n_admit_admitted / self._n_admit_offered,
        }

    def _cache_features_and_pseudo_labels(self):
        """The positive cache as (features, pseudo-labels).

        The "labels" are the cache keys: the classes *the method itself*
        predicted when it cached each item. No ground truth is involved.
        """
        features, pseudo_labels = [], []
        for class_index in sorted(self.pos_cache.keys()):
            for item in self.pos_cache[class_index]:
                features.append(item[0].reshape(1, -1))
                pseudo_labels.append(class_index)
        if len(features) < 2:
            return None, None
        stacked = torch.cat(features, dim=0).float()
        return stacked, torch.tensor(pseudo_labels, device=stacked.device)

    def _estimate_temperature(self):
        """Label-free temperature, as `(temperature, was_estimated)`.

        `was_estimated` is False when there was nothing to estimate from yet, so
        the summary can separate "the search chose 1.0" from "the search never
        ran and 1.0 is the no-op default".
        """
        if not self.config.use_loo_temperature or not self._score_history:
            return 1.0, False
        with torch.no_grad():
            features, pseudo_labels = self._cache_features_and_pseudo_labels()
            if features is None:
                return 1.0, False
            # `loo_accuracy` documents that it needs L2-normalized features but
            # does not enforce it; unnormalized input silently ranks neighbours
            # by magnitude instead of direction. Normalize at the call site.
            estimate = loo_accuracy(F.normalize(features, dim=-1), pseudo_labels)
            history = torch.cat(self._score_history, dim=0)
            return search_temperature(history, estimate), True

    def adaptation_process(self, image, images, args):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
                logits = logit_scale * image_features @ text_features.t()
                softmax0 = logits.softmax(dim=-1)

        ent0 = Entropy(softmax0)
        pred0 = torch.max(logits, 1)[1].item()
        num_classes = logits.size(1)

        # --- Contribution 1: margin-based cache admission --------------------
        # Upstream attempts admission for *every* sample (tda.py:93); its
        # entropy is only an eviction key inside `update_cache`, not a gate. So
        # "margin admission off" has to mean unconditional admission, otherwise
        # all-flags-off would not reduce to upstream TDA. `admit_to_cache`'s
        # entropy fallback is therefore deliberately not reached from here; it
        # stays in the module as the documented single-sample admission rule.
        # `should_admit` reshapes to a single row, so it is called per sample.
        # Counting is instrumentation only: `admit` is the exact same
        # short-circuited expression the branch always evaluated, just bound
        # to a name so `admission_summary()` can report on it.
        admit = not self.config.use_margin_admission or admit_to_cache(logits, self.config)
        self._n_admit_offered += 1
        if admit:
            self._n_admit_admitted += 1
            update_cache(
                self.pos_cache, pred0, [image_features, ent0], self.config.shot_capacity
            )

        # Negative cache: untouched upstream behaviour (normalized-entropy band).
        if 0.2 < ent0 / math.log2(num_classes) and ent0 / math.log2(num_classes) < 0.5:
            update_cache(self.neg_cache, pred0, [image_features, ent0, softmax0], 2, True)

        # --- CLIP evidence, corrected by the negative cache -------------------
        # The negative cache carries its own fixed alpha and is a correction to
        # CLIP's own evidence, so it stays on the CLIP side of the fusion. The
        # positive cache is computed *unscaled* (alpha=1.0) because
        # `finalize_probs` applies `config.alpha` itself in the additive branch.
        base_logits = logits.clone()
        if self.use_neg_cache and len(self.neg_cache) > 0:
            base_logits -= compute_cache_logits(
                image_features, self.neg_cache, self.neg_alpha, self.neg_beta,
                text_features.unsqueeze(0), (0.03, 1.0),
            )

        has_cache = self.use_pos_cache and len(self.pos_cache) > 0
        if has_cache:
            cache_logits = compute_cache_logits(
                image_features, self.pos_cache, 1.0, self.config.beta,
                text_features.unsqueeze(0),
            )
        else:
            cache_logits = torch.zeros_like(base_logits)

        base_logits = base_logits.float()
        cache_logits = cache_logits.float()

        # With an empty cache there is no distribution to fuse with; mixing in
        # softmax(zeros) would inject a uniform prior that upstream never has.
        config = self.config
        fusion_fallback = not has_cache and config.use_prob_fusion
        if fusion_fallback:
            config = replace(config, use_prob_fusion=False)

        # --- Contribution 3: label-free temperature ---------------------------
        # Estimated from the scores collected so far, which do not yet include
        # this sample's, so the current prediction never informs its own
        # temperature. (The cache *has* already seen this sample, as upstream's
        # does — only the score history is held back.)
        self.temperature, estimated = self._estimate_temperature()
        self.temperatures.append(self.temperature)
        self._n_estimated += int(estimated)

        # --- Contribution 2: probabilistic fusion ------------------------------
        probs = finalize_probs(base_logits, cache_logits, config, self.temperature)

        # Only collect scores produced in the configured space. The empty-cache
        # fallback above scores in additive-logit space while fusion scores in
        # log-probability space; mixing the two would hand the temperature
        # search a bimodal history and skew its mean-confidence match.
        if not fusion_fallback:
            self._score_history.append(
                pre_temperature_logits(base_logits, cache_logits, config).detach()
            )

        # `output` is already a probability vector; the driver must not softmax
        # it again. Top-1 accuracy is unaffected — softmax is rank-preserving.
        return {'output': probs}
