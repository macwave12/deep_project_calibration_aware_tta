# The cache that echoes CLIP: agreement-gated TDA — deck v6

**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`
alongside `calibration_aware_tta.pptx`; edit the slide functions in that script instead.

Each section is one slide: its on-screen heading and the speaker notes from its notes pane.

---

## 1. Title

Hello everyone. This talk is about test-time adaptation of CLIP, and in particular about TDA, a popular training-free method that improves CLIP's accuracy but makes its confidence less trustworthy. We show where that comes from, prove the key step, and propose a fix that adds no parameters: on our five design test sets it keeps TDA's accuracy and lowers its ECE from 10.96 to 5.51.

---

## 2. The task: zero-shot CLIP on an unlabeled test stream

The task. We are given the class names, a pretrained CLIP, and a stream of unlabeled test images. CLIP has an image encoder and a text encoder, both frozen; the logit of a class is the cosine between the image feature and the feature of a prompt with the class name. Its argmax is the prediction, which we call CLIP's pseudo-label, and its top probability is the confidence. There are no labels anywhere, each prediction is final before the next image arrives, and we want both the right class and a believable confidence. On textures, flowers, pet breeds, satellite images and aircraft models, our five design datasets, zero-shot CLIP reaches only 53.12 percent on average. Test-time adaptation tries to fix that.

---

## 3. Two ways to adapt CLIP at test time

Test-time adaptation uses the unlabeled stream to improve CLIP, and there are two families. This is the first figure of the TDA paper. On top, test-time prompt tuning: for every image it builds many augmented views and optimizes the prompt by backpropagation, then resets, which is slow. Below, TDA: no training, only a cache of earlier test images labeled by CLIP, used for the next images. It is fast and fully online, which is why it became a widely used baseline of the training-free family and why we study it.

---

## 4. TDA's caches are labeled by CLIP's own predictions

This is TDA's pipeline, from its paper. CLIP produces a prediction. Two caches store test features as keys and pseudo-labels as values: a positive cache of confident samples, and a negative cache of uncertain samples that records which classes are unlikely. Both caches produce predictions that are added to CLIP's. The detail that matters for us is the update path at the bottom: the cache labels are CLIP's own predictions.

---

## 5. TDA in one equation

Here is TDA in one line. The output is b, which is CLIP's logits after the negative cache's penalty, plus alpha times the positive cache's votes. Each cached image votes with a kernel of its cosine to the test image, so each vote is at most one, and with 3 images per class a class gets at most 3. Two facts matter later: the cache is filled with CLIP's argmax, and each image is written before it is read. We use TDA's own single setting, alpha 2 and beta 5, on every dataset, since an online method should not need per-dataset tuning.

---

## 6. Calibration: confidence should equal accuracy

Calibration means that the confidence a model reports matches how often it is right. The expected calibration error bins the predictions by confidence and averages, over the bins, the gap between mean confidence and accuracy, weighted by the number of samples in each bin. The sketch on the right is a concept drawing, not data: below the diagonal is overconfident, above it underconfident. ECE has no sign, so we also report the signed gap, confidence minus accuracy. When the confidence is used to decide whether to trust a prediction, overconfidence is the worse error: the model acts on wrong predictions it trusts, while an underconfident one only gives up some predictions it could have kept.

---

## 7. TDA's catch: more accuracy, worse calibration

Per dataset, on the test splits with 10 stream orders. TDA gains accuracy everywhere, but its ECE grows on three of the five datasets, sharply on DTD, from 8.24 to 17.51. Pets and EuroSAT are the exceptions; on Pets CLIP is underconfident, and TDA's extra confidence happens to correct it, so keep Pets in mind. Which of the two caches is responsible? The negative cache alone lowers ECE; adding the positive cache takes it to 10.9. So the problem sits in the positive cache. Why?

---

## 8. Symptom: each new cache entry votes for itself

The first symptom. Because each image is written before it is read, an image that enters the cache votes for its own pseudo-label with weight exactly one. That vote adds nothing to CLIP's own prediction. It happens for 3 to 41 percent of the samples on the validation splits, and self-voters agree with CLIP far more often than the others, left panel. Reading the cache before writing removes the vote and lowers TDA's ECE, right panel, but not to CLIP's level. So the self-vote matters, but it is not the whole story.

---

## 9. The echo: where the cache agrees, it only adds confidence

Now the general mechanism, with a short proof. Let b be CLIP after the negative cache. If the positive cache's top class is the same as b's, every margin of the output is b's margin plus alpha times a non-negative cache margin: the decision cannot change, and the top probability cannot drop. The data match. On the validation splits 58 percent of the samples agree; there the positive cache leaves accuracy unchanged but turns a slight underconfidence into strong overconfidence. Its whole accuracy gain comes from the samples where it disagrees. Since the cache is filled with CLIP's own labels, agreement is the common case: the cache echoes CLIP.

---

## 10. The agreement gate: use the cache only where it disagrees

The method follows directly. We keep everything in TDA and switch the positive cache off on samples where its top class agrees with CLIP's pseudo-label, because there it can only inflate confidence. Where it disagrees, the cache is used exactly as in TDA. Why CLIP's label and not b's top class, as in the lemma? Because CLIP's label is what the cache is filled with, so agreement with it is the echo; the two differ on only 6.6 percent of agreeing samples. The rule adds no parameter and needs no label, and it was frozen on the validation splits before any test or held-out result.

---

## 11. Main result: TDA's accuracy at CLIP's calibration

The main result on the five design test sets, 10 stream orders. The gate costs a small but consistent 0.09 accuracy points, small next to TDA's gain over CLIP, and it lowers the ECE on every order, to 5.51, just below CLIP. Per dataset it is far better calibrated than TDA on DTD, Flowers, EuroSAT and Aircraft, and worse on Pets, where CLIP is underconfident.

---

## 12. Lower mean ECE than post-hoc calibration, and the two combine

Here we compare with the closest alternatives, paired to TDA on the same stream orders. A single temperature, shared by the five datasets and fitted with labels on their validation splits, reaches an ECE of 6.98; ZAEC, a recent label-free post-hoc method that restores the zero-shot entropy, reaches 5.95. The gate reaches 5.51 without any calibration step, because it removes the cause instead of rescaling afterwards, and the two combine. DMN-weak and ECALP, run like TDA with one view per image, have higher ECE than TDA.

---

## 13. Held-out: same accuracy; lower ECE on 6 of 8 far sets

Now data that played no part in the design, in two tiers defined by what the images are. Near sets are ordinary photos like CLIP's training data: CLIP is already accurate and calibrated, and neither method changes much. Far sets are medical scans, digits, rendered text, sketches, facial expressions and traffic signs: TDA adds accuracy there, and ECE. The gate keeps TDA's accuracy on every held-out set, and on far its mean ECE, 9.74, ends below CLIP's 11.38. The gain is uneven: large on PCam, ImageNet-Sketch and MNIST, small on the others, and TDA stays better calibrated on ImageNet-R and GTSRB.

---

## 14. Ablation: agreement picks the right samples

The ablation tests the diagnosis on the design test sets and both held-out tiers; the far column leaves out ImageNet-Sketch, of which we logged only the first samples, so it differs a little from the previous slide. Adding TDA's positive cache only on the agreeing samples barely moves accuracy but doubles the ECE on the design sets: only confidence. Is the gate just using less cache? No. Adding the cache on as many random samples is worse in both accuracy and ECE, 7.2 against the gate's 5.5 on the design sets, and turning down TDA's alpha until it matches the gate's accuracy leaves its ECE at 10.4. Which samples matters.

---

## 15. When does it help? Where CLIP is not underconfident

When does the gate help? On agreeing samples the echo only adds confidence, so removing it should help where CLIP is calibrated or overconfident, and hurt where CLIP is underconfident, where the added confidence happens to correct it. That is the tendency we see: the more overconfident CLIP is, the more the gate lowers ECE, with a rank correlation of −0.76. That gap needs labels, so it explains the result but cannot drive a switch. The signed gap also shows the direction of the errors: TDA pushes toward overconfidence almost everywhere, and on the sets where TDA wins in ECE, Pets, Caltech101, Food101, ImageNet-R and GTSRB, the gate is underconfident or calibrated in the mean, the safer side.

---

## 16. No label-free score tells TDA's right flips from wrong ones

The gate reads one bit, agreement. Could a finer use of the scores do better? TDA's accuracy gain comes from flips, samples where its decision differs from CLIP's and one of the two is right. An oracle that reverted exactly the wrong flips would add up to 2.3 points, so there is room. We test one score of each kind on the held-out flips: CLIP's confidence, the cache's confidence, and their similarity. Their correlation with a flip being right is at most 0.18 in size, and for the cache's confidence it even changes sign between near and far. In every decile about half or more of the flips are right, so no range is worth reverting. Seven more scores and a regression on all ten do no better.

---

## 17. TDA's cache echoes CLIP; gating it on disagreement removes the echo

To close. TDA's positive cache mostly repeats CLIP's own decision, and where it does, it cannot change the prediction, only inflate the confidence. Gating the cache on disagreement removes that echo: it keeps TDA's accuracy and brings the calibration back to CLIP's level on the design sets, and on far data it removes much of TDA's calibration cost. Beyond that one bit, the scores of CLIP and the cache do not say which corrections are right. The gate is worse calibrated than TDA where CLIP is underconfident, though on the safer side, and telling such data apart without labels is the open problem.

---

## 18. Backup: comparison per dataset

Per-dataset ECE for every method in the comparison, three stream orders each. The gate is best on EuroSAT; the temperature fitted on the validation splits wins on DTD and Aircraft, TDA plus ZAEC on Flowers, and plain TDA on Pets, where removing the echo costs calibration. Combining the gate with ZAEC is the strongest on the mean.

---

## 19. Backup: reproduction and TDA's per-dataset setting

For completeness, the per-dataset alpha and beta that TDA's code release ships, against the single setting from TDA's paper that we use throughout. The per-dataset values do not do better, and the conclusions do not change. Our GPU reproduction of the benchmark is within 0.16 points per dataset; the analyses re-evaluate logged runs of an equivalent implementation, which agree with this direct replay within 0.05 points on these means.

---

## 20. Backup: all ten label-free scores

The same view for all ten label-free scores: three of CLIP, three of the cache and four of their relation. None separates right from wrong flips on the held-out tiers. The best threshold rule per score, chosen on the validation flips, gains at most 0.03 points on a held-out tier, and all ten combined in a logistic regression change accuracy by −0.57 points on the far tier.
