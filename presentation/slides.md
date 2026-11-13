# The cache that echoes CLIP: agreement-gated TDA — deck v5

**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`
alongside `calibration_aware_tta.pptx`; edit the slide functions in that script instead.

Each section is one slide: its on-screen heading and the speaker notes from its notes pane.

---

## 1. Title

Hello everyone. This talk is about test-time adaptation of CLIP, and in particular about TDA, a popular training-free method that improves CLIP's accuracy but makes its confidence less trustworthy. We first diagnose exactly where that comes from, prove it, and then show a fix that needs no parameters and keeps TDA's accuracy while halving its calibration error.

---

## 2. Roadmap

Five parts. First the task: classifying images with CLIP when we only know the class names and the test images arrive one by one without labels. Second, test-time adaptation, the family of methods that uses that stream to adapt, and TDA, the method we study, in some detail. Third, calibration, and why TDA makes CLIP overconfident. Fourth, our diagnosis of where that comes from, with a short proof, and the fix it suggests. Fifth, the results, including where the fix does not help.

---

## 3. The task: classify a stream with nothing but the class names

Let me state the task precisely. We are given the class names and a pretrained model, CLIP, and a stream of test images without labels. There is no labeled data at all, not even for tuning. Each image gets a prediction before the next one arrives. And we want two things: the right class, and a confidence we can believe. On the right is how we evaluate: five core datasets whose validation splits we used for design and whose test splits we use for results, plus further datasets for generalization.

---

## 4. Zero-shot CLIP: compare the image with a prompt per class

This is the model everything builds on. CLIP has an image encoder and a text encoder, both frozen. Each class name is put into a prompt and encoded; the logit of a class is the cosine between the image feature and that text feature, times a learned scale. The softmax gives a distribution, its argmax is the prediction, which we will call CLIP's pseudo-label, and its probability is the confidence. Zero-shot CLIP needs no labels, but it is weak when the images look unlike web photos or the classes are very fine-grained. That is what test-time adaptation tries to fix.

---

## 5. Two ways to adapt CLIP at test time

Test-time adaptation uses the unlabeled stream to improve CLIP. There are two families. This is the first figure of the TDA paper. On top, test-time prompt tuning: for every image it builds many augmented views and optimizes the prompt by backpropagation, then resets, which is slow. Below, TDA: no training, only a cache of earlier test images labelled by CLIP, used for the next images. It is fast and fully online, which is why it became the standard baseline of the training-free family and why we study it.

---

## 6. TDA's two caches

This is TDA's pipeline, from its paper. CLIP produces a prediction. Two caches store test features as keys and pseudo-labels as values: a positive cache of confident samples, and a negative cache of uncertain samples that records which classes are unlikely. Both caches produce predictions that are added to CLIP's. The important detail for us is the update path at the bottom: the cache values are CLIP's own predictions.

---

## 7. TDA in one equation

Here is TDA in one line. The output logits are CLIP's logits, minus the negative cache term, plus alpha times the positive cache's votes. Each positive cache item votes with a kernel of its cosine to the test image, so each vote is at most one, and with 3 items per class a class gets at most 3. Two facts matter later: the cache is filled with CLIP's argmax, and each image is written before it is read. We use alpha 2 and beta 5 on every dataset.

---

## 8. Calibration: confidence should equal accuracy

Calibration means the confidence a model reports matches how often it is right. The expected calibration error bins the predictions by confidence and averages, over the bins, the gap between mean confidence and accuracy, weighted by the number of samples in each bin. The sketch on the right is a concept drawing, not data: points below the diagonal are overconfident. Throughout the talk we also use the signed gap, confidence minus accuracy, on parts of the stream.

---

## 9. TDA's catch: more accuracy, worse calibration

Per dataset, on the test splits with 10 stream orders. TDA gains accuracy everywhere, but the calibration error grows on most datasets; on DTD it goes from 8.24 to 17.51. Pets is the exception: CLIP is underconfident on Pets and TDA's extra confidence happens to correct it. Keep Pets in mind, it will be the one dataset where our method is worse.

---

## 10. Which cache does what (TDA paper's Fig. 3, with ECE)

The TDA paper ablates its two caches for accuracy. We repeat that with calibration, on the val splits. CLIP alone has 52.29 percent accuracy and ECE 6.17. The negative cache alone improves both. The positive cache brings accuracy, 55.60 percent, but its ECE is 11.56. Full TDA ends at 56.29 and ECE 11.19. The overconfidence comes from the positive cache.

---

## 11. Symptom: the self-vote

The first symptom. Because each image is written before it is read, an admitted image votes for its own pseudo-label with weight exactly one. This carries no information about the true label. It happens for 3 to 41 percent of the stream on the val splits, and when it happens the cache almost always agrees with CLIP, left panel. Removing just that term lowers TDA's ECE from 11.19 to 7.52, right panel. So the self-vote matters, but it is not the whole story.

---

## 12. Where the cache agrees with CLIP, it can only add confidence

Now the formal argument. Let b be CLIP plus the negative cache. If the positive cache's top class is the same as b's top class, then every margin of the output is the margin of b plus alpha times a non-negative cache margin. So the decision cannot change, and the top probability can only increase. The gate checks agreement with CLIP's own top class, which differs from the lemma's condition only where the negative cache moved that top class, a small share of samples. So where the cache agrees it mostly adds confidence, and the accuracy gain comes from the samples where it disagrees. Since the cache is filled with CLIP's own labels, agreement is the common case: the cache echoes CLIP.

---

## 13. Ablation: the agreeing positive cache adds only confidence

The ablation that tests the diagnosis on every tier. We add TDA's positive cache to CLIP plus the negative cache only on the samples where it agrees with CLIP, only where it disagrees, which is the gate, or everywhere, which is TDA. On the agreeing samples it changes accuracy by at most 0.09 points, on every tier, but the expected calibration error roughly doubles on val, test and the development sets, from 4.6 to 10.8 on val. So on the agreeing group the positive cache carries no information, only confidence, exactly as the lemma says. The whole accuracy gain comes from the disagreeing samples, and the gate keeps it. On the fresh tier the agreeing echo even helps calibration, because CLIP is underconfident there; that is the limitation we come back to.

---

## 14. The agreement gate: use the cache only where it disagrees

The method follows directly. We keep everything in TDA, and only switch the positive cache off on samples where its top class agrees with CLIP's pseudo-label. There it could only inflate confidence. Where it disagrees, the cache is used exactly as in TDA. The rule has no parameter and needs no label; it costs one comparison per image. It was fixed on the val splits before we looked at any test, development or fresh results.

---

## 15. Main result: TDA's accuracy, half its ECE

The main result on the five test sets, 10 stream orders. The gate keeps TDA's accuracy, a difference of −0.09 points, and lowers ECE by 5.45 points, on every seed. Its mean ECE, 5.51, is below CLIP's 5.65. It is far better than TDA on DTD, Flowers, EuroSAT and Aircraft, and worse on Pets, where, as we saw, CLIP is underconfident and the echo happened to help.

---

## 16. Against calibration fixes and other memory methods

Here we compare with the closest alternatives, three seeds, all paired to TDA. Removing only the self-vote, by reading before writing, reaches 8.77. A single temperature fitted on val with labels reaches 6.98. ZAEC, a recent label-free post-hoc method that restores the zero-shot entropy, reaches 5.95. The gate reaches 5.51 without any calibration step, because it removes the cause instead of rescaling afterwards. Gate plus ZAEC reaches 5.27 here, but that stacking does not hold up on average on the development and fresh sets, so we do not claim it. DMN-weak and ECALP have higher ECE than TDA; ECALP is the most accurate method here, at 58.34 percent, but has an ECE of 32.32.

---

## 17. Generalization: datasets never used in the design

The method was designed on the val splits of five datasets. Here are 14 datasets that played no role in the design: four development sets and the fresh sets, including the full ImageNet validation set. The gate lowers ECE on 7 of them; the mean ECE change is −1.72 points and the mean accuracy change −0.01 points. Accuracy is kept everywhere; ECE is mixed, and the next slide shows when it helps.

---

## 18. Where it helps: mostly where CLIP is not underconfident

When does the gate help? The lemmas say that on agreeing samples the echo only adds confidence. So removing it should help when CLIP is calibrated or overconfident, and hurt when CLIP is underconfident. That is the tendency we see: over 19 datasets, the gate's ECE change follows CLIP's own confidence minus accuracy gap with a rank correlation of −0.89. That variable needs labels, so it explains the result but cannot be used to switch the method. The best label-free proxy we tried, CLIP's mean confidence, reaches only +0.57, because a dataset can be unconfident and still underconfident.

---

## 19. Beyond one bit: CLIP's and the cache's scores say almost nothing

Could a finer use of the scores beat the one-bit gate? TDA's accuracy gain comes from flips, samples where its decision differs from CLIP's. We took ten label-free quantities: CLIP's confidence, margin and the probability it gives the cache's class; the cache's confidence, mass and weight on CLIP's class; and their relation, the correlation over classes, the JS divergence, the weight gap and the alpha needed to flip. On the development and fresh sets every one of them separates right from wrong flips with an AUC between 0.43 and 0.58, close to chance. All ten combined in a logistic regression fitted on val reach 0.75 on test, which has the same datasets as val, but 0.47 and 0.48 on new datasets: they learn which datasets flip well, not which flips are right. Used as a filter, that changes accuracy by at most 0.05 points, while an oracle picking the right one of CLIP's and the cache's class would add 3.2 to 6.2. The headroom exists, but this information is not in the scores.

---

## 20. Limitations and conclusion

To close. The gate is worse on datasets where CLIP is underconfident, because there the echo happened to correct CLIP. A label-free confidence switch recovers the confident ones, but we found it after the fresh results, so it stays an exploratory extension in the backup. The remaining overconfidence is in the disagreeing branch. We tested one backbone and one cache method, and the design was done on val after earlier exploration on test, which is why the development and fresh sets matter. We also tried finer uses of agreement, an alpha graded by disagreement or by the correlation between cache and CLIP, and flip rules; none held up on held-out data. The main message: TDA is overconfident largely because its cache echoes CLIP, and gating it on disagreement removes the echo at unchanged accuracy.

---

## 21. Backup: the echo measured on val, by agreement group

The lemma says what can happen; the data say how much. On the val splits the cache agrees with CLIP on 58 percent of the samples. There, the positive cache leaves accuracy essentially unchanged, but the confidence minus accuracy gap jumps from −2.3 to +12.9 points. Where it disagrees, it carries the accuracy gain with a much smaller change in the gap. That points directly to the method: keep the disagreeing branch, drop the agreeing one.

---

## 22. Backup: reliability, the gate stays near the diagonal

Reliability diagrams on the five test sets, all seeds pooled, ten bins. Points below the diagonal are overconfident. TDA, in red, sits below the diagonal in the confident bins on most datasets. The gate, in green, follows CLIP's calibration where the cache agreed with CLIP and keeps TDA's corrected decisions where it disagreed. Pets is the one panel where TDA's curve is closer to the diagonal.

---

## 23. Backup: protocol P1 (published per-dataset α, β)

For completeness, the published protocol P1 with TDA's per-dataset alpha and beta. The conclusions do not change: the gate keeps TDA's accuracy and lowers its ECE under both protocols. Our GPU reproduction of the benchmark is within 0.16 points per dataset, and every number in this talk comes from an exact CPU replay of cached features.

---

## 24. Backup: proofs

The two proofs in full. For the no-flip region, the negative cache can move a margin by at most its range, and the positive cache by at most alpha times the shot capacity, so a larger CLIP gap cannot be overturned. For the agreement lemma, each output margin is at least the corresponding margin of CLIP plus negative cache, and the top softmax probability is an increasing function of the margins.

---

## 25. Backup: earlier attempts

These are the changes we tried before the diagnosis, all on the same protocol and paired to TDA. Margin admission starves the cache on hard datasets. Probability fusion caps every prediction's confidence. Gating fusion to two-way ties fixed calibration but lost accuracy. A leave-one-out temperature helped calibration partly. Some of this exploration was done on test, which we disclose.

---

## 26. Backup: comparison per dataset (ECE, test, 3 seeds)

Per dataset ECE for every method in the comparison table, three seeds each. The gate is best on EuroSAT; the val-fitted temperature wins on DTD and Aircraft, ZAEC on Flowers, and plain TDA on Pets, where removing the echo costs calibration. Combining the gate with ZAEC is the strongest on the mean.

---

## 27. Backup (exploratory): keep the echo on confident streams

An exploratory extension, not the method. The gate loses where CLIP is underconfident, and on the design sets CLIP is underconfident exactly on the confident ones. Keeping TDA's echo while the running mean CLIP confidence stays above 0.77 lowers mean ECE on the held-out test and fresh tiers with the same accuracy. The threshold was chosen on val plus the development sets, which contain both kinds of stream; any value in the flat region gives the same design result, and the fresh sets alone would have preferred a lower one, 0.61 to 0.62, so it is not tuned to them. Still, the gain comes from a handful of datasets, CIFAR-100 and ImageNet-R are missed, and we found the idea after seeing the fresh results, so it needs new datasets before it can be claimed.

---

## 28. Backup: share of right flips along each label-free score

The same question as a rate. For each of the ten label-free scores, the flips are split into eight equal groups, and we plot the share of them that are right, one line per tier. Where a line slopes on val and test, it is flat or reverses on the fresh tier. The one score whose direction holds everywhere is the alpha needed to flip: harder flips are right less often. But even the worst group is right 49.8 percent of the time, so blocking flips by that score cannot raise accuracy.

---

## 29. Backup: finer uses of agreement that did not hold up

We also tried to get more out of the agreement signal than a binary gate. Making alpha depend on how strongly the cache disagrees with CLIP, or on the correlation between the cache and CLIP logits, including a bump that is strongest at intermediate correlation, gave gains on val that reversed on the test and fresh sets. Softening the disagreeing branch helped on test only. The agreement rate is a worse accuracy estimate than CLIP's own confidence, because the cache is filled with CLIP's own pseudo-labels. And while an oracle that picks the right one of CLIP and the cache per sample would add several points of accuracy, no label-free flip rule captured any of it. So the binary gate is what agreement gives us.
