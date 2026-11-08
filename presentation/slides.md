# Calibration-Aware Test-Time Adaptation for CLIP — draft deck v2

**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`
alongside `calibration_aware_tta.pptx`; edit the slide functions in that script instead.

Each section is one slide: its on-screen heading and the speaker notes from its notes pane.

---

## 1. Title

Hello everyone. This project is about test-time adaptation for CLIP, and specifically about whether the confidence such methods report can be trusted.

The setting is strict: we only know the class names, the test images arrive one at a time without labels, and nothing is retrained. Within that setting, a well-known method called TDA improves accuracy but makes the model overconfident. We study why, propose four small changes, and show which ones work.

This is a draft for structure review. Wherever you see a yellow dashed box, that is a figure or table that will be computed once the image features are extracted.

---

## 2. Roadmap

Six parts. First the background: what exactly the task is, what CLIP does, and where TDA sits among other methods. Then a long section on TDA itself, with the math, because everything we do changes one specific line of it. Then calibration: what we measure and why TDA gets it wrong. Then our four changes, one at a time, each with its motivation, why we expected it to work, the evidence and the outcome. Then results, fairness and baselines, and finally the limitations.

---

## 3. The problem: classify with nothing but the class names

Let me state the task precisely, because it rules out a lot of familiar tools.

We are given the class names, for example the 37 pet breeds, and a stream of test images without labels. There is no labeled set at all, not even for validation. We may not train or backpropagate; each image gets one forward pass, and its prediction is final before the next image arrives.

And we want two things: the right class, and a confidence we can believe.

On the right is our evaluation setup: five datasets, 19,257 test images in total, one image at a time, and calibration measured with 20 bins. It is the same setting as the benchmark paper whose code we forked.

---

## 4. Zero-shot CLIP: two frozen encoders, one prompt

This is the model everything builds on. CLIP has an image encoder and a text encoder. Both are frozen throughout; nothing in this talk changes a single weight of either.

To classify, each class name is put into one prompt, "a photo of a" class, and encoded. We use one template only, the same for every dataset. The logit for class k is the cosine between the image feature and that text feature, times CLIP's logit scale, which is 100. The softmax of the logits is the predicted distribution, the argmax is the prediction, and its probability is what we call the confidence.

Every method later in the talk works on these features and logits; none retrains anything.

---

## 5. Why zero-shot CLIP fails on these datasets

Why is CLIP alone not enough? There are two different reasons, and they matter later.

The first is domain shift. CLIP was trained on web photos; satellite tiles in EuroSAT and texture crops in DTD look nothing like them. CLIP gets 42.06 percent on EuroSAT, which has only 10 classes.

The second is granularity. Web captions rarely name the exact aircraft variant or flower species, so the text features for neighbouring fine-grained classes are close, and CLIP is often torn between them. Aircraft, with 100 variants, is at 23.85 percent. Pets is the counter-example: breed names are common online, and CLIP reaches 88.20.

Keep the second reason in mind: "torn between two neighbouring classes" is exactly the situation our gate later targets.

---

## 6. The method landscape: where TDA sits

Here is the landscape, so you can see why we picked TDA.

Tip-Adapter is the origin of the cache idea, but it needs a few labeled images per class, so it is a different setting; we will use it later only as a reference for what labels buy.

The prompt-tuning family, TPT and its calibration-aware descendants C-TPT and O-TPT, is episodic: for every single image it builds many augmented views, takes gradient steps on the prompt, predicts, and resets. That needs backprop.

The online family, TDA, DMN, BoostAdapter and ECALP, never backpropagates and keeps a memory across the stream. None of them is designed for calibration. UL-TTA is a recent online method that does adjust a temperature without labels; it is the closest prior art to our fourth change and we cite it.

So the gap is: calibration-aware methods exist, but only in the expensive episodic family.

---

## 7. Why we build on TDA

Why TDA, and not one of the others?

First, it fits the setting exactly: online, no gradients, one view per image. That also makes it cheap: since CLIP is frozen and there is a single view, image features can be computed once and replayed, which is how we will run all further experiments on a CPU.

Second, it is the standard baseline of the online family, with official code, and the benchmark we fork already measured both its accuracy and its calibration.

Third, its mechanism is a cache we can open and inspect, so every claim about why it becomes overconfident can be checked against what was actually stored.

We verified that our runs reproduce the published accuracies, so the baseline is sound.

---

## 8. TDA's idea: describe each class with test images, not only text

Before the math, the idea in one picture. In zero-shot CLIP, each class is represented by one point, its text feature, the star. Under domain shift the actual test images of the class form a cloud somewhere else, and one point is a poor description of that cloud.

TDA's idea is to remember a few test images per class, the circled dots, which by construction lie where this domain's images really are. A new image is then compared both with the text features and with these remembered images. It is a nearest-neighbour vote added on top of CLIP.

The catch, which we come back to after the full walkthrough, is that the remembered images are labeled by CLIP's own guess.

---

## 9. TDA, one image at a time

Here is the loop TDA runs for every test image. First CLIP encodes the image and gives logits, a distribution, a pseudo-label which is just CLIP's argmax, and an entropy.

Second, the write step: the image is offered to the positive cache under its pseudo-label, and, if its entropy is in a middle band, also to a negative cache.

Third, the read step: both caches vote, weighted by similarity. Fourth, the prediction adds the positive votes and subtracts the negative ones.

Three facts we verified in the code, because they are often stated wrongly. Every image is offered; the 3 most confident per class are kept. Writing happens before reading, so an image can vote for itself. And the cache labels always come from raw CLIP, never from TDA's output, so there is no feedback loop of TDA reinforcing its own decisions.

---

## 10. Write rule: what enters the positive cache

Now the write rule precisely. The pseudo-label is CLIP's argmax and the entropy is the usual Shannon entropy of CLIP's distribution. The cache for the predicted class keeps the 3 lowest-entropy items it has seen.

Two consequences matter. At the start of the stream, classes are empty, so almost anything gets in, including confident mistakes. Later, an image has to be more confident than the weakest item already stored for its class.

And because the write happens before the read, an admitted image finds itself in the cache with cosine one. Its vote is exactly one, and its own pseudo-label gets plus alpha. That is a self-vote; keep it in mind for the calibration section.

---

## 11. Read rule: cached images vote, weighted by similarity

The read rule. For each class, sum over its cached items an exponential of the cosine between the new image and the item. The exponent is minus beta times one minus the cosine, so an identical item gives exactly one, and less similar items give less.

With at most 3 items per class, a class's cache score is between zero and 3. Classes that have nothing cached get nothing.

The final TDA logit is CLIP's logit, plus alpha times this score, minus a small negative cache term I will show in two slides.

The plot shows the vote weight against cosine for each dataset's beta. A larger beta means only very similar cached images count.

---

## 12. What α and β mean, and where their values come from

What do alpha and beta mean? Beta is the sharpness: how local the nearest-neighbour vote is. A vote halves when the cosine drops to one minus log two over beta, the column in the middle. On EuroSAT, with beta 8.0, only images with cosine above 0.91 count much; on Aircraft, with beta 2.0, the vote is much broader.

Alpha is the weight of the cache against CLIP. The last column is the largest shift the positive cache can apply to a logit difference, alpha times the capacity. On EuroSAT that is 12.0 logit units, which is a lot.

Important for fairness: these values are per dataset and come from TDA's repository, with no stated procedure. The paper itself searched one setting, alpha 2, beta 5. We will report results under both choices.

---

## 13. Negative cache: penalize classes that confused similar images

TDA has a second cache. It stores images whose normalized entropy falls in a middle band, between 0.2 and 0.5: uncertain, but not hopeless. It stores the whole distribution of each.

At read time, if the new image resembles one of these uncertain images, every class that the stored image found plausible, with probability above 0.03, gets a penalty of 0.117 times the vote. It is a small damping term among confusable classes.

Two notes. The paper's equation has a double negative; we follow the code, which masks probabilities above the threshold and subtracts. And none of our changes touch this cache: it stays on CLIP's side in every variant.

---

## 14. Final logits are a normalized product of experts

This slide is the key to the whole talk. Adding alpha times the cache score to the logits and then taking a softmax is exactly the same as multiplying CLIP's distribution by the cache's distribution raised to the power alpha, and renormalizing. The two softmax normalizers simply cancel. That is called a normalized product of experts.

Products of experts have a well-known property: when the experts agree, the result is sharper than either. In this illustrative example, CLIP and the cache both give the top class 0.60. TDA, with alpha 2, reports 0.89. A simple average of the two would stay at 0.60.

And because the cache is filled with CLIP's own guesses, agreement is the normal case, also when CLIP is wrong. Remember the right-hand column; it is our second change.

---

## 15. Worked example: one Pets image through TDA (illustrative)

Let me put numbers on it, with an illustrative Pets image. Three candidate breeds. CLIP's cosines give logits about one point apart, and CLIP says Birman with 0.73.

The cache already holds two Birman images and one Ragdoll. Because the image is written before it is read, it is also in the Birman slot itself, with cosine one and vote one. With Pets' alpha 2.0 and beta 7.0, the Birman logit rises by alpha times its votes.

TDA's confidence becomes 0.99. Without the self-vote it would have been 0.90. In logit terms the self-vote is 54 percent of the Birman cache score: a large part of the boost is the image agreeing with itself. And nothing in this computation knows whether Birman is right. That is the mechanism of overconfidence, in miniature.

---

## 16. What TDA buys: accuracy over zero-shot CLIP

Before any criticism, what TDA achieves. On all five datasets TDA is more accurate than zero-shot CLIP, on average from 53.16 to 56.63 percent.

The gain is very uneven. On EuroSAT it is 11.11 points; there are only 10 classes, so each class's cache fills quickly, and the repository's alpha is the largest. On Aircraft it is only 0.42 points: CLIP is right only about a quarter of the time there, so most cached pseudo-labels are wrong.

These are protocol P1 numbers, with TDA's per-dataset hyperparameters, single seed.

---

## 17. Where TDA's gain comes from: overturning CLIP's top choice

Accuracy changes only when TDA flips CLIP's argmax. A flip from class one to class two happens exactly when CLIP's log-odds between them is smaller than alpha times the cache's preference for class two. Since each class's cache score is at most the capacity, there is a hard limit on how confident a CLIP decision can be overturned.

The table shows that limit. With alpha 2.0, the cache can overturn CLIP decisions up to odds of about 403 to one. On EuroSAT, with alpha 4.0, essentially any decision can be overturned.

What we do not know yet is where the flips actually happen. Evidence A2 will histogram flips by CLIP's top-two gap. Our hypothesis, which the gate relies on, is that the useful flips are concentrated on near-ties.

---

## 18. Confidence, calibration and ECE

Now calibration. The confidence of a prediction is the probability of the predicted class. A model is calibrated if, among all predictions it makes with, say, seventy percent confidence, seventy percent are correct.

Expected calibration error measures the violation. We sort predictions into 20 equal-width bins by confidence, take the absolute difference between accuracy and mean confidence in each bin, and average with bin sizes as weights. We use top-label ECE throughout.

We will also use a simpler number, the gap: mean confidence minus accuracy. By the triangle inequality ECE is at least the absolute gap. When the two are almost equal, every bin errs in the same direction, which means the miscalibration is a uniform shift. That observation will motivate our temperature change.

The diagram on the right is a synthetic reliability diagram: the diagonal is perfect calibration, a curve below it is overconfident, and ECE averages the vertical gaps.

---

## 19. Why calibration matters

Why should anyone care about calibration? Because the confidence is what downstream decisions use. A typical system accepts confident predictions automatically and sends the rest to a human. If the model is overconfident, wrong answers slip through; if it is underconfident, good answers are wasted on human review.

In online test-time adaptation there is a second reason: the confidence also steers the method. TDA's cache keeps the most confident images, so miscalibration changes what it remembers.

The benchmark we build on measured exactly this. For TDA with a ResNet-50 backbone, the average ECE over the fine-grained datasets goes from 5.70 for CLIP to 9.21, and over all datasets from 6.07 to 11.53. They report ECE only for ResNet-50, so our ViT-B/16 numbers, coming next, have no published counterpart.

---

## 20. CLIP vs TDA: accuracy, confidence and ECE per dataset

Here are our own numbers, which is evidence A1, computed from the per-bin data in the run records. For each dataset: accuracy, mean confidence and ECE for CLIP and for TDA, and TDA's gap.

TDA raises mean ECE from 5.57 to 11.24. On 4 of the five datasets TDA is overconfident, and look at the last two columns: the ECE is almost exactly the gap. On DTD, ECE 17.41, gap 17.22. So TDA shifts confidence up nearly uniformly.

Pets is the exception. There CLIP itself is underconfident, and TDA's upward shift happens to land close to calibration. Keep Pets in mind; it behaves differently in every change.

---

## 21. Reliability diagrams: TDA moves the curve below the diagonal

The same story as reliability diagrams, one per dataset. The dashed diagonal is perfect calibration. Blue is CLIP, orange is TDA. Points below the diagonal mean the model claims more confidence than its accuracy supports.

On DTD, EuroSAT and Aircraft, TDA's curve is below CLIP's across the whole range: it pushes confidence up everywhere, and accuracy does not keep pace. On Pets, CLIP's curve is above the diagonal, and TDA's upward push brings it onto the line.

So TDA does not make a few wild mistakes; it inflates confidence across the board. The next slide explains where that inflation comes from.

---

## 22. Anatomy of TDA's overconfidence

Now, after the full walkthrough, the weaknesses. Three mechanisms inflate TDA's confidence.

First, the self-vote we saw: an admitted image votes for its own pseudo-label before it is classified.

Second, the product of experts: the cache is made of CLIP's own guesses, so it tends to agree with CLIP, and agreement multiplies.

Third, a confidently wrong image stays in the cache under the wrong label and keeps voting for it, until it is displaced by an even more confident image of the same predicted class.

None of these depends on whether the prediction is correct. What we already see in the data is that TDA's ECE equals its gap, for example DTD 17.4 vs 17.2, EuroSAT 14.4 vs 13.1, Aircraft 15.5 vs 15.5. What we will add, boxes A3 and A4, is the direct evidence: how much of the lift is self-votes, how pure the cache is, and a concrete poisoned item with the images it pulled.

---

## 23. Our proposal: four training-free changes, one link each

Our proposal: four small changes, each aimed at one link of the diagnosis. Margin admission changes what enters the cache. Probabilistic fusion changes how the cache and CLIP are combined: a mixture instead of a product. Gated fusion changes when the cache is consulted at all. And the leave-one-out temperature rescales the final confidence.

The right column gives away the ending: the first is a negative result, the second fails on its own but leads to the third, which is our recommendation, and the fourth helps alone but does not combine with the third.

Every change is measured on its own against the same baseline, which we call "none": TDA run through our code with every change switched off, in the same sweep. Our knobs were fixed before running, with one exception about the gate that I will disclose.

---

## 24. Change 1 · Margin admission: a negative result

The first change, as one compact slide, because it is a negative result.

The motive: TDA lets every image compete for the cache, including confident mistakes. The idea: require a top-two logit margin above 1.0, one global threshold fixed in advance, so that ambiguous images stay out.

It is worse on every dataset, on both accuracy and ECE, on average ECE 11.28 to 18.06. Two definitions for the table: "admitted" is the share of the stream that passed the gate and competed for a cache slot, which is every image for TDA; "none" is TDA through our code.

Why it fails: the admission rate follows CLIP's accuracy, 15.9 percent on Aircraft, 81.9 on Pets. When few images get in, they come from the classes CLIP already finds easy, and the cache then pulls every prediction toward those classes: on Aircraft the gap goes from 15.6 to 32.6 points. And fundamentally, a confidently wrong image has a large margin too: the margin measures loudness, not correctness. Evidence A5 will show cache purity and coverage directly. We dropped this change.

---

## 25. Change 2 · Probabilistic fusion: motive and why it should work

The second change attacks the product of experts. The table contrasts the two rules. TDA multiplies the distributions; when they agree the result is sharper than either, and there is no upper bound. We instead average them, with weight 0.5. A mixture of two distributions can never be more confident than the more confident of the two.

In the illustrative example with agreeing experts, the product gives 0.89 and the mixture stays at 0.60. When they disagree, the product lets the more peaked one win, while the mixture produces a compromise.

You might ask why not keep a normalized product but with a smaller alpha. That is simply TDA with a different alpha; it still sharpens on agreement. The mixture is a different kind of combination, and that is the point.

---

## 26. Change 2 · Result: fusion makes every dataset underconfident

The result is bad, and instructive. Accuracy barely moves, 56.60 to 55.60, but every dataset becomes underconfident: the gap column goes negative everywhere, by up to 43.8 points on Pets. Mean ECE more than doubles.

The reason is that softmax is not scale-invariant. CLIP's logits are a hundred times a cosine, so its softmax is peaked. The cache logits are sums of at most 3 votes, each at most one, so their softmax over many classes is almost flat. In the illustrative Aircraft case, three votes at cosine 0.80 give a cache logit of 2.01, which over a hundred classes is a top probability of only 0.070. Averaging that with a confident CLIP prediction of 0.90 gives about 0.485, whether CLIP was right or wrong. The mixture is bounded, as promised, but it is dragged down by a flat expert.

Evidence A6 will measure the scale gap directly, and A7 the dilution on images CLIP was already confident about.

---

## 27. Change 3 · Gated fusion: consult the cache only on a two-way tie

The diagnosis of the fusion failure suggests the repair. Fusion only helps where the cache can change the answer. Everywhere else it only dilutes. So we fuse only on a two-way tie, and otherwise output CLIP's own distribution, still with TDA's negative-cache correction.

A tie means the top two CLIP probabilities are within 0.2 of each other, and together hold more than 0.6 of the mass. The second clause excludes images spread thinly over many classes: there CLIP is confused, not torn, and the cache vote is noise. Both thresholds are in probability space, so they mean the same on ten and a hundred classes, and they were fixed in advance.

Why is "clear winner" safe to leave alone? Under the mixture, the cache flips the answer only if its own preference for the runner-up exceeds CLIP's gap. Cache logits are between zero and the capacity, which bounds that preference, as in the formula. On the hundred-class datasets the bound is 0.16, so a CLIP gap of 0.2 or more can never be overturned there; leaving those images to CLIP loses no flip the mixture could have made. On EuroSAT, with ten classes, the bound is 0.66: the gate does give up flips there, and we will see it in the accuracy.

Note this is the same quantity as margin admission, used the opposite way: to decide when the cache speaks, not what it remembers.

---

## 28. Change 3 · Result: gated fusion

With the gate, the blow-up is gone. Mean ECE drops from 11.28 for the baseline to 4.61, which is below CLIP's own mean of 5.57. But I want to be precise: it beats CLIP's ECE on 3 of 5 datasets, not everywhere; on Flowers102 and Pets it is worse than CLIP.

The confidence columns show what the gate does: its mean confidence tracks CLIP's on every dataset. In other words it removes TDA's inflation and returns roughly to CLIP's own calibration, while keeping part of TDA's accuracy.

The cost is accuracy, mostly on EuroSAT, minus 5.92 points, which is what the flip bound on the previous slide predicted: on ten classes the cache could usefully overturn larger gaps, and the gate forbids that.

And a disclosure: during exploration we compared four fusion variants on the test sets and picked this one. That is selection on test. The backup slide lists all four, and the clean protocol and validation-split selection will check whether the choice holds.

---

## 29. Change 4 · LOO temperature: motive and how it works

The fourth change. The calibration section showed that TDA's error is close to a uniform upward shift: ECE about equal to the gap. That is the textbook case for temperature scaling, dividing all logits by one number. Normally the temperature is fitted on labeled validation data, which we do not have.

So we estimate accuracy without labels: for each cached item, find its nearest other cached item and check whether their pseudo-labels agree. The fraction that agree is our estimate. Then we pick the temperature whose mean confidence over the history matches it, from a grid that starts at 1.0, so it can only soften, never sharpen. Since it is a monotone rescaling, accuracy cannot change.

Cost: as implemented, every step re-scores the full history for all 381 grid values, which is quadratic in the stream length. That is an implementation artifact; task T03 makes it linear with identical temperatures, and we will time both.

UL-TTA, a recent paper, also sets a temperature without labels; we cite it as prior art.

---

## 30. Change 4 · Result: helps where TDA is overconfident, hurts where not

Results for the temperature alone. Accuracy is identical on every dataset, as it must be. Mean ECE improves from 11.28 to 8.54, with large wins on DTD and Aircraft, where TDA was most overconfident.

But the estimate is wrong in both directions. On EuroSAT it is too high: the search wants to sharpen, the floor holds it at one, and the temperature sits on the grid edge on 87.5 percent of steps, so EuroSAT stays overconfident. On Pets it is too low, and the search softens a model that was already nearly calibrated. Our hypothesis is that neighbouring breeds look alike, so a correctly labeled cached Birman's nearest neighbour is often a correctly labeled Ragdoll and the agreement understates accuracy. Evidence A9 will compare the estimate with the true running accuracy and test this.

Should we keep it? Given the cost and that it does not combine with the gate, we keep it only as an optional add-on for plain TDA, not in our recommended method.

---

## 31. Headline results (protocol P1, seed 0)

The headline table, protocol P1, seed zero. Top: accuracy; bottom: ECE. The highlighted row is our recommendation, gated fusion.

On average, CLIP reaches 53.16 percent at 5.57 ECE, TDA 56.63 at 11.24. Gated fusion reaches 55.01 at 4.61. So it keeps 1.85 of TDA's 3.47 points of accuracy gain and has the lowest mean ECE of all five, although, as said, not the lowest on every dataset.

The temperature row shows unchanged accuracy and better ECE than TDA except on Pets, and the combination is not better than the gate alone. That is the ablation, two slides on.

---

## 32. Is it fair? Two hyperparameter protocols

Is the comparison fair? Two issues. TDA's per-dataset alpha and beta come from its repository with no stated selection procedure, so they may encode test-set information. And our own knobs could have been tuned.

We therefore report two protocols. P1 uses the published per-dataset values, which makes our numbers comparable with the paper and the benchmark. P2 uses one setting everywhere, alpha 2 and beta 5, the paper's own searched value, for every method; under P2 nothing has seen any of our test sets. Our own knobs are global constants fixed in advance in both.

We agreed a decision rule beforehand: if TDA under P2 still beats CLIP's accuracy, P2 becomes the headline. The P2 numbers, with three seeds, are pending, task T07.

---

## 33. Ablation: gate and temperature do not stack

The two surviving changes, in all four on/off combinations, margin admission off.

The gate alone gives the best ECE. Adding temperature on top leaves accuracy unchanged, as it must, and makes ECE slightly worse, 4.61 to 6.59. The reason is that they fix the same problem: the gate already removes TDA's inflation, and what is left is about CLIP's own calibration. On EuroSAT and Aircraft the temperature search then wants to sharpen, the floor blocks it, and it does nothing on 99.9 and 92.6 percent of steps; on Pets it softens in the wrong direction.

The lower table shows it per dataset. So we recommend the gate alone.

---

## 34. Baselines and positioning

To answer "why only TDA", we will add baselines. Two more zero-shot online methods run by us from the cached features: ECALP, the most accurate online method in the benchmark and also the worst calibrated, and DMN in its zero-shot variant. One few-shot reference, Tip-Adapter with 16 labeled images per class, to show what labels buy; it is not a competitor in our setting. And the calibration-aware prompt-tuning methods, which need backprop and many views, are cited from their papers with the protocol differences stated.

This table is pending, task T08.

---

## 35. Limitations and next steps

What we cannot claim yet. One seed and one backbone; replaying cached features will make three seeds cheap.

Selection history: the gated variant was picked among four fusion variants looked at on the test sets. One of them was never run on EuroSAT, and another one actually beat TDA's accuracy on Flowers, 70.93 against 69.91. We disclose this, and the clean protocol plus selection on the validation split will test whether the choice survives.

TDA's own per-dataset hyperparameters have unstated provenance; protocol P2 removes them.

The temperature estimate is unreliable in both directions, and the mechanism claims so far are supported by math and aggregate statistics; the trace analyses will test them directly.

---

## 36. Conclusion

To conclude. TDA's overconfidence is structural: the cache adds confidence regardless of correctness, through self-votes and a product of experts that agree by construction.

Of our four changes, filtering the cache by margin fails, and mixing instead of multiplying fails because of a scale mismatch. Consulting the cache only where CLIP is torn between two classes works: mean ECE from 11.24 to 4.61, at 55.01 percent accuracy, for one comparison per image. The temperature helps plain TDA but adds nothing on top of the gate.

What remains is to confirm this under the clean protocol, over three seeds, with direct traces of the mechanism and against other methods. Thank you.

---

## 37. Backup B1 · TDA's own ablations, redone with ECE

Backup. We will redo TDA's own ablations with calibration added: which of the two caches causes the overconfidence, how the shot capacity trades purity against coverage, the negative cache thresholds, and alpha-beta heatmaps with both protocols marked. Test-split sweeps here are sensitivity analyses only; no choice is made from them.

---

## 38. Backup B2 · Selection history: the four fusion variants (test sets)

For full disclosure, these are the four fusion variants we looked at on the test sets before settling on the plain gate. One of them, the gated variant with a range-matched cache, was never run on EuroSAT. The ungated range-matched variant beat TDA's accuracy on Flowers. Choosing among these on test is selection on test; the clean protocol and validation-split selection will redo the choice properly.

---

## 39. Backup B3 · Accuracy vs ECE, all headline runs

The headline results as a scatter: accuracy to the right is better, ECE down is better. Each coloured line joins one dataset's methods. The target corner is TDA's accuracy at CLIP's ECE.

---

## 40. Backup B4 · First ablation: all eight on/off combinations

The first ablation, every on/off combination of margin admission, ungated fusion and temperature. Any variant with margin admission or ungated fusion is worse than the baseline on ECE, which is why those were dropped or repaired.

---

## 41. Backup B5 · Implementation notes

A few implementation notes for questions. Our "none" baseline and the separately run upstream TDA agree within 0.12 accuracy points and 0.12 ECE points. TDA's normalized entropy has a quirk: nats divided by log base two of the class count, so it never reaches one; we keep it unchanged for faithfulness. And every number in the deck is produced by code from tracked files.

---

## 42. Backup B6 · Other calibration and selective-prediction metrics

Backup on metrics. ECE depends on binning, so we will also report negative log-likelihood, the Brier score and the area under the risk-coverage curve, which measures the abstention use case directly. Evidence A10.
