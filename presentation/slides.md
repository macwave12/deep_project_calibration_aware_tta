# Calibration-Aware Test-Time Adaptation for CLIP

**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`
alongside `calibration_aware_tta.pptx`; edit the slide functions in that script
instead. The deck is the deliverable, this is the readable mirror of it.

Each section below is one slide: its on-screen heading, and the speaker notes
that live in that slide's PowerPoint notes pane.

---

## 1. Title

Good morning. Our project asks a narrow question about test-time adaptation.

TTA methods make CLIP more accurate on new domains without any training or
labels. We asked whether they also make it more trustworthy - whether the
confidence it reports can still be believed - and whether we could fix that
with three training-free changes.

The short answer, which I will not hide until the end: two of our three ideas
failed, one worked, and the interesting part of this talk is why.

---

## 2. Agenda

Five parts. I will spend the first third on background, because the result
only means something once you know what TDA does and what calibration is.

Sections three and four - our method and our results - are the project
itself, and that is where most of the time goes.

---

## 3. What is CLIP?

First, what CLIP actually is. It is a vision-language model from OpenAI: two
encoders, one for images and one for text, trained so that a picture and a
sentence describing it land close together in one shared embedding space.
The point of training it that way is generality - instead of learning one
fixed set of classes, it learns visual concepts from language, so one frozen
model can recognise anything you can describe in words.

That is what makes zero-shot classification possible. Zero-shot means
classifying into categories the model was never explicitly trained on, with
zero labelled examples. And the reason it is used is economic: a new task
costs nothing. No training data, no classifier head, no fine-tuning - you
write the class names, and the names themselves are the classifier.

Mechanically, each class name is wrapped in a prompt like "a photo of a
Boeing 737", all the prompts are encoded, and an image is assigned to
whichever prompt it sits closest to - the formula on the right. Everything
later in this talk either adds a term to those logits or rescales them; the
encoders stay frozen throughout.

---

## 4. … but CLIP degrades badly out of distribution

So why is CLIP not simply the end of the story? Because its generality comes
from its training data, and that data is web imagery. A domain that looks
nothing like the web - a satellite view, a close-up of an aircraft variant -
is out of distribution for it.

On those domains the zero-shot accuracy collapses. The benchmark paper we
build on reports CLIP at 23.7 percent on EuroSAT satellite scenes and 15.7
percent on fine-grained aircraft - a model that names everyday objects
effortlessly cannot tell a Boeing 737-700 from a 737-800.

The usual remedy would be fine-tuning on the new domain, and that is exactly
what we are not allowed: the new domain has no labels, and retraining a
backbone per deployment target is not realistic. That constraint - recover
the lost accuracy with no labels and no training - is what test-time
adaptation is for, and it is the setting of this whole project.

---

## 5. What is test-time adaptation? TDA, by example

Test-time adaptation means adapting while you are already deployed, using
nothing but the unlabelled test images as they come in. There is no label and
no training run; the test stream itself is the only new information.

There are many TTA methods. The one we build on is TDA, which never
backpropagates. Instead it keeps a small memory of past test images, and the
easiest way to see what that memory does is to follow one photo through it.

Step one: a photo of a plane arrives. Frozen CLIP encodes it into a feature
vector, which is just the point in CLIP's embedding space where this photo
landed, and the usual zero-shot step predicts, say, 737-800 with high
confidence.

Step two: TDA stores that feature in a cache, filed under the class CLIP
itself just predicted. That label is a pseudo-label. Nothing checked it. If
CLIP was wrong and this was really a 737-900, the feature still sits under
737-800.

Step three: a later photo arrives. TDA computes its feature the same way and
compares it to every stored feature by similarity in the embedding space.
Cached photos that look like the new one get a high similarity; unrelated
ones get almost none.

Step four: the similar cached items vote for whatever pseudo-label they were
filed under. If the new photo sits close to three cached 737-800s, the
737-800 logit gets a boost. Those votes are added to CLIP's own zero-shot
logits, and the prediction is the argmax of the sum.

So the cache acts like a nearest-neighbour classifier that grows out of the
test stream, labelled by CLIP's own earlier opinions. That is what makes TDA
cheap and effective, and, as the next slides show, it is also exactly where
its calibration problem comes from.

---

## 6. Accuracy is not the only thing that matters

So far we have only talked about accuracy. But a classifier gives you two
things per prediction: the class it chose, and a confidence - the
probability it assigns to that choice.

A model is well-calibrated when that confidence matches reality. Among all
the predictions it makes with eighty percent confidence, about eighty
percent should actually be correct. An overconfident model - one that is
right eighty percent of the time but routinely claims ninety-five - misleads
anything downstream that acts on the number: a filter that auto-accepts
confident predictions, a human deciding whether to double-check, an
abstention rule.

Expected Calibration Error turns this into one number, in three steps.
First, sort all the predictions into bins by their confidence - we use
twenty equal-width bins. Second, inside each bin, compare the average
confidence to the actual accuracy of those predictions. Third, average those
per-bin gaps, weighting each bin by how many predictions it holds. A
concrete example: if the ninety-to-ninety-five percent bin holds two hundred
predictions averaging ninety-two percent confidence but only seventy-five
percent of them are right, that bin contributes a seventeen-point gap.

The diagram on the right shows the same thing visually, on synthetic curves.
Perfect calibration is the dashed diagonal - accuracy equals confidence
everywhere. An overconfident model sags below it, and the purple segment is
one bin's gap. ECE is just the weighted average of all those gaps: zero
means every confidence level is honest, and higher means the number the
model reports cannot be trusted.

---

## 7. TTA buys accuracy and pays for it in calibration

Here is the problem this project is about, as the literature reports it.

A NeurIPS 2025 benchmark - pointedly titled "The Illusion of Progress?" -
measured test-time adaptation methods across ten datasets, and found that
TDA's accuracy gains come with a hidden bill. Average calibration error
rises from 5.70 to 9.21 percent - it nearly doubles - at the same time as
accuracy improves. The predictions get better while the confidence attached
to them gets less trustworthy.

What makes this interesting is that it is not a tuning accident. It follows
from how TDA is built, and in the next three slides I will show you the
mechanism - because our method is three attempts to break exactly that
mechanism.

The project, in one sentence: close that calibration gap under the same
rules TTA plays by - no retraining, and no labels.

---

## 8. TDA, part 1: a cache of past test features

This is TDA, top to bottom.

An image arrives. Frozen CLIP encodes it and produces zero-shot logits -
exactly the formula from the CLIP slide, unchanged.

Then the cache. TDA keeps two. The positive cache stores features from
samples it is confident about, filed under the class it itself predicted -
these are pseudo-labels, there is no ground truth anywhere - and it keeps at
most three per class. The negative cache stores mid-entropy samples, which
carry evidence about what a class is not.

For a new image, TDA looks up both caches by feature similarity and converts
them into extra logits.

The bottom line is the one to remember: final logits are CLIP logits, plus
the positive cache term, minus the negative cache term. It is a sum of
logits, not a re-normalised mixture of distributions. That detail is where
the calibration problem comes from, and it is what our second contribution
changes.

---

## 9. TDA, part 2: who gets into the cache

Two design choices to pin down, because our method changes both.

First, admission is gated by entropy - low entropy means a peaked, decisive
prediction. To be precise: the negative cache uses an entropy band, and the
positive cache actually admits everything, using entropy only as the
eviction key when a class slot is already full. We verified that in the
source, and it matters later for what turning our own gate off has to mean.

Second, combination is additive: cache logits are added onto CLIP logits
with tuned alpha and beta coefficients.

Now the problem, in the box. Entropy measures how loud a prediction is, not
whether it is right. And there is no label at test time to check against. So
a confidently wrong prediction enters the cache on exactly the same terms as
a confidently right one.

---

## 10. Why that breaks calibration

Put those two design choices together and you get a feedback loop.

A confident prediction - correct or not - gets cached. Later, a similar
image retrieves it. The cache logits agree with CLIP logits, because the
cache was built out of CLIP's own earlier opinion. Adding two agreeing logit
vectors produces a sharper distribution than either one alone.

So confidence goes up. Accuracy does not necessarily follow, because nothing
in that loop ever consulted a label. Confidently wrong predictions are
reinforced exactly as efficiently as confidently right ones.

That is our diagnosis, and each of our three changes attacks one link in
this chain.

---

## 11. Admit by margin, not by entropy

Change one attacks the admission gate.

The trouble with entropy is that it is a property of the whole distribution
across all classes. On a hundred-class problem a long flat tail can drive
entropy down even when the top two classes are essentially tied - and that
tied case is exactly what we want to keep out of the cache.

The top-2 margin asks a narrower and more relevant question: did one class
actually beat the runner-up?

One honest caveat, which becomes important later. We use a single global
threshold of one point zero for every dataset. We could have tuned it per
dataset, but fitting a threshold to each test set margin distribution is
test-set tuning even though no labels are involved. The cost is that logit
margins do not share a scale across datasets with different class counts:
admission runs from 81.9 percent on Pets down to 15.9 percent on
Aircraft, where the cache nearly starves. We report that rather than
retune it.

---

## 12. Fuse probabilities, do not add logits

Change two attacks the combination step.

Adding two logit vectors that agree gives you a distribution sharper than
either one on its own. That is the step that manufactures confidence out of
mere agreement.

A convex mixture of the two probability distributions cannot do that. A
weighted average is bounded by its most confident input, so the cache can
still change which class wins, but it cannot push the winning probability
higher than both sources had.

We weight CLIP and cache equally, at one half. Keep this slide in mind -
this is the change that goes most spectacularly wrong, and the reason is
visible right here in the equation if you look for it.

---

## 13. Pick a temperature with no labels at all

Change three is the one that worked, so let me be precise about it.

A well-calibrated model has mean confidence equal to its accuracy. We can
measure mean confidence directly. The missing half is accuracy, which
normally needs labels.

We estimate it without labels, using the cache itself: take the cached
features and the pseudo-labels the method assigned them, and compute
leave-one-out nearest-neighbour agreement. If the nearest neighbour of a
cached item carries a different pseudo-label, the cache is internally
inconsistent, and that is evidence the predictions are unreliable.

Then we grid-search the temperature that brings mean confidence onto that
estimate. One design decision worth flagging: the grid floor is one, so the
search can only smooth, never sharpen. If our label-free estimate comes out
too optimistic, the worst case is that we do nothing, rather than making
overconfidence worse.

And note the line at the bottom - dividing logits by a positive constant is
monotonic, so it cannot change which class wins. This contribution can move
calibration only. Accuracy is untouchable by construction.

---

## 14. Experimental setup

Five datasets, 19,257 test images per method, standard CoOp splits so
our numbers are comparable to the benchmark paper. ViT-B/16, batch size one
so it is genuinely online, seed zero, twenty-bin ECE.

One thing I want to highlight because it is easy to get wrong in this area:
labels enter our codebase in exactly one function, the scorer that runs at
the end of a run. No adaptation method ever receives a label. We enforce
that with a test that inspects function signatures, so it cannot quietly
decay as the code changes.

Two sweeps: fifteen headline runs, and 40 ablation runs covering all eight
on/off combinations of our three changes on all five datasets.

---

## 15. Headline result: the bundle does not work

This is the result we set out to get, and did not get.

Top table, accuracy: our full method is below TDA on every single dataset.
Bottom table, ECE: it is worse than TDA on four of five - and not slightly.
Flowers goes from 6.39 to 32.73 percent, Pets from
2.52 to 43.98 percent. Aircraft is the one dataset where the
bundle genuinely helps calibration.

Our success criterion was: at least three of five datasets where we match or
beat TDA accuracy while beating its ECE. We hit that on 0 of five.

I want to be direct about this rather than reframe it. The method as
specified - all three contributions enabled together - fails. The
interesting question is why, and that is what the ablation answers.

---

## 16. The ablation: one of the three actually works

Forty runs: all eight on/off combinations of our three changes, on all five
datasets. Left panel is mean ECE, lower is better. Right panel is mean
accuracy. Rows are sorted so the best-calibrated variant is on top.

The orange row is the baseline - all three flags off, which is TDA.

The purple row on top is temperature scaling alone, with margin admission
and fusion both switched off. Mean ECE drops from 11.28 to 8.54
percent, and look at the right panel: accuracy is identical, 56.60
in both. Not approximately - exactly, to two decimals, on all five datasets.
That is the structural guarantee from the previous slide showing up in the
measurements: temperature scaling is monotonic, so it cannot move an argmax.

The hatched row, second from the bottom on ECE, is our full method. Nearly
everything we added on top of temperature made things worse.

One honest qualifier: it wins on 4 of five datasets, not all five. Pets is
the exception, and I will come back to it.

---

## 17. Why the other two failed

Three diagnoses, one per card.

Margin admission: the single global threshold does not port across datasets
with different class counts. On Aircraft, with a hundred classes, it admits
under sixteen percent of samples and the cache starves. Mean accuracy drops
almost three points, from 56.60 to 53.67. The signal it threw away
was being used correctly.

Fusion: this one backfired in the opposite direction from the design intent.
We built it to prevent over-confidence. But the cache logits are small and
flat compared to CLIP's sharp distribution, so averaging them fifty-fifty caps
top-one confidence well below one on every sample, whether it is right or
wrong. The result is severe under-confidence - and ECE punishes that exactly
as hard as over-confidence.

Third card: when margin and fusion are both on, the leave-one-out estimate
is inflated, because a selectively-pure cache agrees with itself more than
the true accuracy would. The search therefore wants to sharpen, and our
floor at T equals one stops it. Between 98 and 100 percent of samples sit
pinned at the no-op. So inside the bundle, contribution three is doing
essentially nothing - which is why the bundle looks like margin plus fusion.

Finally, the box. On Pets, TDA is already extremely well calibrated. Our
method still smooths, because it has no way to tell that smoothing was not
needed, and ECE goes from 2.53 to 14.79 percent. We had developed on DTD,
where the story was clean. Five datasets showed us the exception.

---

## 18. Limitations

Four limitations we would want a reader to know.

The global margin threshold is not portable across class counts, and we
chose not to tune it per dataset because that would be test-set tuning.

Temperature scaling has no stopping rule - it always smooths toward the
estimate, even when the model was already honest. That is the Pets failure.

Five datasets, one backbone, one seed. Enough to establish that the bundle
fails and that temperature alone helps on four of five. Not enough to claim
either holds universally.

And the leave-one-out estimate is only as good as the cache it reads, so it
inherits any bias the admission rule introduces. That coupling is exactly
why contributions one and three interfere with each other.

---

## 19. Conclusion

To close.

The headline claim we can defend is the box: leave-one-out temperature
scaling, applied on its own, cuts TDA mean calibration error from 11.28 to
8.54 percent while leaving accuracy completely unchanged, on 4 of five
datasets. It is training-free, label-free, and costs one grid search.

The bundle we originally proposed does not work, and we are not going to
dress that up. Margin admission and probabilistic fusion are both harmful on
average.

What I would argue we actually got out of this project is the diagnosis.
The ablation - not our intuition - told us margin starves the cache, fusion
manufactures under-confidence, and the two of them together render the one
idea that works inert. We would not have known any of that from the headline
numbers alone.

Happy to take questions.
