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

The short answer, which I will not hide until the end: one of our three ideas
failed and was dropped, one failed and was repaired on the way, and the repaired
one is what we recommend. The interesting part of this talk is why.

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

That is our diagnosis. Each of our three changes attacks one link in this
chain, and we measure each one alone before combining anything.

---

## 11. Change 1: admit by margin — and why it fails

From here on, one change at a time, each measured alone against the same
baseline: our method with every change switched off, which is TDA run
through our code, in the same sweep. The small table on the right also
carries the setup: five datasets, 19,257 test images per method, CoOp
test splits, ViT-B/16, batch size one so it is genuinely online, seed zero,
twenty-bin ECE. Labels are opened once per run, by the final scorer.

Change one replaces entropy admission with a margin gate: admit only if the
top logit beats the runner-up by more than 1. One global threshold,
fixed before any run, because fitting it per test set would be test-set
tuning even without labels.

It loses everywhere. Mean accuracy 56.60 to 53.67, mean ECE
11.28 to 18.06. The admitted column explains the first reason: a
logit gap of one means something different on ten and a hundred classes, so
the gate admits 81.9 percent on Pets and 15.9 percent on
Aircraft, ordered by how well CLIP already does.

Second reason: a starving cache is not a smaller cache, it is a biased one.
Only the easy classes clear the bar, so the cache can only ever vote for
easy classes, whatever the image shows. That is why Aircraft doubles its
ECE while Pets, where almost everything is admitted, barely moves.

Third: margin filters loudness, not correctness. A confidently wrong sample
has a large margin exactly like a confidently right one, so the gate could
never remove the samples it was designed for. We drop it; everything from
here on has margin admission off.

---

## 12. Change 2: fuse probabilities — and why that fails too

Change two attacks the combination step. TDA adds logits and then takes a
softmax, so two sources that agree sharpen each other. A convex mixture of
the two probability distributions cannot do that: it is bounded by its most
confident input. We weight them equally.

As proposed it is the worst single change in the project. Mean ECE
11.28 to 24.54, driven by Flowers 6.51 to 32.46 and Pets
to 43.78. Accuracy barely moves, so the cache is still choosing the
class; what breaks is the confidence.

The mechanism is in the equation once you compare the scales. A cache logit
is a sum of a few exponential affinity terms, about two at most, and most
classes score near zero; softmaxed over a hundred classes that is almost
flat, top entry around seven percent. CLIP's logits are peaked, often above
ninety. Average the two fifty-fifty and you halve the peak, on every sample,
right or wrong. Confidence is capped near one half. ECE punishes
underconfidence as hard as overconfidence, so the cost scales with accuracy:
Pets is right 88.77 percent of the time and reports about forty-five,
and that gap is the number in the table.

Aircraft is the one place fusion looks good, and it is an accident: its true
accuracy happens to sit near where the ceiling landed.

---

## 13. The repair: fuse only on a two-way tie

The diagnosis says fusion fails because it runs on every sample, including
the large majority where CLIP already has a clear winner and the cache can
only dilute it. On those samples the cache cannot change the argmax either
way: adding would only sharpen, which is TDA's overconfidence, and mixing
would only dilute. So leave it out.

The cache has a real job on exactly one kind of sample: a two-way tie. There
it can break the tie, which is where TDA's accuracy gain comes from, and on
a genuine tie a confidence near one half is roughly honest.

So we gate the fusion. Take CLIP's softmax and its two largest entries. A
sample is a two-way tie when the gap between them is below 0.2 and
their sum is above 0.6. The first condition is the close call. The
second is what low entropy buys: a sample spread thinly over many classes
also has a small gap, but there CLIP is confused rather than torn, and the
cache's vote is noise. The three sketches show the three cases.

On a tie the output is the fifty-fifty mixture; otherwise it is CLIP's own
distribution, still corrected by TDA's negative cache, which we never touch.
Both thresholds are in probability space, so one global pair means the same
thing on ten and a hundred classes, which is exactly the portability margin
admission lacked, and both were fixed before any run.

Notice it is the same quantity change one used, the gap between the top two
scores, with the opposite purpose: at prediction time, deciding when the
cache is consulted, instead of at storage time, deciding what it remembers.

---

## 14. Gated fusion: the blow-up is gone

With the gate, the blow-up is gone. Flowers and Pets are back to single-digit
ECE, and gated fusion beats the baseline's ECE on four of five datasets, by
ten or more points on three of them. Mean ECE goes from 11.28 to
4.61, which is below zero-shot CLIP's own 5.57: the method is better
calibrated than the frozen model it started from. Pets is the one loss, on
the dataset where TDA was already nearly perfectly calibrated.

The accuracy side is the cost. Gated fusion matches or fractionally exceeds
the baseline on Pets and Aircraft, gives up 0.77 points on Flowers and
1.35 on DTD, and 5.92 on EuroSAT, where it returns about half of
TDA's gain. EuroSAT is where TDA's cache weighting is strongest, so its flips
presumably come from moderate-margin samples the gate keeps the cache away
from; we did not instrument that.

The contrast with change one is the clearest finding in the project. Margin
admission applied the top-two gap at storage time and threw information
away, unevenly across classes. The gate applies it at prediction time and
leaves every class represented.

---

## 15. Change 3: a label-free temperature — one reversal

Change three. A well-calibrated model has mean confidence equal to its
accuracy. Mean confidence we can measure; accuracy normally needs labels. We
estimate it from the cache itself: leave-one-out nearest-neighbour agreement
among the cached features and their pseudo-labels. Then a grid search picks
the temperature, at least one, that brings mean confidence onto that
estimate. Dividing logits by a positive constant is monotonic, so this change
cannot move an argmax.

The table shows that guarantee in the measurements: accuracy identical to two
decimals on every dataset. On calibration it is a real but uneven gain. Mean
ECE 11.28 to 8.54, with large drops where the baseline was badly
overconfident, DTD and Aircraft, and small ones on EuroSAT and Flowers.

Pets is a clear reversal, 2.48 to 15.54. Two reasons.
The search has no stopping rule: it always moves confidence toward the
estimate, so where TDA was already honest the only thing it can do is
damage. And the estimate is biased low on fine-grained data: on Pets a
cached Birman's nearest cached neighbour is often a Ragdoll even when both
pseudo-labels are correct, so agreement sits well below the true accuracy,
the search reads that as overconfidence, and smooths a model that did not
need it. The floor at T equals one only prevents the opposite mistake.

---

## 16. Putting it together: five methods, five datasets

Now the five headline methods side by side: zero-shot CLIP, TDA, and three
configurations of ours, gated fusion alone, temperature alone, and the two
together, all with margin admission off.

Averaged over the five datasets, CLIP reaches 53.16 percent accuracy at
5.57 ECE, TDA 56.63 at 11.24. Gated fusion alone, the highlighted
row, reaches 55.01 at 4.61: it keeps 1.85 of TDA's 3.47 points of
accuracy gain over CLIP while reporting better-calibrated confidence than
either of them.

The scatter on the right is the same comparison. The target is a point as
far right as TDA and as far down as zero-shot CLIP; the gated-fusion marker
sits below both on three of the five datasets.

---

## 17. The ablation: the two do not stack

The ablation over the two surviving changes: both on/off combinations of
gated fusion and temperature, margin off, on all five datasets, 20
records.

They do not stack. Adding temperature on top of the gate leaves accuracy
unchanged, as it must, and makes mean ECE worse, 4.61 to 6.59.
The reason is that both target the same thing. Temperature scaling was
designed to undo the overconfidence TDA's cache manufactures; the gate
removes that overconfidence at the source, by never letting the cache sharpen
a prediction CLIP had already made. What is left after the gate is CLIP's
own calibration, already around 5.57, and the label-free estimate is not
accurate enough to improve on it. On EuroSAT and Aircraft the search wants
to sharpen, the floor blocks it, and it sits on a grid edge for 99.9 and
92.6 percent of samples; on Pets it acts on every sample, in the wrong
direction, the same Pets failure as before.

The reliability diagrams on the right show the direction of each failure
directly: TDA's curve sits below the identity line, the temperature-only
curve sits above it on Pets, and gated fusion tracks the diagonal most
closely on DTD.

Against the project's own strict criterion, accuracy at least TDA and ECE
below TDA on three of five datasets, gated fusion passes on 1 of five. We
report that as it stands. So we recommend gated fusion alone.

---

## 18. Conclusion

To close. We set out to undo TDA's calibration loss with three training-free,
label-free changes, and measured each one rather than assuming it worked.

Margin admission fails outright, and we dropped it. Probabilistic fusion
fails as proposed, for a reason the equation makes visible once you compare
the two inputs' scales; the repair follows from the diagnosis, and fusing
only where CLIP is torn between two classes is the method we recommend:
mean ECE 11.24 to 4.61, below zero-shot CLIP's own 5.57, at
55.01 percent accuracy against TDA's 56.63. Temperature scaling helps on its
own but adds nothing on top of the gate.

What the result supports is a narrower and, we think, more useful claim than
the one we set out to make: TDA's calibration loss comes from consulting the
cache on samples where it can only add confidence, and the gap between
CLIP's top two scores, applied at prediction time rather than at admission
time, is enough to tell those samples apart.

Happy to take questions.
