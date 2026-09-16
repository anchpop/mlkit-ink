//! Gradient descent on ink: nudge stroke coordinates until the recognizer reads
//! them as the character you wanted.
//!
//! The whole pipeline is differentiable once you hold its *combinatorial*
//! decisions fixed — which samples survive thinning, where a stroke is cut into
//! curves, which curves merge. [`crate::curve::Segmentation`] is exactly that
//! set of decisions, separated out for this purpose. Freeze it, and
//! `points -> features -> logits -> CTC loss` is a smooth function; thaw and
//! recompute it every so often and the optimizer can still change the
//! structure, just not within a single step.
//!
//! # How the three links are differentiated, and why they differ
//!
//! * **CTC loss to logits** — analytic, [`crate::ctc`]. Standard forward-backward.
//! * **Logits to features** — analytic, [`crate::netgrad`]. Hand-written BPTT,
//!   because a tape over six bidirectional 216-unit layers would record tens of
//!   millions of scalar nodes.
//! * **Features to points** — *finite differences*. This is a deliberate
//!   choice, not laziness: the curve fitter is the most intricate and most
//!   precision-sensitive code in the crate, reproduced instruction-faithfully
//!   from the SDK, and a hand-written or generic-scalar differentiable copy of
//!   it would be a second implementation free to drift from the first. Bumping
//!   a coordinate and re-running the *real* fitter cannot drift. It costs two
//!   geometry evaluations per coordinate, which is cheap next to one network
//!   forward pass — the expensive link is the one that is analytic.
//!
//! # This is an adversarial-example generator, and that is the point
//!
//! Without a regularizer the optimizer happily produces scribbles that the
//! network is very confident about and a human cannot read. [`FitOptions`]
//! therefore carries an anchor term (stay near the ink you started with) and a
//! smoothness term (stay a plausible pen trace). Set both to zero and you get
//! the adversarial version, which is interesting for a different reason.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::curve::{self, Segmentation};
use crate::decoder::Alphabet;
use crate::error::Result;
use crate::features;
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::ink::{Stroke, bbox_diagonal};
use crate::mat::Mat;
use crate::netgrad;
use crate::recognizer::Recognizer;
use crate::settings::{CurveSettings, PreprocessingStep};
use crate::{ctc, ink};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitOptions {
    pub steps: usize,
    /// Step size, in units of the ink's bounding-box diagonal, so the same
    /// value behaves the same whether coordinates are in pixels or millimetres.
    pub learning_rate: f64,
    /// Pull back toward the original ink. Zero makes this an adversarial
    /// attack; large values make it refuse to move.
    pub anchor_weight: f64,
    /// Penalize second differences along each stroke, which is what stops the
    /// result from turning into a jagged high-frequency artifact.
    pub smoothness_weight: f64,
    /// Recompute thinning and the curve split after this many steps. The
    /// structure is frozen within a step so the gradient is well defined; it
    /// still has to be allowed to change eventually.
    pub resegment_every: usize,
    /// How many jittered copies of the ink to average the gradient over, and
    /// how far to jitter them (as a fraction of the ink's diagonal).
    ///
    /// This is what separates "labelled A" from "shaped like an A". A genuine
    /// letter keeps reading correctly when its samples are nudged; an
    /// adversarial one sits on a knife-edge and collapses. Measured on our
    /// corpus at 0.5% jitter: a real handwritten A survived 12 times out of 12,
    /// an adversarially-grown one 1 time out of 12. Descending the *expected*
    /// loss under that noise therefore refuses to settle anywhere fragile, and
    /// the only regions left are the ones real handwriting occupies.
    ///
    /// `robust_samples: 1` with `jitter: 0.0` is plain descent, and costs one
    /// forward-backward per step; each extra sample costs another.
    pub robust_samples: usize,
    pub jitter: f64,
    /// Width, as a fraction of the ink's diagonal, of the blur applied to the
    /// descent direction along each stroke. Zero disables it.
    ///
    /// This is what keeps the result looking hand-drawn, and it does more work
    /// than the smoothness penalty does. The raw gradient is free to push
    /// neighbouring samples in unrelated directions — nothing in the loss
    /// couples them — so descending it directly grows exactly the
    /// sample-frequency jitter a person reads as "ugly", even while the
    /// penalty term is objecting. Blurring the *direction* means the optimizer
    /// can only ever move a smooth stretch of stroke at a time, so the jitter
    /// is not penalised into submission; it is unreachable.
    ///
    /// Measured in arc length rather than samples on purpose: a browser
    /// samples a pen roughly four times more densely than our corpus inks, and
    /// a sample-indexed blur would mean something different in each. Tuning
    /// this only against the sparse corpus is exactly how the first version
    /// shipped a default that left browser strokes visibly lumpy.
    ///
    /// It is not a cosmetic trade-off against accuracy: widening this from
    /// 0.03 to 0.4 took a ten-pair spot check from six converged to ten, and
    /// cut the steps needed roughly fivefold. The jitter was wasted motion.
    pub smoothing: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            steps: 200,
            learning_rate: 0.01,
            anchor_weight: 3.0,
            smoothness_weight: 3.0,
            resegment_every: 10,
            robust_samples: 1,
            jitter: 0.0,
            smoothing: 0.4,
        }
    }
}

/// One step's losses, for plotting or for deciding to stop early.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    pub step: usize,
    /// Negative log likelihood of the target under CTC.
    pub ctc_loss: f64,
    /// Anchor plus smoothness.
    pub regularization: f64,
    /// True on the steps where the segmentation was recomputed; the total loss
    /// is allowed to jump at these, because the function being minimized
    /// changed.
    pub resegmented: bool,
}

#[derive(Debug, Clone)]
pub struct FitReport {
    /// The best ink found, not the last one visited. Resegmentation changes
    /// the objective mid-run and Adam can overshoot, so the final iterate is
    /// routinely worse than something the search already passed through.
    pub strokes: Vec<Stroke>,
    pub initial_ctc_loss: f64,
    /// CTC loss of [`FitReport::strokes`] **under the segmentation in force
    /// when it was found**, which is the quantity the descent actually
    /// minimizes. Re-deriving the segmentation from the moved coordinates can
    /// shift it slightly, so treat it as a surrogate: [`FitReport::matched`] is
    /// the claim checked against a fresh, full recognition.
    pub best_ctc_loss: f64,
    /// Which step produced it.
    pub best_step: usize,
    /// The final iterate, which is **not** generally the best one.
    ///
    /// Continue from this, not from [`FitReport::strokes`], when running the
    /// fit in chunks. Resuming from the best iterate makes a chunk that failed
    /// to improve return its own input, so the next chunk starts from an
    /// identical state, takes an identical step, and the loop becomes a fixed
    /// point — the search freezes permanently rather than exploring past a
    /// plateau. Keep your own record of the best across chunks instead.
    pub last: Vec<Stroke>,
    /// Whether [`FitReport::strokes`] actually *reads* as the target under a
    /// greedy decode. A falling CTC loss only means the target got more
    /// probable, which is not the same as it winning — so the search prefers
    /// any iterate that reads correctly over a lower-loss one that does not.
    pub matched: bool,
    pub history: Vec<StepRecord>,
}

/// Move `strokes` so the recognizer reads them as `target`.
///
/// Timestamps are left alone: the three time features are real inputs, but
/// "write the same shape at a different speed" is a strange thing to ask an
/// optimizer for, and letting it rewrite time makes the result much harder to
/// interpret. Only x and y move.
pub fn fit_strokes(
    recognizer: &Recognizer<'_>,
    strokes: &[Stroke],
    target: &str,
    options: &FitOptions,
) -> Result<FitReport> {
    fit_strokes_from(recognizer, strokes, strokes, target, options)
}

/// As [`fit_strokes`], with the ink the regularizer should pull *back toward*
/// supplied separately from the ink to start descending from.
///
/// They differ whenever a fit is resumed. Both penalties measure deformation
/// away from `reference`, so a chunked caller that passes its current ink as
/// both gets a reference that advances in lockstep with the search: the
/// deformation is zero at the start of every chunk, both gradients are zero at
/// every applied update, and the penalties silently do nothing at all. With
/// one step per chunk — which is what a live animation wants — that is *every*
/// update. Pass the user's original drawing here and the current ink as
/// `strokes`.
pub fn fit_strokes_from(
    recognizer: &Recognizer<'_>,
    strokes: &[Stroke],
    reference: &[Stroke],
    target: &str,
    options: &FitOptions,
) -> Result<FitReport> {
    ensure!(options.steps > 0, Invalid, "steps must be positive");
    ensure!(
        options.resegment_every > 0,
        Invalid,
        "resegment_every must be positive"
    );
    let labels = target_indices(&recognizer.alphabet, target)?;
    ensure!(!labels.is_empty(), Invalid, "target is empty");

    let cfg = &recognizer.spec.curve_settings;
    let pipeline = &recognizer.spec.pipeline;
    let original: Vec<Stroke> = reference.to_vec();
    // Both x and y, for both inks. `Stroke::len()` reports the x count alone —
    // deliberately, since a stroke's timestamps are allowed to be missing until
    // preprocessing fills them in — so comparing lengths with it would accept a
    // reference whose y is short and then panic indexing it. The ink being
    // fitted gets validated downstream by the pipeline; the reference is only
    // ever read by the regularizer, so this is its only check.
    ensure!(
        original.len() == strokes.len(),
        Invalid,
        "the regularization reference has {} strokes, the ink has {}",
        original.len(),
        strokes.len()
    );
    for (index, (a, b)) in original.iter().zip(strokes).enumerate() {
        ensure!(
            a.x.len() == a.y.len() && b.x.len() == b.y.len(),
            Invalid,
            "stroke {index}: x and y must be the same length"
        );
        ensure!(
            a.x.len() == b.x.len(),
            Invalid,
            "stroke {index}: the reference has {} samples, the ink has {}",
            a.x.len(),
            b.x.len()
        );
    }
    // From `strokes`, not `original`: those differ whenever a fit is resumed,
    // and starting from the reference would restart the descent from scratch
    // on every chunk.
    let mut current = strokes.to_vec();

    let scale = ink_scale(&original);
    let epsilon = 1e-3 * scale;
    let step_size = options.learning_rate * scale;

    // Segment first: it is what validates the ink, and the regularizer indexes
    // coordinates directly, so a stroke with mismatched x/y must be rejected
    // before anything touches it.
    let mut segmentation = segment_for(&current, cfg, pipeline)?;
    let regularizer = Regularizer::new(&original, scale, options);
    let mut momentum = Momentum::new(coordinate_count(&current));
    let mut history = Vec::with_capacity(options.steps + 1);
    let mut initial_ctc_loss = f64::NAN;
    // Two candidates, because they answer different questions: the lowest loss
    // found, and the lowest loss among iterates that actually decode to the
    // target. The second wins whenever it exists.
    let mut best = (f64::INFINITY, 0usize, current.clone());
    let mut best_matching: Option<(f64, usize, Vec<Stroke>)> = None;

    // `..=` so there is one extra evaluation pass: the ink we return has then
    // actually been scored, instead of being whatever the last update produced.
    for step in 0..=options.steps {
        let resegmented = step > 0 && step % options.resegment_every == 0;
        if resegmented {
            segmentation = segment_for(&current, cfg, pipeline)?;
            // The objective just changed shape, so the accumulated velocity
            // describes a function that no longer exists.
            momentum.reset();
        }

        let features = geometry(&current, &segmentation, cfg, pipeline)?;
        let (logits, activations) =
            netgrad::forward_with_activations(&recognizer.weights, &features)?;
        let loss = ctc::loss_and_grad(&logits, &labels, recognizer.alphabet.blank)?;
        if step == 0 {
            initial_ctc_loss = loss.loss;
        }
        if loss.loss < best.0 {
            best = (loss.loss, step, current.clone());
        }
        // Deliberately re-run the *whole* recognizer rather than decoding the
        // logits we already have. Those were computed against the frozen
        // segmentation, and the real recognizer re-derives it — so a fit can
        // look successful under the surrogate and fail for an actual caller.
        // Asking the real thing is the only check worth reporting.
        if best_matching
            .as_ref()
            .is_none_or(|(best, _, _)| loss.loss < *best)
            && recognizer
                .recognize_greedy(&current)
                .is_ok_and(|candidate| candidate.text == target)
        {
            best_matching = Some((loss.loss, step, current.clone()));
        }

        // An unalignable target offers no gradient; stop rather than step on NaNs.
        if !loss.loss.is_finite() || step == options.steps {
            history.push(StepRecord {
                step,
                ctc_loss: loss.loss,
                regularization: 0.0,
                resegmented,
            });
            break;
        }

        let d_features = netgrad::features_gradient(&recognizer.weights, &activations, &loss.grad)?;
        let mut gradient = geometry_gradient(
            &current,
            &segmentation,
            cfg,
            pipeline,
            &features,
            &d_features,
            epsilon,
        )?;
        // Average in the gradient of a few jittered copies. Each is segmented
        // afresh, because nudging the samples is exactly the kind of change
        // that moves a split — and a perturbation the structure is pinned
        // against would not be testing robustness at all.
        if options.robust_samples > 1 && options.jitter > 0.0 {
            let mut noise = Noise::new(step as u64 + 1);
            let spread = options.jitter * scale;
            // Count what is actually accumulated, not what was asked for: a
            // jittered copy can be rejected (unalignable target, a
            // segmentation that no longer matches), and dividing by the
            // requested count would quietly shrink the acoustic gradient
            // relative to the penalties whenever that happens.
            let mut accumulated = 1usize;
            for _ in 1..options.robust_samples {
                let shaken = noise.shake(&current, spread);
                let Ok(shaken_seg) = segment_for(&shaken, cfg, pipeline) else {
                    continue;
                };
                let Ok(shaken_features) = geometry(&shaken, &shaken_seg, cfg, pipeline) else {
                    continue;
                };
                let Ok((shaken_logits, shaken_acts)) =
                    netgrad::forward_with_activations(&recognizer.weights, &shaken_features)
                else {
                    continue;
                };
                let Ok(shaken_loss) =
                    ctc::loss_and_grad(&shaken_logits, &labels, recognizer.alphabet.blank)
                else {
                    continue;
                };
                if !shaken_loss.loss.is_finite() {
                    continue;
                }
                let Ok(shaken_d) = netgrad::features_gradient(
                    &recognizer.weights,
                    &shaken_acts,
                    &shaken_loss.grad,
                ) else {
                    continue;
                };
                // The jitter is an additive offset, so a derivative with
                // respect to the shaken sample is one with respect to the
                // original too, and the two gradients simply add.
                let Ok(extra) = geometry_gradient(
                    &shaken,
                    &shaken_seg,
                    cfg,
                    pipeline,
                    &shaken_features,
                    &shaken_d,
                    epsilon,
                ) else {
                    continue;
                };
                if extra.len() == gradient.len() {
                    for (total, part) in gradient.iter_mut().zip(&extra) {
                        *total += part;
                    }
                    accumulated += 1;
                }
            }
            let count = accumulated as f64;
            gradient.iter_mut().for_each(|g| *g /= count);
        }
        let regularization = regularizer.add(&mut gradient, &current);

        history.push(StepRecord {
            step,
            ctc_loss: loss.loss,
            regularization,
            resegmented,
        });
        blur_along_strokes(&mut gradient, &current, options.smoothing * scale);
        momentum.step(&mut current, &gradient, step_size);
    }

    let matched = best_matching.is_some();
    let (best_ctc_loss, best_step, strokes) = best_matching.unwrap_or(best);
    Ok(FitReport {
        strokes,
        initial_ctc_loss,
        best_ctc_loss,
        best_step,
        matched,
        last: current,
        history,
    })
}

/// Longest-match the target against the network's own symbols.
///
/// Character by character is wrong for scripts whose symbols span several code
/// points, and those are exactly the scripts where a silent mismatch would be
/// hardest to notice.
pub fn target_indices(alphabet: &Alphabet, target: &str) -> Result<Vec<usize>> {
    let mut labels = Vec::new();
    let mut rest = target;
    'outer: while !rest.is_empty() {
        let mut best: Option<(usize, usize)> = None;
        for (index, text) in alphabet.texts.iter().enumerate() {
            if index == alphabet.blank || text.is_empty() || !rest.starts_with(text.as_str()) {
                continue;
            }
            if best.is_none_or(|(_, len)| text.len() > len) {
                best = Some((index, text.len()));
            }
        }
        match best {
            Some((index, len)) => {
                labels.push(index);
                rest = &rest[len..];
                continue 'outer;
            }
            None => bail!(
                Invalid,
                "target contains {:?}, which this model's charset cannot produce",
                rest.chars().next().unwrap_or_default()
            ),
        }
    }
    Ok(labels)
}

/// Preprocess and choose the discrete structure for the current coordinates.
fn segment_for(
    strokes: &[Stroke],
    cfg: &CurveSettings,
    pipeline: &[PreprocessingStep],
) -> Result<Segmentation> {
    let preprocessed = crate::preprocess::run(strokes, pipeline)?;
    curve::segment(&preprocessed, cfg)
}

/// The numeric half: raw coordinates plus a frozen structure -> features.
fn geometry(
    strokes: &[Stroke],
    segmentation: &Segmentation,
    cfg: &CurveSettings,
    pipeline: &[PreprocessingStep],
) -> Result<Mat> {
    let preprocessed = crate::preprocess::run(strokes, pipeline)?;
    features::encode(&preprocessed, segmentation, cfg)
}

/// `d(loss)/d(x, y)` by central differences through the real fitter.
///
/// Returned flat, in the order [`coordinate_count`] implies: stroke by stroke,
/// point by point, x then y.
/// `d(loss)/d(x, y)` by central differences through the real fitter.
///
/// Returned flat, in the order [`coordinate_count`] implies: stroke by stroke,
/// point by point, x then y.
///
/// This re-runs the whole geometry per bumped coordinate, which looks wasteful
/// and is not. Two attempts to exploit locality — encoding only the curves a
/// sample can reach, and hoisting the preprocessing out of the loop — both
/// measured *zero* speedup, because the geometry is not the cost. The network
/// is: roughly 36 million multiply-accumulates per forward pass, run twice per
/// step for the gradient and once more for the match check. Step time tracks
/// the curve count, not the sample count. Keep this simple; optimise the net
/// if it ever needs to be faster.
fn geometry_gradient(
    strokes: &[Stroke],
    segmentation: &Segmentation,
    cfg: &CurveSettings,
    pipeline: &[PreprocessingStep],
    base: &Mat,
    d_features: &[f64],
    epsilon: f64,
) -> Result<Vec<f64>> {
    debug_assert_eq!(base.as_slice().len(), d_features.len());
    let mut gradient = alloc::vec![0.0; coordinate_count(strokes)];
    let mut scratch = strokes.to_vec();
    let mut slot = 0;

    for stroke in 0..strokes.len() {
        for point in 0..strokes[stroke].len() {
            for axis in 0..2 {
                let original = coordinate(&scratch[stroke], point, axis);
                set_coordinate(&mut scratch[stroke], point, axis, original + epsilon);
                let up = geometry(&scratch, segmentation, cfg, pipeline)?;
                set_coordinate(&mut scratch[stroke], point, axis, original - epsilon);
                let down = geometry(&scratch, segmentation, cfg, pipeline)?;
                set_coordinate(&mut scratch[stroke], point, axis, original);

                // A bump that changes the feature count means the frozen
                // structure no longer describes this ink; treat the coordinate
                // as having no usable gradient rather than comparing matrices
                // of different shapes.
                if up.rows() != base.rows() || down.rows() != base.rows() {
                    slot += 1;
                    continue;
                }
                let mut directional = 0.0;
                for (i, &weight) in d_features.iter().enumerate() {
                    directional += weight * f64::from(up.as_slice()[i] - down.as_slice()[i]);
                }
                gradient[slot] = directional / (2.0 * epsilon);
                slot += 1;
            }
        }
    }
    Ok(gradient)
}

/// Keeps the result recognizable as the ink it started from.
///
/// Both terms are **means**, and both are normalized against a stated reference
/// deformation, so a weight means the same thing regardless of how many samples
/// the ink has or what units it is in:
///
/// * `anchor_weight` is the loss of moving every sample by
///   [`ANCHOR_REFERENCE`] of the ink's diagonal.
/// * `smoothness_weight` is the loss of adding a kink of
///   [`SMOOTHNESS_REFERENCE`] of the mean sample spacing at every sample.
///
/// Both are directly comparable to the CTC loss, which is a negative log
/// likelihood of order 1-10 in practice. Sums rather than means would make the
/// right weight depend on stroke length, which is exactly the kind of hidden
/// coupling that makes a knob impossible to tune.
///
/// The smoothness term measures the *change* in second difference from the
/// original, not its absolute value. Real handwriting is full of curvature and
/// none of it should be penalized; what should be penalized is new curvature
/// the optimizer introduced.
struct Regularizer<'a> {
    original: &'a [Stroke],
    anchor: f64,
    smoothness: f64,
}

/// Anchor reference deformation, as a fraction of the ink's bbox diagonal.
pub const ANCHOR_REFERENCE: f64 = 0.1;
/// Smoothness reference kink, as a fraction of the mean sample spacing.
pub const SMOOTHNESS_REFERENCE: f64 = 0.2;

impl<'a> Regularizer<'a> {
    fn new(original: &'a [Stroke], scale: f64, options: &FitOptions) -> Self {
        let points = coordinate_count(original) / 2;
        let count = points.max(1) as f64;
        let reference = ANCHOR_REFERENCE * scale;
        let spacing = mean_spacing(original, scale);
        let kink = SMOOTHNESS_REFERENCE * spacing;
        Regularizer {
            original,
            anchor: options.anchor_weight / (count * reference * reference),
            smoothness: options.smoothness_weight / (count * kink * kink),
        }
    }

    /// Add the regularizer's gradient, returning its loss contribution.
    fn add(&self, gradient: &mut [f64], current: &[Stroke]) -> f64 {
        let mut loss = 0.0;
        let mut base = 0;
        for (stroke, source) in current.iter().zip(self.original) {
            let n = stroke.len().min(source.len());
            for point in 0..n {
                for axis in 0..2 {
                    let delta = coordinate(stroke, point, axis) - coordinate(source, point, axis);
                    loss += 0.5 * self.anchor * delta * delta;
                    gradient[base + point * 2 + axis] += self.anchor * delta;
                }
            }
            for point in 1..n.saturating_sub(1) {
                for axis in 0..2 {
                    let added = second_difference(stroke, point, axis)
                        - second_difference(source, point, axis);
                    loss += 0.5 * self.smoothness * added * added;
                    let scaled = self.smoothness * added;
                    gradient[base + (point - 1) * 2 + axis] += scaled;
                    gradient[base + point * 2 + axis] -= 2.0 * scaled;
                    gradient[base + (point + 1) * 2 + axis] += scaled;
                }
            }
            base += stroke.len() * 2;
        }
        loss
    }
}

fn second_difference(stroke: &Stroke, point: usize, axis: usize) -> f64 {
    coordinate(stroke, point - 1, axis) - 2.0 * coordinate(stroke, point, axis)
        + coordinate(stroke, point + 1, axis)
}

/// Mean distance between adjacent samples, which is the length scale the
/// smoothness term is measured against. Falls back to the ink's diagonal when
/// there are no adjacent pairs at all.
fn mean_spacing(strokes: &[Stroke], scale: f64) -> f64 {
    let (mut total, mut count) = (0.0, 0usize);
    for stroke in strokes {
        // Zipped rather than indexed: this runs before the pipeline has had a
        // chance to reject a stroke whose x and y disagree in length.
        let points: Vec<_> = stroke.x.iter().zip(&stroke.y).collect();
        for pair in points.windows(2) {
            let (dx, dy) = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
            total += (dx * dx + dy * dy).sqrt();
            count += 1;
        }
    }
    if count == 0 || total <= 0.0 {
        scale
    } else {
        total / count as f64
    }
}

/// A tiny deterministic noise source for the robustness samples.
///
/// Deterministic on purpose: a fit that cannot be reproduced cannot be
/// debugged, and the point of the jitter is to sample a neighbourhood, not to
/// be unpredictable.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Noise(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// xorshift64*, which is plenty for jittering a few hundred points.
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (bits >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn shake(&mut self, strokes: &[Stroke], spread: f64) -> Vec<Stroke> {
        strokes
            .iter()
            .map(|stroke| Stroke {
                x: stroke.x.iter().map(|v| v + self.next() * spread).collect(),
                y: stroke.y.iter().map(|v| v + self.next() * spread).collect(),
                t: stroke.t.clone(),
                pen_up: stroke.pen_up,
            })
            .collect()
    }
}

/// Momentum, with a single global normalization of the step length.
///
/// Deliberately NOT Adam. Adam rescales every coordinate by its own running
/// gradient magnitude, which is the one thing this problem cannot tolerate: a
/// sample that barely affects the loss gets its tiny gradient amplified to the
/// same step as its neighbour's large one, so a perfectly smooth descent
/// direction comes out of the optimizer as jitter. Normalizing by one global
/// RMS keeps Adam's real benefit here — insensitivity to the absolute scale of
/// the loss — while preserving the direction's shape along the stroke.
struct Momentum {
    velocity: Vec<f64>,
    /// Running `beta^step` for the bias correction, accumulated rather than
    /// recomputed — `f64::powi` is a `std` method and this crate is `no_std`.
    decay: f64,
}

impl Momentum {
    const BETA: f64 = 0.9;
    const EPSILON: f64 = 1e-12;

    fn new(size: usize) -> Self {
        Momentum {
            velocity: vec![0.0; size],
            decay: 1.0,
        }
    }

    fn reset(&mut self) {
        self.velocity.fill(0.0);
        self.decay = 1.0;
    }

    fn step(&mut self, strokes: &mut [Stroke], gradient: &[f64], step_size: f64) {
        self.decay *= Self::BETA;
        let bias = 1.0 - self.decay;
        let mut magnitude = 0.0;
        for (velocity, &g) in self.velocity.iter_mut().zip(gradient) {
            *velocity = Self::BETA * *velocity + (1.0 - Self::BETA) * g;
            let corrected = *velocity / bias;
            magnitude += corrected * corrected;
        }
        // One scale for the whole ink, so the step length is predictable but
        // the relative sizes of individual displacements survive.
        let rms = (magnitude / self.velocity.len().max(1) as f64).sqrt();
        let scale = step_size / (rms + Self::EPSILON);

        let mut slot = 0;
        for stroke in strokes.iter_mut() {
            for point in 0..stroke.len() {
                for axis in 0..2 {
                    let delta = scale * self.velocity[slot] / bias;
                    let updated = coordinate(stroke, point, axis) - delta;
                    set_coordinate(stroke, point, axis, updated);
                    slot += 1;
                }
            }
        }
    }
}

/// Blur the descent direction along each stroke with a Gaussian of the given
/// arc-length width.
///
/// Each stroke is blurred independently — a pen lift is a real discontinuity
/// and smoothing across one would drag unrelated strokes together. Endpoints
/// use a renormalized truncated kernel rather than padding, so a stroke's tips
/// stay as free to move as its middle.
fn blur_along_strokes(gradient: &mut [f64], strokes: &[Stroke], sigma: f64) {
    // NaN or non-positive means "no blur"; written positively so the NaN case
    // is a deliberate choice rather than a negation clippy has to guess at.
    if !sigma.is_finite() || sigma <= 0.0 {
        return;
    }
    let mut base = 0;
    for stroke in strokes {
        let n = stroke.len();
        if n < 3 {
            base += n * 2;
            continue;
        }
        let mut arc = Vec::with_capacity(n);
        arc.push(0.0);
        for i in 1..n {
            let dx = stroke.x[i] - stroke.x[i - 1];
            let dy = stroke.y[i] - stroke.y[i - 1];
            arc.push(arc[i - 1] + (dx * dx + dy * dy).sqrt());
        }
        for axis in 0..2 {
            let source: Vec<f64> = (0..n).map(|i| gradient[base + i * 2 + axis]).collect();
            let blurred = blur_scalars(&source, &arc, sigma);
            for i in 0..n {
                gradient[base + i * 2 + axis] = blurred[i];
            }
        }
        base += n * 2;
    }
}

/// Gaussian-blur a scalar field along arc length, renormalizing at the ends.
fn blur_scalars(values: &[f64], arc: &[f64], sigma: f64) -> Vec<f64> {
    // NaN or non-positive means "no blur", stated positively so the NaN case
    // reads as a decision rather than a negation.
    if !sigma.is_finite() || sigma <= 0.0 {
        return values.to_vec();
    }
    let (cutoff, denominator) = (3.0 * sigma, 2.0 * sigma * sigma);
    (0..values.len())
        .map(|i| {
            let (mut total, mut weight) = (0.0, 0.0);
            for j in 0..values.len() {
                let distance = arc[i] - arc[j];
                if distance.abs() > cutoff {
                    continue;
                }
                let w = (-distance * distance / denominator).exp();
                total += w * values[j];
                weight += w;
            }
            if weight > 0.0 {
                total / weight
            } else {
                values[i]
            }
        })
        .collect()
}

fn coordinate_count(strokes: &[Stroke]) -> usize {
    strokes.iter().map(|s| s.len() * 2).sum()
}

fn coordinate(stroke: &Stroke, point: usize, axis: usize) -> f64 {
    if axis == 0 {
        stroke.x[point]
    } else {
        stroke.y[point]
    }
}

fn set_coordinate(stroke: &mut Stroke, point: usize, axis: usize, value: f64) {
    if axis == 0 {
        stroke.x[point] = value;
    } else {
        stroke.y[point] = value;
    }
}

/// Bounding-box diagonal of the whole ink, used to make the step size and both
/// regularizer weights independent of the coordinate units.
fn ink_scale(strokes: &[Stroke]) -> f64 {
    let points: Vec<ink::Point> = strokes
        .iter()
        .flat_map(|s| {
            s.x.iter()
                .zip(&s.y)
                .map(|(&x, &y)| ink::Point::new(x as f32, y as f32, 0.0))
        })
        .collect();
    let diagonal = f64::from(bbox_diagonal(&points));
    if diagonal > 0.0 && diagonal.is_finite() {
        diagonal
    } else {
        1.0
    }
}

/// Render a target back out of network indices, for reporting.
pub fn render(alphabet: &Alphabet, labels: &[usize]) -> String {
    labels
        .iter()
        .filter_map(|&index| alphabet.texts.get(index))
        .map(String::as_str)
        .collect()
}
