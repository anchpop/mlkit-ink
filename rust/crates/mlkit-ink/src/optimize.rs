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
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            steps: 200,
            learning_rate: 0.01,
            anchor_weight: 3.0,
            smoothness_weight: 3.0,
            resegment_every: 10,
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
    let original: Vec<Stroke> = strokes.to_vec();
    let mut current = original.clone();

    let scale = ink_scale(&original);
    let epsilon = 1e-3 * scale;
    let step_size = options.learning_rate * scale;

    // Segment first: it is what validates the ink, and the regularizer indexes
    // coordinates directly, so a stroke with mismatched x/y must be rejected
    // before anything touches it.
    let mut segmentation = segment_for(&current, cfg, pipeline)?;
    let regularizer = Regularizer::new(&original, scale, options);
    let mut adam = Adam::new(coordinate_count(&current));
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
            // The objective just changed shape, so Adam's accumulated moments
            // describe a function that no longer exists.
            adam.reset();
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
        let regularization = regularizer.add(&mut gradient, &current);

        history.push(StepRecord {
            step,
            ctc_loss: loss.loss,
            regularization,
            resegmented,
        });
        adam.step(&mut current, &gradient, step_size);
    }

    let matched = best_matching.is_some();
    let (best_ctc_loss, best_step, strokes) = best_matching.unwrap_or(best);
    Ok(FitReport {
        strokes,
        initial_ctc_loss,
        best_ctc_loss,
        best_step,
        matched,
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
    let mut gradient = vec![0.0; coordinate_count(strokes)];
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

/// Adam, because the three gradient terms differ in scale by orders of
/// magnitude and a single global step size cannot serve all of them.
struct Adam {
    mean: Vec<f64>,
    variance: Vec<f64>,
    /// Running `beta^step` for the bias correction, accumulated rather than
    /// recomputed — `f64::powi` is a `std` method and this crate is `no_std`.
    decay: (f64, f64),
}

impl Adam {
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPSILON: f64 = 1e-8;

    fn new(size: usize) -> Self {
        Adam {
            mean: vec![0.0; size],
            variance: vec![0.0; size],
            decay: (1.0, 1.0),
        }
    }

    fn reset(&mut self) {
        self.mean.fill(0.0);
        self.variance.fill(0.0);
        self.decay = (1.0, 1.0);
    }

    fn step(&mut self, strokes: &mut [Stroke], gradient: &[f64], step_size: f64) {
        self.decay = (self.decay.0 * Self::BETA1, self.decay.1 * Self::BETA2);
        let bias1 = 1.0 - self.decay.0;
        let bias2 = 1.0 - self.decay.1;
        let mut slot = 0;
        for stroke in strokes.iter_mut() {
            for point in 0..stroke.len() {
                for axis in 0..2 {
                    let g = gradient[slot];
                    self.mean[slot] = Self::BETA1 * self.mean[slot] + (1.0 - Self::BETA1) * g;
                    self.variance[slot] =
                        Self::BETA2 * self.variance[slot] + (1.0 - Self::BETA2) * g * g;
                    let mean = self.mean[slot] / bias1;
                    let variance = self.variance[slot] / bias2;
                    let delta = step_size * mean / (variance.sqrt() + Self::EPSILON);
                    let updated = coordinate(stroke, point, axis) - delta;
                    set_coordinate(stroke, point, axis, updated);
                    slot += 1;
                }
            }
        }
    }
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
