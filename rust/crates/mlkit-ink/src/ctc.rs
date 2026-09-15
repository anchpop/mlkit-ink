//! CTC loss and its gradient with respect to the logits.
//!
//! Decoding asks "what did this ink say?"; this module asks the inverse, "how
//! wrong is this ink for the text I wanted?", and — more usefully — "which way
//! should the logits move to fix it". [`crate::optimize`] chains that backwards
//! through the network and the curve fitter to get a gradient on the ink
//! itself.
//!
//! Standard Graves et al. 2006 forward-backward over the blank-interleaved
//! label sequence, in log space throughout. Nothing here is recovered from
//! Google's binary — the SDK only ever runs the forward direction.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Result;
// `cfg(test)` because the unit-test harness links `std` even when the feature
// is off, which would make this import dead and trip clippy.
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::mat::Mat;

/// Numerically safe `log(exp(a) + exp(b))`, with `-inf` absorbing correctly.
pub fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// `[T, C]` logits -> log probabilities, row-normalized, in f64.
pub fn log_softmax(logits: &Mat) -> Vec<f64> {
    let cols = logits.cols();
    let mut out = vec![0.0f64; logits.rows() * cols];
    for (t, row) in logits.iter_rows().enumerate() {
        let max = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let mut total = 0.0;
        for (c, &v) in row.iter().enumerate() {
            let shifted = f64::from(v) - max;
            out[t * cols + c] = shifted;
            total += shifted.exp();
        }
        let log_total = total.ln();
        for c in 0..cols {
            out[t * cols + c] -= log_total;
        }
    }
    out
}

/// The result of one forward-backward pass.
#[derive(Debug, Clone)]
pub struct CtcLoss {
    /// Negative log likelihood of the target under the CTC model.
    pub loss: f64,
    /// `[T, C]`, row-major: d(loss) / d(logit).
    pub grad: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

/// Blank-interleaved target: `- y1 - y2 - ... - yU -`.
fn extend(target: &[usize], blank: usize) -> Vec<usize> {
    let mut z = Vec::with_capacity(2 * target.len() + 1);
    z.push(blank);
    for &label in target {
        z.push(label);
        z.push(blank);
    }
    z
}

/// CTC negative log likelihood of `target` (network output indices) and its
/// gradient with respect to `logits`.
///
/// Returns an infinite loss with a zero gradient when the target cannot be
/// aligned at all — too long for the frame count, which is a legitimate state
/// for an optimizer to pass through rather than an error.
pub fn loss_and_grad(logits: &Mat, target: &[usize], blank: usize) -> Result<CtcLoss> {
    let (rows, cols) = (logits.rows(), logits.cols());
    ensure!(
        blank < cols,
        Invalid,
        "blank index {blank} outside {cols} classes"
    );
    for &label in target {
        ensure!(
            label < cols,
            Invalid,
            "target label {label} outside {cols} classes"
        );
        ensure!(label != blank, Invalid, "target contains the CTC blank");
    }

    let z = extend(target, blank);
    let (t_len, s_len) = (rows, z.len());
    // Every emitted label needs its own frame, and a repeated pair needs a
    // blank between them.
    let minimum = target.len() + target.windows(2).filter(|w| w[0] == w[1]).count();
    if t_len == 0 || minimum > t_len {
        return Ok(CtcLoss {
            loss: f64::INFINITY,
            grad: vec![0.0; rows * cols],
            rows,
            cols,
        });
    }

    let log_probs = log_softmax(logits);
    let lp = |t: usize, c: usize| log_probs[t * cols + c];

    let mut alpha = vec![f64::NEG_INFINITY; t_len * s_len];
    alpha[0] = lp(0, z[0]);
    if s_len > 1 {
        alpha[1] = lp(0, z[1]);
    }
    for t in 1..t_len {
        for s in 0..s_len {
            let mut total = alpha[(t - 1) * s_len + s];
            if s >= 1 {
                total = log_add_exp(total, alpha[(t - 1) * s_len + s - 1]);
            }
            // A label may be reached directly from two positions back only when
            // that would not silently merge a repeated label.
            if s >= 2 && z[s] != blank && z[s] != z[s - 2] {
                total = log_add_exp(total, alpha[(t - 1) * s_len + s - 2]);
            }
            alpha[t * s_len + s] = total + lp(t, z[s]);
        }
    }

    let last = (t_len - 1) * s_len;
    let total_log_prob = if s_len > 1 {
        log_add_exp(alpha[last + s_len - 1], alpha[last + s_len - 2])
    } else {
        alpha[last]
    };
    if total_log_prob == f64::NEG_INFINITY {
        return Ok(CtcLoss {
            loss: f64::INFINITY,
            grad: vec![0.0; rows * cols],
            rows,
            cols,
        });
    }

    let mut beta = vec![f64::NEG_INFINITY; t_len * s_len];
    beta[last + s_len - 1] = 0.0;
    if s_len > 1 {
        beta[last + s_len - 2] = 0.0;
    }
    for t in (0..t_len - 1).rev() {
        for s in (0..s_len).rev() {
            let mut total = beta[(t + 1) * s_len + s] + lp(t + 1, z[s]);
            if s + 1 < s_len {
                total = log_add_exp(total, beta[(t + 1) * s_len + s + 1] + lp(t + 1, z[s + 1]));
            }
            if s + 2 < s_len && z[s + 2] != blank && z[s + 2] != z[s] {
                total = log_add_exp(total, beta[(t + 1) * s_len + s + 2] + lp(t + 1, z[s + 2]));
            }
            beta[t * s_len + s] = total;
        }
    }

    // d(loss)/d(logit[t][k]) = p[t][k] - (1/P) * sum over positions emitting k
    // of alpha*beta. The subtracted term is the posterior probability that
    // frame t aligns to label k, so the gradient pushes mass toward whichever
    // alignment the model already finds most plausible.
    let mut grad = vec![0.0f64; rows * cols];
    let mut occupancy = vec![f64::NEG_INFINITY; cols];
    for t in 0..t_len {
        occupancy.iter_mut().for_each(|v| *v = f64::NEG_INFINITY);
        for s in 0..s_len {
            let contribution = alpha[t * s_len + s] + beta[t * s_len + s];
            if contribution != f64::NEG_INFINITY {
                occupancy[z[s]] = log_add_exp(occupancy[z[s]], contribution);
            }
        }
        for k in 0..cols {
            let posterior = (occupancy[k] - total_log_prob).exp();
            grad[t * cols + k] = lp(t, k).exp() - posterior;
        }
    }

    Ok(CtcLoss {
        loss: -total_log_prob,
        grad,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, data: &[f32]) -> Mat {
        Mat::from_vec(rows, cols, data.to_vec())
    }

    #[test]
    fn single_frame_single_label_is_just_the_log_probability() {
        let logits = mat(1, 3, &[0.0, 1.0, 2.0]);
        let result = loss_and_grad(&logits, &[1], 2).unwrap();
        let expected = -log_softmax(&logits)[1];
        assert!(
            (result.loss - expected).abs() < 1e-12,
            "{} vs {expected}",
            result.loss
        );
    }

    #[test]
    fn target_longer_than_the_frame_count_is_unalignable() {
        let logits = mat(2, 3, &[0.0, 1.0, 2.0, 0.5, 0.5, 0.5]);
        // "aaa" needs 5 frames: a, blank, a, blank, a.
        assert!(
            loss_and_grad(&logits, &[0, 0, 0], 2)
                .unwrap()
                .loss
                .is_infinite()
        );
    }

    #[test]
    fn repeated_labels_need_a_separating_blank() {
        let logits = mat(3, 3, &[0.0; 9]);
        let repeated = loss_and_grad(&logits, &[0, 0], 2).unwrap().loss;
        let distinct = loss_and_grad(&logits, &[0, 1], 2).unwrap().loss;
        // With uniform logits the only alignment of "aa" is a-blank-a, while
        // "ab" has three, so "aa" must be strictly less likely.
        assert!(repeated > distinct, "{repeated} vs {distinct}");
    }

    /// The gradient is the part most likely to be subtly wrong, and it is the
    /// only part the optimizer actually uses, so check it numerically.
    #[test]
    fn gradient_matches_central_differences() {
        let data: Vec<f32> = (0..4 * 5)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3)
            .collect();
        let logits = mat(4, 5, &data);
        let target = [0usize, 2, 0];
        let analytic = loss_and_grad(&logits, &target, 4).unwrap();

        let epsilon = 1e-4f32;
        for i in 0..data.len() {
            let mut bumped = data.clone();
            bumped[i] += epsilon;
            let up = loss_and_grad(&mat(4, 5, &bumped), &target, 4).unwrap().loss;
            bumped[i] -= 2.0 * epsilon;
            let down = loss_and_grad(&mat(4, 5, &bumped), &target, 4).unwrap().loss;
            let numeric = (up - down) / (2.0 * f64::from(epsilon));
            let delta = (numeric - analytic.grad[i]).abs();
            assert!(
                delta < 1e-4,
                "index {i}: analytic {} vs numeric {numeric}",
                analytic.grad[i]
            );
        }
    }
}
