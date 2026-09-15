//! Thin, split, fit and merge: raw samples -> a sequence of cubic Beziers.
//!
//! The discrete decisions (which samples survive thinning, where a stroke is
//! cut) are separated from the numeric ones (the least-squares fit) and
//! returned as a [`Segmentation`]. That split is what makes
//! [`crate::optimize`] possible: gradients flow through the numeric stages
//! while the combinatorial structure is held fixed and recomputed between
//! steps. It also makes the fitter testable a stage at a time.
//!

use alloc::vec::Vec;

use crate::error::Result;
// The unit-test harness supplies std even with the std feature disabled.
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::ink::{Point, Stroke};
use crate::settings::CurveSettings;

/// Power-basis coefficients of one cubic: `p(s) = w[0] + w[1] s + w[2] s^2 + w[3] s^3`,
/// with the three components being (x, y, t).
///
/// f64 because the least-squares solve runs in f64; consumers that mirror a
/// native f32 computation cast on the way in.
pub type Omega = [[f64; 3]; 4];

/// Power basis -> Bezier control points.
pub fn control_points(omega: &Omega) -> [[f64; 3]; 4] {
    let [a0, a1, a2, a3] = *omega;
    let mut out = [[0.0; 3]; 4];
    for d in 0..3 {
        out[0][d] = a0[d];
        out[1][d] = a0[d] + a1[d] / 3.0;
        out[2][d] = a0[d] + 2.0 * a1[d] / 3.0 + a2[d] / 3.0;
        out[3][d] = a0[d] + a1[d] + a2[d] + a3[d];
    }
    out
}

/// Which samples of one preprocessed stroke survive, and how they group into curves.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeSegments {
    pub pen_up: bool,
    /// Indices into the preprocessed stroke that survived thinning, ascending.
    pub kept: Vec<usize>,
    /// Inclusive `[start, end]` ranges into `kept`. Adjacent curves share an
    /// endpoint, exactly as the native recursive split does.
    pub curves: Vec<(usize, usize)>,
}

/// The discrete structure chosen for a whole ink.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Segmentation {
    pub strokes: Vec<StrokeSegments>,
}

impl Segmentation {
    /// Total number of curves, i.e. the time dimension of the feature matrix.
    pub fn num_curves(&self) -> usize {
        self.strokes.iter().map(|s| s.curves.len()).sum()
    }
}

/// Choose thinning and split points for every stroke of a *preprocessed* ink.
pub fn segment(strokes: &[Stroke], cfg: &CurveSettings) -> Result<Segmentation> {
    let mut segmentation = Segmentation::default();
    for (index, stroke) in strokes.iter().enumerate() {
        let raw = stroke_points(stroke, index)?;
        // The RAW whole-stroke scale survives thinning, every split, and every merge.
        let diagonal = crate::ink::bbox_diagonal(&raw) as f64;
        let kept = thin_points(&raw);
        let mut points: Vec<_> = kept.iter().map(|&i| raw[i]).collect();
        normalize_curve_time(&mut points);
        let mut curves = Vec::new();
        if !points.is_empty() {
            split_points(&points, 0, cfg, diagonal, &mut curves);
        }
        loop {
            let mut merged = Vec::new();
            let mut i = 0;
            let mut changed = false;
            while i < curves.len() {
                if i + 1 < curves.len() {
                    let range = (curves[i].0, curves[i + 1].1);
                    let combined = &points[range.0..=range.1];
                    let (omega, _, s) = fit_cubic(combined);
                    if fit_failures(combined, &omega, &s, cfg, diagonal) == (false, false) {
                        merged.push(range);
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
                merged.push(curves[i]);
                i += 1;
            }
            curves = merged;
            if !changed {
                break;
            }
        }
        segmentation.strokes.push(StrokeSegments {
            pen_up: stroke.pen_up,
            kept,
            curves,
        });
    }
    Ok(segmentation)
}

pub(crate) fn stroke_points(stroke: &Stroke, index: usize) -> Result<Vec<Point>> {
    ensure!(
        stroke.is_timed(),
        Invalid,
        "Malformed input (x.size != t.size). stroke={index}: x={}, y={}, t={}",
        stroke.x.len(),
        stroke.y.len(),
        stroke.t.len()
    );
    Ok((0..stroke.len())
        .map(|i| Point::new(stroke.x[i] as f32, stroke.y[i] as f32, stroke.t[i] as f32))
        .collect())
}

/// Native maximum-cardinality minimum-spacing chain, including its DP tie order.
pub fn thin_points(points: &[Point]) -> Vec<usize> {
    let n = points.len();
    if n < 2 {
        return (0..n).collect();
    }
    let mut cumulative = alloc::vec![0.0f32; n];
    for i in 1..n {
        cumulative[i] = cumulative[i - 1] + distance(points[i], points[i - 1]);
    }
    let e = crate::ink::bbox_diagonal(points) * 0.000_345_266_98_f32;
    let e2 = e * e;
    let mut length = alloc::vec![1; n];
    let mut pred = alloc::vec![0; n];
    let mut prefixmax = alloc::vec![1; n];
    let mut best = 0;
    for i in 1..n {
        pred[i] = i;
        for j in (0..i).rev() {
            if length[i] > prefixmax[j] + 1 {
                break;
            }
            let candidate = length[j] + 1;
            if candidate < length[i] {
                continue;
            }
            // Ordered strict gates skip NaNs too.
            if (cumulative[i] - cumulative[j]).partial_cmp(&e) != Some(core::cmp::Ordering::Greater)
            {
                continue;
            }
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            if (dx * dx + dy * dy).partial_cmp(&e2) != Some(core::cmp::Ordering::Greater) {
                continue;
            }
            // Equal predecessor chains prefer earliest j; final ties prefer latest i.
            // If no valid transition exists, index zero is retained.
            length[i] = candidate;
            pred[i] = j;
            if length[i] >= length[best] {
                best = i;
            }
        }
        prefixmax[i] = length[best];
    }
    let mut indices = Vec::with_capacity(length[best]);
    let mut index = best;
    for _ in 0..length[best] {
        indices.push(index);
        index = pred[index];
    }
    indices.reverse();
    indices
}

fn distance(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

// NumPy's contiguous sum uses eight accumulators and pairwise blocks of 128.
// A left-to-right f32 sum changes the time fit, hence sometimes segmentation.
fn sum_f32(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(-0.0, |a, &b| a + b);
    }
    if n <= 128 {
        let mut r: [f32; 8] = values[..8].try_into().unwrap();
        let end = n - n % 8;
        for chunk in values[8..end].as_chunks::<8>().0 {
            for j in 0..8 {
                r[j] += chunk[j];
            }
        }
        let mut result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[end..] {
            result += v;
        }
        result
    } else {
        let half = (n / 2) / 8 * 8;
        sum_f32(&values[..half]) + sum_f32(&values[half..])
    }
}

/// Native per-stroke fitter rescale, after thinning and before splitting.
pub fn normalize_curve_time(points: &mut [Point]) {
    if points.is_empty() {
        return;
    }
    let duration = points[points.len() - 1].t - points[0].t;
    // Nonpositive/NaN duration zeroes the ENTIRE column. No local origin subtraction.
    if duration > 0.0 {
        let distances: Vec<_> = points.windows(2).map(|p| distance(p[1], p[0])).collect();
        let scale = sum_f32(&distances) / duration;
        for p in points {
            p.t *= scale;
        }
    } else {
        for p in points {
            p.t = 0.0;
        }
    }
}

fn vandermonde(s: f64) -> [f64; 4] {
    [1.0, s, s * s, s * s * s]
}

/// Householder QR, applying each reflector to all three right-hand sides.
/// For N<4 fit degree N-1 and zero-pad, NOT the cubic minimum-norm solution:
/// that made two-point pen-up d1=0.111, d2=0.667 instead of a straight line.
fn solve_coeffs(points: &[Point], s: &[f64]) -> Omega {
    let n = points.len();
    let mut omega = [[0.0; 3]; 4];
    if n == 0 {
        return omega;
    }
    if n == 1 {
        omega[0][0] = points[0].x as f64;
        omega[0][1] = points[0].y as f64;
        // Native singleton path leaves the entire time polynomial zero.
        return omega;
    }
    let cols = n.min(4);
    let mut a: Vec<_> = s.iter().map(|&s| vandermonde(s)).collect();
    let mut b: Vec<_> = points
        .iter()
        .map(|p| [p.x as f64, p.y as f64, p.t as f64])
        .collect();
    for k in 0..cols {
        let norm = a[k..].iter().map(|r| r[k] * r[k]).sum::<f64>().sqrt();
        let alpha = -norm.copysign(a[k][k]);
        let mut v: Vec<_> = a[k..].iter().map(|r| r[k]).collect();
        v[0] -= alpha;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        if vv == 0.0 {
            continue;
        }
        for j in k..cols {
            let scale = 2.0 * v.iter().zip(&a[k..]).map(|(v, r)| v * r[j]).sum::<f64>() / vv;
            for (r, v) in a[k..].iter_mut().zip(&v) {
                r[j] -= scale * v;
            }
        }
        for d in 0..3 {
            let scale = 2.0 * v.iter().zip(&b[k..]).map(|(v, r)| v * r[d]).sum::<f64>() / vv;
            for (r, v) in b[k..].iter_mut().zip(&v) {
                r[d] -= scale * v;
            }
        }
        a[k][k] = alpha;
    }
    for k in (0..cols).rev() {
        for d in 0..3 {
            let tail: f64 = ((k + 1)..cols).map(|j| a[k][j] * omega[j][d]).sum();
            omega[k][d] = (b[k][d] - tail) / a[k][k];
        }
    }
    omega
}

fn evaluate(omega: &Omega, basis: [f64; 4]) -> [f64; 3] {
    core::array::from_fn(|d| {
        ((basis[0] * omega[0][d] + basis[1] * omega[1][d]) + basis[2] * omega[2][d])
            + basis[3] * omega[3][d]
    })
}

fn sse(points: &[Point], omega: &Omega, s: &[f64]) -> f64 {
    points
        .iter()
        .zip(s)
        .map(|(p, &s)| {
            let q = evaluate(omega, vandermonde(s));
            [p.x as f64 - q[0], p.y as f64 - q[1], p.t as f64 - q[2]]
                .iter()
                .map(|r| r * r)
                .sum::<f64>()
        })
        .sum()
}

fn initial_s(points: &[Point]) -> Vec<f64> {
    let mut s = alloc::vec![0.0; points.len()];
    // Although the parameters are f64, Python receives f32 points here:
    // diff/hypot/cumsum stay f32, then concatenate([[0.0], cumulative])
    // widens to f64 BEFORE the division. Widening earlier changes the fit.
    let mut cumulative = 0.0f32;
    for i in 1..points.len() {
        let dx = points[i].x - points[i - 1].x;
        let dy = points[i].y - points[i - 1].y;
        cumulative += dx.hypot(dy);
        s[i] = cumulative as f64;
    }
    let total = s.last().copied().unwrap_or(0.0);
    let n = s.len();
    // Deliberate safety divergence: native all-identical points divide by zero.
    // Keep finite uniform parameters for zero or numerically tiny extent.
    for (i, v) in s.iter_mut().enumerate() {
        *v = if total > 1e-12 {
            *v / total
        } else if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
    }
    s
}

/// Fit one cubic by alternating f64 least squares with Newton reprojection.
/// Returns coefficients, SSE (including time), and final parameters.
pub fn fit_cubic(points: &[Point]) -> (Omega, f64, Vec<f64>) {
    let mut s = initial_s(points);
    let mut omega = solve_coeffs(points, &s);
    let mut error = sse(points, &omega, &s);
    if points.len() <= 1 {
        return (omega, error, s);
    }
    for _ in 0..8 {
        let updated: Vec<_> = points
            .iter()
            .zip(&s)
            .map(|(p, &s)| {
                let q = evaluate(&omega, vandermonde(s));
                let d1 = evaluate(&omega, [0.0, 1.0, 2.0 * s, 3.0 * s * s]);
                let d2 = evaluate(&omega, [0.0, 0.0, 2.0, 6.0 * s]);
                let r = [p.x as f64 - q[0], p.y as f64 - q[1]];
                let f = d1[0] * r[0] + d1[1] * r[1];
                let fp = (d2[0] * r[0] + d2[1] * r[1]) - (d1[0] * d1[0] + d1[1] * d1[1]);
                let step = if fp.abs() > 1e-12 { f / fp } else { 0.0 };
                (s - step).clamp(0.0, 1.0)
            })
            .collect();
        let next = solve_coeffs(points, &updated);
        let next_error = sse(points, &next, &updated);
        // Stop overshooting Newton steps rather than letting alternating minimisation wander.
        if next_error > error - 1e-12 {
            break;
        }
        s = updated;
        omega = next;
        error = next_error;
    }
    (omega, error, s)
}

// Casting the OUTPUT of control_points would be too late: these operations must be f32.
pub(crate) fn control_points_f32(omega: &Omega) -> [[f32; 3]; 4] {
    let w = omega.map(|r| r.map(|x| x as f32));
    let mut out = [[0.0; 3]; 4];
    for d in 0..3 {
        out[0][d] = w[0][d];
        out[1][d] = w[0][d] + w[1][d] / 3.0;
        out[2][d] = w[0][d] + 2.0 * w[1][d] / 3.0 + w[2][d] / 3.0;
        out[3][d] = w[0][d] + w[1][d] + w[2][d] + w[3][d];
    }
    out
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm(a: [f32; 2]) -> f32 {
    dot(a, a).sqrt()
}

fn fit_failures(
    points: &[Point],
    omega: &Omega,
    s: &[f64],
    cfg: &CurveSettings,
    diagonal: f64,
) -> (bool, bool) {
    let mut max_squared = 0.0f64;
    let mut total = 0.0;
    for (p, &s) in points.iter().zip(s) {
        let q = evaluate(omega, vandermonde(s));
        let dx = p.x as f64 - q[0];
        let dy = p.y as f64 - q[1];
        let squared = dx * dx + dy * dy;
        if squared > max_squared || squared.is_nan() {
            max_squared = squared;
        }
        total += squared;
    }
    // Time influences the optimizer, but never these strict spatial gates.
    let wrong = max_squared.sqrt() > cfg.tol2 * diagonal
        || (total / points.len() as f64).sqrt() > cfg.tol3 * diagonal;
    if points.len() < 4 {
        return (wrong, false);
    }
    let [p0, p1, p2, p3] = control_points_f32(omega);
    let v = sub(p1, p0);
    let middle = sub(p2, p1);
    let w = sub(p2, p3);
    let (lv, lm, lw) = (norm(v), norm(middle), norm(w));
    let chord = norm(sub(p3, p0));
    let ratio = ((lv + lm) + lw) / chord;
    let a = dot(v, middle) / lv;
    let b = -dot(w, middle) / lw;
    // x86 MINSS chooses its source operand a when either operand is NaN.
    // Neither minimum nor fmin has these NaN semantics.
    let minimum = if b < a { b } else { a };
    let score = minimum / lm;
    // NumPy f32 scalars also round Python-float thresholds to f32 for comparison.
    // Ordered comparisons reject zero/NaN chords and unordered ratios/cosines.
    let bend = !(chord > 0.0
        && ratio <= cfg.max_arc_ratio as f32
        && score >= cfg.split_cos_threshold as f32);
    (wrong, bend)
}

fn split_at_min_angle(points: &[Point], radius: f64) -> Option<usize> {
    // NumPy 2 weak promotion compares its f32 norm with the radius rounded to f32.
    let radius = radius as f32;
    let mut best_score = 1.0f32;
    let mut best = None;
    for i in 1..points.len() - 1 {
        let mut left = i as isize - 1;
        let mut right = i + 1;
        while left >= 0 && distance(points[i], points[left as usize]) < radius {
            left -= 1;
        }
        while right < points.len() && distance(points[right], points[i]) < radius {
            right += 1;
        }
        if left < 0 || right >= points.len() {
            continue;
        }
        let a = [
            points[i].x - points[left as usize].x,
            points[i].y - points[left as usize].y,
        ];
        let b = [points[right].x - points[i].x, points[right].y - points[i].y];
        let score = dot(a, b) / (norm(a) * norm(b));
        // Strict comparison gives earliest argmin; no score below 1 means no corner.
        if score < best_score {
            best_score = score;
            best = Some(i);
        }
    }
    best
}

fn split_at_max_curvature(omega: &Omega, s: &[f64]) -> usize {
    let s: Vec<_> = s.iter().map(|&v| v as f32).collect();
    let w = omega.map(|r| r.map(|v| v as f32));
    let (a, b) = (s[1], s[s.len() - 2]);
    let h = (b - a) / 99.0f32;
    let mut best = -1.0f32;
    let mut star = a;
    for j in 0..100 {
        let t = if a.abs() <= b.abs() {
            if j == 99 { b } else { a + j as f32 * h }
        } else if j == 0 {
            a
        } else {
            b + (j as f32 - 99.0) * h
        };
        let d1: [f32; 2] = core::array::from_fn(|d| {
            ((0.0 * w[0][d] + w[1][d]) + (2.0 * t) * w[2][d]) + (3.0 * t * t) * w[3][d]
        });
        let d2: [f32; 2] = core::array::from_fn(|d| {
            ((0.0 * w[0][d] + 0.0 * w[1][d]) + 2.0 * w[2][d]) + (6.0 * t) * w[3][d]
        });
        let num = (d1[0] * d2[1] - d1[1] * d2[0]).abs();
        let speed2 = dot(d1, d1);
        let den = ((speed2 * speed2) * speed2).sqrt();
        // Deliberate safety divergence from native unguarded division: degenerate
        // samples get zero curvature, not NaN. Even all-zero curvature schedules a split.
        let kappa = if den > 0.0 { num / den } else { 0.0 };
        if kappa > best || (kappa.is_nan() && !best.is_nan()) {
            best = kappa;
            star = t;
        }
    }
    let mut index = 1;
    let mut nearest = (s[1] - star).abs();
    for (i, &t) in s.iter().enumerate().take(s.len() - 1).skip(2) {
        let delta = (t - star).abs();
        if delta < nearest || (delta.is_nan() && !nearest.is_nan()) {
            nearest = delta;
            index = i;
        }
    }
    index
}

fn split_points(
    points: &[Point],
    start: usize,
    cfg: &CurveSettings,
    diagonal: f64,
    out: &mut Vec<(usize, usize)>,
) {
    // N<4 bypasses the geometry gate, including non-monotone quadratics.
    if points.len() < 4 {
        out.push((start, start + points.len() - 1));
        return;
    }
    let (omega, _, s) = fit_cubic(points);
    let (wrong, bend) = fit_failures(points, &omega, &s, cfg, diagonal);
    let corner = if wrong {
        split_at_min_angle(points, cfg.tol1 * diagonal)
    } else {
        None
    };
    let index = match corner {
        Some(i) => i,
        // No eligible residual corner is insufficient to accept: fall through
        // to the independent geometry gate, then always split on rejection.
        None if bend => split_at_max_curvature(&omega, &s),
        None => {
            out.push((start, start + points.len() - 1));
            return;
        }
    }
    .clamp(1, points.len() - 2);
    split_points(&points[..=index], start, cfg, diagonal, out);
    split_points(&points[index..], start + index, cfg, diagonal, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_exact_cubic() {
        let expected = [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 4.0],
            [3.0, 2.0, -2.0],
            [4.0, 1.0, 1.0],
        ];
        let s = [0.0, 0.25, 0.5, 0.75, 1.0];
        let points: Vec<_> = s
            .iter()
            .map(|&s| {
                let p = evaluate(&expected, vandermonde(s));
                Point::new(p[0] as f32, p[1] as f32, p[2] as f32)
            })
            .collect();
        let actual = solve_coeffs(&points, &s);
        for (a, e) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!((a - e).abs() < 1e-12, "{actual:?}");
        }
    }

    #[test]
    fn thinning_ties_and_stationary_points() {
        let p: Vec<_> = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
            .iter()
            .map(|&x| Point::new(x, 0.0, 0.0))
            .collect();
        assert_eq!(thin_points(&p), alloc::vec![0, 2, 5]);
        assert_eq!(thin_points(&[p[0]; 4]), alloc::vec![0]);
        assert!(thin_points(&[]).is_empty());
    }

    #[test]
    fn control_points_round_trip() {
        let w = [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 4.0],
            [3.0, 2.0, -2.0],
            [4.0, 1.0, 1.0],
        ];
        let p = control_points(&w);
        for d in 0..3 {
            let actual = [
                p[0][d],
                3.0 * (p[1][d] - p[0][d]),
                3.0 * (p[2][d] - 2.0 * p[1][d] + p[0][d]),
                p[3][d] - 3.0 * p[2][d] + 3.0 * p[1][d] - p[0][d],
            ];
            for k in 0..4 {
                assert!((actual[k] - w[k][d]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn time_rescale_keeps_global_origin_and_zeroes_bad_duration() {
        let mut p = [Point::new(0.0, 0.0, 20.0), Point::new(3.0, 4.0, 40.0)];
        normalize_curve_time(&mut p);
        assert_eq!([p[0].t, p[1].t], [5.0, 10.0]);
        p[1].t = f32::NAN;
        normalize_curve_time(&mut p);
        assert_eq!([p[0].t, p[1].t], [0.0, 0.0]);
    }

    #[test]
    fn corner_and_curvature_ties_choose_earliest() {
        let p = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 1.0),
            Point::new(1.0, 1.0, 2.0),
            Point::new(2.0, 1.0, 3.0),
        ];
        assert_eq!(split_at_min_angle(&p, 0.0), Some(1));
        assert_eq!(split_at_min_angle(&p, 10.0), None);
        let line = [[0.0; 3], [1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]];
        assert_eq!(split_at_max_curvature(&line, &[0.0, 0.25, 0.75, 1.0]), 1);
        assert_eq!(
            split_at_max_curvature(&[[0.0; 3]; 4], &[0.0, 0.25, 0.75, 1.0]),
            1
        );
    }

    #[test]
    fn geometry_gate_keeps_minss_nan_source_semantics() {
        let cfg = CurveSettings::handwriting();
        let points = [Point::default(); 4];
        let s = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        // P0=0, P1=1, P2=P3=2 gives a finite source a and NaN b.
        // MINSS retains a, so a zero final leg alone does not reject geometry.
        let zero_last_leg = [[0.0; 3], [3.0, 0.0, 0.0], [0.0; 3], [-1.0, 0.0, 0.0]];
        assert!(!fit_failures(&points, &zero_last_leg, &s, &cfg, 1000.0).1);
        // P0=P1=0, P2=1, P3=2 makes a NaN: reversing min operands would accept.
        let zero_first_leg = [[0.0; 3], [0.0; 3], [3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        assert!(fit_failures(&points, &zero_first_leg, &s, &cfg, 1000.0).1);
        assert!(fit_failures(&points, &[[0.0; 3]; 4], &s, &cfg, 1000.0).1);
        assert!(!fit_failures(&points[..3], &[[0.0; 3]; 4], &s[..3], &cfg, 1000.0).1);
    }

    #[test]
    fn singleton_zeros_time_polynomial_and_includes_time_in_sse() {
        let (omega, error, s) = fit_cubic(&[Point::new(2.0, 3.0, 4.0)]);
        assert_eq!(omega, [[2.0, 3.0, 0.0], [0.0; 3], [0.0; 3], [0.0; 3]]);
        assert_eq!(error, 16.0);
        assert_eq!(s, alloc::vec![0.0]);
    }
}
