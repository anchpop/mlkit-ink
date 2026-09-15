//! Input ink: strokes of (x, y, t) samples.
//!
//! Timestamps are deliberately optional and are *not* validated on
//! construction. A stroke whose `t` length disagrees with its `x` length is the
//! native trigger to regenerate timestamps for the entire ink (see
//! [`crate::preprocess::hallucinate_time`]), not an input error, so the
//! mismatch has to survive until preprocessing runs.

use alloc::vec::Vec;

// `cfg(test)` because the unit-test harness links `std` even when the feature
// is off, which would make this import dead and trip clippy.
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;

/// One pen trace.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stroke {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Empty when the caller supplied no timestamps.
    #[cfg_attr(feature = "serde", serde(default))]
    pub t: Vec<f64>,
    /// True for the synthetic strokes that bridge a gap between two real ones.
    /// Their `pen_down` feature is 0.0 where a real trace has 1.0.
    #[cfg_attr(feature = "serde", serde(default))]
    pub pen_up: bool,
}

impl Stroke {
    pub fn new(x: Vec<f64>, y: Vec<f64>, t: Vec<f64>) -> Self {
        Stroke {
            x,
            y,
            t,
            pen_up: false,
        }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// True once `t` has been completed by preprocessing.
    pub fn is_timed(&self) -> bool {
        self.t.len() == self.x.len() && self.x.len() == self.y.len()
    }
}

/// One (x, y, t) sample, in the f32 precision the native fitter uses.
///
/// The fitter is float32 throughout; only the least-squares solve steps up to
/// f64. Keeping that split explicit in the types is the cheapest defence
/// against silently "improving" the precision and diverging from the SDK.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub t: f32,
}

impl Point {
    pub fn new(x: f32, y: f32, t: f32) -> Self {
        Point { x, y, t }
    }
}

/// Bounding-box diagonal of the spatial extent, the scale every acceptance
/// gate is measured against. Computed over the *whole original stroke* and
/// carried down the split recursion unchanged.
pub fn bbox_diagonal(points: &[Point]) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    // `f32::min`/`max` discard NaN; numpy's `ptp` propagates it. That
    // difference is observable: a NaN diagonal makes every downstream
    // threshold comparison false, so the reference keeps a single thinned
    // point where silently-finite bounds would keep several. Comparisons are
    // written the long way to preserve it.
    let (mut x0, mut x1) = (points[0].x, points[0].x);
    let (mut y0, mut y1) = (points[0].y, points[0].y);
    for p in points {
        x0 = if p.x < x0 || p.x.is_nan() { p.x } else { x0 };
        x1 = if p.x > x1 || p.x.is_nan() { p.x } else { x1 };
        y0 = if p.y < y0 || p.y.is_nan() { p.y } else { y0 };
        y1 = if p.y > y1 || p.y.is_nan() { p.y } else { y1 };
    }
    ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)).sqrt()
}
