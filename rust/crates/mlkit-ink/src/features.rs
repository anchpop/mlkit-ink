//! The 10 Bezier features, in the layout the native encoder actually uses.
//!
//! `CoeffsToFeaturesAnglesRatios` emits
//!
//! ```text
//! (pen_down, dx, dy, a1, d1, a2, d2, dt, start_time_leg, end_time_leg)
//! ```
//!
//! which is NOT the order printed in Carbune et al. 2020. Blind search over
//! plausible orderings topped out at 8 of 61 corpus inks; reading the encoder
//! out of the binary gave 61 of 61. Angles are signed radians, ratios use the
//! fixed corresponding endpoint, and the time features are computed straight
//! from the f32 power-basis coefficients so they never cancel against the
//! absolute timestamp.
//!

use alloc::vec::Vec;

use crate::curve::{Omega, Segmentation};
use crate::error::Result;
// The unit-test harness supplies std even with the std feature disabled.
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::ink::Stroke;
use crate::mat::Mat;
use crate::settings::{CurveSettings, PreprocessingStep};

/// One curve -> its feature row.
pub fn curve_features(omega: &Omega, pen_up: bool, cfg: &CurveSettings) -> Vec<f32> {
    let [p0, p1, p2, p3] = crate::curve::control_points_f32(omega);
    let (dx, dy) = (p3[0] - p0[0], p3[1] - p0[1]);
    let (vx, vy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let (wx, wy) = (p2[0] - p3[0], p2[1] - p3[1]);
    let length = (dx * dx + dy * dy).sqrt();
    // The native ratio guard is strictly L>0, not an epsilon comparison.
    let d1 = if length > 0.0 {
        (vx * vx + vy * vy).sqrt() / length
    } else {
        0.0
    };
    let d2 = if length > 0.0 {
        (wx * wx + wy * wy).sqrt() / length
    } else {
        0.0
    };
    // atan2f is unconditional, including signed zeros and degenerate chords.
    // The second angle uses the REVERSED chord, not the first angle's reference.
    let a1 = (dx * vy - dy * vx).atan2(dx * vx + dy * vy);
    let a2 = (dy * wx - dx * wy).atan2(-dx * wx - dy * wy);
    let mut values = alloc::vec![if pen_up { 0.0 } else { 1.0 }, dx, dy, a1, d1, a2, d2];
    if cfg.interpolate_time {
        let [g1, g2, g3] = [omega[1][2] as f32, omega[2][2] as f32, omega[3][2] as f32];
        // Avoid cancellation against absolute timestamps: operate on coefficients,
        // not control-point differences, preserving this f32 operation order.
        values.extend_from_slice(&[
            (g1 + g2) + g3,
            g1 / 3.0,
            (-g1 / 3.0 - (2.0 * g2) / 3.0) - g3,
        ]);
    }
    if cfg.normalize_outputs_to_zero_one {
        for i in [1, 2, 7, 8, 9] {
            if i < values.len() {
                values[i] = ((values[i] + 1.0) / 2.0).clamp(0.0, 1.0);
            }
        }
        for i in [3, 5] {
            let pi = core::f32::consts::PI;
            values[i] = ((values[i] + pi) / (2.0 * pi)).clamp(0.0, 1.0);
        }
        for i in [4, 6] {
            values[i] = values[i].clamp(0.0, 1.0);
        }
    }
    values
}

/// Preprocessed strokes plus a fixed segmentation -> `[T, num_features]`.
pub fn encode(strokes: &[Stroke], seg: &Segmentation, cfg: &CurveSettings) -> Result<Mat> {
    ensure!(
        cfg.use_angles_ratios,
        Unsupported,
        "only angles/ratios curve features are implemented"
    );
    ensure!(
        !cfg.generate_second_order_features,
        Unsupported,
        "second-order curve features are not implemented"
    );
    ensure!(
        strokes.len() == seg.strokes.len(),
        Invalid,
        "segmentation has {} strokes for {} input strokes",
        seg.strokes.len(),
        strokes.len()
    );
    let mut data = Vec::with_capacity(seg.num_curves() * cfg.num_features());
    for (index, (stroke, structure)) in strokes.iter().zip(&seg.strokes).enumerate() {
        let raw = crate::curve::stroke_points(stroke, index)?;
        let mut points = Vec::with_capacity(structure.kept.len());
        for &i in &structure.kept {
            let point = raw.get(i).ok_or_else(|| {
                err!(
                    Invalid,
                    "stroke {index}: kept index {i} outside {} samples",
                    raw.len()
                )
            })?;
            points.push(*point);
        }
        ensure!(
            structure.kept.windows(2).all(|w| w[0] < w[1]),
            Invalid,
            "stroke {index}: kept indices must increase strictly"
        );
        // This is intentionally repeated from segment: only combinatorial choices
        // are frozen; coordinates, time rescale, and coefficients remain numeric.
        crate::curve::normalize_curve_time(&mut points);
        for &(start, end) in &structure.curves {
            ensure!(
                start <= end && end < points.len(),
                Invalid,
                "stroke {index}: invalid inclusive curve range ({start}, {end}) for {} retained points",
                points.len()
            );
            let (omega, _, _) = crate::curve::fit_cubic(&points[start..=end]);
            data.extend(curve_features(&omega, structure.pen_up, cfg));
        }
    }
    Ok(Mat::from_vec(seg.num_curves(), cfg.num_features(), data))
}

/// Raw ink -> `[T, num_features]`. Runs preprocessing, segmentation and encoding.
pub fn extract(
    strokes: &[Stroke],
    cfg: &CurveSettings,
    pipeline: &[PreprocessingStep],
) -> Result<Mat> {
    // Python returns the correctly shaped empty matrix before consulting the pipeline.
    if strokes.is_empty() {
        return Ok(Mat::zeros(0, cfg.num_features()));
    }
    let strokes = crate::preprocess::run(strokes, pipeline)?;
    let seg = crate::curve::segment(&strokes, cfg)?;
    encode(&strokes, &seg, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ink::Point;

    #[test]
    fn two_point_pen_up_is_a_straight_line() {
        // Endpoints chosen so both f32 control-leg subtractions give 1/3
        // exactly; arbitrary translations incur the reference's f32 cancellation.
        let points = [Point::new(-1.0, 0.0, 0.0), Point::new(2.0, 0.0, 3.0)];
        let (omega, _, _) = crate::curve::fit_cubic(&points);
        assert_eq!(omega[2], [0.0; 3]);
        assert_eq!(omega[3], [0.0; 3]);
        let row = curve_features(&omega, true, &CurveSettings::handwriting());
        assert_eq!(row[0], 0.0);
        assert_eq!(row[4], 1.0f32 / 3.0);
        assert_eq!(row[6], 1.0f32 / 3.0);
        assert!(row[3].abs() < 1e-6);
        assert!(row[5].abs() < 1e-6);
    }

    #[test]
    fn native_layout_time_legs_and_normalization() {
        let w = [
            [0.0, 0.0, 1e20],
            [3.0, 0.0, 3.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 9.0],
        ];
        let mut cfg = CurveSettings::handwriting();
        let row = curve_features(&w, false, &cfg);
        assert_eq!(&row[7..], &[18.0, 1.0, -14.0]);
        assert_eq!(row[1], 3.0);
        assert_eq!(row[2], 0.0);
        cfg.interpolate_time = false;
        cfg.normalize_outputs_to_zero_one = true;
        let row = curve_features(&w, false, &cfg);
        assert_eq!(row.len(), 7);
        assert_eq!(row[1], 1.0);
        assert_eq!(row[2], 0.5);
        assert_eq!(row[3], 0.5);
        assert_eq!(row[5], 0.5);
    }

    #[test]
    fn fixed_segmentation_encodes_changed_coordinates_without_resegmenting() {
        let mut strokes = alloc::vec![Stroke::new(
            alloc::vec![0.0, 1.0, 2.0],
            alloc::vec![0.0, 0.0, 0.0],
            alloc::vec![0.0, 1.0, 2.0]
        )];
        let cfg = CurveSettings::handwriting();
        let seg = crate::curve::segment(&strokes, &cfg).unwrap();
        strokes[0].y[1] = 1.0;
        let result = encode(&strokes, &seg, &cfg).unwrap();
        assert_eq!(result.rows(), seg.num_curves());
        assert!(result[(0, 3)].abs() > 0.1);
    }

    #[test]
    fn empty_input_and_mismatched_timestamps_keep_reference_ordering() {
        let cfg = CurveSettings::handwriting();
        let unsupported = [PreprocessingStep::Unsupported {
            field: 99,
            name: "unknown".into(),
        }];
        assert_eq!(extract(&[], &cfg, &unsupported).unwrap(), Mat::zeros(0, 10));
        let stroke = Stroke::new(alloc::vec![0.0, 1.0], alloc::vec![0.0, 1.0], alloc::vec![]);
        assert!(extract(core::slice::from_ref(&stroke), &cfg, &[]).is_err());
        assert_eq!(
            extract(&[stroke], &cfg, &crate::settings::handwriting_pipeline())
                .unwrap()
                .rows(),
            1
        );
    }

    #[test]
    fn invalid_fixed_indices_return_errors_not_panics() {
        let strokes = [Stroke::new(
            alloc::vec![0.0, 1.0],
            alloc::vec![0.0, 1.0],
            alloc::vec![0.0, 1.0],
        )];
        let cfg = CurveSettings::handwriting();
        let mut seg = crate::curve::segment(&strokes, &cfg).unwrap();
        seg.strokes[0].kept[1] = 2;
        assert!(encode(&strokes, &seg, &cfg).is_err());
        seg.strokes[0].kept[1] = 0;
        assert!(encode(&strokes, &seg, &cfg).is_err());
        seg.strokes[0].kept[1] = 1;
        seg.strokes[0].curves[0] = (1, 0);
        assert!(encode(&strokes, &seg, &cfg).is_err());
        assert!(encode(&[], &seg, &cfg).is_err());
    }
}
