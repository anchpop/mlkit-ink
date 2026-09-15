//! The named preprocessing steps the recospec's `InkPreprocessorSpec` lists.
//!
//! Modelled as a driver over an ordered list, exactly like the proto, so a spec
//! that uses one of the other ~26 native steps needs a new match arm here and
//! nothing else.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Result;
use crate::ink::Stroke;
use crate::settings::PreprocessingStep;

/// Subtract the first stroke's first timestamp once, in f32, across all
/// strokes. Native preprocessing neither scales time nor repairs its ordering;
/// the path-length rescale lives in the fitter instead.
pub fn normalize_time(strokes: &[Stroke]) -> Vec<Stroke> {
    let Some(&origin) = strokes.first().and_then(|s| s.t.first()) else {
        return strokes.to_vec();
    };
    let origin = origin as f32;
    let mut out = strokes.to_vec();
    for stroke in &mut out {
        for t in &mut stroke.t {
            *t = (*t as f32 - origin) as f64;
        }
    }
    out
}

/// If any stroke's timestamp count mismatches its point count, regenerate all
/// timestamps at a global fixed step. Matching arrays are untouched even when
/// times are constant, decreasing or nonfinite.
pub fn hallucinate_time(strokes: &[Stroke], interval: f32, force: bool) -> Vec<Stroke> {
    let mut out = strokes.to_vec();
    if !force && strokes.iter().all(|s| s.t.len() == s.len()) {
        return out;
    }
    let mut index = 0;
    for stroke in &mut out {
        stroke.t = hallucinated_times(index, stroke.len(), interval);
        index += stroke.len();
    }
    out
}

fn hallucinated_times(index: usize, len: usize, interval: f32) -> Vec<f64> {
    // np.arange(start, stop, dtype=float32) uses the rounded start and the
    // difference of its first two values as its step, even beyond 2^24.
    let start = index as f32;
    let step = (index + 1) as f32 - start;
    (0..len)
        .map(|i| ((start + i as f32 * step) * interval) as f64)
        .collect()
}

/// Native no-guide normalization: x/y only, divisor `max(height, width / 100)`.
/// Like the Python reference, use f32 division, not reciprocal multiplication.
/// Coordinate lengths must agree; [`run`] validates this for pipeline callers.
pub fn normalize_size(strokes: &[Stroke], margin: f32, first_point_origin: bool) -> Vec<Stroke> {
    let Some(first) = strokes.iter().find(|s| !s.is_empty()) else {
        return strokes.to_vec();
    };
    let nonempty = || strokes.iter().filter(|s| !s.is_empty());
    let (xmin, xmax) = coordinate_bounds(nonempty().flat_map(|s| s.x.iter().copied()));
    let (ymin, ymax) = coordinate_bounds(nonempty().flat_map(|s| s.y.iter().copied()));
    let mut height = ymax - ymin;
    let width_height = (xmax - xmin) / 100.0;
    // Python's max(height, width / 100) keeps height if either comparison
    // operand is NaN; f32::max would instead discard a NaN height.
    if width_height > height {
        height = width_height;
    }
    if height < f32::EPSILON {
        height = 1.0;
    }
    let margin = margin * height;
    let denominator = height + 2.0 * margin;
    let y0 = ymin - margin;
    let x0 = if first_point_origin {
        first.x[0] as f32
    } else {
        xmin - margin
    };
    let mut out = strokes.to_vec();
    for stroke in &mut out {
        for x in &mut stroke.x {
            *x = ((*x as f32 - x0) / denominator) as f64;
        }
        for y in &mut stroke.y {
            *y = ((*y as f32 - y0) / denominator) as f64;
        }
    }
    out
}

fn coordinate_bounds(values: impl Iterator<Item = f64>) -> (f32, f32) {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in values {
        let value = value as f32;
        // NumPy reductions propagate NaNs rather than ignoring them.
        if value <= min || value.is_nan() {
            min = value;
        }
        if value >= max || value.is_nan() {
            max = value;
        }
    }
    (min, max)
}

/// Insert synthetic strokes bridging each stroke's end to the next one's start.
/// Empty strokes are removed and all retained input strokes become pen-down.
/// Coordinate lengths must agree; [`run`] validates this for pipeline callers.
pub fn add_pen_up_strokes(strokes: &[Stroke]) -> Vec<Stroke> {
    let mut real = strokes.iter().filter(|s| !s.is_empty()).peekable();
    let mut out = Vec::new();
    while let Some(stroke) = real.next() {
        let mut down = stroke.clone();
        down.pen_up = false;
        out.push(down);
        if let Some(next) = real.peek() {
            let last = stroke.len() - 1;
            let t = match (stroke.t.last(), next.t.first()) {
                (Some(&end), Some(&start)) => vec![end, start],
                _ => Vec::new(),
            };
            out.push(Stroke {
                x: vec![stroke.x[last], next.x[0]],
                y: vec![stroke.y[last], next.y[0]],
                t,
                pen_up: true,
            });
        }
    }
    out
}

/// Apply an ordered pipeline, rejecting mismatched x/y lengths but preserving
/// timestamp mismatches until a HallucinateTime step handles them.
pub fn run(strokes: &[Stroke], steps: &[PreprocessingStep]) -> Result<Vec<Stroke>> {
    for (index, stroke) in strokes.iter().enumerate() {
        ensure!(
            stroke.x.len() == stroke.y.len(),
            Invalid,
            "stroke {index}: x/y lengths differ ({} vs {})",
            stroke.x.len(),
            stroke.y.len()
        );
    }
    let mut out = strokes.to_vec();
    for step in steps {
        out = match step {
            PreprocessingStep::NormalizeTime => normalize_time(&out),
            PreprocessingStep::HallucinateTime { interval, force } => {
                hallucinate_time(&out, *interval, *force)
            }
            // Raw ink has no guide: native explicitly logs
            // "Ink doesn't have writing guide. Use NormalizeSize."
            PreprocessingStep::NormalizeSize {
                margin,
                first_point_origin,
            }
            | PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
                margin,
                first_point_origin,
            } => normalize_size(&out, *margin, *first_point_origin),
            PreprocessingStep::AddPenUpStrokes => add_pen_up_strokes(&out),
            PreprocessingStep::Unsupported { field, name } => {
                bail!(Unsupported, "preprocessing step {name:?} (field {field})")
            }
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::settings::handwriting_pipeline;

    fn stroke(x: &[f64], y: &[f64], t: &[f64]) -> Stroke {
        Stroke::new(x.to_vec(), y.to_vec(), t.to_vec())
    }

    #[test]
    fn normalize_time_uses_one_origin_and_f32_subtraction() {
        let mut input = vec![
            stroke(&[1.25, 2.5], &[3.75, 4.0], &[16_777_217.0, 16_777_219.0]),
            stroke(&[5.0, 6.0], &[7.0, 8.0], &[16_777_215.0]),
            stroke(&[9.0], &[10.0], &[]),
        ];
        input[1].pen_up = true;
        let out = normalize_time(&input);
        assert_eq!(out[0].t, vec![0.0, 4.0]);
        assert_eq!(out[1].t, vec![-1.0]);
        assert!(out[2].t.is_empty());
        for (before, after) in input.iter().zip(&out) {
            assert_eq!(before.x, after.x);
            assert_eq!(before.y, after.y);
            assert_eq!(before.pen_up, after.pen_up);
        }
        assert_eq!(input[0].t[0], 16_777_217.0);
    }

    #[test]
    fn normalize_time_does_not_seek_a_later_origin() {
        let input = vec![Stroke::default(), stroke(&[1.0], &[2.0], &[0.1])];
        assert_eq!(normalize_time(&input), input);
        assert!(normalize_time(&[]).is_empty());
        let input = vec![stroke(&[], &[], &[10.0]), stroke(&[1.0], &[2.0], &[11.0])];
        let out = normalize_time(&input);
        assert_eq!(out[0].t, vec![0.0]);
        assert_eq!(out[1].t, vec![1.0]);
    }

    #[test]
    fn hallucinate_time_regenerates_every_stroke_on_any_length_mismatch() {
        for bad_times in [vec![], vec![9.0, 10.0]] {
            let mut input = vec![
                stroke(&[1.0, 2.0], &[3.0, 4.0], &[100.0, 99.0]),
                Stroke::default(),
                stroke(&[5.0], &[6.0], &bad_times),
                stroke(&[7.0], &[8.0], &[200.0]),
            ];
            input[2].pen_up = true;
            let out = hallucinate_time(&input, 20.0, false);
            assert_eq!(out[0].t, vec![0.0, 20.0]);
            assert!(out[1].t.is_empty());
            assert_eq!(out[2].t, vec![40.0]);
            assert_eq!(out[3].t, vec![60.0]);
            for (before, after) in input.iter().zip(&out) {
                assert_eq!(before.x, after.x);
                assert_eq!(before.y, after.y);
                assert_eq!(before.pen_up, after.pen_up);
            }
        }
    }

    #[test]
    fn hallucinate_time_preserves_matching_arrays_bit_for_bit() {
        let input = vec![stroke(
            &[0.0; 7],
            &[0.0; 7],
            &[
                0.1,
                0.1,
                -2.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                -0.0,
            ],
        )];
        let out = hallucinate_time(&input, 20.0, false);
        for (before, after) in input[0].t.iter().zip(&out[0].t) {
            assert_eq!(before.to_bits(), after.to_bits());
        }
    }

    #[test]
    fn hallucinate_time_force_and_fractional_interval_use_f32() {
        let input = vec![stroke(&[0.0; 4], &[0.0; 4], &[9.0; 4])];
        let out = hallucinate_time(&input, 0.1, true);
        assert_eq!(
            out[0].t,
            vec![0.0, 0.1f32 as f64, 0.2f32 as f64, 0.3f32 as f64]
        );
        assert!(hallucinate_time(&[], 20.0, true).is_empty());
        // An empty stroke with extraneous timestamps still triggers regeneration.
        let input = vec![stroke(&[], &[], &[9.0]), stroke(&[0.0], &[0.0], &[8.0])];
        let out = hallucinate_time(&input, 20.0, false);
        assert!(out[0].t.is_empty());
        assert_eq!(out[1].t, vec![0.0]);
    }

    #[test]
    fn hallucinated_times_matches_numpy_arange_at_f32_integer_boundary() {
        assert_eq!(
            hallucinated_times(16_777_216, 3, 1.0),
            vec![16_777_216.0; 3]
        );
        assert_eq!(
            hallucinated_times(16_777_217, 3, 1.0),
            vec![16_777_216.0, 16_777_218.0, 16_777_220.0]
        );
    }

    #[test]
    fn normalize_size_uses_global_extent_first_nonempty_origin_and_margin() {
        let mut input = vec![
            Stroke::default(),
            stroke(&[4.0, 2.0], &[2.0, 6.0], &[0.1]),
            stroke(&[8.0], &[4.0], &[]),
        ];
        input[1].pen_up = true;
        let first = normalize_size(&input, 0.5, true);
        assert_eq!(first[1].x, vec![0.0, -0.25]);
        assert_eq!(first[1].y, vec![0.25, 0.75]);
        assert_eq!(first[2].x, vec![0.5]);
        assert_eq!(first[2].y, vec![0.5]);
        assert_eq!(first[1].t, input[1].t);
        assert!(first[1].pen_up);
        assert!(first[0].is_empty());
        let bbox = normalize_size(&input, 0.5, false);
        assert_eq!(bbox[1].x, vec![0.5, 0.25]);
        assert_eq!(bbox[2].x, vec![1.0]);
    }

    #[test]
    fn normalize_size_guards_stationary_and_tiny_ink_but_not_epsilon() {
        for height in [0.0, f32::EPSILON as f64 / 2.0] {
            let input = vec![stroke(&[3.0, 3.0], &[0.0, height], &[1.0, 2.0])];
            let out = normalize_size(&input, 0.0, true);
            assert_eq!(out[0].x, vec![0.0, 0.0]);
            assert_eq!(out[0].y, vec![0.0, height]);
        }
        let input = vec![stroke(&[0.0, 0.0], &[0.0, f32::EPSILON as f64], &[])];
        assert_eq!(normalize_size(&input, 0.0, true)[0].y, vec![0.0, 1.0]);
        let empty = vec![stroke(&[], &[], &[0.1])];
        assert_eq!(normalize_size(&empty, 0.0, true), empty);
        assert!(normalize_size(&[], 0.0, true).is_empty());
    }

    #[test]
    fn normalize_size_caps_horizontal_extent_and_divides_in_f32() {
        let horizontal = vec![stroke(&[0.0, 200.0], &[3.0, 3.0], &[])];
        let out = normalize_size(&horizontal, 0.0, true);
        assert_eq!(out[0].x, vec![0.0, 100.0]);
        assert_eq!(out[0].y, vec![0.0, 0.0]);
        let input = vec![stroke(&[0.0, 5.0], &[0.0, 6.0], &[0.1])];
        let out = normalize_size(&input, 0.0, true);
        assert_eq!(out[0].x[1], 0.8333333134651184);
        assert_ne!(out[0].x[1], (5.0f32 * (1.0f32 / 6.0)) as f64);
        assert_eq!(out[0].t, vec![0.1]);
        let input = vec![stroke(&[16_777_217.0, 16_777_219.0], &[0.0, 4.0], &[])];
        assert_eq!(normalize_size(&input, 0.0, true)[0].x, vec![0.0, 1.0]);
    }

    #[test]
    fn normalize_size_matches_numpy_nan_reductions_and_python_max() {
        let input = vec![stroke(&[0.0, 1.0], &[f64::NAN, 1.0], &[])];
        let out = normalize_size(&input, 0.0, true);
        assert!(out[0].x.iter().chain(&out[0].y).all(|v| v.is_nan()));
        let input = vec![stroke(&[0.0, f64::NAN], &[0.0, 1.0], &[])];
        let out = normalize_size(&input, 0.0, true);
        assert_eq!(out[0].x[0], 0.0);
        assert!(out[0].x[1].is_nan());
        assert_eq!(out[0].y, vec![0.0, 1.0]);
    }

    #[test]
    fn add_pen_up_strokes_filters_empty_and_resets_input_flags() {
        let mut input = vec![
            Stroke::default(),
            stroke(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]),
            Stroke::default(),
            stroke(&[7.0], &[8.0], &[9.0]),
            stroke(&[10.0], &[11.0], &[12.0]),
            Stroke::default(),
        ];
        input[1].pen_up = true;
        let out = add_pen_up_strokes(&input);
        assert_eq!(out.len(), 5);
        assert!(!out[0].pen_up);
        assert_eq!(
            out[1],
            Stroke {
                x: vec![2.0, 7.0],
                y: vec![4.0, 8.0],
                t: vec![6.0, 9.0],
                pen_up: true
            }
        );
        assert_eq!(out[2], input[3]);
        assert_eq!(
            out[3],
            Stroke {
                x: vec![7.0, 10.0],
                y: vec![8.0, 11.0],
                t: vec![9.0, 12.0],
                pen_up: true
            }
        );
        assert_eq!(out[4], input[4]);
        assert!(input[1].pen_up);
        assert!(add_pen_up_strokes(&[Stroke::default()]).is_empty());
        assert!(add_pen_up_strokes(&[]).is_empty());
    }

    #[test]
    fn add_pen_up_strokes_uses_available_time_endpoints_even_if_mismatched() {
        for (left, right, expected) in [
            (vec![1.0, 2.0, 3.0], vec![4.0], vec![3.0, 4.0]),
            (vec![], vec![4.0], vec![]),
            (vec![1.0], vec![], vec![]),
        ] {
            let input = vec![
                stroke(&[0.0, 1.0], &[0.0, 1.0], &left),
                stroke(&[2.0, 3.0], &[2.0, 3.0], &right),
            ];
            let out = add_pen_up_strokes(&input);
            assert_eq!(out[1].t, expected);
            assert_eq!(out[0].t, left);
            assert_eq!(out[2].t, right);
        }
    }

    #[test]
    fn run_applies_ordered_steps_and_writing_guide_fallback() {
        let input = vec![
            stroke(&[2.0, 4.0], &[1.0, 3.0], &[]),
            stroke(&[6.0], &[2.0], &[99.0]),
        ];
        let out = run(&input, &handwriting_pipeline()).unwrap();
        assert_eq!(
            out,
            vec![
                stroke(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 20.0]),
                Stroke {
                    x: vec![1.0, 2.0],
                    y: vec![1.0, 0.5],
                    t: vec![20.0, 40.0],
                    pen_up: true
                },
                stroke(&[2.0], &[0.5], &[40.0]),
            ]
        );
        let steps = [
            PreprocessingStep::AddPenUpStrokes,
            PreprocessingStep::HallucinateTime {
                interval: 2.0,
                force: false,
            },
        ];
        let reversed = run(&input, &steps).unwrap();
        assert_eq!(reversed[1].t, vec![4.0, 6.0]);
        assert_eq!(reversed[2].t, vec![8.0]);
        for first_point_origin in [true, false] {
            let normal = PreprocessingStep::NormalizeSize {
                margin: 0.25,
                first_point_origin,
            };
            let guide = PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
                margin: 0.25,
                first_point_origin,
            };
            assert_eq!(
                run(&input, &[normal]).unwrap(),
                run(&input, &[guide]).unwrap()
            );
        }
        assert_eq!(run(&input, &[]).unwrap(), input);
        assert!(run(&[], &handwriting_pipeline()).unwrap().is_empty());
    }

    #[test]
    fn run_rejects_invalid_xy_even_without_steps_but_allows_mismatched_times() {
        for invalid in [stroke(&[1.0], &[], &[]), stroke(&[], &[1.0], &[])] {
            let error = run(&[Stroke::default(), invalid], &[]).unwrap_err();
            assert!(
                matches!(error, Error::Invalid(ref message) if message.contains("stroke 1") && message.contains("x/y"))
            );
        }
        let input = vec![stroke(&[1.0, 2.0], &[3.0, 4.0], &[7.0])];
        assert_eq!(
            run(&input, &[PreprocessingStep::NormalizeTime]).unwrap()[0].t,
            vec![0.0]
        );
    }

    #[test]
    fn run_reports_unsupported_step_name_and_field_even_for_empty_ink() {
        let steps = [PreprocessingStep::Unsupported {
            field: 42,
            name: "unrecovered_step".into(),
        }];
        let error = run(&[], &steps).unwrap_err();
        assert!(
            matches!(error, Error::Unsupported(ref message) if message.contains("unrecovered_step") && message.contains("42"))
        );
    }
}
