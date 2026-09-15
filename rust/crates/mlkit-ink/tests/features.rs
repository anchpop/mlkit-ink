mod common;

use mlkit_ink::{CurveSettings, features, settings};

#[test]
fn all_74_feature_goldens() {
    let cfg = CurveSettings::handwriting();
    let pipeline = settings::handwriting_pipeline();
    let trials = common::trials();
    assert_eq!(trials.len(), 74);
    let mut matched_rows = 0;
    let mut shape_failures = Vec::new();
    let mut comparisons = Vec::new();
    let mut worst = (0.0f32, String::new(), 0usize, 0usize);
    for trial in trials {
        let ink_path = trial["ink"].as_str().expect("trial ink path");
        let ink = common::load_ink(ink_path);
        let actual = features::extract(&ink, &cfg, &pipeline).expect(ink_path);
        let rows = trial["features"].as_array().expect("feature rows");
        if actual.rows() != rows.len() {
            shape_failures.push(format!(
                "{ink_path}: got {} rows, want {}",
                actual.rows(),
                rows.len()
            ));
            continue;
        }
        matched_rows += 1;
        let expected: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                let row = row.as_array().expect("feature row");
                assert_eq!(row.len(), 10);
                row.iter()
                    .map(|v| v.as_f64().expect("feature value") as f32)
            })
            .collect();
        assert_eq!(actual.cols(), 10);
        for (i, (&a, &e)) in actual.as_slice().iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite(),
                "{ink_path}: nonfinite feature at row {}, column {}",
                i / 10,
                i % 10
            );
            let delta = (a - e).abs();
            if delta > worst.0 {
                worst = (delta, ink_path.to_owned(), i / 10, i % 10);
            }
        }
        comparisons.push((ink_path, actual, expected));
    }
    eprintln!(
        "{matched_rows}/74 exact row counts; worst feature deviation {} at {} row {}, column {}",
        worst.0, worst.1, worst.2, worst.3
    );
    assert!(
        shape_failures.is_empty(),
        "row count mismatches:\n{}",
        shape_failures.join("\n")
    );
    for (path, actual, expected) in comparisons {
        common::assert_close(actual.as_slice(), &expected, 1e-5, path);
    }
}
