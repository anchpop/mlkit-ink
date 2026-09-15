mod common;

use mlkit_ink::{Error, mat::Mat, net, tflite};
use std::path::Path;

#[test]
fn english_weights_and_all_golden_logits() {
    let Some(path) = common::model_path(common::EN_US_TFLITE) else {
        eprintln!("skipping network goldens: en-US TFLite model pack is absent");
        return;
    };
    let weights = tflite::load_weights(&std::fs::read(path).unwrap()).unwrap();
    let model = &common::goldens()["model"];
    assert_eq!(
        weights.layers.len(),
        model["layers"].as_u64().unwrap() as usize
    );
    assert_eq!(
        weights.input_size,
        model["input_size"].as_u64().unwrap() as usize
    );
    assert_eq!(
        weights.num_classes,
        model["num_classes"].as_u64().unwrap() as usize
    );
    for (i, layer) in weights.layers.iter().enumerate() {
        let hidden = model["hidden_sizes"][i].as_u64().unwrap() as usize;
        assert_eq!(layer.forward.hidden, hidden);
        assert_eq!(layer.backward.hidden, hidden);
        assert!(layer.forward.diagonal && layer.backward.diagonal);
        assert_eq!(
            layer.cell_clip,
            model["cell_clips"][i].as_f64().unwrap() as f32
        );
    }
    let empty = net::forward(&weights, &Mat::zeros(0, weights.input_size)).unwrap();
    assert_eq!((empty.rows(), empty.cols()), (0, weights.num_classes));

    let mut worst = (0.0f32, "", 0usize, 0.0f32, 0.0f32);
    let mut frames = 0;
    let mut matches = 0;
    let mut all_actual = Vec::new();
    let mut all_expected = Vec::new();
    for trial in common::trials() {
        let id = trial["id"].as_str().unwrap();
        let rows = trial["features"].as_array().unwrap();
        let features = Mat::from_rows(
            weights.input_size,
            rows.iter().map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            }),
        );
        let actual = net::forward(&weights, &features).unwrap();
        assert_eq!(
            actual.rows(),
            trial["logits_shape"][0].as_u64().unwrap() as usize
        );
        assert_eq!(
            actual.cols(),
            trial["logits_shape"][1].as_u64().unwrap() as usize
        );
        let expected = common::golden_logits(id);
        assert_eq!(actual.as_slice().len(), expected.len());
        assert!(
            actual.as_slice().iter().all(|x| x.is_finite()),
            "{id}: nonfinite logits"
        );
        for (i, (&a, &e)) in actual.as_slice().iter().zip(&expected).enumerate() {
            let delta = (a - e).abs();
            if delta > worst.0 {
                worst = (delta, id, i, a, e);
            }
        }
        for (frame, (a, e)) in actual
            .iter_rows()
            .zip(expected.chunks_exact(weights.num_classes))
            .enumerate()
        {
            frames += 1;
            if argmax(a) == argmax(e) {
                matches += 1;
            } else {
                eprintln!(
                    "{id} frame {frame}: argmax got {}, expected {}",
                    argmax(a),
                    argmax(e)
                );
            }
        }
        all_actual.extend_from_slice(actual.as_slice());
        all_expected.extend_from_slice(&expected);
    }
    eprintln!(
        "{} trials: worst absolute logit deviation {} at {} frame {} class {} (got {}, expected {}); argmax {matches}/{frames}",
        common::trials().len(),
        worst.0,
        worst.1,
        worst.2 / weights.num_classes,
        worst.2 % weights.num_classes,
        worst.3,
        worst.4
    );
    assert_eq!(matches, frames, "per-frame argmax agreement");
    common::assert_close(
        &all_actual,
        &all_expected,
        1e-3,
        &format!(
            "worst logit: {} frame {} class {}",
            worst.1,
            worst.2 / weights.num_classes,
            worst.2 % weights.num_classes
        ),
    );
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

#[test]
fn all_model_packs_load_only_handwriting_nets() {
    let root = common::repo_root().join("models");
    if !root.is_dir() {
        eprintln!("skipping model sweep: models directory is absent");
        return;
    }
    let mut files = Vec::new();
    collect_tflite(&root, &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!("skipping model sweep: no .tflite model packs are present");
        return;
    }
    let mut loaded = 0;
    let mut rejected = 0;
    for path in files {
        let name = path.to_string_lossy().to_lowercase();
        let special = ["emoji", "autodraw", "shapes", "scribe"]
            .iter()
            .any(|kind| name.contains(kind));
        let result = tflite::load_weights(&std::fs::read(&path).unwrap());
        if special {
            let error = result.expect_err("gesture/scribe net must not load as handwriting");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "{}: {error}",
                path.display()
            );
            assert!(
                error.to_string().contains("gesture") || error.to_string().contains("scribe"),
                "{}: unclear rejection: {error}",
                path.display()
            );
            eprintln!("rejected {}: {error}", path.display());
            rejected += 1;
        } else {
            let weights = result.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(!weights.layers.is_empty());
            assert_eq!(weights.input_size, 10);
            assert!(weights.num_classes > 0);
            loaded += 1;
        }
    }
    eprintln!(
        "model sweep: loaded {loaded} handwriting files, rejected {rejected} gesture/scribe files"
    );
}

fn collect_tflite(dir: &Path, paths: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            collect_tflite(&path, paths);
        } else if path.extension().is_some_and(|ext| ext == "tflite") {
            paths.push(path);
        }
    }
}

#[test]
fn truncated_models_return_offset_errors() {
    let Some(path) = common::model_path(common::EN_US_TFLITE) else {
        eprintln!("skipping truncated model checks: en-US TFLite model pack is absent");
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    for len in (0..128).chain([bytes.len() / 4, bytes.len() / 2, bytes.len() * 3 / 4]) {
        assert!(
            matches!(tflite::load_weights(&bytes[..len]), Err(Error::Format(message)) if message.contains("byte ")),
            "truncated model of {len} bytes must fail with a byte offset"
        );
    }
}
