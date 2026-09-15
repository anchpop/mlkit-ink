//! End-to-end checks for gradient descent on ink.
//!
//! The optimizer's failure mode is silence: a wrong gradient still runs, still
//! prints a loss, and simply never converges. So these assert on movement of
//! the actual objective, not on the code path completing.

mod common;

use mlkit_ink::optimize::{self, FitOptions};
use mlkit_ink::{Recognizer, Stroke};

/// Few steps, because this runs in a debug build. Enough to show the loss is
/// being driven down, not enough to reach a target.
const STEPS: usize = 12;

fn recognizer_and_ink(trial: &str) -> Option<(Vec<u8>, Vec<u8>, Vec<Stroke>)> {
    let recospec = std::fs::read(common::model_path(common::EN_US_RECOSPEC)?).ok()?;
    let tflite = std::fs::read(common::model_path(common::EN_US_TFLITE)?).ok()?;
    let ink = common::load_ink(&format!("testdata/corpus/trials/{trial}/ink.json"));
    Some((recospec, tflite, ink))
}

#[test]
fn descent_reduces_the_ctc_loss_of_the_target() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");

    let options = FitOptions {
        steps: STEPS,
        ..FitOptions::default()
    };
    let report = optimize::fit_strokes(&recognizer, &ink, "h", &options).expect("fit");

    assert!(
        report.best_ctc_loss < report.initial_ctc_loss,
        "loss did not improve: {} -> {}",
        report.initial_ctc_loss,
        report.best_ctc_loss
    );
    // A working gradient moves this a long way in a dozen steps; a broken one
    // drifts. The margin is deliberately loose, the direction is not.
    assert!(
        report.best_ctc_loss < 0.6 * report.initial_ctc_loss,
        "loss barely moved: {} -> {}",
        report.initial_ctc_loss,
        report.best_ctc_loss
    );
    assert_eq!(
        report.history.len(),
        STEPS + 1,
        "one evaluation per step, plus the final one"
    );
}

#[test]
fn the_result_is_the_same_ink_moved_not_a_different_ink() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-i") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let options = FitOptions {
        steps: STEPS,
        ..FitOptions::default()
    };
    let report = optimize::fit_strokes(&recognizer, &ink, "j", &options).expect("fit");

    assert_eq!(report.strokes.len(), ink.len(), "stroke count changed");
    for (after, before) in report.strokes.iter().zip(&ink) {
        assert_eq!(after.x.len(), before.x.len(), "point count changed");
        assert_eq!(after.t, before.t, "timestamps must not be touched");
        assert!(
            after.x.iter().chain(&after.y).all(|v| v.is_finite()),
            "non-finite coordinate"
        );
    }
}

/// `matched` is the claim a caller will actually rely on, and it is checked
/// against a full fresh recognition rather than the frozen-structure surrogate
/// the gradient uses. Verify that independently here.
#[test]
fn a_reported_match_really_reads_as_the_target() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let options = FitOptions {
        steps: 60,
        ..FitOptions::default()
    };
    let report = optimize::fit_strokes(&recognizer, &ink, "h", &options).expect("fit");

    let text = recognizer
        .recognize_greedy(&report.strokes)
        .expect("decode")
        .text;
    if report.matched {
        assert_eq!(
            text, "h",
            "reported a match that does not decode to the target"
        );
    } else {
        eprintln!("did not converge in 60 steps; decoded {text:?}");
    }
}

/// A stroke whose x and y disagree in length must come back as an error, not
/// a panic: the regularizer indexes coordinates directly, so it has to run
/// after the pipeline has had its say.
#[test]
fn malformed_ink_is_an_error_not_a_panic() {
    let Some((recospec, tflite, _)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let ragged = vec![Stroke {
        x: vec![0.0, 1.0],
        y: vec![0.0],
        t: Vec::new(),
        pen_up: false,
    }];
    let options = FitOptions {
        steps: 1,
        ..FitOptions::default()
    };
    assert!(optimize::fit_strokes(&recognizer, &ragged, "h", &options).is_err());
}

#[test]
fn a_target_outside_the_charset_is_rejected_by_name() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let options = FitOptions {
        steps: 1,
        ..FitOptions::default()
    };
    // The Latin charset has no Devanagari.
    let error =
        optimize::fit_strokes(&recognizer, &ink, "\u{0915}", &options).expect_err("should reject");
    assert!(format!("{error}").contains("charset"), "{error}");
}
