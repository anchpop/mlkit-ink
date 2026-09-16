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

/// A smiley: two short vertical eyes and a wide U mouth, sampled densely the
/// way a browser pen stream is. Deliberately not a corpus ink — the corpus is
/// sparse and every target reachable from it improves on every chunk, which is
/// exactly why a corpus-based version of the test below passed even with the
/// bug present.
fn smiley() -> Vec<Stroke> {
    let line = |x: f64, n: usize| -> Stroke {
        let pts: Vec<f64> = (0..n)
            .map(|i| 100.0 + 65.0 * i as f64 / (n - 1) as f64)
            .collect();
        Stroke {
            x: vec![x; n],
            y: pts,
            t: (0..n).map(|i| i as f64 * 8.0).collect(),
            pen_up: false,
        }
    };
    let mouth: Vec<(f64, f64)> = (0..91)
        .map(|i| {
            let t = i as f64 / 90.0;
            (
                120.0 + 150.0 * t,
                205.0 + 95.0 * (core::f64::consts::PI * t).sin(),
            )
        })
        .collect();
    vec![
        line(150.0, 31),
        line(230.0, 31),
        Stroke {
            x: mouth.iter().map(|p| p.0).collect(),
            y: mouth.iter().map(|p| p.1).collect(),
            t: (0..mouth.len()).map(|i| i as f64 * 8.0).collect(),
            pen_up: false,
        },
    ]
}

/// The regression that matters most in practice: driving the fit in small
/// chunks, as a UI must, and continuing from the wrong iterate.
///
/// Resuming from `strokes` (the best) rather than `last` makes a chunk that
/// failed to improve hand back its own input. The next chunk then starts from
/// an identical state, takes an identical step, and the loop is a fixed point —
/// the search freezes permanently. It looks exactly like "it optimised partway
/// and then stopped", and a whole-run test never sees it.
#[test]
fn chunked_fitting_does_not_freeze_on_a_plateau() {
    let Some((recospec, tflite, _)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let options = FitOptions {
        steps: 4,
        resegment_every: 5,
        ..FitOptions::default()
    };

    let mut current = smiley();
    let mut frozen = 0;
    for _ in 0..20 {
        let report = optimize::fit_strokes(&recognizer, &current, "Q", &options).expect("fit");
        if report.last == current {
            frozen += 1;
        }
        current = report.last;
        if report.matched {
            break;
        }
    }
    assert!(
        frozen <= 2,
        "chunked fitting froze on {frozen} of 20 chunks; it is resuming from the best \
         iterate instead of the last one"
    );
}

/// `last` is the final iterate and `strokes` is the best one, and they are
/// allowed to differ — that difference is the whole reason both exist.
#[test]
fn the_report_exposes_both_the_best_and_the_final_iterate() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let options = FitOptions {
        steps: 20,
        ..FitOptions::default()
    };
    let report = optimize::fit_strokes(&recognizer, &ink, "h", &options).expect("fit");

    assert_eq!(report.last.len(), ink.len());
    for (after, before) in report.last.iter().zip(&ink) {
        assert_eq!(after.x.len(), before.x.len());
        assert!(after.x.iter().chain(&after.y).all(|v| v.is_finite()));
    }
    // The best was found before the end, so the two must not be the same ink.
    if report.best_step < options.steps {
        assert_ne!(
            report.last, report.strokes,
            "best and final should differ here"
        );
    }
}

/// The robustness option is opt-in, so the thing most worth pinning is that
/// leaving it off changes nothing at all — and that turning it on still
/// converges rather than averaging itself into paralysis.
#[test]
fn robustness_sampling_is_off_by_default_and_works_when_on() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");
    let base = FitOptions {
        steps: STEPS,
        ..FitOptions::default()
    };
    assert_eq!(base.robust_samples, 1, "robustness must default to off");
    assert_eq!(base.jitter, 0.0, "jitter must default to off");

    let plain = optimize::fit_strokes(&recognizer, &ink, "h", &base).expect("fit");
    // One sample with no jitter has to be bit-identical to not asking at all.
    let explicit = FitOptions {
        robust_samples: 1,
        jitter: 0.02,
        ..base
    };
    let same = optimize::fit_strokes(&recognizer, &ink, "h", &explicit).expect("fit");
    assert_eq!(
        plain.strokes, same.strokes,
        "one sample must skip the jitter path entirely"
    );

    let robust = FitOptions {
        robust_samples: 3,
        jitter: 0.006,
        ..base
    };
    let averaged = optimize::fit_strokes(&recognizer, &ink, "h", &robust).expect("fit");
    assert!(
        averaged.best_ctc_loss < averaged.initial_ctc_loss,
        "averaged gradient stopped making progress: {} -> {}",
        averaged.initial_ctc_loss,
        averaged.best_ctc_loss
    );
    assert!(
        averaged
            .strokes
            .iter()
            .all(|s| s.x.iter().all(|v| v.is_finite()))
    );
}

/// The penalties have to bite when a fit is resumed, which is how every live
/// caller drives it.
///
/// Passing the current ink as its own reference makes the measured deformation
/// zero at the start of each chunk, so both gradients vanish on every applied
/// update and the anchor silently does nothing — invisible in a whole-run test,
/// and it made the demo's slider inert.
#[test]
fn the_anchor_still_bites_when_a_fit_is_resumed() {
    let Some((recospec, tflite, ink)) = recognizer_and_ink("lower-n") else {
        eprintln!("skipping: en-US packs are not fetched");
        return;
    };
    let recognizer = Recognizer::load(&recospec, &tflite, None).expect("load");

    // One step per chunk, as the animation uses, holding the original ink as
    // the reference throughout.
    let chunked = |anchor: f64| -> Vec<Stroke> {
        let options = FitOptions {
            steps: 1,
            anchor_weight: anchor,
            resegment_every: 2,
            ..FitOptions::default()
        };
        let mut current = ink.clone();
        for _ in 0..60 {
            let report = optimize::fit_strokes_from(&recognizer, &current, &ink, "h", &options)
                .expect("fit");
            if anchor > 0.0 {
                let moved: f64 = report
                    .last
                    .iter()
                    .zip(&current)
                    .map(|(a, b)| {
                        a.x.iter()
                            .zip(&b.x)
                            .map(|(x, y)| (x - y).abs())
                            .sum::<f64>()
                    })
                    .sum();
                eprintln!(
                    "    reg={:.6} ctc={:.4} moved={:.6} steps_in_history={}",
                    report.history[0].regularization,
                    report.history[0].ctc_loss,
                    moved,
                    report.history.len()
                );
            }
            current = report.last;
        }
        current
    };

    let drift = |after: &[Stroke]| -> f64 {
        let mut total = 0.0;
        for (a, b) in after.iter().zip(&ink) {
            for i in 0..a.x.len() {
                total += (a.x[i] - b.x[i]).powi(2) + (a.y[i] - b.y[i]).powi(2);
            }
        }
        total.sqrt()
    };

    let loose = drift(&chunked(0.0));
    let tight = drift(&chunked(40.0));
    assert!(
        tight < loose * 0.9,
        "a heavy anchor barely restrained a resumed fit ({tight:.3} vs {loose:.3}); \
         the regularization reference is probably moving with the search"
    );
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
