//! Numerical checks of feature-space backpropagation, independent of curve fitting.
mod common;

use mlkit_ink::{
    ctc,
    mat::Mat,
    net, netgrad,
    tflite::{self, DirectionWeights, GATES, LayerWeights, NetworkWeights},
};

// A local, fixed-seed generator keeps these tests reproducible without a rand
// dependency. Use nonsymmetric weights and nonzero initial states so swapped
// gates, transposes, and accidentally discarded recurrent paths are visible.
struct Random(u64);

impl Random {
    fn value(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| scale * self.value()).collect()
    }

    fn unit_vector(&mut self, len: usize) -> Vec<f64> {
        let mut values: Vec<f64> = self.values(len, 1.0).into_iter().map(f64::from).collect();
        let norm = dot(&values, &values).sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    fn direction(&mut self, input: usize, hidden: usize, diagonal: bool) -> DirectionWeights {
        DirectionWeights {
            input_kernels: self.values(GATES * hidden * input, 0.6),
            recurrent_kernels: self.values(GATES * hidden * if diagonal { 1 } else { hidden }, 0.5),
            diagonal,
            biases: self.values(GATES * hidden, 0.3),
            initial_activation: self.values(hidden, 0.2),
            initial_cell: self.values(hidden, 0.2),
            hidden,
            input,
        }
    }

    fn network(&mut self, depth: usize, diagonal: bool) -> NetworkWeights {
        let mut layers = Vec::new();
        let mut input = 3;
        for _ in 0..depth {
            // Unequal direction widths also exercise the concatenation offset.
            layers.push(LayerWeights {
                forward: self.direction(input, 2, diagonal),
                backward: self.direction(input, 3, diagonal),
                cell_clip: 0.0,
                custom_options: vec![],
            });
            input = 5;
        }
        NetworkWeights {
            layers,
            fc_weights: self.values(4 * input, 0.7),
            fc_bias: self.values(4, 0.2),
            input_size: 3,
            num_classes: 4,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn weighted_logits(logits: &Mat, weights: &[f64]) -> f64 {
    assert_eq!(logits.as_slice().len(), weights.len());
    logits
        .as_slice()
        .iter()
        .zip(weights)
        .map(|(&value, weight)| f64::from(value) * weight)
        .sum()
}

fn exact_forward<'w>(
    weights: &'w NetworkWeights,
    features: &Mat,
) -> (Mat, netgrad::Activations<'w>) {
    let (logits, activations) = netgrad::forward_with_activations(weights, features).unwrap();
    let ordinary = net::forward(weights, features).unwrap();
    assert_eq!(
        (logits.rows(), logits.cols()),
        (ordinary.rows(), ordinary.cols())
    );
    // Compare bits, not a tolerance (or float equality, which conflates +/-0).
    assert!(logits.as_slice().iter().all(|v| v.is_finite()));
    assert_eq!(
        logits
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        ordinary
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "recording activations must not change forward arithmetic"
    );
    (logits, activations)
}

fn every_feature_difference(
    weights: &NetworkWeights,
    features: &Mat,
    upstream: &[f64],
    name: &str,
) {
    let (_, activations) = exact_forward(weights, features);
    let analytic = netgrad::features_gradient(weights, &activations, upstream).unwrap();
    assert_eq!(analytic.len(), features.as_slice().len());
    assert!(analytic.iter().all(|v| v.is_finite()));
    let epsilon = 2e-3f32;
    let mut worst = 0.0f64;
    for (index, &expected) in analytic.iter().enumerate() {
        let (row, col) = (index / features.cols(), index % features.cols());
        let mut plus = features.clone();
        let mut minus = features.clone();
        plus[(row, col)] += epsilon;
        minus[(row, col)] -= epsilon;
        let upper = weighted_logits(&net::forward(weights, &plus).unwrap(), upstream);
        let lower = weighted_logits(&net::forward(weights, &minus).unwrap(), upstream);
        // Divide by the actual representable f32 displacement.
        let numerical =
            (upper - lower) / (f64::from(plus[(row, col)]) - f64::from(minus[(row, col)]));
        let error = (numerical - expected).abs();
        worst = worst.max(error);
        assert!(
            error <= 2e-5 + 3e-3 * expected.abs(),
            "{name} feature {index}: analytic={expected:.9e}, numerical={numerical:.9e}, epsilon={epsilon}, error={error:.9e}"
        );
    }
    eprintln!(
        "{name}: all {} features agree, epsilon={epsilon}, worst absolute error={worst:.9e}",
        analytic.len()
    );
}

#[test]
fn synthetic_one_and_two_layers_match_every_feature_central_difference() {
    let mut random = Random(0x831c_409f_712a_568d);
    for diagonal in [true, false] {
        for depth in [1, 2] {
            let weights = random.network(depth, diagonal);
            let unchanged = weights.clone();
            let features = Mat::from_vec(5, 3, random.values(15, 0.7));
            let upstream = random.unit_vector(5 * weights.num_classes);
            every_feature_difference(
                &weights,
                &features,
                &upstream,
                &format!("synthetic depth={depth} diagonal={diagonal}"),
            );
            assert_eq!(weights, unchanged, "backpropagation must not train weights");
        }
    }
}

fn reverse_rows(matrix: &Mat) -> Mat {
    Mat::from_rows(
        matrix.cols(),
        (0..matrix.rows()).rev().map(|t| matrix.row(t).to_vec()),
    )
}

#[test]
fn dense_backward_direction_restores_feature_gradient_time_order() {
    let mut random = Random(0x596a_a03e_6e8b_1171);
    let direction = random.direction(3, 2, false);
    let mut weights = NetworkWeights {
        layers: vec![LayerWeights {
            forward: direction.clone(),
            backward: direction,
            cell_clip: 0.0,
            custom_options: vec![],
        }],
        fc_weights: vec![0.0, 0.0, 0.7, -0.4],
        fc_bias: vec![0.1],
        input_size: 3,
        num_classes: 1,
    };
    let features = Mat::from_vec(5, 3, random.values(15, 0.7));
    let upstream = random.unit_vector(5);
    every_feature_difference(&weights, &features, &upstream, "isolated dense reverse");
    let (backward_logits, backward_cache) = exact_forward(&weights, &features);
    let backward_grad = netgrad::features_gradient(&weights, &backward_cache, &upstream).unwrap();

    // Select only the forward half of the very same layer, with both the input
    // and the upstream cotangent reversed. The gradients must reverse too.
    weights.fc_weights = vec![0.7, -0.4, 0.0, 0.0];
    let (forward_logits, forward_cache) = exact_forward(&weights, &reverse_rows(&features));
    let reversed_upstream: Vec<f64> = upstream.iter().rev().copied().collect();
    let forward_grad =
        netgrad::features_gradient(&weights, &forward_cache, &reversed_upstream).unwrap();
    assert_eq!(backward_logits, reverse_rows(&forward_logits));
    for t in 0..features.rows() {
        for j in 0..features.cols() {
            assert_eq!(
                backward_grad[t * features.cols() + j],
                forward_grad[(features.rows() - 1 - t) * features.cols() + j]
            );
        }
    }
}

#[test]
fn clipped_cell_blocks_candidate_but_preserves_output_gate_and_its_recurrence() {
    for sign in [-1.0f32, 1.0] {
        for reverse in [false, true] {
            // Feature 0 enters only the candidate; feature 1 only the output
            // gate. Every raw cell is well beyond +/-0.1, away from the kink.
            let direction = DirectionWeights {
                input_kernels: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                recurrent_kernels: vec![0.0, 0.0, 0.4, 0.7],
                diagonal: true,
                biases: vec![1.0, 0.0, sign * 2.0, 0.0],
                initial_activation: vec![0.1],
                initial_cell: vec![0.0],
                hidden: 1,
                input: 2,
            };
            let mut weights = NetworkWeights {
                layers: vec![LayerWeights {
                    forward: direction.clone(),
                    backward: direction,
                    cell_clip: 0.1,
                    custom_options: vec![],
                }],
                fc_weights: if reverse {
                    vec![0.0, 1.0]
                } else {
                    vec![1.0, 0.0]
                },
                fc_bias: vec![0.0],
                input_size: 2,
                num_classes: 1,
            };
            let one_step = Mat::from_vec(1, 2, vec![0.2, -0.3]);
            let (_, cache) = exact_forward(&weights, &one_step);
            let gradient = netgrad::features_gradient(&weights, &cache, &[1.0]).unwrap();
            let output_gate = 1.0 / (1.0 + (0.3f64 - 0.7 * 0.1).exp());
            let expected = (f64::from(sign) * 0.1).tanh() * output_gate * (1.0 - output_gate);
            assert_eq!(
                gradient[0], 0.0,
                "candidate derivative crosses a clipped cell"
            );
            assert!(
                (gradient[1] - expected).abs() < 1e-8,
                "output gate must survive clipping: {} vs {expected}",
                gradient[1]
            );
            assert!(gradient[1].abs() > 0.02);

            let features = Mat::from_vec(3, 2, vec![0.2, -0.3, -0.1, 0.4, 0.15, 0.1]);
            let upstream = if reverse {
                vec![1.0, 0.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            };
            every_feature_difference(
                &weights,
                &features,
                &upstream,
                &format!("clipped sign={sign} reverse={reverse}"),
            );
            let (_, cache) = exact_forward(&weights, &features);
            let clipped = netgrad::features_gradient(&weights, &cache, &upstream).unwrap();
            for row in clipped.as_chunks::<2>().0 {
                assert_eq!(row[0], 0.0);
                assert!(
                    row[1].abs() > 1e-6,
                    "output-gate recurrence must cross clipped timesteps"
                );
            }
            weights.layers[0].cell_clip = 0.0;
            let (_, cache) = exact_forward(&weights, &features);
            let unclipped = netgrad::features_gradient(&weights, &cache, &upstream).unwrap();
            assert!(
                unclipped
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|row| row[0].abs() > 1e-5),
                "disabling clipping must restore the candidate path"
            );
        }
    }
}

fn english_weights() -> Option<NetworkWeights> {
    let Some(path) = common::model_path(common::EN_US_TFLITE) else {
        eprintln!("skipping real network gradients: en-US TFLite model pack is absent");
        return None;
    };
    Some(tflite::load_weights(&std::fs::read(path).unwrap()).unwrap())
}

fn trial(id: &str) -> &'static serde_json::Value {
    common::trials()
        .iter()
        .find(|trial| trial["id"] == id)
        .expect("named golden trial")
}

fn trial_features(trial: &serde_json::Value, input_size: usize) -> Mat {
    Mat::from_rows(
        input_size,
        trial["features"].as_array().unwrap().iter().map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        }),
    )
}

fn oracle_target(trial: &serde_json::Value) -> (Vec<usize>, usize) {
    let decoder: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(common::goldens_dir().join("decoder_goldens.json")).unwrap(),
    )
    .unwrap();
    let alphabet = &decoder["alphabet"];
    let texts = alphabet["texts"].as_array().unwrap();
    // CTC consumes NETWORK indices, not the FST's alphabet.labels values.
    let target = trial["oracle_top"]
        .as_str()
        .unwrap()
        .chars()
        .map(|character| {
            let character = character.to_string();
            texts
                .iter()
                .position(|text| text.as_str().unwrap() == character)
                .unwrap_or_else(|| panic!("oracle character {character:?} absent from alphabet"))
        })
        .collect();
    (target, alphabet["blank"].as_u64().unwrap() as usize)
}

fn displaced(features: &Mat, direction: &[f64], amount: f64) -> Mat {
    assert_eq!(features.as_slice().len(), direction.len());
    Mat::from_vec(
        features.rows(),
        features.cols(),
        features
            .as_slice()
            .iter()
            .zip(direction)
            .map(|(&value, delta)| (f64::from(value) + amount * delta) as f32)
            .collect(),
    )
}

/// The forward computation is f32: small epsilons hit rounding noise and large
/// ones see curvature. Require agreement at multiple scales, not one lucky h.
fn direction_sweep(
    features: &Mat,
    gradient: &[f64],
    direction: &[f64],
    objective: impl Fn(&Mat) -> f64,
    absolute_tolerance: f64,
    epsilons: [f64; 5],
    name: &str,
) -> bool {
    let analytic = dot(gradient, direction);
    assert!(analytic.is_finite());
    let tolerance = absolute_tolerance + 5e-3 * analytic.abs();
    let mut agreements = 0;
    let mut best = (f64::INFINITY, 0.0);
    let mut chosen_agrees = false;
    // The first epsilon is fixed by the caller, never chosen from the errors.
    let chosen_epsilon = epsilons[0];
    for epsilon in epsilons {
        let upper = objective(&displaced(features, direction, epsilon));
        let lower = objective(&displaced(features, direction, -epsilon));
        let numerical = (upper - lower) / (2.0 * epsilon);
        let error = (numerical - analytic).abs();
        assert!(numerical.is_finite(), "{name}: nonfinite finite difference");
        if error <= tolerance {
            agreements += 1;
        }
        if epsilon == chosen_epsilon {
            chosen_agrees = error <= tolerance;
        }
        if error < best.0 {
            best = (error, epsilon);
        }
        eprintln!(
            "{name}: epsilon={epsilon:.1e}, analytic={analytic:.9e}, numerical={numerical:.9e}, absolute_error={error:.3e}, relative_error={:.3e}",
            error / analytic.abs().max(1e-15)
        );
    }
    eprintln!(
        "{name}: agreements={agreements}/5, chosen epsilon={chosen_epsilon:.1e} agrees={chosen_agrees}, best epsilon={:.1e}, best absolute error={:.3e}, tolerance={tolerance:.3e}",
        best.1, best.0
    );
    agreements >= 2 && chosen_agrees
}

#[test]
fn english_weighted_logits_and_oracle_ctc_directional_derivatives() {
    let Some(weights) = english_weights() else {
        return;
    };
    let mut random = Random(0xb30c_f8a6_0459_213d);
    let mut all_agree = true;
    for id in ["reference-hi", "reference-cat", "word-so"] {
        let trial = trial(id);
        let features = trial_features(trial, weights.input_size);
        let (logits, cache) = exact_forward(&weights, &features);
        let upstream = random.unit_vector(logits.as_slice().len());
        let weighted_gradient = netgrad::features_gradient(&weights, &cache, &upstream).unwrap();
        assert_eq!(weighted_gradient.len(), features.as_slice().len());
        assert!(weighted_gradient.iter().all(|v| v.is_finite()));
        assert!(dot(&weighted_gradient, &weighted_gradient) > 1e-8);
        // The long cat fixture has higher curvature: h=.003 is too large.
        // Shorter trials need h=.003 to rise above f32 cancellation noise.
        // Fix one step per trial, shared by ALL its weighted directions.
        let weighted_epsilons = if id == "reference-cat" {
            [1e-3, 1e-2, 3e-3, 2e-3, 1e-4]
        } else {
            [3e-3, 1e-2, 2e-3, 1e-3, 1e-4]
        };
        for index in 0..3 {
            let direction = random.unit_vector(features.as_slice().len());
            all_agree &= direction_sweep(
                &features,
                &weighted_gradient,
                &direction,
                |x| weighted_logits(&net::forward(&weights, x).unwrap(), &upstream),
                1e-4,
                weighted_epsilons,
                &format!("{id} weighted direction={index}"),
            );
        }
        let (target, blank) = oracle_target(trial);
        assert_eq!(blank + 1, weights.num_classes);
        let loss = ctc::loss_and_grad(&logits, &target, blank).unwrap();
        assert!(loss.loss.is_finite() && loss.loss > 0.0);
        let ctc_gradient = netgrad::features_gradient(&weights, &cache, &loss.grad).unwrap();
        assert_eq!(ctc_gradient.len(), features.as_slice().len());
        assert!(ctc_gradient.iter().all(|v| v.is_finite()));
        assert!(dot(&ctc_gradient, &ctc_gradient) > 1e-12);
        eprintln!(
            "{id}: exact logits, oracle={:?}, CTC loss={:.9e}",
            trial["oracle_top"], loss.loss
        );
        // A common h=.001 balances CTC curvature and f32 forward noise;
        // coarse/fine endpoints remain diagnostic, not required to agree.
        for index in 0..2 {
            let direction = random.unit_vector(features.as_slice().len());
            all_agree &= direction_sweep(
                &features,
                &ctc_gradient,
                &direction,
                |x| {
                    ctc::loss_and_grad(&net::forward(&weights, x).unwrap(), &target, blank)
                        .unwrap()
                        .loss
                },
                2e-6,
                [1e-3, 1e-2, 5e-3, 5e-4, 1e-4],
                &format!("{id} oracle CTC direction={index}"),
            );
        }
    }
    assert!(
        all_agree,
        "every direction must agree at its fixed epsilon and at least two sweep scales; see agreements above"
    );
}

#[test]
fn fifty_plain_feature_gradient_steps_substantially_reduce_oracle_ctc_loss() {
    let Some(weights) = english_weights() else {
        return;
    };
    // A real, moderately uncertain two-letter trial, without manufactured noise
    // or a changed target. No Adam, line search, clipping, or feature projection.
    let trial = trial("word-so");
    let mut features = trial_features(trial, weights.input_size);
    let (target, blank) = oracle_target(trial);
    let initial = ctc::loss_and_grad(&net::forward(&weights, &features).unwrap(), &target, blank)
        .unwrap()
        .loss;
    let learning_rate = 2e-2;
    eprintln!("word-so plain GD: step=0 loss={initial:.9e}, learning_rate={learning_rate}");
    for step in 1..=50 {
        let (logits, cache) = netgrad::forward_with_activations(&weights, &features).unwrap();
        let loss = ctc::loss_and_grad(&logits, &target, blank).unwrap();
        assert!(loss.loss.is_finite());
        let gradient = netgrad::features_gradient(&weights, &cache, &loss.grad).unwrap();
        assert_eq!(gradient.len(), features.as_slice().len());
        assert!(gradient.iter().all(|value| value.is_finite()));
        features = displaced(&features, &gradient, -learning_rate);
        if step == 1 || step % 10 == 0 {
            let loss =
                ctc::loss_and_grad(&net::forward(&weights, &features).unwrap(), &target, blank)
                    .unwrap()
                    .loss;
            eprintln!("word-so plain GD: step={step} loss={loss:.9e}");
        }
    }
    let final_loss =
        ctc::loss_and_grad(&net::forward(&weights, &features).unwrap(), &target, blank)
            .unwrap()
            .loss;
    eprintln!(
        "word-so plain GD: before={initial:.9e}, after={final_loss:.9e}, remaining={:.3}%",
        100.0 * final_loss / initial
    );
    assert!(initial.is_finite() && initial > 0.0);
    assert!(
        final_loss.is_finite() && final_loss < initial * 0.25,
        "50 plain GD steps must reduce CTC loss by at least 75%: {initial} -> {final_loss}"
    );
}
