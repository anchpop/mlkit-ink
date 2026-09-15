//! Forward pass for the stacked bidirectional IndyLSTM.
//!
//! IndyLSTM (Gonnet & Deselaers, 2019) is an LSTM whose recurrent weight matrix
//! is *diagonal*, so the recurrence is an elementwise product rather than a
//! matvec. Full recurrent matrices use the same gate and state updates here.
//!
//! This is the dequantized f32 reference, not TFLite's dynamic activation
//! quantization or approximate activation kernels.

use crate::error::Result;
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::mat::Mat;
use crate::tflite::{DirectionWeights, GATES, LayerWeights, NetworkWeights};

/// Evaluate one direction. Backward outputs are restored to original time
/// order. Cell clipping applies after the cell update and before `tanh(cell)`.
pub fn direction(
    features: &Mat,
    weights: &DirectionWeights,
    cell_clip: f32,
    reverse: bool,
) -> Result<Mat> {
    ensure!(
        features.cols() == weights.input,
        Invalid,
        "expected features [T, {}], got [T, {}]",
        weights.input,
        features.cols()
    );
    finite(features.as_slice(), "features")?;
    ensure!(
        cell_clip.is_finite() && cell_clip >= 0.0,
        Invalid,
        "cell clipping threshold must be finite and nonnegative"
    );
    let hidden = weights.hidden;
    ensure!(
        hidden > 0 && weights.input > 0,
        Invalid,
        "LSTM dimensions must be positive"
    );
    let gates = size(GATES, hidden)?;
    tensor(
        &weights.input_kernels,
        size(gates, weights.input)?,
        "input kernels",
    )?;
    tensor(
        &weights.recurrent_kernels,
        if weights.diagonal {
            gates
        } else {
            size(gates, hidden)?
        },
        "recurrent kernels",
    )?;
    tensor(&weights.biases, gates, "gate biases")?;
    tensor(&weights.initial_activation, hidden, "initial activation")?;
    tensor(&weights.initial_cell, hidden, "initial cell")?;

    // Project every timestep first. Bias is added AFTER the input dot product,
    // and recurrence after that, matching the NumPy reference's f32 rounding.
    let mut projected = project(features, &weights.input_kernels, &weights.biases)?;
    let mut h = weights.initial_activation.clone();
    let mut c = weights.initial_cell.clone();
    let mut output = Mat::zeros(features.rows(), hidden);
    for step in 0..features.rows() {
        let t = if reverse {
            features.rows() - 1 - step
        } else {
            step
        };
        let gates = projected.row_mut(t);
        // Complete ALL recurrent projections before replacing any h entries:
        // a full matrix must see the previous timestep's entire activation.
        for (gate, value) in gates.iter_mut().enumerate() {
            let recurrent = if weights.diagonal {
                weights.recurrent_kernels[gate] * h[gate % hidden]
            } else {
                dot(
                    &weights.recurrent_kernels[gate * hidden..(gate + 1) * hidden],
                    &h,
                )
            };
            *value += recurrent;
        }
        for j in 0..hidden {
            let input = sigmoid(gates[j]);
            let forget = sigmoid(gates[hidden + j]);
            let candidate = gates[2 * hidden + j].tanh();
            let out = sigmoid(gates[3 * hidden + j]);
            c[j] = forget * c[j] + input * candidate;
            if cell_clip > 0.0 {
                c[j] = c[j].clamp(-cell_clip, cell_clip);
            }
            h[j] = out * c[j].tanh();
        }
        output.row_mut(t).copy_from_slice(&h);
    }
    Ok(output)
}

/// Concatenate forward and backward hidden outputs. Never cell states.
pub fn bidirectional(features: &Mat, layer: &LayerWeights) -> Result<Mat> {
    let forward = direction(features, &layer.forward, layer.cell_clip, false)?;
    let backward = direction(features, &layer.backward, layer.cell_clip, true)?;
    let width = forward
        .cols()
        .checked_add(backward.cols())
        .ok_or_else(|| err!(Invalid, "bidirectional output width overflows usize"))?;
    size(features.rows(), width)?;
    let mut output = Mat::zeros(features.rows(), width);
    for t in 0..features.rows() {
        let (f, b) = output.row_mut(t).split_at_mut(forward.cols());
        f.copy_from_slice(forward.row(t));
        b.copy_from_slice(backward.row(t));
    }
    Ok(output)
}

/// `[T, input_size]` features -> `[T, num_classes]` unnormalized logits.
///
/// Batch size is one, so the graph's `time_major` flag makes no difference to
/// this 2-D API. Empty sequences return `[0, num_classes]`.
pub fn forward(weights: &NetworkWeights, features: &Mat) -> Result<Mat> {
    ensure!(
        features.cols() == weights.input_size,
        Invalid,
        "expected features [T, {}], got [T, {}]",
        weights.input_size,
        features.cols()
    );
    finite(features.as_slice(), "features")?;
    ensure!(
        weights.input_size > 0 && weights.num_classes > 0,
        Invalid,
        "network dimensions must be positive"
    );
    // Borrow the input until the first layer, rather than copying features.
    let mut activations;
    let mut x = features;
    for layer in &weights.layers {
        activations = bidirectional(x, layer)?;
        x = &activations;
    }
    tensor(
        &weights.fc_weights,
        size(weights.num_classes, x.cols())?,
        "fully connected weights",
    )?;
    tensor(
        &weights.fc_bias,
        weights.num_classes,
        "fully connected bias",
    )?;
    project(x, &weights.fc_weights, &weights.fc_bias)
}

fn size(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| err!(Invalid, "matrix dimensions {rows}x{cols} overflow usize"))
}

fn finite(values: &[f32], name: &str) -> Result<()> {
    ensure!(
        values.iter().all(|x| x.is_finite()),
        Invalid,
        "{name} must be finite"
    );
    Ok(())
}

fn tensor(values: &[f32], expected: usize, name: &str) -> Result<()> {
    ensure!(
        values.len() == expected,
        Invalid,
        "{name}: expected {expected} values, got {}",
        values.len()
    );
    finite(values, name)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |sum, (a, b)| sum + a * b)
}

/// Row-major linear projection. The caller has checked the tensor lengths.
fn project(input: &Mat, kernels: &[f32], biases: &[f32]) -> Result<Mat> {
    size(input.rows(), biases.len())?;
    let mut output = Mat::zeros(input.rows(), biases.len());
    for t in 0..input.rows() {
        for (j, value) in output.row_mut(t).iter_mut().enumerate() {
            *value = dot(
                &kernels[j * input.cols()..(j + 1) * input.cols()],
                input.row(t),
            ) + biases[j];
        }
    }
    Ok(output)
}

// exp(-abs(x)) cannot overflow, even for extreme finite preactivations.
fn sigmoid(x: f32) -> f32 {
    let e = (-x.abs()).exp();
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn tiny() -> DirectionWeights {
        DirectionWeights {
            input_kernels: vec![0.0, 0.0, 1.0, 0.0],
            recurrent_kernels: vec![0.0; 4],
            diagonal: true,
            biases: vec![0.0; 4],
            initial_activation: vec![0.0],
            initial_cell: vec![0.0],
            hidden: 1,
            input: 1,
        }
    }

    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 2e-7,
            "got {actual}, expected {expected}"
        );
    }

    #[test]
    fn extreme_sigmoid() {
        for x in [
            f32::MIN,
            -1000.0,
            -100.0,
            -50.0,
            0.0,
            50.0,
            100.0,
            1000.0,
            f32::MAX,
        ] {
            let y = sigmoid(x);
            assert!(
                y.is_finite() && (0.0..=1.0).contains(&y),
                "sigmoid({x}) = {y}"
            );
        }
        assert_eq!(sigmoid(f32::MIN), 0.0);
        assert_eq!(sigmoid(f32::MAX), 1.0);
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(-100.0) > 0.0, "do not flush subnormals");
        assert_eq!(sigmoid(f32::NEG_INFINITY), 0.0);
        assert_eq!(sigmoid(f32::INFINITY), 1.0);
        assert!(sigmoid(f32::NAN).is_nan());
    }

    #[test]
    fn sigmoid_is_monotone_symmetric_and_matches_known_values() {
        close(sigmoid(1.098_612_3), 0.75);
        close(sigmoid(-1.098_612_3), 0.25);
        close(sigmoid(1.0), 0.731_058_6);
        close(sigmoid(-1.0), 0.268_941_43);
        let mut previous = 0.0;
        for i in -12_000..=12_000 {
            let x = i as f32 / 1000.0;
            let value = sigmoid(x);
            assert!(value >= previous, "sigmoid is not monotone at {x}");
            close(value + sigmoid(-x), 1.0);
            previous = value;
        }
    }

    #[test]
    fn hand_computed_one_step_and_clipping() {
        let mut weights = tiny();
        weights.initial_cell[0] = 0.25;
        weights.initial_activation[0] = 0.75;
        // tanh(atanh(0.5)) = 0.5; i = f = o = 0.5.
        // c = 0.5 * 0.25 + 0.5 * 0.5 = 0.375; h = tanh(c)/2.
        let x = Mat::from_vec(1, 1, vec![0.549_306_15]);
        close(
            direction(&x, &weights, 0.0, false).unwrap()[(0, 0)],
            0.179_178_7,
        );
        // Clip the cell, not the output: h = tanh(0.1)/2.
        close(
            direction(&x, &weights, 0.1, false).unwrap()[(0, 0)],
            0.049_833_998,
        );
        let before = weights.clone();
        assert_eq!(
            direction(&x, &weights, 0.0, false).unwrap(),
            direction(&x, &weights, 0.0, false).unwrap()
        );
        assert_eq!(weights, before);
    }

    #[test]
    fn reverse_restores_time_and_bidirectional_concatenates_hidden() {
        let mut weights = tiny();
        weights.recurrent_kernels[2] = 0.5;
        let x = Mat::from_vec(3, 1, vec![0.2, -0.4, 0.8]);
        let reversed = Mat::from_vec(3, 1, vec![0.8, -0.4, 0.2]);
        let backward = direction(&x, &weights, 0.0, true).unwrap();
        let reference = direction(&reversed, &weights, 0.0, false).unwrap();
        let f = direction(&x, &weights, 0.0, false).unwrap();
        let layer = LayerWeights {
            forward: weights.clone(),
            backward: weights,
            cell_clip: 0.0,
            custom_options: vec![],
        };
        let both = bidirectional(&x, &layer).unwrap();
        for t in 0..3 {
            assert_eq!(backward[(t, 0)], reference[(2 - t, 0)]);
            assert_eq!(both.row(t), &[f[(t, 0)], backward[(t, 0)]]);
        }
        assert_ne!(f, backward);
    }

    #[test]
    fn full_recurrence_uses_previous_state_and_row_major_gates() {
        let mut weights = DirectionWeights {
            input_kernels: vec![0.0; 8],
            recurrent_kernels: vec![0.0; 16],
            diagonal: false,
            biases: vec![0.0; 8],
            initial_activation: vec![0.25, -0.5],
            initial_cell: vec![0.0; 2],
            hidden: 2,
            input: 1,
        };
        // Candidate rows [0, 1] and [2, 0] mix the OTHER previous unit.
        weights.recurrent_kernels[9] = 1.0;
        weights.recurrent_kernels[10] = 2.0;
        let x = Mat::zeros(2, 1);
        let actual = direction(&x, &weights, 0.0, false).unwrap();
        let mut h: [f32; 2] = [0.25, -0.5];
        let mut c: [f32; 2] = [0.0; 2];
        for t in 0..2 {
            let candidates = [h[1].tanh(), (2.0 * h[0]).tanh()];
            for j in 0..2 {
                c[j] = 0.5 * c[j] + 0.5 * candidates[j];
                h[j] = 0.5 * c[j].tanh();
                close(actual[(t, j)], h[j]);
            }
        }
    }

    #[test]
    fn diagonal_and_full_recurrence_agree() {
        let diagonal = DirectionWeights {
            input_kernels: vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8],
            recurrent_kernels: vec![0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9],
            diagonal: true,
            biases: vec![0.1; 8],
            initial_activation: vec![0.75, -0.25],
            initial_cell: vec![-0.2, 0.3],
            hidden: 2,
            input: 1,
        };
        let mut full = diagonal.clone();
        full.diagonal = false;
        full.recurrent_kernels = vec![0.0; 16];
        for row in 0..8 {
            full.recurrent_kernels[row * 2 + row % 2] = diagonal.recurrent_kernels[row];
        }
        let x = Mat::from_vec(3, 1, vec![0.3, 0.8, -0.7]);
        for reverse in [false, true] {
            assert_eq!(
                direction(&x, &diagonal, 0.0, reverse).unwrap(),
                direction(&x, &full, 0.0, reverse).unwrap()
            );
        }
    }

    #[test]
    fn forward_stacks_layers_and_leaves_logits_unnormalized() {
        let first = LayerWeights {
            forward: tiny(),
            backward: tiny(),
            cell_clip: 0.0,
            custom_options: vec![],
        };
        let mut second_direction = tiny();
        second_direction.input = 2;
        second_direction.input_kernels = vec![0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0];
        let second = LayerWeights {
            forward: second_direction.clone(),
            backward: second_direction,
            cell_clip: 0.0,
            custom_options: vec![],
        };
        let weights = NetworkWeights {
            layers: vec![first, second],
            fc_weights: vec![2.0, -3.0, 0.0, 0.0],
            fc_bias: vec![1.0, -2.0],
            input_size: 1,
            num_classes: 2,
        };
        let x = Mat::from_vec(2, 1, vec![0.2, 0.5]);
        let intermediate = bidirectional(&x, &weights.layers[0]).unwrap();
        let hidden = bidirectional(&intermediate, &weights.layers[1]).unwrap();
        let actual = forward(&weights, &x).unwrap();
        for t in 0..2 {
            close(
                actual[(t, 0)],
                2.0 * hidden[(t, 0)] - 3.0 * hidden[(t, 1)] + 1.0,
            );
            assert_eq!(actual[(t, 1)], -2.0);
        }
        let empty = forward(&weights, &Mat::zeros(0, 1)).unwrap();
        assert_eq!((empty.rows(), empty.cols()), (0, 2));
    }

    #[test]
    fn invalid_inputs_and_weights_return_errors() {
        let x = Mat::zeros(1, 1);
        for clip in [-1.0, f32::INFINITY, f32::NAN] {
            assert!(direction(&x, &tiny(), clip, false).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(direction(&Mat::from_vec(1, 1, vec![value]), &tiny(), 0.0, false).is_err());
        }
        assert!(direction(&Mat::zeros(0, 2), &tiny(), 0.0, false).is_err());
        for field in 0..5 {
            let mut weights = tiny();
            match field {
                0 => weights.input_kernels.clear(),
                1 => weights.recurrent_kernels.clear(),
                2 => weights.biases.clear(),
                3 => weights.initial_activation.clear(),
                _ => weights.initial_cell.clear(),
            }
            assert!(direction(&x, &weights, 0.0, false).is_err());
        }
        let mut weights = tiny();
        weights.initial_cell[0] = f32::NAN;
        assert!(direction(&x, &weights, 0.0, false).is_err());
        weights.hidden = usize::MAX;
        assert!(direction(&x, &weights, 0.0, false).is_err());
        let mut network = NetworkWeights {
            layers: vec![],
            fc_weights: vec![2.0],
            fc_bias: vec![1.0],
            input_size: 1,
            num_classes: 1,
        };
        assert_eq!(forward(&network, &x).unwrap()[(0, 0)], 1.0);
        network.fc_weights.clear();
        assert!(forward(&network, &Mat::zeros(0, 1)).is_err());
    }
}
