//! Backpropagation through the frozen bidirectional (Indy)LSTM stack.
//!
//! The forward arithmetic stays f32 and in the same order as [`crate::net`].
//! Only adjoints widen to f64: rounding those at every layer and timestep
//! would lose accuracy before the gradient ever reached the ink.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Result;
#[cfg(all(not(feature = "std"), not(test)))]
use crate::float::Float;
use crate::mat::Mat;
use crate::net;
use crate::tflite::{DirectionWeights, GATES, NetworkWeights};

/// Values needed to differentiate one forward pass. Borrowing the weights
/// prevents them from being mutated between that pass and its backward sweep.
/// No layer inputs are retained: those are needed for weight gradients only.
#[derive(Debug)]
pub struct Activations<'w> {
    weights: &'w NetworkWeights,
    rows: usize,
    layers: Vec<[DirectionActivations; 2]>,
}

#[derive(Debug)]
struct DirectionActivations {
    // Original time order, including for the reverse direction.
    cells: Vec<Cell>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    gates: [f32; GATES],
    previous_cell: f32,
    tanh_cell: f32,
    unclipped: bool,
}

/// `[T, input_size]` features -> logits identical to [`net::forward`], plus
/// the per-timestep values needed by [`features_gradient`]. Empty sequences
/// are supported. The returned activations borrow the frozen model, not ink.
pub fn forward_with_activations<'w>(
    weights: &'w NetworkWeights,
    features: &Mat,
) -> Result<(Mat, Activations<'w>)> {
    // net's validation and projection helpers are private. An empty forward
    // reuses all its model checks without evaluating the network twice.
    net::forward(weights, &Mat::zeros(0, features.cols()))?;
    ensure!(
        features.as_slice().iter().all(|x| x.is_finite()),
        Invalid,
        "features must be finite"
    );
    let mut layers = Vec::with_capacity(weights.layers.len());
    let mut hidden;
    let mut x = features;
    for layer in &weights.layers {
        let (forward, f) = direction(x, &layer.forward, layer.cell_clip, false)?;
        let (backward, b) = direction(x, &layer.backward, layer.cell_clip, true)?;
        let width = forward.cols() + backward.cols();
        size(features.rows(), width)?;
        hidden = Mat::zeros(features.rows(), width);
        for t in 0..features.rows() {
            let (left, right) = hidden.row_mut(t).split_at_mut(forward.cols());
            left.copy_from_slice(forward.row(t));
            right.copy_from_slice(backward.row(t));
        }
        layers.push([f, b]);
        x = &hidden;
    }
    let logits = project(x, &weights.fc_weights, &weights.fc_bias)?;
    Ok((
        logits,
        Activations {
            weights,
            rows: features.rows(),
            layers,
        },
    ))
}

/// `d(loss)/d(logits)` as `[T, num_classes]` row-major ->
/// `d(loss)/d(features)` as `[T, input_size]` row-major, accumulated in f64.
///
/// Pass the same weight object used by [`forward_with_activations`]. Reusing
/// activations with different weights would silently differentiate a different
/// function, so this is checked even when the two models have the same shape.
/// At the cell-clip boundary we take the inside derivative (one); outside it,
/// the cell-update derivative is zero. The output gate remains differentiable.
pub fn features_gradient(
    weights: &NetworkWeights,
    activations: &Activations<'_>,
    dlogits: &[f64],
) -> Result<Vec<f64>> {
    ensure!(
        core::ptr::eq(weights, activations.weights),
        Invalid,
        "activations belong to a different weight object"
    );
    ensure!(
        dlogits.len() == size(activations.rows, weights.num_classes)?,
        Invalid,
        "expected {} logit gradients, got {}",
        activations.rows * weights.num_classes,
        dlogits.len()
    );
    ensure!(
        dlogits.iter().all(|x| x.is_finite()),
        Invalid,
        "logit gradients must be finite"
    );
    let rows = activations.rows;
    let width = weights.layers.last().map_or(weights.input_size, |layer| {
        layer.forward.hidden + layer.backward.hidden
    });
    let mut dx = vec![0.0; size(rows, width)?];
    for t in 0..rows {
        transpose_add(
            &weights.fc_weights,
            &dlogits[t * weights.num_classes..(t + 1) * weights.num_classes],
            &mut dx[t * width..(t + 1) * width],
        );
    }
    for (layer, saved) in weights.layers.iter().zip(&activations.layers).rev() {
        let mut input_gradient = vec![0.0; size(rows, layer.forward.input)?];
        backward_direction(
            &layer.forward,
            &saved[0],
            false,
            &dx,
            0,
            layer.forward.hidden + layer.backward.hidden,
            &mut input_gradient,
        );
        backward_direction(
            &layer.backward,
            &saved[1],
            true,
            &dx,
            layer.forward.hidden,
            layer.forward.hidden + layer.backward.hidden,
            &mut input_gradient,
        );
        dx = input_gradient;
    }
    ensure!(
        dx.iter().all(|x| x.is_finite()),
        Invalid,
        "feature gradients must be finite"
    );
    Ok(dx)
}

fn size(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| err!(Invalid, "matrix dimensions {rows}x{cols} overflow usize"))
}

// These three f32 operations mirror net's private helpers. In particular,
// adding the bias inside the fold would change the validated forward result.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |sum, (a, b)| sum + a * b)
}

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

fn sigmoid(x: f32) -> f32 {
    let e = (-x.abs()).exp();
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

fn direction(
    input: &Mat,
    weights: &DirectionWeights,
    clip: f32,
    reverse: bool,
) -> Result<(Mat, DirectionActivations)> {
    ensure!(
        input.as_slice().iter().all(|x| x.is_finite()),
        Invalid,
        "features must be finite"
    );
    let hidden = weights.hidden;
    let mut projected = project(input, &weights.input_kernels, &weights.biases)?;
    let mut output = Mat::zeros(input.rows(), hidden);
    let mut cells = vec![Cell::default(); size(input.rows(), hidden)?];
    let mut h = weights.initial_activation.clone();
    let mut c = weights.initial_cell.clone();
    for step in 0..input.rows() {
        let t = if reverse {
            input.rows() - 1 - step
        } else {
            step
        };
        let gates = projected.row_mut(t);
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
            let previous_cell = c[j];
            c[j] = forget * c[j] + input * candidate;
            let unclipped = clip == 0.0 || (-clip..=clip).contains(&c[j]);
            if clip > 0.0 {
                c[j] = c[j].clamp(-clip, clip);
            }
            let tanh_cell = c[j].tanh();
            h[j] = out * tanh_cell;
            cells[t * hidden + j] = Cell {
                gates: [input, forget, candidate, out],
                previous_cell,
                tanh_cell,
                unclipped,
            };
        }
        output.row_mut(t).copy_from_slice(&h);
    }
    Ok((output, DirectionActivations { cells }))
}

/// Accumulate W^T * upstream without a transposed copy of the frozen weights.
fn transpose_add(weights: &[f32], upstream: &[f64], output: &mut [f64]) {
    for (&gradient, kernel) in upstream.iter().zip(weights.chunks_exact(output.len())) {
        for (value, &weight) in output.iter_mut().zip(kernel) {
            *value += gradient * f64::from(weight);
        }
    }
}

fn backward_direction(
    weights: &DirectionWeights,
    saved: &DirectionActivations,
    reverse: bool,
    upstream: &[f64],
    offset: usize,
    width: usize,
    dx: &mut [f64],
) {
    let hidden = weights.hidden;
    let rows = saved.cells.len() / hidden;
    let mut dh = vec![0.0; hidden];
    let mut dc = vec![0.0; hidden];
    let mut dg = vec![0.0; GATES * hidden];
    for step in 0..rows {
        // BPTT reverses the recurrence, not necessarily the input's time axis.
        let t = if reverse { step } else { rows - 1 - step };
        for j in 0..hidden {
            let cell = saved.cells[t * hidden + j];
            let [input, forget, candidate, out] = cell.gates.map(f64::from);
            let tanh_cell = f64::from(cell.tanh_cell);
            let dh = dh[j] + upstream[t * width + offset + j];
            let update = if cell.unclipped {
                dc[j] + dh * out * (1.0 - tanh_cell * tanh_cell)
            } else {
                0.0
            };
            dg[j] = update * candidate * input * (1.0 - input);
            dg[hidden + j] = update * f64::from(cell.previous_cell) * forget * (1.0 - forget);
            dg[2 * hidden + j] = update * input * (1.0 - candidate * candidate);
            dg[3 * hidden + j] = dh * tanh_cell * out * (1.0 - out);
            dc[j] = update * forget;
        }
        transpose_add(
            &weights.input_kernels,
            &dg,
            &mut dx[t * weights.input..(t + 1) * weights.input],
        );
        // All gate derivatives must be ready before replacing recurrent dh,
        // since a dense recurrence can couple any two hidden units.
        dh.fill(0.0);
        if weights.diagonal {
            for (gate, &gradient) in dg.iter().enumerate() {
                dh[gate % hidden] += gradient * f64::from(weights.recurrent_kernels[gate]);
            }
        } else {
            transpose_add(&weights.recurrent_kernels, &dg, &mut dh);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear() -> NetworkWeights {
        NetworkWeights {
            layers: vec![],
            fc_weights: vec![1.0, 2.0, -3.0, 4.0],
            fc_bias: vec![0.5, -0.25],
            input_size: 2,
            num_classes: 2,
        }
    }

    #[test]
    fn linear_head_and_empty_sequence() {
        let weights = linear();
        let features = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let (logits, saved) = forward_with_activations(&weights, &features).unwrap();
        assert_eq!(logits, net::forward(&weights, &features).unwrap());
        assert_eq!(
            features_gradient(&weights, &saved, &[2.0, 3.0, -1.0, 2.0]).unwrap(),
            [-7.0, 16.0, -7.0, 6.0]
        );
        let (logits, saved) = forward_with_activations(&weights, &Mat::zeros(0, 2)).unwrap();
        assert_eq!(logits, Mat::zeros(0, 2));
        assert!(features_gradient(&weights, &saved, &[]).unwrap().is_empty());
    }

    #[test]
    fn invalid_inputs_and_mismatched_activations_return_errors() {
        let weights = linear();
        assert!(forward_with_activations(&weights, &Mat::zeros(1, 3)).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                forward_with_activations(&weights, &Mat::from_vec(1, 2, vec![value, 0.0])).is_err()
            );
        }
        let (_, saved) = forward_with_activations(&weights, &Mat::zeros(1, 2)).unwrap();
        assert!(features_gradient(&weights, &saved, &[0.0]).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(features_gradient(&weights, &saved, &[value, 0.0]).is_err());
        }
        assert!(features_gradient(&weights, &saved, &[f64::MAX, f64::MAX]).is_err());
        assert!(features_gradient(&weights.clone(), &saved, &[0.0, 0.0]).is_err());
        let mut malformed = linear();
        malformed.fc_weights.pop();
        assert!(forward_with_activations(&malformed, &Mat::zeros(0, 2)).is_err());
    }
}
