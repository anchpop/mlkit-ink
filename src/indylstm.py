"""NumPy float32 reference forward pass for ML Kit handwriting sequence nets.

This evaluates dequantized weights, NOT TFLite's dynamic activation quantization
or its approximate activation kernels. Differences can accumulate through the
stack (no small-error guarantee); custom-op agreement needs an on-device oracle.
"""
import numpy as np

from .tflite_weights import DirectionWeights, LayerWeights, NetworkWeights, load_weights


def _sigmoid(x):
    # exp(-abs(x)) cannot overflow, even for unusually large finite preactivations.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def direction(features, weights: DirectionWeights, cell_clip=50.0, *, reverse=False):
    """One LSTM direction; backward outputs are restored to original time order.

    Recurrence uses a product for Indy diagonals, a matvec for full LSTMs. Biases
    are already trained/exported: do not add a separate forget-bias constant.
    Cell clipping applies AFTER the cell update and BEFORE tanh(cell).
    Initial states are copied on each call, never retained between sequences.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != weights.input_kernels.shape[2]:
        raise ValueError("Features must have shape [time, input_size]")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    if not np.isfinite(cell_clip) or cell_clip < 0:
        raise ValueError("Cell clipping threshold must be finite and nonnegative")
    hidden = weights.hidden_size
    # Project all input timesteps/gates in one BLAS call; recurrence stays serial.
    projected = (x @ weights.input_kernels.reshape(4 * hidden, x.shape[1]).T)
    projected = projected.reshape(len(x), 4, hidden) + weights.biases
    h = weights.initial_activation.copy()
    c = weights.initial_cell.copy()
    output = np.empty((len(x), hidden), dtype=np.float32)
    recurrent = weights.recurrent_kernels
    if not weights.diagonal:
        recurrent = recurrent.reshape(4 * hidden, hidden)
    timesteps = range(len(x) - 1, -1, -1) if reverse else range(len(x))
    for t in timesteps:
        rec = recurrent * h if weights.diagonal else (recurrent @ h).reshape(4, hidden)
        gates = projected[t] + rec
        i, f, candidate, o = gates
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(candidate)
        if cell_clip:
            c = np.clip(c, -cell_clip, cell_clip)
        h = _sigmoid(o) * np.tanh(c)
        output[t] = h
    return output


def bidirectional(features, weights: LayerWeights):
    """Concatenate forward/backward hidden outputs, never cell states."""
    return np.concatenate((direction(features, weights.forward, weights.cell_clip),
                           direction(features, weights.backward, weights.cell_clip,
                                     reverse=True)), axis=1)


def forward(weights: NetworkWeights, features):
    """Evaluate preloaded weights: [T, 10] -> unnormalized logits [T, classes].

    Batch size is one. Thus the graph's time_major flag makes no difference to
    this 2-D API (the corresponding 3-D tensors are [T,1,D] or [1,T,D]). Empty
    sequences return [0, classes]. No softmax, CTC decoding or state caching.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != weights.input_size:
        raise ValueError(f"Expected features [T, {weights.input_size}], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    for layer in weights.layers:
        x = bidirectional(x, layer)
    return x @ weights.fc_weights.T + weights.fc_bias


def logits(tflite_path, features):
    """Convenience entry point. Use load_weights + forward to reuse a model."""
    return forward(load_weights(tflite_path), features)
