"""Read ML Kit's handwriting sequence nets without a TFLite interpreter.

Gate axis 0 is ALWAYS (input, forget, cell, output), taken from operator slots,
not tensor names or tensor indices. All returned weights are float32.

Important correction to SPEC.md: historical hybrid UINT8 tensors with zp=0
store signed int8 *bit patterns*. Reinterpret those bytes before multiplying
by scale; neither unsigned v*scale nor (v-128)*scale is correct. See TensorFlow
v1.13.1 tensorflow/lite/tools/optimize/quantize_weights.cc,
SymmetricQuantizeTensor, and kernels/fully_connected.cc, EvalHybrid.
"""
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import tflite

GATES = ("input", "forget", "cell", "output")


@dataclass(frozen=True)
class DirectionWeights:
    input_kernels: np.ndarray       # [4, hidden, input]
    recurrent_kernels: np.ndarray   # [4, hidden] (Indy) or [4, hidden, hidden]
    biases: np.ndarray              # [4, hidden]
    initial_activation: np.ndarray  # [hidden]
    initial_cell: np.ndarray        # [hidden]

    @property
    def hidden_size(self):
        return self.biases.shape[1]

    @property
    def diagonal(self):
        return self.recurrent_kernels.ndim == 2


@dataclass(frozen=True)
class LayerWeights:
    forward: DirectionWeights
    backward: DirectionWeights
    cell_clip: float
    time_major: bool
    custom_options: bytes = b""     # preserved for eventual oracle verification


@dataclass(frozen=True)
class NetworkWeights:
    layers: tuple[LayerWeights, ...]
    fc_weights: np.ndarray          # [classes, 2 * last_hidden]
    fc_bias: np.ndarray             # [classes]
    input_size: int

    @property
    def num_classes(self):
        return self.fc_bias.size


def dequantize(values, scale, zero_point, *, legacy_hybrid_uint8=False):
    """Per-tensor affine dequantization, with explicit legacy hybrid handling.

    The legacy flag is only appropriate for float-input hybrid weight tensors;
    UINT8 with a nonzero zero point retains ordinary affine semantics. Never
    choose an encoding heuristically from the mean of an individual tensor.
    """
    values = np.asarray(values)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")
    if legacy_hybrid_uint8 and values.dtype == np.uint8 and zero_point == 0:
        values = values.view(np.int8)
    return (values.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def _custom_options(data):
    """Hypothesis: little-endian {float cell_clip; u8 activation; bools; pad}.

    [4,1,1] matches TANH/merge_outputs/time_major in the builtin models.
    The final byte is treated as padding: Kannada has 0x02 in one layer while
    the others have 0x00. This is not an established custom-op ABI; keep the
    original bytes on LayerWeights so an on-device oracle can check it.
    """
    if len(data) != 8 or data[4:7] != bytes((4, 1, 1)):
        raise ValueError(f"Unsupported IndyLSTM custom options: {data.hex()}")
    clip = struct.unpack("<f", data[:4])[0]
    if not np.isfinite(clip) or clip < 0:
        raise ValueError(f"Invalid cell clipping threshold: {clip}")
    return clip, True


class _Tensors:
    def __init__(self, model, graph):
        self.model, self.graph = model, graph

    def tensor(self, index):
        if not 0 <= index < self.graph.TensorsLength():
            raise ValueError(f"Missing or invalid tensor index: {index}")
        return self.graph.Tensors(index)

    def shape(self, index):
        return tuple(int(v) for v in self.tensor(index).ShapeAsNumpy())

    def read(self, index, *, state=False):
        tensor = self.tensor(index)
        shape = self.shape(index)
        types = {tflite.TensorType.FLOAT32: "<f4", tflite.TensorType.UINT8: "u1",
                 tflite.TensorType.INT8: "i1"}
        if tensor.Type() not in types:
            raise ValueError(f"Unsupported weight type {tensor.Type()} at tensor {index}")
        buffer = self.model.Buffers(tensor.Buffer())
        if not buffer.DataLength():
            if state and tensor.Type() == tflite.TensorType.FLOAT32:
                return np.zeros(shape, np.float32)
            raise ValueError(f"Weight tensor {index} has no data")
        data = np.frombuffer(buffer.DataAsNumpy(), dtype=types[tensor.Type()])
        if data.size != int(np.prod(shape)):
            raise ValueError(f"Tensor {index}: buffer size does not match shape {shape}")
        data = data.reshape(shape)
        if tensor.Type() != tflite.TensorType.FLOAT32:
            quant = tensor.Quantization()
            if quant is None or quant.ScaleLength() != 1 or quant.ZeroPointLength() != 1:
                raise ValueError(f"Tensor {index}: expected per-tensor quantization")
            data = dequantize(data, quant.Scale(0), quant.ZeroPoint(0),
                              legacy_hybrid_uint8=True)
        if not np.isfinite(data).all():
            raise ValueError(f"Tensor {index} contains nonfinite weights")
        return data.copy()

    def direction(self, inputs, start, bias_start, state_start, width, diagonal):
        # The nonconsecutive underlying tensor indices must remain in slot order.
        kernels = np.stack([self.read(inputs[start + g]) for g in range(4)])
        rec = np.stack([self.read(inputs[start + 4 + g]) for g in range(4)])
        biases = np.stack([self.read(inputs[bias_start + g]) for g in range(4)])
        if biases.ndim != 2 or biases.shape[0] != 4:
            raise ValueError("Expected four vector gate biases")
        hidden = biases.shape[1]
        if hidden == 0 or kernels.shape != (4, hidden, width):
            raise ValueError(f"Input kernels {kernels.shape} do not match hidden/input {hidden}/{width}")
        expected_rec = (4, hidden) if diagonal else (4, hidden, hidden)
        if rec.shape != expected_rec:
            raise ValueError(f"Recurrent kernels {rec.shape} do not match {expected_rec}")
        states = [self.read(inputs[state_start + g], state=True) for g in range(2)]
        if any(s.shape != (1, hidden) for s in states):
            raise ValueError("Expected single-batch activation and cell states")
        return DirectionWeights(kernels, rec, biases, states[0][0], states[1][0])


def _options(op, cls, expected_type):
    if op.BuiltinOptionsType() != expected_type or op.BuiltinOptions() is None:
        raise ValueError(f"Missing or unsupported {cls.__name__}")
    options = cls()
    table = op.BuiltinOptions()
    options.Init(table.Bytes, table.Pos)
    return options


def load_weights(tflite_path):
    """Extract a stack of bidirectional (Indy)LSTMs followed by one linear FC.

    All 27 legacy handwriting script nets use this topology. Gesture classifiers
    (emoji, autodraw, shapes) and newer scribe networks are deliberately rejected:
    their reduction/dense heads are not per-timestep handwriting logits.
    Unsupported peepholes, projections, CIFG and auxiliary inputs fail loudly.
    """
    data = Path(tflite_path).read_bytes()
    if len(data) < 8 or data[4:8] != b"TFL3":
        raise ValueError("Not a TFLite flatbuffer")
    model = tflite.Model.GetRootAsModel(data, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("Expected exactly one subgraph")
    graph = model.Subgraphs(0)
    if graph.InputsLength() != 1 or graph.OutputsLength() != 1:
        raise ValueError("Expected one input and one output")
    tensors = _Tensors(model, graph)
    current = graph.Inputs(0)
    shape = tensors.shape(current)
    if len(shape) != 3 or shape[-1] != 10 or shape[:2] != (1, 1):
        raise ValueError(f"Expected a handwriting input [1, 1, 10], got {shape}")
    if tensors.tensor(current).Type() != tflite.TensorType.FLOAT32:
        raise ValueError("Only float-input hybrid networks are supported")
    width = input_size = shape[-1]
    layers = []
    fc_weights = fc_bias = None
    for k in range(graph.OperatorsLength()):
        op = graph.Operators(k)
        code = model.OperatorCodes(op.OpcodeIndex())
        builtin = code.BuiltinCode()
        inputs = list(map(int, op.InputsAsNumpy()))
        if not inputs or inputs[0] != current or op.OutputsLength() != 1:
            raise ValueError(f"Operator {k}: expected a connected, single-output chain")
        if code.CustomCode() == b"bidirectional_sequence_indylstm":
            if len(inputs) != 29:
                raise ValueError("Expected 29 IndyLSTM inputs")
            raw_options = bytes(op.CustomOptionsAsNumpy())
            clip, time_major = _custom_options(raw_options)
            fw = tensors.direction(inputs, 1, 9, 25, width, True)
            bw = tensors.direction(inputs, 13, 21, 27, width, True)
            layers.append(LayerWeights(fw, bw, clip, time_major, raw_options))
            width = fw.hidden_size + bw.hidden_size
        elif builtin == tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM:
            if len(inputs) != 48:
                raise ValueError("Expected 48 builtin bidirectional LSTM inputs")
            absent = (9, 10, 11, 16, 17, 26, 27, 28, 33, 34, *range(39, 48))
            if any(inputs[j] != -1 for j in absent):
                raise ValueError("Peepholes, projections and auxiliary LSTM inputs are unsupported")
            options = _options(op, tflite.BidirectionalSequenceLSTMOptions,
                               tflite.BuiltinOptions.BidirectionalSequenceLSTMOptions)
            clip = options.CellClip()
            if (options.FusedActivationFunction() != tflite.ActivationFunctionType.TANH
                    or not options.MergeOutputs() or options.ProjClip() != 0
                    or options.AsymmetricQuantizeInputs()
                    or not np.isfinite(clip) or clip < 0):
                raise ValueError("Unsupported builtin bidirectional LSTM options")
            fw = tensors.direction(inputs, 1, 12, 35, width, False)
            bw = tensors.direction(inputs, 18, 29, 37, width, False)
            layers.append(LayerWeights(fw, bw, clip, options.TimeMajor()))
            width = fw.hidden_size + bw.hidden_size
        elif builtin == tflite.BuiltinOperator.FULLY_CONNECTED:
            if not layers or k != graph.OperatorsLength() - 1 or len(inputs) != 3:
                raise ValueError("Expected exactly one final fully connected layer")
            options = _options(op, tflite.FullyConnectedOptions,
                               tflite.BuiltinOptions.FullyConnectedOptions)
            if (options.FusedActivationFunction() != tflite.ActivationFunctionType.NONE
                    or options.WeightsFormat() != tflite.FullyConnectedOptionsWeightsFormat.DEFAULT
                    or options.AsymmetricQuantizeInputs()):
                raise ValueError("Expected a plain linear fully connected layer")
            fc_weights, fc_bias = tensors.read(inputs[1]), tensors.read(inputs[2])
            if fc_bias.ndim != 1 or fc_bias.size == 0 or fc_weights.shape != (fc_bias.size, width):
                raise ValueError("Fully connected weights/bias do not match the recurrent output")
            width = fc_bias.size
        else:
            raise ValueError(f"Unsupported operator {k}: {code.CustomCode() or builtin!r}")
        current = op.Outputs(0)
        # Intermediate shapes are absent in the shipped flatbuffers. In that
        # case the next layer's kernels validate the inferred output width.
        output_shape = tensors.shape(current)
        if output_shape and output_shape[-1] != width:
            raise ValueError(f"Operator {k}: output width does not match extracted weights")
        if tensors.tensor(current).Type() != tflite.TensorType.FLOAT32:
            raise ValueError(f"Operator {k}: expected float output")
    if fc_weights is None or current != graph.Outputs(0):
        raise ValueError("Network does not end in a fully connected logits output")
    return NetworkWeights(tuple(layers), fc_weights, fc_bias, input_size)
