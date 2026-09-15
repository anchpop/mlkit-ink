"""Offline math/flatbuffer tests, plus integration checks for downloaded script nets.

Run with PYTHONPATH set to the project root:
    .venv/bin/python -m unittest discover -s tests -p test_indylstm.py -v
No tests download models. Set INK_MODEL_DIAGNOSTICS=1 for per-net measurements.
"""
from dataclasses import replace
import os
from pathlib import Path
import struct
import tempfile
import unittest

import flatbuffers
import numpy as np
import tflite

from src.indylstm import _sigmoid, bidirectional, direction, forward, logits
from src.tflite_weights import (DirectionWeights, LayerWeights, NetworkWeights,
                                _custom_options, dequantize, load_weights)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_features(length=96):
    """Smooth, bounded curve-like coefficients; NOT a verified feature encoder."""
    t = np.linspace(0, 8 * np.pi, length, dtype=np.float32)
    return np.column_stack((.12 + .1 * np.cos(t), .3 * np.sin(t),
                            .2 * np.cos(.5 * t), .2 * np.sin(.7 * t),
                            .05 * np.cos(2 * t), .05 * np.sin(2 * t),
                            np.cos(t), np.sin(t), np.full(length, .04),
                            np.arange(length) % 12 != 11)).astype(np.float32)


def small_direction(diagonal=True):
    rng = np.random.default_rng(7)
    kernels = rng.normal(0, .2, (4, 3, 2)).astype(np.float32)
    rec_shape = (4, 3) if diagonal else (4, 3, 3)
    return DirectionWeights(kernels, rng.normal(0, .3, rec_shape).astype(np.float32),
                            rng.normal(0, .1, (4, 3)).astype(np.float32),
                            np.array([.1, -.2, .3], np.float32),
                            np.array([.2, -.1, -.3], np.float32))


def scalar_reference(x, weights, clip, reverse=False):
    """Independent float64 scalar loops, including non-diagonal recurrent terms."""
    h = weights.initial_activation.astype(float).copy()
    c = weights.initial_cell.astype(float).copy()
    out = np.zeros((len(x), len(h)))
    for t in (range(len(x) - 1, -1, -1) if reverse else range(len(x))):
        old_h, old_c = h.copy(), c.copy()
        for j in range(len(h)):
            gates = []
            for g in range(4):
                v = float(weights.biases[g, j])
                for k in range(x.shape[1]):
                    v += float(weights.input_kernels[g, j, k]) * float(x[t, k])
                if weights.diagonal:
                    v += float(weights.recurrent_kernels[g, j]) * old_h[j]
                else:
                    for k in range(len(h)):
                        v += float(weights.recurrent_kernels[g, j, k]) * old_h[k]
                gates.append(np.tanh(v) if g == 2 else 1 / (1 + np.exp(-v)))
            i, f, candidate, o = gates
            c[j] = f * old_c[j] + i * candidate
            if clip:
                c[j] = max(-clip, min(clip, c[j]))
            h[j] = o * np.tanh(c[j])
        out[t] = h
    return out


def fixture_model(*, diagonal=True, optional=False, padding=0):
    """Tiny genuine flatbuffer with misleading names and interleaved gate indices."""
    b = flatbuffers.Builder(4096)
    tensors, buffers = [], []
    tflite.BufferStart(b)
    buffers.append(tflite.BufferEnd(b))

    def tensor(shape, data=None, name="misleading/output", quantized=False):
        buffer_index = 0
        quant = 0
        if data is not None:
            values = np.asarray(data, dtype=np.uint8 if quantized else np.float32)
            raw = b.CreateByteVector(values.tobytes())
            tflite.BufferStart(b)
            tflite.BufferAddData(b, raw)
            buffers.append(tflite.BufferEnd(b))
            buffer_index = len(buffers) - 1
        if quantized:
            scale = b.CreateNumpyVector(np.array([.125], np.float32))
            zp = b.CreateNumpyVector(np.array([0], np.int64))
            tflite.QuantizationParametersStart(b)
            tflite.QuantizationParametersAddScale(b, scale)
            tflite.QuantizationParametersAddZeroPoint(b, zp)
            quant = tflite.QuantizationParametersEnd(b)
        shp = b.CreateNumpyVector(np.array(shape, np.int32))
        nm = b.CreateString(name)
        tflite.TensorStart(b)
        tflite.TensorAddShape(b, shp)
        tflite.TensorAddName(b, nm)
        tflite.TensorAddType(b, tflite.TensorType.UINT8 if quantized else tflite.TensorType.FLOAT32)
        tflite.TensorAddBuffer(b, buffer_index)
        if quant:
            tflite.TensorAddQuantization(b, quant)
        tensors.append(tflite.TensorEnd(b))
        return len(tensors) - 1

    inp = tensor((1, 1, 10))
    slots = [inp] + [-1] * (28 if diagonal else 47)
    for start, bias_start, state_start in ((1, 9, 25), (13, 21, 27)) if diagonal else ((1, 12, 35), (18, 29, 37)):
        for g in (0, 2, 1, 3):
            # Distinct values per gate; 255 represents -1, not +255 or +127.
            slots[start + g] = tensor((2, 10), np.full((2, 10), 255 - g), quantized=True)
            rec_shape = (2,) if diagonal else (2, 2)
            slots[start + 4 + g] = tensor(rec_shape, np.full(rec_shape, .1 * (g + 1)))
            slots[bias_start + g] = tensor((2,), [g + .25, g + .5])
        slots[state_start] = tensor((1, 2))
        slots[state_start + 1] = tensor((1, 2))
    if optional and not diagonal:
        slots[9] = slots[12]
    recurrent_output = tensor(())  # Missing intermediate shape in real nets too.
    fc_w = tensor((3, 4), np.arange(12).reshape(3, 4), quantized=True)
    fc_b = tensor((3,), [1, 2, 3])
    out = tensor((1, 1, 3))

    def operator(inputs, output, opcode, opts=0, opt_type=0, custom=b""):
        ins = b.CreateNumpyVector(np.array(inputs, np.int32))
        outs = b.CreateNumpyVector(np.array([output], np.int32))
        raw = b.CreateByteVector(custom) if custom else 0
        tflite.OperatorStart(b)
        tflite.OperatorAddOpcodeIndex(b, opcode)
        tflite.OperatorAddInputs(b, ins)
        tflite.OperatorAddOutputs(b, outs)
        if opts:
            tflite.OperatorAddBuiltinOptions(b, opts)
            tflite.OperatorAddBuiltinOptionsType(b, opt_type)
        if raw:
            tflite.OperatorAddCustomOptions(b, raw)
        return tflite.OperatorEnd(b)

    if diagonal:
        rnn = operator(slots, recurrent_output, 0,
                       custom=struct.pack("<fBBBB", 50., 4, 1, 1, padding))
    else:
        tflite.BidirectionalSequenceLSTMOptionsStart(b)
        tflite.BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(b, 4)
        tflite.BidirectionalSequenceLSTMOptionsAddCellClip(b, 50.)
        tflite.BidirectionalSequenceLSTMOptionsAddMergeOutputs(b, True)
        tflite.BidirectionalSequenceLSTMOptionsAddTimeMajor(b, True)
        opts = tflite.BidirectionalSequenceLSTMOptionsEnd(b)
        rnn = operator(slots, recurrent_output, 0, opts,
                       tflite.BuiltinOptions.BidirectionalSequenceLSTMOptions)
    tflite.FullyConnectedOptionsStart(b)
    fc_opts = tflite.FullyConnectedOptionsEnd(b)
    fc = operator([recurrent_output, fc_w, fc_b], out, 1, fc_opts,
                  tflite.BuiltinOptions.FullyConnectedOptions)

    def offsets(values):
        b.StartVector(4, len(values), 4)
        for value in reversed(values):
            b.PrependUOffsetTRelative(value)
        return b.EndVector()

    ops = offsets([rnn, fc])
    ts = offsets(tensors)
    ins = b.CreateNumpyVector(np.array([inp], np.int32))
    outs = b.CreateNumpyVector(np.array([out], np.int32))
    tflite.SubGraphStart(b)
    tflite.SubGraphAddTensors(b, ts)
    tflite.SubGraphAddInputs(b, ins)
    tflite.SubGraphAddOutputs(b, outs)
    tflite.SubGraphAddOperators(b, ops)
    sg = tflite.SubGraphEnd(b)
    custom_name = b.CreateString("bidirectional_sequence_indylstm")
    codes = []
    for code in (32 if diagonal else 52, 9):
        tflite.OperatorCodeStart(b)
        tflite.OperatorCodeAddBuiltinCode(b, code)
        tflite.OperatorCodeAddDeprecatedBuiltinCode(b, code)
        if code == 32:
            tflite.OperatorCodeAddCustomCode(b, custom_name)
        codes.append(tflite.OperatorCodeEnd(b))
    codes, graphs, bufs = offsets(codes), offsets([sg]), offsets(buffers)
    tflite.ModelStart(b)
    tflite.ModelAddVersion(b, 3)
    tflite.ModelAddSubgraphs(b, graphs)
    tflite.ModelAddBuffers(b, bufs)
    tflite.ModelAddOperatorCodes(b, codes)
    model = tflite.ModelEnd(b)
    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())


class MathTests(unittest.TestCase):
    def test_scalar_reference_both_recurrences_and_directions(self):
        x = np.random.default_rng(9).normal(size=(11, 2)).astype(np.float32)
        for diagonal in (True, False):
            for reverse in (True, False):
                for clip in (0, .1, 50):
                    with self.subTest(diagonal=diagonal, reverse=reverse, clip=clip):
                        w = small_direction(diagonal)
                        np.testing.assert_allclose(direction(x, w, clip, reverse=reverse),
                                                   scalar_reference(x, w, clip, reverse),
                                                   atol=1e-7, rtol=1e-5)

    def test_diagonal_matches_full_diagonal_matrices(self):
        w = small_direction()
        full = replace(w, recurrent_kernels=np.stack([np.diag(g) for g in w.recurrent_kernels]))
        x = np.random.default_rng(8).normal(size=(17, 2)).astype(np.float32)
        np.testing.assert_allclose(direction(x, w), direction(x, full), atol=1e-7)

    def test_clips_cell_before_output_not_hidden(self):
        w = DirectionWeights(np.zeros((4, 1, 1), np.float32), np.zeros((4, 1), np.float32),
                             np.array([[30], [30], [2], [30]], np.float32),
                             np.zeros(1, np.float32), np.array([3], np.float32))
        x = np.zeros((2, 1), np.float32)
        np.testing.assert_allclose(direction(x, w, .1), np.tanh(.1), atol=1e-7)
        self.assertGreater(float(direction(x, w, 0)[0, 0]), .99)

    def test_bidirectional_and_linear_head_and_state_reset(self):
        fw = small_direction()
        bw = replace(fw, biases=-fw.biases)
        layer = LayerWeights(fw, bw, .2, True)
        w = NetworkWeights((layer,), np.arange(24, dtype=np.float32).reshape(4, 6),
                           np.array([.1, .2, .3, .4], np.float32), 2)
        x = np.random.default_rng(3).normal(size=(13, 2)).astype(np.float32)
        expected_hidden = np.concatenate((scalar_reference(x, fw, .2),
                                          scalar_reference(x, bw, .2, True)), axis=1)
        np.testing.assert_allclose(bidirectional(x, layer), expected_hidden, atol=1e-7)
        expected = expected_hidden @ w.fc_weights.T + w.fc_bias
        np.testing.assert_allclose(forward(w, x), expected, atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(forward(w, x), forward(w, x))
        np.testing.assert_array_equal(fw.initial_cell, np.array([.2, -.1, -.3], np.float32))
        self.assertEqual(forward(w, np.zeros((0, 2))).shape, (0, 4))
        for bad in (np.zeros((1, 1, 2)), np.zeros((2, 3)), [[np.nan, 0]], [[np.inf, 0]]):
            with self.assertRaises(ValueError):
                forward(w, bad)

    def test_stable_sigmoid(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_array_equal(_sigmoid(np.array([-1e30, 0, 1e30], np.float32)), [0, .5, 1])


class LoaderTests(unittest.TestCase):
    def test_affine_and_legacy_dequantization(self):
        raw = np.array([0, 1, 127, 128, 129, 255], np.uint8)
        np.testing.assert_array_equal(dequantize(raw, .5, 0, legacy_hybrid_uint8=True),
                                      [0, .5, 63.5, -64, -63.5, -.5])
        np.testing.assert_array_equal(dequantize(raw, .5, 128, legacy_hybrid_uint8=True),
                                      [-64, -63.5, -.5, 0, .5, 63.5])
        np.testing.assert_array_equal(dequantize(raw, .5, 0), raw.astype(np.float32) * .5)
        np.testing.assert_array_equal(dequantize(raw.view(np.int8), .5, 0),
                                      dequantize(raw, .5, 0, legacy_hybrid_uint8=True))
        for scale in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                dequantize(raw, scale, 0)

    def test_custom_options_including_kannada_padding(self):
        for pad in (0, 2):
            self.assertEqual(_custom_options(struct.pack("<fBBBB", 50, 4, 1, 1, pad)), (50., True))
        for data in (b"", struct.pack("<fBBBB", 50, 3, 1, 1, 0),
                     struct.pack("<fBBBB", -1, 4, 1, 1, 0)):
            with self.assertRaises(ValueError):
                _custom_options(data)

    def test_real_flatbuffer_positional_gates_and_convenience(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.tflite"
            for diagonal in (True, False):
                path.write_bytes(fixture_model(diagonal=diagonal))
                w = load_weights(path)
                self.assertEqual(w.num_classes, 3)
                self.assertEqual(len(w.layers), 1)
                for d in (w.layers[0].forward, w.layers[0].backward):
                    self.assertEqual(d.diagonal, diagonal)
                    for g in range(4):
                        np.testing.assert_array_equal(d.input_kernels[g], -.125 * (g + 1))
                        np.testing.assert_allclose(d.recurrent_kernels[g], .1 * (g + 1))
                        np.testing.assert_array_equal(d.biases[g], [g + .25, g + .5])
                    np.testing.assert_array_equal(d.initial_cell, [0, 0])
                np.testing.assert_array_equal(w.fc_weights, np.arange(12).reshape(3, 4) * .125)
                np.testing.assert_array_equal(w.fc_bias, [1, 2, 3])
                x = synthetic_features(7)
                np.testing.assert_array_equal(logits(path, x), forward(w, x))

    def test_unsupported_graph_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.tflite"
            path.write_bytes(fixture_model(diagonal=False, optional=True))
            with self.assertRaisesRegex(ValueError, "Peepholes"):
                load_weights(path)
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "Not a TFLite"):
                load_weights(path)


class DownloadedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the old sequence nets; scribe and special classifier heads are not
        # handwriting sequence models. Catalog downloads are an explicit action.
        cls.paths = [p for p in sorted((ROOT / "models").glob("*.tflite"))
                     if not any(s in p.name for s in ("scribe", "autodraw", "emoji", "shapes"))]
        if not cls.paths:
            raise unittest.SkipTest("No downloaded handwriting sequence models")

    def test_all_downloaded_shape_flow_and_numerics(self):
        x = synthetic_features()
        diagnostics = bool(os.environ.get("INK_MODEL_DIAGNOSTICS"))
        nondegenerate = 0
        for path in self.paths:
            with self.subTest(model=path.name):
                w = load_weights(path)
                hidden = x
                for layer in w.layers:
                    for d in (layer.forward, layer.backward):
                        self.assertEqual(d.input_kernels.shape, (4, d.hidden_size, hidden.shape[1]))
                        self.assertTrue(np.isfinite(d.input_kernels).all())
                        self.assertTrue(np.isfinite(d.recurrent_kernels).all())
                        self.assertGreater(float(np.std(d.input_kernels)), 0)
                        np.testing.assert_array_equal(d.initial_activation, 0)
                        np.testing.assert_array_equal(d.initial_cell, 0)
                    hidden = bidirectional(hidden, layer)
                    self.assertEqual(hidden.shape, (len(x), layer.forward.hidden_size + layer.backward.hidden_size))
                    self.assertTrue(np.isfinite(hidden).all())
                    self.assertLessEqual(float(np.max(np.abs(hidden))), 1.)
                y = hidden @ w.fc_weights.T + w.fc_bias
                self.assertEqual(y.shape, (len(x), w.num_classes))
                self.assertTrue(np.isfinite(y).all())
                self.assertGreater(float(y.std()), 0)
                self.assertGreater(float(np.std(y, axis=0).max()), 1e-4)
                classes = len(np.unique(y.argmax(axis=1)))
                nondegenerate += classes > 1
                # Softmax is measured, not asserted: CTC blanks and pretrained
                # large FC scales legitimately yield very peaked distributions.
                probs = np.exp(y - y.max(axis=1, keepdims=True))
                probs /= probs.sum(axis=1, keepdims=True)
                if diagnostics:
                    print(f"\n{path.name}: layers={len(w.layers)} hidden={w.layers[0].forward.hidden_size} "
                          f"indy={w.layers[0].forward.diagonal} shape={y.shape} "
                          f"range=[{y.min():.6g},{y.max():.6g}] std={y.std():.6g} "
                          f"argmax_classes={classes} median_maxprob={np.median(probs.max(axis=1)):.8g}")
        # This is not an accuracy assertion, nor evidence of a correct feature
        # encoder. Even plausible synthetic input can be all-blank for a script.
        self.assertGreater(nondegenerate, 0)

    def test_quantized_distributions(self):
        diagnostics = bool(os.environ.get("INK_MODEL_DIAGNOSTICS"))
        for path in self.paths:
            m = tflite.Model.GetRootAsModel(path.read_bytes(), 0)
            sg = m.Subgraphs(0)
            for index in range(sg.TensorsLength()):
                t = sg.Tensors(index)
                if t.Type() != tflite.TensorType.UINT8:
                    continue
                with self.subTest(model=path.name, tensor=index):
                    q = t.Quantization()
                    raw = np.asarray(m.Buffers(t.Buffer()).DataAsNumpy())
                    self.assertEqual(q.ZeroPoint(0), 0)
                    signed = raw.view(np.int8)
                    decoded = dequantize(raw, q.Scale(0), 0, legacy_hybrid_uint8=True)
                    self.assertTrue(np.isfinite(decoded).all())
                    self.assertGreater(float(decoded.std()), 0)
                    self.assertGreater(float(decoded.max()), 0)
                    self.assertLess(float(decoded.min()), 0)
                    # 0/255 are signed 0/-1, NOT saturation. Signed extremes
                    # should be rare, while the distribution clusters at zero.
                    self.assertLess(float(np.mean(np.abs(signed.astype(int)) >= 127)), .02)
                    self.assertLess(abs(float(signed.mean())), float(signed.std()))
                    self.assertLess(float(signed.std()), float(raw.std()))
                    if diagnostics and index in (sg.Operators(0).Inputs(1),
                                                  sg.Operators(sg.OperatorsLength() - 1).Inputs(1)):
                        scale = q.Scale(0)
                        print(f"\n{path.name} tensor={index} scale={scale:.9g} zp={q.ZeroPoint(0)} "
                              f"raw_mean/std={raw.mean():.6g}/{raw.std():.6g} "
                              f"unsigned_mean/std={raw.mean()*scale:.6g}/{raw.std()*scale:.6g} "
                              f"offset128_mean/std={(raw.mean()-128)*scale:.6g}/{raw.std()*scale:.6g} "
                              f"signed_mean/std={decoded.mean():.6g}/{decoded.std():.6g} "
                              f"raw_p0/255={np.mean(raw==0):.6g}/{np.mean(raw==255):.6g} "
                              f"signed_extremes={np.mean(np.abs(signed.astype(int))>=127):.6g}")


if __name__ == "__main__":
    unittest.main()
