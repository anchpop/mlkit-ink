//! Read ML Kit's handwriting nets straight out of the TFLite flatbuffer.
//!
//! There is no interpreter here and no TFLite dependency: the nets use a custom
//! `bidirectional_sequence_indylstm` op that stock LiteRT does not implement
//! anyway, so we read the weight tensors and evaluate them ourselves ([`crate::net`]).
//!
//! The quantization detail that cost the most to establish: hybrid tensors
//! declared `UINT8` with `zero_point == 0` hold *signed int8 bit patterns*.
//! Neither `v * scale` nor `(v - 128) * scale` is correct; the bytes must be
//! reinterpreted as `i8` first.
//!

use alloc::vec::Vec;

use crate::error::Result;

/// Gate order on axis 0 is always (input, forget, cell, output), taken from
/// operator input slots rather than tensor names.
pub const GATES: usize = 4;

/// One LSTM direction's weights, all dequantized to f32.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionWeights {
    /// `[GATES][hidden][input]`, flattened row-major.
    pub input_kernels: Vec<f32>,
    /// `[GATES][hidden]` for IndyLSTM's diagonal recurrence, or
    /// `[GATES][hidden][hidden]` for a full one.
    pub recurrent_kernels: Vec<f32>,
    pub diagonal: bool,
    /// `[GATES][hidden]`. Already trained and exported: do not add a separate
    /// forget-bias constant.
    pub biases: Vec<f32>,
    pub initial_activation: Vec<f32>,
    pub initial_cell: Vec<f32>,
    pub hidden: usize,
    pub input: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerWeights {
    pub forward: DirectionWeights,
    pub backward: DirectionWeights,
    pub cell_clip: f32,
    /// Kept verbatim so the custom-op ABI stays checkable against the oracle.
    pub custom_options: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkWeights {
    pub layers: Vec<LayerWeights>,
    /// `[num_classes][2 * last_hidden]`, flattened row-major.
    pub fc_weights: Vec<f32>,
    pub fc_bias: Vec<f32>,
    pub input_size: usize,
    pub num_classes: usize,
}

/// Extract a stack of bidirectional (Indy)LSTMs followed by one linear head.
///
/// Gesture classifiers (emoji, autodraw, shapes) and newer scribe networks are
/// rejected on purpose: their heads are not per-timestep handwriting logits.
/// Peepholes, projections, CIFG and auxiliary inputs fail loudly.
pub fn load_weights(data: &[u8]) -> Result<NetworkWeights> {
    let fb = Flatbuffer(data);
    ensure!(
        fb.bytes(4, 4)? == b"TFL3",
        Format,
        "byte 4: expected TFL3 identifier"
    );
    let model = fb.table(fb.indirect(0)?)?;
    let graphs = model.vector(8, 4)?;
    ensure!(
        graphs.len == 1,
        Unsupported,
        "expected exactly one subgraph (gesture/scribe nets are unsupported)"
    );
    let graph = graphs.table(0)?;
    let graph_inputs = graph.vector(6, 4)?;
    let graph_outputs = graph.vector(8, 4)?;
    ensure!(
        graph_inputs.len == 1 && graph_outputs.len == 1,
        Unsupported,
        "expected one handwriting input and output (gesture/scribe nets are unsupported)"
    );
    let tensors = Tensors {
        tensors: graph.vector(4, 4)?,
        buffers: model.vector(12, 4)?,
    };
    let mut current = graph_inputs.i32(0)?;
    ensure!(
        tensors.shape(current)? == [1, 1, 10],
        Unsupported,
        "expected handwriting input [1, 1, 10] (gesture/scribe nets are unsupported)"
    );
    ensure!(
        tensors.tensor(current)?.u8(6, 0)? == 0,
        Unsupported,
        "only float-input hybrid networks are supported"
    );
    let codes = model.vector(6, 4)?;
    let ops = graph.vector(10, 4)?;
    let mut width = 10;
    let mut layers = Vec::new();
    let mut head = None;
    for k in 0..ops.len {
        let op = ops.table(k)?;
        let code = codes.table(op.u32(4, 0)? as usize)?;
        // The generated accessor falls back to the old int8 field below 127.
        let builtin = code.i32(10, 0)?;
        let builtin = if builtin < 127 {
            i32::from(code.u8(4, 0)? as i8)
        } else {
            builtin
        };
        let custom = code.string(6)?;
        let inputs = op.vector(6, 4)?;
        let outputs = op.vector(8, 4)?;
        ensure!(
            inputs.len > 0 && inputs.i32(0)? == current && outputs.len == 1,
            Format,
            "byte {}: operator {k} is not a connected single-output chain",
            op.pos
        );
        if custom == b"bidirectional_sequence_indylstm" {
            ensure!(
                inputs.len == 29,
                Format,
                "byte {}: expected 29 IndyLSTM inputs",
                op.pos
            );
            let raw = op.vector(14, 1)?;
            let raw_options = fb.bytes(raw.pos, raw.len)?;
            let cell_clip = custom_options(raw_options, raw.pos)?;
            let forward = tensors.direction(inputs, (1, 9, 25), width, true)?;
            let backward = tensors.direction(inputs, (13, 21, 27), width, true)?;
            width = forward
                .hidden
                .checked_add(backward.hidden)
                .ok_or_else(|| err!(Format, "byte {}: output width overflow", op.pos))?;
            layers.push(LayerWeights {
                forward,
                backward,
                cell_clip,
                custom_options: raw_options.to_vec(),
            });
        } else if builtin == 52 {
            ensure!(
                inputs.len == 48,
                Format,
                "byte {}: expected 48 builtin bidirectional LSTM inputs",
                op.pos
            );
            for j in [9, 10, 11, 16, 17, 26, 27, 28, 33, 34]
                .into_iter()
                .chain(39..48)
            {
                ensure!(
                    inputs.i32(j)? == -1,
                    Unsupported,
                    "operator {k}: peepholes, projections and auxiliary LSTM inputs are unsupported"
                );
            }
            for j in [1, 5, 12, 18, 22, 29] {
                ensure!(
                    inputs.i32(j)? >= 0,
                    Unsupported,
                    "operator {k}: CIFG LSTM inputs are unsupported"
                );
            }
            let options = builtin_options(op, 69)?;
            let cell_clip = options.f32(6, 0.0)?;
            ensure!(
                options.u8(4, 0)? == 4
                    && options.u8(10, 0)? != 0
                    && options.f32(8, 0.0)? == 0.0
                    && options.u8(14, 0)? == 0
                    && cell_clip.is_finite()
                    && cell_clip >= 0.0,
                Unsupported,
                "operator {k}: unsupported builtin bidirectional LSTM options"
            );
            // Batch size one makes TimeMajor immaterial to our [T, D] API.
            let _time_major = options.u8(12, 1)?;
            let forward = tensors.direction(inputs, (1, 12, 35), width, false)?;
            let backward = tensors.direction(inputs, (18, 29, 37), width, false)?;
            width = forward
                .hidden
                .checked_add(backward.hidden)
                .ok_or_else(|| err!(Format, "byte {}: output width overflow", op.pos))?;
            layers.push(LayerWeights {
                forward,
                backward,
                cell_clip,
                custom_options: Vec::new(),
            });
        } else if builtin == 9 {
            ensure!(
                !layers.is_empty() && k == ops.len - 1 && inputs.len == 3,
                Unsupported,
                "expected exactly one final fully connected layer (gesture/scribe heads are unsupported)"
            );
            let options = builtin_options(op, 8)?;
            ensure!(
                options.u8(4, 0)? == 0 && options.u8(6, 0)? == 0 && options.u8(10, 0)? == 0,
                Unsupported,
                "expected a plain linear fully connected layer"
            );
            let bias_index = inputs.i32(2)?;
            let bias_shape = tensors.shape(bias_index)?;
            ensure!(
                bias_shape.len() == 1 && bias_shape[0] > 0,
                Format,
                "byte {}: expected nonempty fully connected bias vector",
                op.pos
            );
            let num_classes = bias_shape[0];
            let fc_bias = tensors.read(bias_index, &bias_shape, false)?;
            let fc_weights = tensors.read(inputs.i32(1)?, &[num_classes, width], false)?;
            head = Some((fc_weights, fc_bias));
            width = num_classes;
        } else {
            bail!(
                Unsupported,
                "operator {k}: unsupported operator {:?} (builtin {builtin}); gesture/scribe nets are not handwriting sequence nets",
                core::str::from_utf8(custom).unwrap_or("<non-UTF8 custom code>")
            );
        }
        current = outputs.i32(0)?;
        let shape = tensors.shape(current)?;
        ensure!(
            shape.is_empty() || shape.last() == Some(&width),
            Format,
            "byte {}: operator {k} output width does not match weights",
            op.pos
        );
        ensure!(
            tensors.tensor(current)?.u8(6, 0)? == 0,
            Unsupported,
            "operator {k}: expected float output"
        );
    }
    ensure!(
        current == graph_outputs.i32(0)?,
        Format,
        "byte {}: graph output is not the final operator output",
        graph.pos
    );
    let (fc_weights, fc_bias) = head.ok_or_else(|| err!(Unsupported, "network does not end in fully connected handwriting logits; gesture/scribe nets are unsupported"))?;
    Ok(NetworkWeights {
        layers,
        num_classes: fc_bias.len(),
        fc_weights,
        fc_bias,
        input_size: 10,
    })
}

fn custom_options(data: &[u8], offset: usize) -> Result<f32> {
    ensure!(
        data.len() == 8 && data[4..7] == [4, 1, 1],
        Format,
        "byte {offset}: unsupported IndyLSTM custom options {data:?}"
    );
    // Kannada has 0x02 in byte 7: padding is not a fourth option.
    let clip = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    ensure!(
        clip.is_finite() && clip >= 0.0,
        Format,
        "byte {offset}: invalid cell clipping threshold {clip}"
    );
    Ok(clip)
}

fn dequantize(
    data: &[u8],
    tensor_type: u8,
    scale: f32,
    zero_point: i64,
    offset: usize,
) -> Result<Vec<f32>> {
    ensure!(
        scale.is_finite() && scale > 0.0,
        Format,
        "byte {offset}: invalid quantization scale {scale}"
    );
    Ok(data
        .iter()
        .map(|&byte| {
            // Historical hybrid UINT8 is signed storage, not an affine shift by 128.
            let value = if tensor_type == 9 || zero_point == 0 {
                (byte as i8) as f32
            } else {
                byte as f32
            };
            (value - zero_point as f32) * scale
        })
        .collect())
}

fn builtin_options(op: Table<'_>, expected: u8) -> Result<Table<'_>> {
    ensure!(
        op.u8(10, 0)? == expected,
        Unsupported,
        "byte {}: missing or unsupported builtin options (expected {expected})",
        op.pos
    );
    let field = op
        .field(12, 4)?
        .ok_or_else(|| err!(Format, "byte {}: missing builtin options table", op.pos))?;
    op.fb.table(op.fb.indirect(field)?)
}

struct Tensors<'a> {
    tensors: Vector<'a>,
    buffers: Vector<'a>,
}

impl Tensors<'_> {
    fn tensor(&self, index: i32) -> Result<Table<'_>> {
        ensure!(
            index >= 0,
            Format,
            "byte {}: missing or invalid tensor index {index}",
            self.tensors.pos
        );
        self.tensors.table(index as usize)
    }

    fn shape(&self, index: i32) -> Result<Vec<usize>> {
        let shape = self.tensor(index)?.vector(4, 4)?;
        (0..shape.len)
            .map(|i| {
                let dim = shape.i32(i)?;
                ensure!(
                    dim >= 0,
                    Format,
                    "byte {}: negative tensor {index} dimension {dim}",
                    shape.pos + i * 4
                );
                Ok(dim as usize)
            })
            .collect()
    }

    fn read(&self, index: i32, expected: &[usize], state: bool) -> Result<Vec<f32>> {
        let tensor = self.tensor(index)?;
        ensure!(
            self.shape(index)? == expected,
            Format,
            "byte {}: tensor {index} shape does not match {expected:?}",
            tensor.pos
        );
        let count = expected
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| err!(Format, "byte {}: tensor shape overflow", tensor.pos))?;
        let kind = tensor.u8(6, 0)?;
        ensure!(
            matches!(kind, 0 | 3 | 9),
            Unsupported,
            "tensor {index}: unsupported weight type {kind}"
        );
        let buffer = self
            .buffers
            .table(tensor.u32(8, 0)? as usize)?
            .vector(4, 1)?;
        if buffer.len == 0 && state && kind == 0 {
            return Ok(alloc::vec![0.0; count]);
        }
        let element_size = if kind == 0 { 4 } else { 1 };
        ensure!(
            buffer.len > 0 && count.checked_mul(element_size) == Some(buffer.len),
            Format,
            "byte {}: tensor {index} buffer size does not match {expected:?}",
            buffer.pos
        );
        let bytes = tensor.fb.bytes(buffer.pos, buffer.len)?;
        let values = if kind == 0 {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        } else {
            let field = tensor.field(12, 4)?.ok_or_else(|| {
                err!(
                    Format,
                    "byte {}: tensor {index} has no quantization",
                    tensor.pos
                )
            })?;
            let quant = tensor.fb.table(tensor.fb.indirect(field)?)?;
            let scale = quant.vector(8, 4)?;
            let zero_point = quant.vector(10, 8)?;
            ensure!(
                scale.len == 1 && zero_point.len == 1,
                Format,
                "byte {}: tensor {index} expected per-tensor quantization",
                quant.pos
            );
            dequantize(
                bytes,
                kind,
                f32::from_bits(tensor.fb.u32(scale.pos)?),
                i64::from_le_bytes(tensor.fb.array(zero_point.pos)?),
                scale.pos,
            )?
        };
        ensure!(
            values.iter().all(|x| x.is_finite()),
            Format,
            "byte {}: tensor {index} has nonfinite weights",
            buffer.pos
        );
        Ok(values)
    }

    fn direction(
        &self,
        inputs: Vector<'_>,
        slots: (usize, usize, usize),
        width: usize,
        diagonal: bool,
    ) -> Result<DirectionWeights> {
        let (start, bias_start, state_start) = slots;
        let bias_shape = self.shape(inputs.i32(bias_start)?)?;
        ensure!(
            bias_shape.len() == 1 && bias_shape[0] > 0,
            Format,
            "byte {}: expected nonempty gate bias vector",
            inputs.pos
        );
        let hidden = bias_shape[0];
        let mut input_kernels = Vec::new();
        let mut recurrent_kernels = Vec::new();
        let mut biases = Vec::new();
        let rec_shape = if diagonal {
            alloc::vec![hidden]
        } else {
            alloc::vec![hidden, hidden]
        };
        for gate in 0..GATES {
            input_kernels.extend(self.read(inputs.i32(start + gate)?, &[hidden, width], false)?);
            recurrent_kernels.extend(self.read(
                inputs.i32(start + 4 + gate)?,
                &rec_shape,
                false,
            )?);
            biases.extend(self.read(inputs.i32(bias_start + gate)?, &[hidden], false)?);
        }
        Ok(DirectionWeights {
            input_kernels,
            recurrent_kernels,
            diagonal,
            biases,
            initial_activation: self.read(inputs.i32(state_start)?, &[1, hidden], true)?,
            initial_cell: self.read(inputs.i32(state_start + 1)?, &[1, hidden], true)?,
            hidden,
            input: width,
        })
    }
}

// Slot numbers below are the generated Python accessors' vtable offsets, not
// schema field indices. Every pointer and extent is checked before indexing.
#[derive(Clone, Copy)]
struct Flatbuffer<'a>(&'a [u8]);

impl<'a> Flatbuffer<'a> {
    fn bytes(self, pos: usize, len: usize) -> Result<&'a [u8]> {
        let end = pos
            .checked_add(len)
            .ok_or_else(|| err!(Format, "byte {pos}: extent overflow ({len} bytes)"))?;
        self.0.get(pos..end).ok_or_else(|| {
            err!(
                Format,
                "byte {pos}: need {len} bytes, file length {}",
                self.0.len()
            )
        })
    }

    fn array<const N: usize>(self, pos: usize) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.bytes(pos, N)?);
        Ok(out)
    }

    fn u16(self, pos: usize) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array(pos)?))
    }
    fn u32(self, pos: usize) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(pos)?))
    }

    fn indirect(self, pos: usize) -> Result<usize> {
        let distance = self.u32(pos)? as usize;
        ensure!(
            distance >= 4,
            Format,
            "byte {pos}: invalid relative offset {distance}"
        );
        let target = pos
            .checked_add(distance)
            .ok_or_else(|| err!(Format, "byte {pos}: relative offset overflow"))?;
        self.bytes(target, 4)?;
        Ok(target)
    }

    fn table(self, pos: usize) -> Result<Table<'a>> {
        // Shared vtables can be on either side of a table: the offset is signed.
        let distance = self.u32(pos)? as i32;
        let vtable = (pos as i128) - i128::from(distance);
        ensure!(
            vtable >= 0 && vtable <= usize::MAX as i128,
            Format,
            "byte {pos}: invalid vtable offset {distance}"
        );
        let vtable = vtable as usize;
        let vsize = self.u16(vtable)? as usize;
        self.bytes(vtable, 4)?;
        let size = self.u16(vtable + 2)? as usize;
        ensure!(
            vsize >= 4 && vsize.is_multiple_of(2) && size >= 4,
            Format,
            "byte {vtable}: invalid vtable/table size {vsize}/{size}"
        );
        self.bytes(vtable, vsize)?;
        self.bytes(pos, size)?;
        Ok(Table {
            fb: self,
            pos,
            vtable,
            vsize,
            size,
        })
    }
}

#[derive(Clone, Copy)]
struct Table<'a> {
    fb: Flatbuffer<'a>,
    pos: usize,
    vtable: usize,
    vsize: usize,
    size: usize,
}

impl<'a> Table<'a> {
    fn field(self, slot: usize, width: usize) -> Result<Option<usize>> {
        if slot >= self.vsize {
            return Ok(None);
        }
        let offset = self.fb.u16(self.vtable + slot)? as usize;
        if offset == 0 {
            return Ok(None);
        }
        ensure!(
            offset >= 4
                && offset
                    .checked_add(width)
                    .is_some_and(|end| end <= self.size),
            Format,
            "byte {}: field points outside table at byte {}",
            self.vtable + slot,
            self.pos
        );
        Ok(Some(self.pos + offset))
    }

    fn u8(self, slot: usize, default: u8) -> Result<u8> {
        self.field(slot, 1)?
            .map_or(Ok(default), |p| Ok(self.fb.bytes(p, 1)?[0]))
    }
    fn u32(self, slot: usize, default: u32) -> Result<u32> {
        self.field(slot, 4)?.map_or(Ok(default), |p| self.fb.u32(p))
    }
    fn i32(self, slot: usize, default: i32) -> Result<i32> {
        Ok(self.u32(slot, default as u32)? as i32)
    }
    fn f32(self, slot: usize, default: f32) -> Result<f32> {
        Ok(f32::from_bits(self.u32(slot, default.to_bits())?))
    }

    fn vector(self, slot: usize, width: usize) -> Result<Vector<'a>> {
        let Some(field) = self.field(slot, 4)? else {
            return Ok(Vector {
                fb: self.fb,
                pos: self.pos,
                len: 0,
                width,
            });
        };
        let start = self.fb.indirect(field)?;
        let len = self.fb.u32(start)? as usize;
        let extent = len
            .checked_mul(width)
            .ok_or_else(|| err!(Format, "byte {start}: vector size overflow"))?;
        let pos = start + 4;
        self.fb.bytes(pos, extent)?;
        Ok(Vector {
            fb: self.fb,
            pos,
            len,
            width,
        })
    }

    fn string(self, slot: usize) -> Result<&'a [u8]> {
        let vector = self.vector(slot, 1)?;
        if self.field(slot, 4)?.is_some() {
            ensure!(
                self.fb.bytes(vector.pos + vector.len, 1)? == [0],
                Format,
                "byte {}: string lacks terminator",
                vector.pos
            );
        }
        self.fb.bytes(vector.pos, vector.len)
    }
}

#[derive(Clone, Copy)]
struct Vector<'a> {
    fb: Flatbuffer<'a>,
    pos: usize,
    len: usize,
    width: usize,
}

impl<'a> Vector<'a> {
    fn element(self, index: usize) -> Result<usize> {
        ensure!(
            index < self.len,
            Format,
            "byte {}: vector index {index} exceeds length {}",
            self.pos,
            self.len
        );
        Ok(self.pos + index * self.width)
    }
    fn i32(self, index: usize) -> Result<i32> {
        Ok(self.fb.u32(self.element(index)?)? as i32)
    }
    fn table(self, index: usize) -> Result<Table<'a>> {
        self.fb.table(self.fb.indirect(self.element(index)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn signed_hybrid_bytes_and_affine_uint8() {
        let bytes = [0, 1, 127, 128, 255];
        assert_eq!(
            dequantize(&bytes, 3, 0.5, 0, 0).unwrap(),
            [0.0, 0.5, 63.5, -64.0, -0.5]
        );
        assert_eq!(
            dequantize(&bytes, 9, 0.5, 0, 0).unwrap(),
            [0.0, 0.5, 63.5, -64.0, -0.5]
        );
        assert_eq!(
            dequantize(&bytes, 3, 0.5, 128, 0).unwrap(),
            [-64.0, -63.5, -0.5, 0.0, 63.5]
        );
        assert_eq!(dequantize(&[255], 9, 0.5, -1, 0).unwrap(), [0.0]);
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(dequantize(&bytes, 3, scale, 0, 42).is_err());
        }
    }

    #[test]
    fn custom_options_allow_kannada_padding() {
        for padding in [0, 2, 255] {
            assert_eq!(
                custom_options(&[0, 0, 72, 66, 4, 1, 1, padding], 0).unwrap(),
                50.0
            );
        }
        for data in [
            &[0, 0, 72, 66, 4, 1, 1][..],
            &[0, 0, 72, 66, 4, 0, 1, 0],
            &[0, 0, 128, 191, 4, 1, 1, 0],
            &[0, 0, 128, 127, 4, 1, 1, 0],
        ] {
            assert!(
                matches!(custom_options(data, 42), Err(Error::Format(m)) if m.contains("byte 42"))
            );
        }
    }

    #[test]
    fn vectors_and_strings_validate_their_entire_extent() {
        // A one-field table, followed by a length-one string/vector.
        let valid = [
            6, 0, 8, 0, 4, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, b'x', 0,
        ];
        let table = Flatbuffer(&valid).table(8).unwrap();
        assert_eq!(table.string(4).unwrap(), b"x");
        assert!(table.vector(4, 4).is_err());
        assert!(table.vector(4, 1).unwrap().element(1).is_err());
        let mut missing_terminator = valid;
        missing_terminator[21] = 1;
        assert!(
            Flatbuffer(&missing_terminator)
                .table(8)
                .unwrap()
                .string(4)
                .is_err()
        );
        let mut huge_vector = valid;
        huge_vector[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            Flatbuffer(&huge_vector)
                .table(8)
                .unwrap()
                .vector(4, 8)
                .is_err()
        );
        let mut invalid_field = valid;
        invalid_field[4] = 7;
        assert!(
            Flatbuffer(&invalid_field)
                .table(8)
                .unwrap()
                .vector(4, 1)
                .is_err()
        );
    }

    #[test]
    fn malformed_offsets_are_format_errors() {
        for bytes in [
            &[][..],
            b"bad file",
            &[255, 255, 255, 255, b'T', b'F', b'L', b'3'],
            &[8, 0, 0, 0, b'T', b'F', b'L', b'3', 255, 255, 255, 127],
        ] {
            assert!(matches!(load_weights(bytes), Err(Error::Format(m)) if m.contains("byte ")));
        }
        assert!(Flatbuffer(&[0; 8]).bytes(usize::MAX, 2).is_err());
        // Vtable follows the table, exercising negative signed displacement.
        let bytes = [252, 255, 255, 255, 4, 0, 4, 0];
        assert!(
            Flatbuffer(&bytes)
                .table(0)
                .unwrap()
                .field(4, 4)
                .unwrap()
                .is_none()
        );
        let bytes = [8, 0, 4, 0, 3, 0, 0, 0, 8, 0, 0, 0];
        assert!(Flatbuffer(&bytes).table(8).unwrap().field(4, 4).is_err());
        let bytes = [255, 255, 255, 255];
        assert!(Flatbuffer(&bytes).indirect(0).is_err());
    }
}
