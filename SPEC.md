# ML Kit Digital Ink Recognition — reverse-engineering spec

Goal: run Google's ML Kit Digital Ink Recognition **outside the ML Kit SDK**, from the
original model artifacts. Python reference implementation first, Rust port after.

Everything below is *established fact*, verified from the shipped artifacts — not guesswork.
Open questions are called out explicitly at the bottom.

## 1. Where the models come from

The `digital-ink-recognition` AAR ships the full catalog:

    https://dl.google.com/dl/android/maven2/com/google/mlkit/digital-ink-recognition/19.0.0/digital-ink-recognition-19.0.0.aar

Relevant contents:

| path | what |
|---|---|
| `assets/manifest.json` | catalog of **808 packs**: name, `download_urls`, size, md5, sha1 |
| `assets/packmapping.pb` | BCP-47 language tag -> which packs to use |
| `jni/*/libdigitalink.so` | native recognizer (the thing we are replacing) |

All downloads are **plain unauthenticated** `https://dl.google.com/handwriting/models/...`.
No Play Services, no auth, no device required. `manifest.json` and `packmapping.pb` are
copied into the repo root.

### packmapping.pb
Repeated field 1, each entry:
- `1` = BCP-47 tag (e.g. `"en-US"`, `"aa-Latn"`, `"zh-Hani-CN"`)
- `5` = recospec pack name
- `6` = tflite (neural net) pack name
- `7` = FST language-model pack name (absent for gesture models)

Tags ending `-x-gesture` map to the newer 2022 `scribe_*` models (gesture/autodraw) and have no FST.

### Three artifact kinds
1. **recospec** (~4 KB) — protobuf config: preprocessing pipeline, charset, decoder params. Per language.
2. **tflite** (1–8 MB) — the neural net. **Per script, not per language.** e.g.
   `indy_lstm_latin_6x216_tflite_20191208_zip` serves English and ~60 other Latin languages.
   29 script nets total (latin, cyrillic, arabic, devanagari, chinese, japanese, korean, ...).
3. **compact_fst** (~22 MB) — OpenFst word/char language model. Per language.

## 2. The neural net

Confirmed by parsing `latin_indy_lstm_6x216_20191208.tflite`:

    input [1, T, 10] float32
      -> 6 x bidirectional_sequence_indylstm  (216 units per direction, output 432)
      -> FULLY_CONNECTED (weights [314, 432], bias [314])
      -> logits [1, T, 314]

- **Custom op `bidirectional_sequence_indylstm`** — NOT in stock TFLite; implemented in
  `libdigitalink.so` (`research/handwriting/mobile/inference/bidirectional_sequence_indylstm.cc`).
  Stock LiteRT fails with `Encountered unresolved custom op`. We must reimplement it.
  IndyLSTM = LSTM whose **recurrent weight matrix is diagonal** — hence `kernel_w_rec_to_*`
  tensors are shape `[216]` (a vector) while `kernel_w_input_to_*` are `[216, input]`.
  Reference: Gonnet & Deselaers, "IndyLSTMs: Independently Recurrent LSTMs" (2019).
- Per-layer tensor inputs, in op input order: input, then for **forward** then **backward**:
  `kernel_w_input_to_{input,forget,cell,output}`, `kernel_w_rec_to_{input,forget,cell,output}`,
  `bias_{input,forget,cell,output}`; then 4 initial-state tensors
  (fwd activation, fwd cell, bwd activation, bwd cell), all zero-filled.
  NOTE the op input ordering is i,f,c,o for kernels but the tensor *indices* are interleaved —
  read them by position in the op's input list, never by tensor name order.
- **Quantization: hybrid, and the dtype label is a lie.** `kernel_w_input_to_*` and the FC
  weights are declared `UINT8`, per-tensor `scale`, `zero_point=0`
  (e.g. 0.044183675; FC 0.26830208). Recurrent diagonals and all biases are `FLOAT32`.
  In TFLite's historical hybrid format the stored bytes are **signed int8 reinterpreted as
  uint8** (symmetric, zp=0). Verified in TF v1.13.1 `tools/optimize/quantize_weights.cc`
  (builds a `vector<int8_t>`, then labels it `TensorType_UINT8`) and `fully_connected.cc`
  EvalHybrid, which does `reinterpret_cast<int8_t*>(filter->data.uint8)`.
  Dequantize as `int8(byte) * scale` == `(v - 256 if v > 127 else v) * scale`.
  NOT `(v - zero_point) * scale`, NOT `(v - 128) * scale` — both yield garbage.
- `custom_options` (8 bytes, identical for all 6 layers):
  `00 00 48 42 04 01 01 00` = float32 `50.0` then bytes `[4, 1, 1, 0]`.
  The 50.0 is almost certainly **cell clip**. The `4` is likely activation=TANH;
  the two `1`s likely merge_outputs / time_major. Treat as a hypothesis to validate.

## 3. Charset / CTC mapping  (CORRECTED - blank is LAST, not first)

The recospec carries an FST symbol table of **317** symbols:

    <epsilon>=0, <reserved>=1, [[space]]=2, !=3, ... ✡=314, <S>=315, </S>=316

The processor charset (recospec field `.2.1`) holds **313 characters**, and
`charset[i] == FST symbol i+2` exactly (`[[space]]` being a literal space).

The net emits **314** logits = 313 characters + 1 CTC blank. The mapping is:

    net index k, 0 <= k <= 312   ->  charset[k]  ->  FST symbol k+2
    net index 313                ->  CTC BLANK

i.e. **net index == charset index**, and the blank sits at the **end**.

An earlier hypothesis here had blank=0 with net k -> FST symbol k+1. That was **wrong**.
The correction comes from the native `NetworkScoreCache` in `libdigitalink.so`: it treats a
dimension-1 special index of **313** (for Latin) as the blank, and keys normal symbols by
`index + 2` into the FST. Raw tensor indexing is otherwise unchanged.
Evidence addresses: ctor `0x4f132b..32`, score `0x50a224..261`, diagnostic `0x4dfa4c..aae`.

Note 313 is exactly `num_classes - 1`, so the blank index is script-dependent (it is the last
output, whatever the charset size). Derive it as `num_classes - 1`; do not hardcode 313.

Still worth confirming against the oracle, but the static evidence here is strong and
self-consistent with `charset[i] == symbol i+2`.

## 4. The recospec protobuf

`evidence/en_us.recospec.raw.txt` is a full `protoc --decode_raw`. Structure:

    1: "tf_reco"                  # recognizer type
    2: "en"                       # language
    27: "codepoints"
    158518157 { ... }             # proto2 EXTENSION -> TfRecognizerSpec
      2 { ... }                   # ProcessorConfig / LabeledInkCurveProcessor
        1: <each codepoint>      # repeated; the 314-symbol charset (field .2.1, NOT .2.2.1)
        6 { 1:0.05 2:0.02 3:0.01 4:3.0 5:-0.8 6:0 7:1 9:1 }   # CurveSettings (doubles)
        7: 10                     # >>> number of input features = 10 <<<
        8: "codepoints"
        12 { repeated InkPreprocessingStepSpec }   # the preprocessing pipeline
        13: "LabeledInkCurveProcessor"
      3 { 5 { 1:"BUNDLED_ONDEVICE_MODEL" ... } }   # ModelConfig -> the tflite
      4 { ... }                   # DecoderConfig
        6 { 1 {beam params} 3 { 1:"BUNDLED_LM_FST" 4:<symbol table> } }  # FstDecoderConfig.WordLmFst
        9 { ... }                 # CharClassesBeamScorerSpec (char classes + weights)

For en-US the pipeline `12` has 4 steps, each a different oneof field number:
`5:""`, `6 {1:20.0 2:0}`, `8 {1:0.0 2:1 3:0.5}`, `11:""`.
Mapping oneof field number -> settings type is an open question; infer it by diffing
many of the 391 recospecs (different languages exercise different steps).

## 5. Native library evidence

`libdigitalink.so` is stripped of symbols but **retains C++ RTTI names and assertion strings**.
- `evidence/cpp_class_names.txt` — 142 demangled `research_handwriting::*` classes
- `evidence/proto_message_names.txt` — 74 proto message names
- `evidence/libdigitalink.strings.txt` — raw strings

Key preprocessing step classes (each has a matching `InkPreprocessingStepSpec_*Settings`):
`NormalizeSize`, `NormalizeSizeByTime`, `NormalizeSizeFirstStroke`, `NormalizeMultilineSize`,
`NormalizeTimeBySize`, `Resampling`, `ResamplingByTime`, `RamerResampling`, `ResamplePenupStrokes`,
`AddPenUpStrokes`, `FilterStrokes`, `HallucinateTime`, `SanitizeTime`, `InkBasedSlopeCorrection`,
`DetectAndRearrangeMultiLine`, `ScaleShiftForRendering`, `SetWritingGuide`, `WritingGuideFromBoundingBox`.

Two feature encoders exist (assertion strings give exact signatures):

    CoeffsToFeatures(vecs, settings.interpolate_time(), stroke.pen_down(),
                     settings.generate_second_order_features(), &timesteps.back())
                     == expected_num_coeff_features
    CoeffsToFeaturesAnglesRatios(vecs, settings.interpolate_time(), stroke.pen_down(),
                     settings.normalize_outputs_to_zero_one(), &timesteps.back())
                     == expected_num_coeff_features

Source layout (from assertion paths): `research/handwriting/features/curves.cc`,
`ink_preprocessor.cc`, `tensorflow/labeled_ink_curve_processor.cc`,
`service/fst_decoder.cc`, `service/char_classes_rescoring_lm.cc`.

The published description of this exact system is
Carbune et al., "Fast multi-language LSTM-based online handwriting recognition", IJDAR 2020 —
ink is fit with **cubic Bézier curves**, each curve encoded as ~10 features.

## 6. Ground truth

`android-oracle/` contains a data-driven harness that runs the **real** ML Kit SDK on the
connected device against a JSON ink file, so any reimplementation can be diffed against
Google's own output on identical input. Use it — do not guess at correctness.

## 7. Open questions
- Exact semantics + ordering of the 10 Bézier features.
- oneof field-number -> preprocessing settings type mapping.
- Exact meaning of `custom_options` bytes `[4,1,1,0]`.
- CTC blank index / symbol off-by-one (stated above as hypothesis).

## 8. The 10 Bézier features (RECOVERED FROM THE BINARY — the paper's order is wrong)

Recovered by disassembling `CoeffsToFeaturesAnglesRatios` in the x86_64 `libdigitalink.so`
(binary sha256 `d7b2873f...`; assertion string at `0x83e5c` referenced from `0x5a0ba3`
identifies the inlined function). Output stores are at `+0x00,0x04,...,0x24` off `%r13`.

**The real layout is NOT the paper's `(dx,dy,d1,d2,α1,α2,γ1,γ2,γ3,p)`:**

| idx | value | store addr |
|---:|---|---|
| 0 | `pen_down ? 1.0f : 0.0f` | `0x5a07fa` |
| 1 | `dx = (P3-P0).x` | `0x5a0821` |
| 2 | `dy = (P3-P0).y` | `0x5a0841` |
| 3 | `α1` | `0x5a09af` |
| 4 | `d1` | `0x5a09cb` |
| 5 | `α2` | `0x5a0ad5` |
| 6 | `d2` | `0x5a0af1` |
| 7 | `Δt` | `0x5a0b1c` |
| 8 | `start_time_leg` | `0x5a0b42` |
| 9 | `end_time_leg` | `0x5a0b68` |

Three differences from the paper, each of which was a live bug in our implementation:
1. **The pen flag is FIRST, not last — and it is `pen_down`, not `pen_up`.** We emitted 1.0 for
   pen-up, i.e. that bit was inverted on every timestep. The assertion signature
   (`stroke.pen_down()`) hinted at this.
2. **Angles and ratios interleave** (`α1, d1, α2, d2`), they are not grouped.
3. **The time features are transformed, not raw γ coefficients** (see below).

With `D = P3-P0`, `V = P1-P0`, `W = P2-P3`, `L = ‖D‖` (2D spatial norm, **time excluded**):

    d1 = ‖V‖ / L        d2 = ‖W‖ / L          fixed pairing, no nearest-endpoint search
    α1 = atan2f(D.x*V.y - D.y*V.x,   D.x*V.x + D.y*V.y)     signed angle from  D to V
    α2 = atan2f(D.y*W.x - D.x*W.y,  -D.x*W.x - D.y*W.y)     signed angle from -D to W

`atan2` throughout — signed, no `acos`, no negation. The **reversed chord for α2 is confirmed**
(sign flip of `D.x` via the mask at `0xa1b20`, address `0x5a0a07`).

Degenerate case: ratios are computed only when `L > 0`, else both are exactly `0.0`. There is
**no epsilon** — a tiny positive `L` still divides. Both `atan2` calls run unconditionally;
there is no "zero chord ⇒ zero angles" override.

### Time features (indices 7–9)
For `t(s) = γ0 + γ1 s + γ2 s² + γ3 s³`, the emitted values are

    idx 7 = (γ1 + γ2) + γ3          == T3 - T0   (Δt)
    idx 8 = γ1 / 3                  == T1 - T0
    idx 9 = -γ1/3 - (2 γ2)/3 - γ3   == T2 - T3

i.e. **Bézier time control-point legs**, mirroring the spatial part's use of control points —
not the raw power-basis coefficients. γ0 drops out. Conversion helper at `0x5c7c80`; constants
`0x9f2c0 = 3.0f` and `0xa09f0 = (-3,+3)`. Preserve this operation order in float32.

If `interpolate_time` is false the encoder emits **7** features, not 10 (`0x5a0af7`).

`normalize_outputs_to_zero_one` is **false** for en-US, so nothing is clipped (ratios may
exceed 1). When true (callback `0x5a1b60`) it is a clamped affine map: `(v+1)/2` for dx,dy and
the time features; `(v+π)/(2π)` for angles; `clamp(v,0,1)` for ratios; flag untouched.

### Splitting thresholds (also recovered)
Let `B` = the diagonal of the whole stroke's bounding box, `r_i` the **spatial** residuals:

| field | value | recovered role |
|---|---|---|
| 1 | 0.05 | corner-selection neighbour distance, `0.05 * B` — **not** an SSE tolerance |
| 2 | 0.02 | split if `max_i ‖r_i‖ > 0.02 * B` |
| 3 | 0.01 | split if `sqrt(Σ‖r_i‖²/N) > 0.01 * B` (**RMS**, not total or mean SSE) |
| 4 | 3.0 | `(‖V‖ + ‖P2-P1‖ + ‖W‖) / ‖D‖ <= 3.0` — **control-polygon** length, not arc length |
| 5 | -0.8 | `min(cos(V, P2-P1), cos(-W, P2-P1)) >= -0.8` |

Evidence: bbox diagonal `0x5c3a5a`, threshold scaling `0x5c3f19`, RMS `0x5c1ff0`, max-error
`0x5c20da`, split gate `0x5c4626`, merge-rejection `0x5c6ce7`, geometry check `0x5c918b`.

So SPEC's earlier "arc length > 3× endpoint distance" reading (from the paper) is the right
*idea* but the implementation uses the control-polygon length, and the error gate is RMS +
max-error scaled by the bbox diagonal rather than a per-point SSE constant.

### Curve fitting (paper eq. 1–6)
Curves are cubic polynomials **in the power basis**, in s ∈ [0,1]:
`x(s) = α0 + α1·s + α2·s² + α3·s³`, likewise `y` (β) and `t` (γ).
Note the features reference *control points*, so you must convert power basis -> Bézier
control points: `P0=α0`, `P1=α0+α1/3`, `P2=α0+2α1/3+α2/3`, `P3=α0+α1+α2+α3`.

1. **Normalize size** — scale the whole ink so y ∈ [0,1].
2. **Normalize time** — per stroke, rescale t linearly so `t_last - t_first` equals the stroke's
   total spatial path length (eq. 2), putting t in the same numerical range as x,y.
3. **Fit** — solve eq. 4/5 (`VᵀZ = VᵀV·Ω`) by least squares for the coefficients, alternating
   with a Newton update of the projection parameters `s_i` via eq. 6, until convergence.
4. **Split** recursively if (a) SSE too large, or (b) arc length > **3×** the endpoint distance.
   Split point: for (a) the triplet of consecutive points with the smallest angle; for (b) the
   point of maximum local curvature. **The 3.0 here is `CurveSettings` field 4 = 3.0** — a direct
   confirmation that our CurveSettings field reading is right.
5. **Merge** — repeatedly re-fit consecutive curve pairs as one; keep the merge if it still
   satisfies the criteria. Removes spurious breakpoints.

`CurveSettings` field names were recovered by **disassembling the proto serializer** in the
x86_64 `libdigitalink.so` (tag -> struct-offset -> assertion-string name):

| field | name | en-US value |
|---|---|---|
| 1 | tolerance (role unconfirmed) | 0.05 |
| 2 | tolerance (role unconfirmed) | 0.02 |
| 3 | tolerance (role unconfirmed) | 0.01 |
| 4 | max arc-length / endpoint-distance ratio | 3.0 |
| 5 | split angle threshold (cosine) | -0.8 |
| 6 | `generate_second_order_features` | **false** |
| 7 | selects the angles/ratios encoder | **true** |
| 9 | `interpolate_time` | **true** |
| 10 | `normalize_outputs_to_zero_one` | **absent -> false** |

Three independent confirmations fall out of this:
- field 4 = 3.0 matches the paper's "3 times the endpoint distance" split rule;
- field 7 = true confirms `CoeffsToFeaturesAnglesRatios` (not `CoeffsToFeatures`);
- field 6 = false is consistent with 10 features rather than a second-order-expanded set;
- field 9 = true is consistent with the γ time coefficients being present in the 10-vector.

Note `normalize_outputs_to_zero_one` is **false** for en-US, so angles stay in radians and
values sit in [-1,1] as the paper describes — they are NOT remapped into [0,1].

Pen-up curves: the native lib has `AddPenUpStrokesPreprocessingStep` /
`ResamplePenupStrokesPreprocessingStep`, i.e. synthetic strokes bridging one stroke's end to the
next stroke's start, flagged `p=1`. That is what the `p` feature distinguishes.

## 9. InkPreprocessingStepSpec oneof map (COMPLETE, 30/30)

Recovered by disassembling the preprocessing-step factory + proto serializer in
`libdigitalink.so` (tag -> struct offset -> step class). Unless noted, the settings message is
`InkPreprocessingStepSpec.<Name>Settings`.

| # | step | # | step |
|---|---|---|---|
| 1 | Resampling | 16 | NormalizeMultilineSize |
| 2 | NormalizeSize | 17 | FilterStrokes |
| 3 | StrokeOrder *(top-level `StrokeOrderSettings`)* | 18 | RemoveTime *(no settings)* |
| 4 | SanitizeTime | 19 | RemoveGuide *(no settings)* |
| 5 | NormalizeTime *(no settings)* | 20 | NormalizeSizeForScribe *(no settings)* |
| 6 | HallucinateTime | 21 | NormalizeSizeByTime |
| 7 | InkBasedSlopeCorrection | 22 | StartAtOrigin *(no settings)* |
| 8 | NormalizeSizeWritingGuideFirstStroke | 23 | ResamplePenupStrokes |
| 9 | DetectAndRearrangeMultiLine | 24 | RemoveTimeOrder *(no settings)* |
| 10 | RemovePressure *(no settings)* | 25 | ScaleShiftForRendering |
| 11 | AddPenUpStrokes *(no settings)* | 26 | ShiftInkBasedOnWritingGuide *(no settings)* |
| 12 | TimeMsToS *(no settings)* | 27 | WritingGuideFromBoundingBox |
| 13 | RamerResampling | 28 | SetWritingGuide |
| 14 | ResamplingByTime | 29 | RandomizeInkPositionWithinWritingGuide |
| 15 | NormalizeTimeBySize | 30 | NormalizeSizeFirstStroke |

Caveat: the "no settings" fields are message-typed weak placeholders
(`proto2::internal::ImplicitWeakMessage`); the original target message is unknown and is
**not** proven to be `google.protobuf.Empty`.

### The actual en-US pipeline
    5  NormalizeTime
    6  HallucinateTime                        {1: 20.0, 2: 0}
    8  NormalizeSizeWritingGuideFirstStroke   {1: 0.0, 2: 1, 3: 0.5}
    11 AddPenUpStrokes

Note the order: time is normalised **before** size, and size normalisation is the
writing-guide/first-stroke variant. Since ink built from raw stroke JSON carries no writing
guide, the native code path falls back to plain NormalizeSize — it carries the literal string
`"Ink doesn't have writing guide. Use NormalizeSize."`. `src/features.py` implements this as a
named-step registry driven by an ordered pipeline, mirroring `InkPreprocessorSpec`.

### CurveSettings proto defaults (vs en-US overrides)
Defaults: fields 1-5 = `[0.05, 0.02, 0.01, 4, -0.9]`; fields 6,7,9,10,11,12 = bool false;
field 13 = int32 80; field 8 absent (not proven reserved).
en-US overrides field 4 -> **3.0** and field 5 -> **-0.8**, and sets 7 and 9 true.

## 10. The model zoo, precisely (corrects earlier "29 scripts")

`models/` holds **30** TFLite files: **27 handwriting sequence nets** + **3 special classifiers**
(`autodraw`, `emoji`, `shapes`). The classifiers are NOT per-timestep recognizers — they have
reduction/dense heads, and `autodraw`/`emoji` take **5 input features, not 10**. Do not feed them
through the CTC path.

Of the 27 sequence nets, two different ops are used:

| op | count | runnable by stock LiteRT? |
|---|---|---|
| `bidirectional_sequence_indylstm` (custom, 29 inputs) | 14 | **no** — must reimplement |
| `BIDIRECTIONAL_SEQUENCE_LSTM` (builtin, 48 inputs) | 13 | **yes** |

Custom/Indy: latin, cyrillic, arabic, devanagari, bengali, gujarati, kannada, sinhala, ethiopic,
greek, armenian, hebrew, myanmar, tibetan, vietnamese.
Builtin/standard: chinese, japanese, korean, thai, lao, khmer, georgian, malayalam, odia,
punjabi, tamil, telugu.

**This split is a gift.** The 13 builtin nets run natively in LiteRT, so they are a free,
exact numerical oracle for the LSTM cell math — which is shared with the Indy path apart from
the recurrent term (diagonal vs full matrix). Validating against them validates most of the
custom implementation without needing the device at all.

### Options are uniform across every net
Every builtin net reports: `activation=TANH`, `cell_clip=50.0`, `proj_clip=0`,
`merge_outputs=true`, `time_major=true`.

That is **exactly** what the custom op's 8 option bytes decode to
(`00 00 48 42 | 04 | 01 | 01 | 00` -> float32 50.0, activation 4=TANH, merge_outputs=1,
time_major=1, one byte padding). The custom-op layout is therefore corroborated independently,
not just guessed: it mirrors the builtin options struct.

### No optional tensors anywhere
Across all builtin nets the absent (index -1) inputs are identical:
`[9,10,11, 16,17, 26,27,28, 33,34, 39..47]` — i.e. **no peephole weights (`cell_to_*`), no
projection, no auxiliary inputs**, in either direction. So the cell is plain LSTM:

    c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)      clipped to +-50
    h_t = sigmoid(o) * tanh(c_t)

with gate order **i, f, cell, o** and no separate forget-bias constant (biases are trained).

### Quantization, settled with numbers
Dequantizing as signed int8 yields plausible trained-weight statistics; the unsigned reading
does not:

| tensor | unsigned mean/std | minus-128 mean/std | **signed** mean/std |
|---|---:|---:|---:|
| Latin first input kernel | 5.255 / 4.996 | -0.401 / 4.996 | **-0.018 / 1.117** |
| Latin FC weights | 30.75 / 30.75 | -3.589 / 30.75 | **-0.194 / 6.805** |

Raw bytes 0 and 255 are common (4.95% / 4.35%) because they are signed 0 and -1 — not
saturation. Confirmed by TFLite's own regression test `quantize_weights_test.cc`
(`SymmetricDequantizeAndCompare` reads via `reinterpret_cast<const int8_t*>`).

Caveat: real TFLite *also* dynamically quantizes activations per batch
(range/127, round-half-away-from-zero). Our float path dequantizes weights and does float
matmuls, so it is an approximation, not bit-exact.

## 11. Network implementation is VALIDATED (against native LiteRT)

The 13 builtin-op nets run natively in LiteRT, giving an exact numerical oracle with no device
needed. Comparing `src/indylstm.forward` against LiteRT on identical random input, T=24:

| net | classes | mean abs err | max | argmax agreement |
|---|---:|---:|---:|---:|
| korean_5x160 | 3319 | 0.00996 | 0.072 | **100%** |
| chinese_lstm_4x192 | 12362 | 0.02064 | 0.172 | **100%** |
| malayalam_lstm_6x128 | 188 | 0.571 | 3.795 | 100% |
| telugu_5x160 | 159 | 0.250 | 11.57 | 95.8% |
| thai_5x128 | 196 | 24.23 | 133.4 | 95.8% |
| georgian_5x128 | 111 | 44.06 | 238.5 | 95.8% |

The outliers are **not** a bug. All six nets have identical structure (`N x BiLSTM -> FC`), and
the divergence tracks weight **scale**, not architecture: georgian's largest per-tensor scale is
**5.27** vs korean's 0.066. TFLite's hybrid kernel quantizes activations to int8 at runtime;
that rounding error gets multiplied by the weight scale, so a huge-scale tensor amplifies it.
Our float path dequantizes weights and does float matmuls, which is arguably *more* accurate
but necessarily differs from TFLite.

**What matters for Latin:** its scales are 0.017-0.268 — the same regime as korean/chinese,
which agree to ~0.01-0.02 mean error and 100% argmax. So the network implementation, including
the reimplemented IndyLSTM cell, is sound for the script we care about. Any end-to-end
mismatch is therefore in **feature extraction**, not the net or the decoder.

## 12. Ground truth harness (working)

`android-oracle/run_oracle.sh <ink.json>` runs the real ML Kit SDK on-device and writes n-best
JSON. It handles the locked-device case by showing its Activity above the keyguard (an earlier
version silently stalled: Android put the backgrounded UID under `APP_BACKGROUND` network
restriction and ML Kit only retries model downloads on a `CONNECTIVITY_CHANGE` broadcast).

Confirmed reference results:

    testdata/hi.json  -> ["hi","hil","his","h i","hir", ...]
    testdata/cat.json -> ["cat","Cat","cal","cot","cant", ...]

`getScore()` returns **null** for the English model — the SDK exposes no numeric scores here,
so only candidate *order* can be compared, not magnitudes.

The models ML Kit downloaded are **byte-identical** to the archives we fetched independently
from dl.google.com, which closes the loop on the acquisition path:
tflite 3,992,904 B, FST 21,798,892 B, recospec 3,929 B.
(The English FST archive is actually named `en_us.20191208.compact.fst.zip`.)

## 13. Feature bugs found and fixed so far

1. **Time scale** — `NormalizeTime` sets each stroke's time span to its path length in original
   pixel units, but `NormalizeSize` scaled only x and y. gamma came out ~50x too large (54.0
   where the paper says [-1,1]), saturating the net. Fix: scale t by the same factor.
2. **alpha2 reference vector** — measuring the second control leg against the forward chord puts
   alpha2 at +-pi for *every* straight segment (the control points lie on the line). It must be
   measured against the reversed chord `P0-P3`, so a straight segment gives alpha1 = alpha2 = 0.
3. **Underdetermined fits** — a cubic through fewer than 4 points is underdetermined, and plain
   lstsq returns a minimum-norm solution with arbitrary control points; 2-point pen-up segments
   produced `d1=0.111, d2=0.667` instead of the required 1/3, 1/3. Fix: fit the highest
   exactly-determined degree and zero-pad.

Still wrong end to end, so at least one convention remains incorrect. See section 7.

## 14. Preprocessing, recovered from the binary

The en-US pipeline is `NormalizeTime` -> `HallucinateTime` -> `NormalizeSizeWritingGuideFirstStroke`
-> `AddPenUpStrokes` (recospec fields 5, 6, 8, 11). Settings: HallucinateTime `{1: 20.0, 2: 0}`,
NormalizeSizeWritingGuideFirstStroke `{1: 0.0, 2: 1, 3: 0.5}`.

### NormalizeSizeWritingGuideFirstStroke (SOLVED)

**Size normalisation never touches `t`.** The shared transform (`0x5b1aaa` -> `0x5b1852`) loops
over each stroke writing **only** the x array (`+0x30`) and y array (`+0x40`); the t array
(`+0x50`) is left untouched (`0x5b18e3..0x5b1905`).

This **disproves SPEC §13 item 1**. We had inferred "scale t with x and y" empirically because
γ otherwise came out ~50x too large — that inference was wrong, and it was masking an incorrect
`NormalizeTime`. Since size normalisation provably never rescales time, whatever `NormalizeTime`
leaves in `t` is what the fitter sees.

Constants: `eps = 0x34000000` = 2^-23 (`0x9f0b0`); width guard `100.0f` (`0x9f38c`).

Guide selection (`0x5b1dc2`, `0x5b1dcd`): fall back if the guide's width `< eps` OR height
`< eps`; a null guide selects a default at `0x85b9a0`. **That default is all zeros**
(width `+0x20` = 0.0f, height `+0x24` = 0.0f, top `+0x58` = 0.0f, and constructor `0x5dd370`
zeroes them too), so `0 < eps` and **ink with no writing guide definitively takes the fallback
path.** Accessor names proven via assertions: `top()` `0x5aeb4d`, `width()` `0x5ae97f`,
`height()` `0x5ae9be`. Settings layout: field1 float `+0x18`, field2 bool `+0x1c`,
field3 float `+0x20`; all default 0/false, en-US supplies the overrides.

**Fallback path** (`0x5b1e28` -> `0x5b1c76` -> `0x5b1abf`), bbox over **all** stroke points:

    h = max(H, W/100);   if h < eps: h = 1
    margin      = field1 * h
    y0          = ymin - margin
    denominator = (margin + margin) + h        # this float32 operation order
    x0          = xmin, unless field2 and stroke0 is non-empty, then stroke0.x[0]
    x' = (x - x0) * (1/denominator)    y' = (y - y0) * (1/denominator)    t' = t

Note the native code computes the **reciprocal once and multiplies** (`0x5b1be1..0x5b1bfc`)
rather than dividing per point; in float32 that is not bit-identical to division. `W` and `H` are
**whole-ink** bbox extents. Evidence: denominator `0x5b1b04..59`, first-x anchor `0x5b1b5d..bd3`.

**Guide path:** `denominator = guide.height`, `y0 = guide+0x58`. Only **stroke0** is consulted
(not the first non-empty stroke). If stroke0 is non-empty, `x0 = field2 ? stroke0.x[0] : stroke0
bbox xmin`; with `h0 = stroke0.ymax - stroke0.ymin`, if `h0 > eps` and `guide.height > eps` then

    denominator = (1 - field3)*guide.height + field3*h0      (this operation order)

Guide path evidence: blend `0x5b1fa8..0x5b1fe7`, transform `0x5b1ffd..0x5b201a`; the wrapper
fixes the initial stroke count to 1 at `0x5b1dc5`. A guide is valid iff width and height are both
`>= eps` and finite. In guide mode `y0 = guide.top` and `x0` is stroke0's first x when field2 is
true, else stroke0's bbox xmin.

Note that is a blend of **heights**, not of reciprocal scales — with field3 = 0.5 it is the mean
height, which is *not* the mean scale. `field1` is ignored in guide mode. Empty stroke0 or no
strokes leaves `x0 = 0`.

**What this means for us:** for en-US ink with no writing guide, `field1 = 0` so `margin = 0` and
the divisor is just `h = max(H, W/100)` — i.e. the bbox height for any normal ink, which is what
we already do. The `x0`/`y0` origin choice is irrelevant regardless, because all 10 features are
differences, ratios or angles, so translation cancels. **Size normalisation is therefore not our
bug**; the erroneous t-scaling was.

### AddPenUpStrokes (SOLVED — matches our implementation)

Each bridge is **exactly two points**, copied from the previous stroke's last point and the next
stroke's first point, carrying **those endpoints' own t values**. It is *not* densified or
resampled. The pen flag is **stroke-level** (`pen_down = false`, at `+0x78`), not per point.
**No bridge is added before the first stroke or after the last** (loop bound `0x5b3c79`).
This matches what we already do. Route: RTTI `0xc722b` -> typeinfo `0x83d548` -> vtable apply
slot `0x83d540` = `0x5ad798` -> `0x5b38da`.

Edge cases worth honouring:
- Real strokes have `pen_down` **forced true** on copy (`0x5b3bc9`); bridges store false (`0x5b3bc1`).
- The bridge gets t values **only if BOTH neighbouring strokes have non-empty t arrays**
  (`0x5b3ae6..0x5b3b16`); otherwise the bridge's t array is **absent entirely**, not zero-filled.
  The processor then **rejects** such ink rather than filling it in: it compares `x_size` vs
  `t_size` at `0x5a037f` and on mismatch jumps to `0x5a11d0`, raising
  `"Malformed input (x.size != t.size). key = "` (literal `0x936d8`). There is no fallback
  timestamp path. So the right behaviour to mirror is **raise**, not degrade gracefully.
  Unreachable in the real en-US pipeline, since `HallucinateTime` makes every stroke
  count-complete before bridges are built — but reachable if `AddPenUpStrokes` is run standalone,
  which is presumably why native validates instead of trusting the invariant.
- An empty stroke logs `"Empty stroke"` and is skipped (`0x5b3a27`).
- `N <= 1` just copies the original.
- Pathological guard: if `N >= 2` and the **second** original stroke already has `pen_down = false`,
  the routine returns immediately — but *after* clearing the output strokes, so it is NOT a
  harmless no-op (`0x5b3924..0x5b3942`). Do not feed it already-bridged ink.

The pipeline applies steps in strictly the recospec's repeated-field order (`0x5bbb13..0x5bbb34`),
confirming we can drive it straight from the parsed spec.

Corroboration for the t-offset finding: `TimeMsToS` independently proves `+0x50` is the t array,
which is the same offset the size transform provably leaves untouched. Two independent routes to
the same conclusion.

### NormalizeTime (SOLVED — and it is NOT the paper's eq. 2)

`0x5b4b85..0x5b4ca5`, wrapper `0x5ad4f8`. It performs **global origin subtraction and nothing
else**. The sole floating-point operation is a `subss` at `0x5b4c67` using the ORIGINAL
`strokes[0].t[0]` (captured `0x5b4c11/18`, reloaded `0x5b4c41`), applied to every existing t
across every stroke. **No division, no path length, no bbox, no target range, no per-stroke
reset.** Returns unchanged if the ink is empty or stroke0 has no t. It preserves global
non-decreasing order if the input had it; it does not repair bad ordering, and float rounding
can collapse distinct times.

Independently cross-checked by a second agent reading the full routine. Treat as settled.

**Consequence:** time stays **globally monotonic across strokes**. Our per-stroke reset to zero
was inventing a problem — it made all 103 pen-up bridges in the corpus run *backwards* in time.

### HallucinateTime (SOLVED)

Gate `0x5ad516..0x5ad59b` compares each stroke's `x_size` against its `t_size`. If **any**
mismatch, ALL strokes' t arrays are regenerated; if all match, field2 (force, = 0 for en-US)
decides. The generator (`0x5b4ca6..0x5b4dfc`) writes `global_point_index * interval` in float32
(`0x5b4d3e..46`), with the counter advancing across strokes (`0x5b4d6d`) and **no extra
inter-stroke gap** — so with interval = 20.0 the whole ink gets `0, 20, 40, ...` straight
through. No geometry, and no test for constant/decreasing/NaN input times.

So **field 1 (20.0) is a per-point time increment**, not a sampling rate in the usual sense.

### The time-magnitude question: RESOLVED — eq. 2 lives inside the fitter

The paper's eq. 2 normalisation is **not** a preprocessing step. It is applied **per stroke
inside the curve fitter** (`0x5c3700`, called from `0x5a0508`), after point selection/thinning:

    duration = t_last - t_first                    # 0x5c3f72..7c
    if not (duration > 0):   t_column = 0          # strict gate 0x5c3f84..87; memset 0x5c3fef..4003
    else:                    L = sum of spatial segment lengths          # 0x5c3f60
                             t *= L / duration                           # 0x5c417e, 0x5c7233..3c

This reconciles everything: by the time the fitter runs, x and y are already size-normalised into
~[0,1], so `L` is a normalised path length and the rescaled time span is O(1) — exactly the
magnitude the network needs — while every preprocessing step provably leaves t alone.

Three details that matter:
- **No local-origin subtraction.** t is multiplied only; it keeps the offset left by
  `NormalizeTime`'s global shift. The emitted time legs are differences so the offset cancels
  there, but it does change `γ0` and hence least-squares conditioning and Newton projection.
- **`L` is over the RETAINED/thinned points**, not necessarily the raw stroke.
- **The degenerate branch zeroes the entire t column** rather than skipping the rescale, and the
  gate is strictly `> 0` so NaN takes that branch too. A pen-up bridge whose two endpoints share
  a timestamp therefore gets time legs of exactly 0.

The pen flag is read only AFTER the fit (`0x5a07a2`) and there is **no bridge exemption** —
bridges are rescaled identically to real strokes.

So SPEC §13 item 1 was wrong in mechanism but accidentally close in magnitude: scaling t by
`1/height` approximates `pathlen/duration` on already-normalised coordinates. That coincidence is
precisely why the bad fix looked convincing.

### Superseded: the earlier open question

Putting the proven steps together: t ends up in raw milliseconds (or `0,20,40,...`) while x,y are
normalised to ~[0,1]. That should make the encoder's time legs enormous — precisely the ~50x
blowup originally measured. Yet the network plainly needs O(1) values there.

SPEC §13 item 1 (our "scale t inside NormalizeSize" fix) is **wrong** — disproved twice over. It
happened to approximate the right magnitude by accident, since `path_len/height` is close to the
normalised path length.

**Leading hypothesis, not yet confirmed:** the paper's eq. 2 time normalisation lives **inside
the curve fitter**, not in preprocessing. The paper introduces it in section 2.1.2 (the Bézier
section) rather than as a preprocessing step, and there it would act on already-size-normalised
x,y, naturally yielding a time span of O(1). A disassembly pass over
`labeled_ink_curve_processor.cc` / `curves.cc` is checking for it.

Explicitly flagged by the cross-checking agent, and worth repeating: **do not reinterpret the
time normalisation to fit the expectation that gamma should be small.** If no such rescale exists
downstream, the correct conclusion is that the network really is fed raw-millisecond time legs
and we must accept that.

### Splitter corrections: no effect

Both rounds of binary-derived splitter corrections — the corner/degenerate semantics, and then
the 100-sample interior curvature grid with residual-no-corner fallthrough — produced a **delta
of exactly zero** on both corpus splits, with every confusion unchanged. Both selector branches
are now binary-settled, and segmentation is conclusively **not** the source of the residual
errors.

## 15. Fitter point thinning (recovered) — completes the chain

Before the time rescale, the fitter thins the stroke's points. This determines both which points
are fitted AND which points `L` is summed over.

`e = bbox_diagonal * 0.0003452669770922512` (float32 at `0x9f27c`, ~sqrt(float epsilon); loaded
`0x5c3a82`, squared `0x5c3a91`). `cum` is the cumulative spatial segment length array
(`0x5c3834`, `0x5c3880..0x5c38e0`).

A **maximum-cardinality predecessor chain** over the original ordered points, by DP
(`0x5c3a9b..0x5c3b8b`) — optimal, not greedy:

    length = [1]*N;  pred = [0]*N;  prefixmax = [1]*N;  best = 0
    for i in 1..N-1:
        pred[i] = i
        for j in i-1..0:                            # DESCENDING
            if length[i] > prefixmax[j] + 1: break  # early exit, 0x3af0..3b00
            cand = length[j] + 1
            if cand < length[i]:      continue      # >= is accepted
            if cum[i] - cum[j] <= e:  continue
            if dist2(i, j) <= e*e:    continue
            length[i] = cand;  pred[i] = j
            if length[i] >= length[best]: best = i
        prefixmax[i] = length[best]
    # walk back from `best` via pred for exactly length[best] indices (0x3bdc..3c7b)

Both gates must hold: arc-length separation AND Euclidean separation, the latter compared
squared. Both are strict, and skip unordered/NaN comparisons. Tie-breaking falls out of the loop
shape: `pred[i]` ends as the **earliest** eligible j (descending loop overwrites equal-length
candidates), while **latest** i wins ties for `best`. If no transition is ever taken, the result
is the singleton index 0.

`dist2` is squared Euclidean in x,y only — time excluded, like `L`.

For a two-point pen-up bridge with distinct finite endpoints both points survive; coincident
endpoints collapse to one, which then takes the singleton path (all-zero time coefficients).

Since `e` is ~3.45e-4 of the bbox diagonal, this is a near-duplicate filter that should be a
no-op on clean ink and only bites on real ink with repeated or jittery samples. It also makes our
earlier `initial-s` guard redundant: thinning removes coincident points before the
cumulative-distance normalisation, so the native unguarded `0/0` cannot fire except for a wholly
degenerate stroke.

### The complete fitter chain, all binary-proven

    thin (DP above)
      -> L        = Σ hypot(Δx, Δy) over RETAINED points        (t excluded)
      -> duration = t_last - t_first over RETAINED points
      -> if duration > 0:  t *= L/duration   else:  t = 0       (whole column, strict gate)
         (the length sum uses a SIMD reduction for larger arrays, `0x5c4008..0x5c4065`,
          so it is not invariably left-to-right float32 accumulation)
      -> fit                                                    (no local origin subtraction)

Every stage from ink to logits is now recovered from the binary rather than inferred from the
paper — preprocessing, thinning, time rescale, curve fitting, splitting, feature encoding,
network, CTC mapping and LM decoding.

## 16. Result and ablation

On a 74-ink corpus labelled by the real ML Kit SDK on-device (61 discovery / 13 held-out,
frozen before any scoring), the reimplementation reproduces the SDK's **top candidate on 74/74**
inks with greedy CTC decoding — 61/61 discovery, 13/13 held-out.

### The ablation that proves the mechanism

Disabling only the fitter-side eq. 2 time rescale, leaving everything else identical:

| configuration | discovery | holdout | max time leg |
|---|---:|---:|---:|
| **with** fitter rescale (native) | **61/61** | **13/13** | 1.78 / 2.15 |
| without it (raw time straight through) | 1/61 | 0/13 | **1493 / 1596** |

Accuracy collapses to chance and the time legs are ~1000x out of range. This is the strongest
available confirmation that the rescale is real and that we located it correctly — it is not a
parameter that merely happened to improve a score.

Feature magnitudes with the native configuration: `dt` in [0, 1.781], `start_leg` in [0, 0.670],
`end_leg` in [-0.680, 0], means 0.677 / 0.240 / -0.239 — comparable to dx/dy, exactly as the
paper describes ("most of the resulting values are in the range [-1, 1]").

### Contribution of each fix

| change | discovery delta |
|---|---:|
| feature layout from `CoeffsToFeaturesAnglesRatios` (order, pen polarity, time legs) | 0 -> 52/61 |
| corner + degenerate splitter semantics | **0** |
| 100-sample interior curvature grid + residual fallthrough | **0** |
| proven preprocessing + fitter time rescale | +9 -> **61/61** |
| exact DP thinning | **0** |

Segmentation turned out to be irrelevant for this corpus: both splitter corrections and the
thinning DP each moved nothing. Thinning removes 7 points total, all from coincident two-point
pen-up bridges (which then collapse to singletons); **no real stroke loses any point**. Our
`initial-s` guard fired 0/1572 times on discovery and 0/620 on holdout — effectively dead for
real ink, retained only for degenerate traces and direct helper calls.

### Char-class rescoring: the weights are COSTS

The recospec's `CharClassesBeamScorerSpec` weights are applied as **costs (subtracted), not
bonuses**. Tested with both fixed signs, using Google's values verbatim and no tuning:

| sign | discovery | holdout |
|---|---:|---:|
| word LM only (no rescoring) | 55/61 | 13/13 |
| added as a bonus (+1) | 53/61 | 12/13 |
| **subtracted as a cost (-1)** | **59/61** | **13/13** |

Greedy is 61/61 and 13/13 under both signs — rescoring is decoder-side only, as expected.

The cost interpretation is principled rather than fitted: the FST LM is already tropical, where
weights are negative log probabilities, so a rescoring LM combining with it must use the same
convention. Under it, `number: +1.30` *penalises* digits and `lower: -0.99` *rewards* lowercase.

That fixes `l->|`, `g->9`, `y->Y` and `up->Up`. Two misses remain, `0->o` and `s->S`, and they are
genuine glyph ambiguities the class weights cannot resolve — `0/o` in particular demands the
opposite of what `g/9` demands on the very same `number` weight, so no value of it satisfies both.

Still unmapped: the `upper_` / `lower_` classes, whose membership is language-specific and comes
from somewhere we have not located (the table carries bare `upper_en_us` / `lower_en_us` entries
with no character list). We applied only the classes with explicit membership rather than guess.

### Final numbers

| decode | discovery | holdout | total |
|---|---:|---:|---:|
| greedy (acoustic only) | 61/61 | 13/13 | **74/74** |
| full LM + char-class rescoring | 59/61 | 13/13 | **72/74** |

Greedy matching more often than the LM path is an artifact of the corpus being 51/74 single
characters, which is the worst case for a word LM; the two remaining LM misses are near-ties the
LM tips and a purely acoustic decode never does.

### Known gap: n-best ordering below rank 1

Top-1 always matches, and often rank 2 (`cat` -> `["cat","Cat",...]` in both). Lower ranks
diverge, because the decoder runs with a neutral LM weight (1.0) and insertion bonus (0.0): the
corresponding floats in the recospec `DecoderConfig` (e.g. `4.6.7 = 0.6152`, `4.6.10 = -1.2286`)
and the `CharClassesBeamScorerSpec` weights are present but their meanings were **not** recovered,
and we declined to invent them. That is the obvious next thread to pull.

## 17. Char-class rescoring, verified against the binary

This was the one component implemented from inference rather than disassembly. A dedicated pass
settled the membership semantics; the sign/magnitude is addressed below.

### Membership parsing (SOLVED — our implementation was wrong for most languages)

Parser at `0x50bb80`. Behaviours we did not have, each of which breaks non-English languages:

- **A class line with no members is skipped entirely** (`0x50bd51..0x50bd6f` branches straight to
  the next line at `0x50c061`, *before* underscore handling). This is why en-US's bare
  `upper_en_us` / `lower_en_us` lines contribute nothing and those classes are inert there,
  despite `lower_` carrying the table's largest weight (2.0). There is no runtime locale lookup
  on that path. Our choice to leave them unapplied was correct — but for the wrong reason.
- **Suffixed names truncate to the prefix INCLUDING the underscore** (`0x50bda0..0x50be09`), so
  `upper_be <chars>` feeds class `upper_`. **163 of the 361** shipped tables have suffixed
  classes that DO carry members (`upper_`/`lower_`, plus `punct_` in 5 and `quote_` in 3). en-US
  is the unusual one.
- **Overlap is last-write-wins**, never additive; the code logs literally
  `"overriding character class from ... to ..."` (`0x50bf53`, `0x50bf81`). **78 of the 361**
  tables contain characters in more than one class — we previously *raised* on this, which would
  have failed outright on all 78.
- **A missing weight defaults to 0.0** (`0x508db0`, constant `0x9ef64`), not an error.
- **Lookup is by the whole symbol string**, not a per-codepoint sum (`0x508c11`, equality via
  `0x1d1dfd` length + memcmp). Multi-codepoint symbols simply miss and fall back to
  `no_char_class` — there is no Unicode upper/lower heuristic.
- Member text is split by single UTF-8 codepoint, with a correct 4-byte/non-BMP path
  (`0x50cb58`), so non-BMP characters are not split into surrogates.
- The table also supports `[[HEX]]` / `[[HEX-HEX]]` inclusive scalar ranges (`0x50b438`,
  expansion `0x50b697..0x50b7d3`) — **we do not implement this**; check whether any shipped table
  uses it before relying on non-Latin scripts.
- Loading a class table from a file is unsupported on Android (`0x50af6a`); only the inline
  table is used.

`src/decoder.py` now matches all of the above except the `[[HEX]]` range syntax.

### Combination (sign CONFIRMED CONTRADICTED — we had it backwards)

The scorer's runtime slot (`0x508e78`) loads the class weight indexed by `label - 2` — an
independent corroboration of our net→FST `+2` mapping, arrived at from a different part of the
binary — XORs it with the sign mask at `0xa1b20` and appends the **negated** value.

The `SumWeightCombiner` (`0x593898`, vptr `0x83b520`) then forms, per term,
`out = (entry+0x30) * original_scalar + (entry+0x44) * component_score`, with `+0x30` from
decoder field **f6 (default 1.0)** and `+0x44` from **f7 = 0.615223527**. So:

    cost = 1.0 * original_transition_cost + 0.615223527 * (-raw_class_weight)

i.e. the class contribution to **cost** is `-0.6152 * weight`. In our parameterisation that is
`char_class_weight = +0.6152`; we currently ship **-1**, which is the opposite direction and the
wrong magnitude.

Uncomfortably, the native direction scores **worse** on our corpus (+1 -> 53/61, vs -1 -> 59/61,
vs 55/61 with rescoring disabled). We follow the binary regardless — picking -1 because it scores
better is the same error as the time-scaling fix in §13, which produced good numbers from a wrong
mechanism.

**Settled.** The word component is the plain FST cost with tropical sign preserved — arc and
backoff weights are ADDED, never negated (`0x592ae2`, `0x592b4a`) — and the en-US CTC topology
has zero arc weights (`0x593f0b`, constant `0x9fd70`), so `f6`'s graph-cost term contributes
nothing. Smaller costs win (`0x5160ad`). The full rescoring contribution to cost is therefore

    0.615223527 * (word_FST_cost - class_value)

i.e. in maximised-score convention the class value is **added** with coefficient +0.6152, and the
**same** coefficient scales the word-LM cost (`0x4e049e` for the class component, `0x4e0690` for
the word component — the shared multiplier is the point). `f7 = 0.615223527` maps to proto offset
`+0x44` / decoder offset `+0x8c` (`0x59ca5e`, `0x4de0c6`).

So the faithful setting is `word_weight == char_class_weight == 0.615223527`, with the class term
subtracted from the word cost. We previously shipped `lm_weight = 1.0`, `char_class_weight = -1`:
wrong sign AND wrong scale, on both terms.

Also corrected: **f10 = -1.2286 is NOT the char-class weight.** It is a
`PreSpaceAwareNetworkScoreCache` synthetic first-frame space penalty (`0x4f1182`,
`cost = -acoustic_scale * (first_frame_max + f10)`) — a separate mechanism we do not implement.

Blanks and collapsed repeats are not charged (CTC output 0 guards skip rescoring, `0x593edc`),
matching our implementation. The combiner also snaps sub-epsilon changes back to the original
scalar (`0x593768..0x593896`, eps `0x9f30c` = 3.814697265625e-6) — tolerance suppression, not
normalisation, and irrelevant at our O(1) weight magnitudes.


## 18. Two further decoder parameters recovered

Both were sitting at neutral defaults in our decoder because their meanings were unknown.

### `4.6.1.7 = -1.901311993598938` — per-emitted-label cost (insertion penalty)

A cost added once per emitted label, **including space**, exempted when the output label is zero
(blank/collapsed repeat). Serializer `0x59ed26..0x59ed31`, options transfer `0x4ea5dc`,
conditional addition `0x506f43..0x506f66`, zero-output exemption `0x4e8954`.

Being negative, it *reduces* cost per token, i.e. it is an insertion **bonus** of +1.9013 in our
maximised-score convention. We had `insertion_bonus = 0.0`.

### `4.6.10 = -1.228597641` — synthetic initial-space penalty

NOT the char-class multiplier, as we first suspected. It configures a conditional synthetic
leading-space alternative in `PreSpaceAwareNetworkScoreCache`: gate `0x4dfce9..0x4dfd25`, stored
`0x4dff6f..0x4dffc8`, added before acoustic conversion `0x4f1182..0x4f118c`. At acoustic scale 1
the synthetic space costs 1.2286 more than blank. It applies when the preceding context is
non-empty and does not end in a space. **We do not implement this**; it only matters for
continuation contexts, which our single-shot recognition never supplies.

### Verdict on the char-class component

Correct in our implementation: per-emitted-label granularity, blanks/collapsed repeats exempt,
en-US's empty `upper_`/`lower_` left unapplied, `no_char_class` fallback, whole-symbol lookup.

Wrong, now fixed: the sign and scale of both the class term and the word-LM weight; membership
parsing (suffix truncation, last-write-wins, `[[HEX]]` ranges, 0.0 default weight).

Still UNKNOWN: separate final/EOS handling of the class component, and any candidate-level
normalisation beyond the traced path.

## 19. Decoder weights: shipped faithful, measured worse, and why

With the sign and scale corrected per §17, the full corpus measures:

| configuration | total | singles | words |
|---|---:|---:|---:|
| native (`lm = class = 0.615223527`) | 68/74 | 47/51 | 21/23 |
| native + insertion bonus (+1.9013) | 68/74 | 47/51 | 21/23 |
| previous empirical (`lm=1.0, class=-1`) | **72/74** | 49/51 | **23/23** |

The insertion bonus changes nothing — it cancels between equal-length hypotheses.

The binary-faithful configuration is **worse on our corpus, on both slices**. An early hypothesis
that this was a corpus artifact (51/74 single characters, where a word LM has little to say) is
**refuted**: the empirical config also wins on words, 23/23 vs 21/23.

We ship the faithful values anyway:

- The `-1` sign is **verified wrong** (§17). It wins here largely by biasing toward lowercase on a
  lowercase-heavy corpus — five of the six faithful misses are lowercase-truth/uppercase-output.
- Every other time we preferred the binary over our own inference, accuracy *improved*; the one
  time we preferred a good-looking number over the mechanism (§13's time scaling) we lost hours.
- There is a concrete **unmodelled** quantity that plausibly explains the whole gap: native applies
  an `acoustic_scale` (visible in the pre-space penalty, `cost = -acoustic_scale * (first_frame_max
  + f10)`). We assume 1.0. If it is not 1.0, our LM-to-acoustic balance is wrong by that factor no
  matter how correct the LM-internal weights are — so "faithful" is currently *partial*.

**Recovering `acoustic_scale` is the single highest-value open thread.** Until then the empirical
values remain available:

    Recognizer.recognize(..., lm_weight=1.0, char_class_weight=-1.0)

Greedy decoding is unaffected by all of this and remains **74/74**.
