# mlkit-ink

Google's **ML Kit Digital Ink Recognition**, running locally outside the ML Kit SDK.

This loads Google's own shipped model artifacts — the recognizer spec, the neural net and the
language model — and runs the full recognition pipeline in Python. No ML Kit, no Play Services,
no Android device at runtime.

> **Status: working.** On a 74-ink corpus labelled by the real ML Kit SDK running on-device,
> this reproduces the SDK's top candidate on **74/74** inks (61/61 discovery, 13/13 held-out).
> Every stage — preprocessing, point thinning, time normalisation, Bézier fitting, both split
> selectors, the feature encoder, the network, CTC mapping and FST decoding — was recovered from
> Google's own binary and verified, not inferred from the paper. See "Status" below for the one
> remaining known gap (n-best ordering below rank 1).

## Why this is possible

The `digital-ink-recognition` AAR ships `assets/manifest.json`: a catalog of all **808 model
packs**, each with a plain, unauthenticated `https://dl.google.com/handwriting/models/...` URL
and a SHA-1. Nothing is device-bound or account-bound.

Each language resolves to three artifacts:

| artifact | size | scope |
|---|---|---|
| **recospec** | ~4 KB | per language — protobuf config: preprocessing pipeline, charset, decoder params |
| **tflite net** | 1–8 MB | **per script** — one Latin net serves English and ~60 other languages |
| **compact FST** | ~22 MB | per language — the n-gram language model |

The catalog covers 725 language tags across 27 scripts.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install ai-edge-litert numpy tflite protobuf

# resolve + download everything for a language (sha1-verified)
PYTHONPATH=. .venv/bin/python src/packs.py en-US

# recognize
PYTHONPATH=. .venv/bin/python -m src.recognize testdata/hi.json -n 5
```

```python
from src.recognize import Recognizer
from src.features import Stroke

r = Recognizer.for_language("en-US")
r.recognize([Stroke(x=[...], y=[...], t=[...])])
```

## What had to be reverse-engineered

**The network uses a custom op.** `bidirectional_sequence_indylstm` is implemented inside
`libdigitalink.so` and stock LiteRT refuses to load it, so it is reimplemented in numpy.
IndyLSTM is an LSTM whose recurrent weight matrix is *diagonal* (Gonnet & Deselaers 2019) —
hence `kernel_w_rec_*` tensors are `[216]` vectors rather than matrices. Weights are hybrid
quantized: the bytes are declared `UINT8` but are really **signed int8 reinterpreted**.

**The recospec schema.** `libdigitalink.so` is stripped, but retains C++ RTTI names and
assertion strings; disassembling its protobuf serializer recovered field-tag -> struct-offset
-> name bindings, including all 30 `InkPreprocessingStepSpec` branches. The reconstructed
schema round-trips all **391** shipped recospecs byte-identically.

**The language model format.** `compact_lm` v2 — a LOUDS-based succinct n-gram FST, not a
stock OpenFst format. Parsed from scratch: 1,921,076 states, 6,708,178 arcs, 100% reachable.

**The input encoding.** Ink is fit with cubic Bézier curves, each encoded as a 10-vector
`(dx, dy, d1, d2, α1, α2, γ1, γ2, γ3, p)`, per Carbune et al., *Fast multi-language LSTM-based
online handwriting recognition* (IJDAR 2020), which describes this exact system.

Full details, with evidence for every claim, are in [`SPEC.md`](SPEC.md).

## Layout

```
SPEC.md              the reverse-engineering writeup — read this first
manifest.json        Google's pack catalog, lifted from the AAR
packmapping.pb       BCP-47 tag -> pack names
src/
  packs.py           language -> pack resolution + verified download
  recospec.py        protobuf loader for the recognizer spec
  features.py        preprocessing pipeline + Bezier fitting + the 10 features
  tflite_weights.py  weight extraction incl. hybrid dequantization
  indylstm.py        bidirectional IndyLSTM forward pass (numpy)
  fst.py             compact_lm parser + backoff LM
  decoder.py         CTC greedy + LM-fused prefix beam search
  recognize.py       end-to-end entry point
proto/recospec.proto recovered schema
evidence/            RTTI names, strings and raw decodes from libdigitalink.so
android-oracle/      on-device harness running the REAL SDK, for ground truth
```

## Status

| stage | state | how it was verified |
|---|---|---|
| pack catalog + download | working | sha1-verified; files byte-identical to what ML Kit itself downloads on-device |
| recospec parsing | working | 391/391 byte-identical protobuf round-trips |
| charset / CTC mapping | working | 360/360 symbol tables; blank is the LAST index, not 0 |
| preprocessing + thinning | working | recovered from the binary, independently cross-checked by a second pass |
| Bézier features | working | recovered from `CoeffsToFeaturesAnglesRatios`; the paper's field order is wrong |
| network forward pass | working | 100% argmax agreement, ~0.01 mean error vs native LiteRT |
| FST language model | working | 100% state reachability; real words cheap, nonsense costly; flips `cax`->`cat` |
| CTC decoder | working | exhaustive alignment tests vs independent enumeration |
| **end-to-end top-1** | **74/74 vs the real SDK** | 61/61 discovery + 13/13 held-out, greedy decode |
| n-best ordering below rank 1 | approximate | top-1 always matches; lower ranks differ (see below) |

**The known gap.** Our top candidate matches ML Kit's on every test case under greedy decoding.
The LM-fused path sits at 68/74 with the binary-faithful decoder weights, or 72/74 with an
empirically-chosen sign that the binary says is wrong.

| decode | discovery | holdout | total |
|---|---:|---:|---:|
| greedy (acoustic only) | 61/61 | 13/13 | **74/74** |
| full LM, binary-faithful weights *(default)* | 56/61 | 12/13 | 68/74 |
| full LM, empirical weights | 59/61 | 13/13 | 72/74 |

We default to the faithful weights even though they measure worse, because the alternative's sign
is verified incorrect and there is a concrete unmodelled quantity (native's `acoustic_scale`,
which we assume is 1.0) that plausibly explains the gap. See SPEC.md §19. To use the empirical
values: `Recognizer.recognize(..., lm_weight=1.0, char_class_weight=-1.0)`.

Run the tests with:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

## Ground truth

`android-oracle/` builds a small app that runs the **real** ML Kit SDK on a connected device
against a JSON ink file, so any change here can be diffed against Google's own output on
identical input:

```bash
android-oracle/run_oracle.sh /absolute/path/to/ink.json
```

Note the English model returns `null` from `getScore()`, so only candidate order is comparable.

## Legal

The model artifacts are Google's, downloaded from Google's own public endpoints. They are
committed here **encrypted with git-crypt** (private repo), alongside the catalog and the
strings/decodes extracted from `libdigitalink.so`. Everything authored by us — source, tests,
the ink corpus, and these writeups — is plaintext. Only the `.zip` archives are stored; their
extracted contents are reproducible from them, so a fresh clone works offline after
`git-crypt unlock`.

This is interoperability work on a shipped format.
