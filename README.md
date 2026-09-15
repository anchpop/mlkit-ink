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

**The input encoding.** Ink is fit with cubic Bézier curves, each encoded as a 10-vector. Carbune
et al., *Fast multi-language LSTM-based online handwriting recognition* (IJDAR 2020) describes this
exact system — but its **field order, time-normalisation placement and pen-flag polarity are all
wrong** relative to the shipped code. The real layout, read out of `CoeffsToFeaturesAnglesRatios`,
is `(pen_down, dx, dy, a1, d1, a2, d2, dt, t1-t0, t2-t3)`. Searching over plausible orderings
scored 8/61 on our corpus; reading the binary scored 61/61.

Full details, with evidence for every claim, are in [`SPEC.md`](SPEC.md).

## The Rust port

`rust/` is a second, independent implementation: same pipeline, no Python, no NumPy, and — in the
core crate — **no dependencies at all**. It reproduces the Python stage for stage against frozen
fixtures, so "the port works" is a measured claim rather than an impression:

| stage | agreement with the Python reference |
|---|---|
| Bézier features | 74/74 inks with an **exact** curve count; worst value difference 1.2e-7 (one f32 ulp) |
| network forward pass | **437/437** frame argmaxes; worst logit difference 3.6e-4 |
| CTC greedy, LM beam, empirical beam | 74/74 each, full 5-best lists; worst score difference 7.1e-15 |
| FST traversal | state ids and per-step costs exact over 17 transitions |
| recospec + pack resolution | all **725** catalog tags, all 391 recospecs |

The core is `#![no_std]` + `alloc` and `#![forbid(unsafe_code)]`, and reads every artifact out of a
borrowed `&[u8]`. It never opens a file or allocates a thread, so the same code runs against an
mmap on a desktop, a `Uint8Array` in a browser, and an asset handle on a phone.

```bash
cd rust
cargo run --release -p mlkit-ink-cli -- recognize ../testdata/hi.json -n 5
cargo run --release -p mlkit-ink-cli -- languages
```

```
crates/mlkit-ink       the whole pipeline, dependency-free and no_std
crates/mlkit-ink-cli   fetch/unzip/mmap, recognize, fit  (the host lives here)
crates/mlkit-ink-wasm  wasm-bindgen bindings
crates/mlkit-ink-ffi   C ABI + header for Android (JNI) and iOS (Swift)
```

All of `wasm32-unknown-unknown`, `aarch64-apple-ios`, `aarch64-apple-ios-sim`,
`aarch64-linux-android`, `armv7-linux-androideabi` and `x86_64-linux-android` compile from the same
source; `rust-toolchain.toml` installs them.

**On the browser, skip the language model.** The English FST is 22 MB and the greedy decode is the
configuration that matches the SDK 74/74, so `Recognizer.greedy_only(recospec, tflite)` ships about
5 MB instead of 27 MB and loses nothing measurable.

### Gradient descent on ink

Because the pipeline is differentiable once its *combinatorial* decisions are held fixed, the Rust
port can run the recognizer backwards: nudge stroke coordinates until the ink reads as a character
you choose.

```bash
cargo run --release -p mlkit-ink-cli --     fit ../testdata/corpus/trials/lower-n/ink.json h --svg /tmp/n-to-h.svg
```

```
target      "h"
before      "n"
after       "h"
CTC loss    6.8578 -> 0.0981  (best at step 138 of 150)
moved       4.0% of the ink's diagonal (rms)
```

Three links, differentiated three different ways, for reasons worth stating:

- **CTC loss → logits**: analytic forward-backward (`ctc.rs`).
- **logits → features**: hand-written BPTT (`netgrad.rs`), because a tape over six bidirectional
  216-unit layers would record tens of millions of scalar nodes. Verified by directional
  derivatives against the real network, agreeing to 0.01–0.6%.
- **features → points**: *finite differences* through the real fitter (`optimize.rs`). This is
  deliberate. The curve fitter is the most precision-sensitive code in the crate, reproduced
  instruction-faithfully from the SDK; a hand-written differentiable copy of it would be a second
  implementation free to drift from the first. Bumping a coordinate and re-running the actual
  fitter cannot drift, and it is cheap next to the network pass — the expensive link is the
  analytic one.

`--anchor 0 --smoothness 0` turns it into an adversarial-example generator: the result scores
beautifully and is unreadable. The defaults keep it a plausible pen trace, at the cost of failing
on transformations the ink cannot reach — on a ten-pair spot check, six converged, all moving
points by only 2–4% of the ink's diagonal, and the four failures say so rather than reporting a
match they did not achieve.

## Layout

```
SPEC.md              the reverse-engineering writeup — read this first
manifest.json        Google's pack catalog, lifted from the AAR
packmapping.pb       BCP-47 tag -> pack names
src/                 the Python reference implementation
  packs.py           language -> pack resolution + verified download
  recospec.py        protobuf loader for the recognizer spec
  features.py        preprocessing pipeline + Bezier fitting + the 10 features
  tflite_weights.py  weight extraction incl. hybrid dequantization
  indylstm.py        bidirectional IndyLSTM forward pass (numpy)
  fst.py             compact_lm parser + backoff LM
  decoder.py         CTC greedy + LM-fused prefix beam search
  recognize.py       end-to-end entry point
rust/                the Rust port (see above)
tools/               fixture generators and the acoustic_scale sweep
testdata/goldens/    frozen Python output, stage by stage, that the port is tested against
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
is verified incorrect. To use the empirical values:
`Recognizer.recognize(..., lm_weight=1.0, char_class_weight=-1.0)`.

**The unmodelled quantity is now identified.** Native multiplies every neural-net log-posterior by
an `acoustic_scale` before combining it with the (unscaled) LM costs — proven at `0x42ed74`, where
the score accessor reads straight out of the network's output array and multiplies. Its *value* is
not recoverable from the binary: it is a runtime field that nothing in `.text` ever writes, so the
configure path was never found.

Only its ratio to the LM weights matters, so it can be swept rather than guessed. Doing that
(SPEC.md §20, and note the subtlety there about *where* it has to be applied) gives a curve that
**rises monotonically to the edge of the sweep**: 68/74 at the 1.0 we ship, 71/74 from 1.25, 73/74
by 8, against the 74/74 that dropping the language model entirely already achieves.

That shape is the finding, and it is not "the constant is 8". A parameter whose optimum sits at the
boundary of the search is the signature of a **missing mechanism rather than a mis-set constant** —
the sweep is rewarding "use less of our language model", which is what you would expect if our LM
integration is incomplete. We don't implement the synthetic leading-space penalty, and native's
pruning semantics are still unknown. So we ship `acoustic_scale = 1.0` and treat the curve as a
bound on the remaining error, not as a recovered value.

One thing it does settle: every miss is a *ranking* failure — the correct transcript is present in
the n-best and placed second — not the FST refusing to offer it.

Run the tests with:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests   # 96 tests
cd rust && cargo test --workspace --release                   # 104 tests
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
