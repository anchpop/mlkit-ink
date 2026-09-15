//! ML Kit Digital Ink Recognition, reimplemented outside the SDK.
//!
//! ```text
//! ink strokes
//!   -> preprocess      named pipeline steps from the recospec
//!   -> curve           thinning, splitting, cubic fitting, merging
//!   -> features        the 10 Bezier features
//!   -> net             stacked bidirectional IndyLSTM + linear head
//!   -> decoder         CTC beam search, shallow-fused with the n-gram FST
//!   -> candidates
//! ```
//!
//! Every model artifact is Google's own, fetched from the public
//! `dl.google.com` catalog; only the inference code here is ours. How each
//! format was recovered is written up in SPEC.md, with each claim anchored to
//! an address in `libdigitalink.so`.
//!
//! # Portability
//!
//! The core is `no_std` + `alloc` and reads every artifact out of a borrowed
//! `&[u8]`. It never opens a file, allocates a thread, or touches the clock, so
//! the same code runs on a desktop against an mmap, in a browser against a
//! `Uint8Array`, and on a phone against an asset handle. Host concerns live in
//! the `mlkit-ink-cli`, `mlkit-ink-wasm` and `mlkit-ink-ffi` crates.
//!
//! # Numeric precision is load-bearing
//!
//! The SDK's fitter is f32 and its decoder is f64, and the two are not
//! interchangeable: widening the fitter changes which curves get split, which
//! changes the feature count, which changes the transcript. Precision here
//! matches the native code deliberately. Do not "improve" it.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
pub mod error;

pub mod ctc;
pub mod curve;
pub mod decoder;
pub mod features;
pub mod float;
pub mod fst;
pub mod ink;
pub mod mat;
pub mod net;
pub mod netgrad;
pub mod optimize;
pub mod packs;
pub mod preprocess;
pub mod proto;
pub mod recognizer;
pub mod recospec;
pub mod settings;
pub mod tflite;

pub use error::{Error, Result};
pub use ink::{Point, Stroke};
pub use recognizer::{Candidate, Recognizer};
pub use settings::{CurveSettings, DecoderSettings, NUM_FEATURES};
