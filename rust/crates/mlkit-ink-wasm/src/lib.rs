//! WebAssembly bindings: handwriting recognition in a browser, offline.
//!
//! ```js
//! import init, { Recognizer } from "./pkg/mlkit_ink_wasm.js";
//! await init();
//!
//! // Greedy needs no language model, and greedy is the configuration that
//! // matches the real ML Kit SDK on all 74 of our oracle-labelled inks.
//! const r = Recognizer.greedy_only(recospecBytes, tfliteBytes);
//! r.greedy({ strokes: [{ x: [...], y: [...], t: [...] }] });  // -> "hi"
//! ```
//!
//! # Why the language model is optional
//!
//! The English n-gram FST is 22 MB, and loading it is most of this module's
//! cost. It buys you n-best alternatives and word-level context — but not
//! accuracy on our corpus, where the plain acoustic decode already matches the
//! SDK exactly. So a web page that only needs the top answer should skip it,
//! and ship about 5 MB of model instead of 27 MB.

use mlkit_ink::ink::Stroke;
use mlkit_ink::settings::DecoderSettings;
use mlkit_ink::{Candidate, Recognizer as Core};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Model bytes owned on the Rust side, with a recognizer borrowing from them.
///
/// The core recognizer deliberately borrows its model bytes so a desktop
/// caller can hand it a 22 MB mmap without copying. In a browser there is no
/// mmap and nothing else to own the bytes, so this cell does: `self_cell` makes
/// the self-reference safe without the core crate giving up its borrow-based
/// API or this crate reaching for `unsafe`.
struct Model {
    recospec: Vec<u8>,
    tflite: Vec<u8>,
    fst: Vec<u8>,
    /// Empty `fst` means "no language model", distinguishable from an empty
    /// borrow because `self_cell` cannot hold an `Option` of a borrow.
    has_fst: bool,
}

self_cell::self_cell! {
    struct Loaded {
        owner: Model,
        #[covariant]
        dependent: Core,
    }
}

/// A loaded recognizer for one language.
#[wasm_bindgen]
pub struct Recognizer {
    loaded: Loaded,
    settings: DecoderSettings,
}

/// The ink schema: `{ strokes: [{ x: [...], y: [...], t: [...] }] }`.
///
/// `t` may be omitted. Timestamps are only used for the three time features,
/// and the pipeline synthesizes them at a fixed 20 ms step when they are
/// missing — so leaving them out costs a little accuracy, not correctness.
#[derive(Deserialize)]
struct Ink {
    strokes: Vec<Stroke>,
}

#[derive(Serialize)]
struct JsCandidate {
    text: String,
    score: f64,
}

impl From<&Candidate> for JsCandidate {
    fn from(candidate: &Candidate) -> Self {
        JsCandidate {
            text: candidate.text.clone(),
            score: candidate.score,
        }
    }
}

#[wasm_bindgen]
impl Recognizer {
    /// Load a language from raw pack bytes, with the n-gram language model.
    #[wasm_bindgen(constructor)]
    pub fn new(recospec: Vec<u8>, tflite: Vec<u8>, fst: Vec<u8>) -> Result<Recognizer, JsError> {
        Recognizer::build(Model {
            recospec,
            tflite,
            fst,
            has_fst: true,
        })
    }

    /// Load without the language model. Decoding is then best-path only, which
    /// is both far cheaper and, on our corpus, exactly as accurate.
    pub fn greedy_only(recospec: Vec<u8>, tflite: Vec<u8>) -> Result<Recognizer, JsError> {
        Recognizer::build(Model {
            recospec,
            tflite,
            fst: Vec::new(),
            has_fst: false,
        })
    }

    fn build(model: Model) -> Result<Recognizer, JsError> {
        let loaded = Loaded::try_new(model, |model| {
            Core::load(
                &model.recospec,
                &model.tflite,
                model.has_fst.then_some(model.fst.as_slice()),
            )
        })
        .map_err(js_error)?;
        let settings = loaded.borrow_dependent().settings;
        Ok(Recognizer { loaded, settings })
    }

    /// Best-path decode. Returns the recognized text.
    pub fn greedy(&self, ink: JsValue) -> Result<String, JsError> {
        let strokes = parse_ink(ink)?;
        self.loaded
            .borrow_dependent()
            .recognize_greedy(&strokes)
            .map(|candidate| candidate.text)
            .map_err(js_error)
    }

    /// Full decode. Returns `[{ text, score }]`, best first. Without a language
    /// model this returns the single greedy candidate.
    pub fn recognize(&self, ink: JsValue, nbest: usize) -> Result<JsValue, JsError> {
        let strokes = parse_ink(ink)?;
        let candidates = self
            .loaded
            .borrow_dependent()
            .recognize_with(&strokes, nbest, &self.settings)
            .map_err(js_error)?;
        let converted: Vec<JsCandidate> = candidates.iter().map(JsCandidate::from).collect();
        serde_wasm_bindgen::to_value(&converted).map_err(|e| JsError::new(&e.to_string()))
    }

    /// The `[T, 10]` Bezier curve features, flattened row-major. Useful for
    /// drawing what the network actually sees.
    pub fn features(&self, ink: JsValue) -> Result<Vec<f32>, JsError> {
        let strokes = parse_ink(ink)?;
        let features = self
            .loaded
            .borrow_dependent()
            .features(&strokes)
            .map_err(js_error)?;
        Ok(features.as_slice().to_vec())
    }

    /// Number of feature columns, so a caller can reshape [`Recognizer::features`].
    #[wasm_bindgen(getter)]
    pub fn feature_width(&self) -> usize {
        self.loaded
            .borrow_dependent()
            .spec
            .curve_settings
            .num_features()
    }

    #[wasm_bindgen(getter)]
    pub fn has_language_model(&self) -> bool {
        self.loaded.borrow_owner().has_fst
    }

    /// Switch to the empirically tuned decoder weights. They score better on
    /// our corpus (72/74 against 68/74) with a sign the disassembly proves
    /// wrong, so they are opt-in rather than the default.
    pub fn use_empirical_weights(&mut self) {
        let beam_width = self.settings.beam_width;
        self.settings = DecoderSettings {
            beam_width,
            ..DecoderSettings::empirical()
        };
    }
}

fn parse_ink(value: JsValue) -> Result<Vec<Stroke>, JsError> {
    let ink: Ink =
        serde_wasm_bindgen::from_value(value).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(ink.strokes)
}

fn js_error(error: mlkit_ink::Error) -> JsError {
    JsError::new(&error.to_string())
}
