//! The end-to-end recognizer: bytes in, candidates out.
//!
//! Model artifacts are borrowed, never owned. The FST is 22 MB for English and
//! the caller almost always has it mapped or already resident, so copying it
//! into the recognizer would be the single largest avoidable allocation in the
//! crate.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::decoder::{self, Alphabet, CharClassRescoringLm, CharClassScorer};
use crate::error::Result;
use crate::features;
use crate::fst::{BackoffLm, CompactFst};
use crate::ink::Stroke;
use crate::mat::Mat;
use crate::net;
use crate::recospec::{CtcMapping, RecoSpec};
use crate::settings::DecoderSettings;
use crate::tflite::{self, NetworkWeights};

pub use crate::decoder::Candidate;

/// Everything one language needs, loaded and validated.
pub struct Recognizer<'a> {
    pub spec: RecoSpec,
    pub weights: NetworkWeights,
    pub mapping: CtcMapping,
    pub alphabet: Alphabet,
    lm: Option<BackoffLm<'a>>,
    char_classes: Option<CharClassScorer>,
    pub settings: DecoderSettings,
}

impl<'a> Recognizer<'a> {
    /// Load from raw pack contents. `fst` is optional: without it the decoder
    /// falls back to greedy, which is what the FST-less CTC configs need.
    pub fn load(recospec: &[u8], tflite_model: &[u8], fst: Option<&'a [u8]>) -> Result<Self> {
        let spec = RecoSpec::parse(recospec)?;
        let weights = tflite::load_weights(tflite_model)?;

        let mapping = spec.ctc_mapping(weights.num_classes)?.ok_or_else(|| {
            err!(
                Unsupported,
                "this pack is a gesture/classifier model, not a CTC recognizer"
            )
        })?;

        let expected = spec.curve_settings.num_features();
        if let Some(declared) = spec.num_features {
            ensure!(
                declared == expected,
                Unsupported,
                "recospec expects {declared} input features; the Bezier encoder produces {expected}"
            );
        }
        ensure!(
            weights.input_size == expected,
            Unsupported,
            "network takes {} inputs; the Bezier encoder produces {expected}",
            weights.input_size
        );

        let alphabet = build_alphabet(&mapping);
        let classes = spec.decoder.char_classes();
        let char_classes = if classes.is_empty() || spec.decoder.char_class_weights.is_empty() {
            None
        } else {
            Some(CharClassScorer::from_tables(
                &classes,
                &spec.decoder.char_class_weights,
                &alphabet,
            )?)
        };

        let lm = match fst {
            Some(bytes) => Some(BackoffLm::new(CompactFst::parse(bytes)?)),
            None => None,
        };

        let mut settings = DecoderSettings::default();
        if let Some(width) = spec.decoder.beam_width {
            settings.beam_width = width;
        }

        Ok(Recognizer {
            spec,
            weights,
            mapping,
            alphabet,
            lm,
            char_classes,
            settings,
        })
    }

    /// Ink -> `[T, 10]` Bezier curve features.
    pub fn features(&self, strokes: &[Stroke]) -> Result<Mat> {
        features::extract(strokes, &self.spec.curve_settings, &self.spec.pipeline)
    }

    /// Ink -> `[T, num_classes]` unnormalized logits.
    pub fn logits(&self, strokes: &[Stroke]) -> Result<Mat> {
        net::forward(&self.weights, &self.features(strokes)?)
    }

    /// Best-path decode, ignoring the LM entirely. This is the configuration
    /// that matches the real SDK on 74 of 74 corpus inks.
    pub fn recognize_greedy(&self, strokes: &[Stroke]) -> Result<Candidate> {
        decoder::greedy_decode(&self.logits(strokes)?, &self.alphabet)
    }

    /// Full decode: beam search, shallow-fused with the word LM and rescored by
    /// character class when the pack carries those tables.
    pub fn recognize(&self, strokes: &[Stroke], nbest: usize) -> Result<Vec<Candidate>> {
        self.recognize_with(strokes, nbest, &self.settings)
    }

    /// As [`Recognizer::recognize`], with the decoder weights supplied per call
    /// rather than taken from the recognizer. Everything expensive to load —
    /// the net and the language model — is independent of these weights, so
    /// callers that want to compare configurations should vary them here rather
    /// than reloading a second recognizer.
    pub fn recognize_with(
        &self,
        strokes: &[Stroke],
        nbest: usize,
        settings: &DecoderSettings,
    ) -> Result<Vec<Candidate>> {
        self.decode_logits(&self.logits(strokes)?, nbest, settings)
    }

    /// Decode logits this recognizer produced earlier.
    ///
    /// Splitting this out is what makes a decoder parameter sweep affordable:
    /// the network is by far the expensive stage and is entirely independent of
    /// the decoder weights, so there is no reason to re-run it per setting.
    pub fn decode_logits(
        &self,
        logits: &Mat,
        nbest: usize,
        settings: &DecoderSettings,
    ) -> Result<Vec<Candidate>> {
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        let Some(lm) = self.lm.as_ref() else {
            return Ok(alloc::vec![decoder::greedy_decode(logits, &self.alphabet)?]);
        };
        match &self.char_classes {
            Some(scorer) if settings.char_class_weight != 0.0 => {
                // The class term rides along on the LM so the two weights stay
                // independent; the wrapper applies both, which is why the search
                // itself is handed a neutral `lm_weight`.
                let fused = CharClassRescoringLm::new(lm, scorer, settings)?;
                let neutral = DecoderSettings {
                    lm_weight: 1.0,
                    ..*settings
                };
                decoder::prefix_beam_search(logits, &self.alphabet, Some(&fused), &neutral, nbest)
            }
            _ => decoder::prefix_beam_search(logits, &self.alphabet, Some(lm), settings, nbest),
        }
    }
}

/// Render the CTC mapping's symbols into the decoder's alphabet.
fn build_alphabet(mapping: &CtcMapping) -> Alphabet {
    Alphabet {
        blank: mapping.blank_index,
        labels: mapping.net_to_fst.clone(),
        texts: mapping
            .net_symbols
            .iter()
            .map(|s| {
                if s == "[[space]]" {
                    String::from(" ")
                } else {
                    s.to_string()
                }
            })
            .collect(),
    }
}
