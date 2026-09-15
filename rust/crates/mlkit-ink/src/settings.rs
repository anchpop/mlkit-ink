//! Configuration recovered from the recospec protobuf.
//!
//! Field numbers are the recospec's own. Names for the fields whose accessors
//! we recovered from `libdigitalink.so` use those names; the rest keep
//! descriptive names and a comment recording the wire identity, because the
//! wire number is the part we actually verified. See SPEC.md sections 8 and 9.

use alloc::string::String;
use alloc::vec::Vec;

/// `research_handwriting.CurveSettings`. Defaults here are the *proto*
/// defaults, not en-US's values: a spec that omits a field really does mean the
/// proto default, and en-US explicitly overrides fields 4 and 5.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSettings {
    /// Field 1: corner-neighbour distance, as a fraction of the stroke bbox diagonal.
    pub tol1: f64,
    /// Field 2: maximum spatial residual / bbox diagonal.
    pub tol2: f64,
    /// Field 3: RMS spatial residual / bbox diagonal.
    pub tol3: f64,
    /// Field 4: control-*polygon* length / endpoint distance. Not arc length.
    pub max_arc_ratio: f64,
    /// Field 5: minimum adjacent control-leg cosine.
    pub split_cos_threshold: f64,
    /// Field 6.
    pub generate_second_order_features: bool,
    /// Field 7: selects the angles/ratios encoder this crate implements.
    pub use_angles_ratios: bool,
    /// Field 9: emit the three time features.
    pub interpolate_time: bool,
    /// Field 10: squash outputs into [0, 1]. Absent (false) for en-US.
    pub normalize_outputs_to_zero_one: bool,
}

impl Default for CurveSettings {
    fn default() -> Self {
        CurveSettings {
            tol1: 0.05,
            tol2: 0.02,
            tol3: 0.01,
            max_arc_ratio: 4.0,
            split_cos_threshold: -0.9,
            generate_second_order_features: false,
            use_angles_ratios: false,
            interpolate_time: false,
            normalize_outputs_to_zero_one: false,
        }
    }
}

impl CurveSettings {
    /// The values every shipped handwriting recospec carries, useful for tests
    /// and for callers that fit curves without loading a model.
    pub fn handwriting() -> Self {
        CurveSettings {
            max_arc_ratio: 3.0,
            split_cos_threshold: -0.8,
            use_angles_ratios: true,
            interpolate_time: true,
            ..CurveSettings::default()
        }
    }

    /// Feature width: 10 with time interpolation, 7 without.
    pub fn num_features(&self) -> usize {
        if self.interpolate_time { 10 } else { 7 }
    }
}

/// Widest feature vector the encoder produces.
pub const NUM_FEATURES: usize = 10;

/// One entry of `InkPreprocessorSpec.steps`, identified by the oneof branch the
/// native step factory dispatches on. Unrecognised branches are preserved so a
/// spec using a step we have not implemented fails loudly and by name.
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessingStep {
    /// Field 5. Subtract the first stroke's first timestamp, in f32.
    NormalizeTime,
    /// Field 6. Regenerate all timestamps at a fixed interval.
    HallucinateTime { interval: f32, force: bool },
    /// Field 2. (Field 7 is `ink_based_slope_correction`, which we do not implement.)
    NormalizeSize {
        margin: f32,
        first_point_origin: bool,
    },
    /// Field 8. Degrades to `NormalizeSize` when no writing guide is attached,
    /// which is always the case for raw stroke input.
    NormalizeSizeWritingGuideFirstStroke {
        margin: f32,
        first_point_origin: bool,
    },
    /// Field 11. Insert the synthetic pen-up bridges between real strokes.
    AddPenUpStrokes,
    /// A branch we parsed but do not implement.
    Unsupported { field: u32, name: String },
}

/// The pipeline 362 of the 391 shipped recospecs carry verbatim. The other 29
/// are gesture/scribe models, which are not CTC recognizers at all.
pub fn handwriting_pipeline() -> Vec<PreprocessingStep> {
    alloc::vec![
        PreprocessingStep::NormalizeTime,
        PreprocessingStep::HallucinateTime {
            interval: 20.0,
            force: false
        },
        PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
            margin: 0.0,
            first_point_origin: true,
        },
        PreprocessingStep::AddPenUpStrokes,
    ]
}

/// Decoder parameters recovered from the recospec's `FstDecoderConfig`.
///
/// The defaults are the *native* values (SPEC.md sections 17 and 18), which are
/// deliberately what we ship even though an empirically tuned alternative
/// scores better on our 74-ink corpus: the empirical sign is verified wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderSettings {
    /// Search field 1.
    pub beam_width: usize,
    /// Decoder field 4.6.7. Scales the word-FST cost *and* the char-class term.
    pub lm_weight: f64,
    /// Decoder field 4.6.1.7, negated: native stores a cost, we maximise a score.
    pub insertion_bonus: f64,
    /// Same native multiplier as `lm_weight`; separable here so callers can
    /// disable rescoring (0.0) or reproduce the empirical configuration.
    pub char_class_weight: f64,
    /// Native multiplies every neural-net log-posterior by this before
    /// combining it with the (unscaled) LM costs — proven at `0x42ed74`, but
    /// its value was not recoverable from the binary. See SPEC.md section 20.
    ///
    /// Modelled explicitly, rather than silently assumed to be 1.0, so the
    /// assumption is visible and adjustable. It is applied to the per-frame log
    /// probabilities before the alignment sum, which is where native applies
    /// it. Folding it into the LM weights instead gives the same ratio but is
    /// *not* equivalent, because CTC sums over alignments — see
    /// [`crate::decoder::prefix_beam_search`].
    ///
    /// Our corpus prefers values above 1, monotonically, all the way to the
    /// edge of the sweep: 68/74 here at 1.0, 73/74 by 8, against the 74/74 that
    /// dropping the language model entirely already gets. An optimum at the
    /// boundary points at a missing mechanism rather than a mis-set constant,
    /// so no value is read off that curve and we ship 1.0.
    pub acoustic_scale: f64,
}

/// Decoder field 4.6.7.
pub const NATIVE_LM_WEIGHT: f64 = 0.615223527;
/// Decoder field 4.6.1.7 as a maximised-score bonus.
pub const NATIVE_INSERTION_BONUS: f64 = 1.901311993598938;

impl Default for DecoderSettings {
    fn default() -> Self {
        DecoderSettings {
            beam_width: 1000,
            lm_weight: NATIVE_LM_WEIGHT,
            insertion_bonus: NATIVE_INSERTION_BONUS,
            char_class_weight: NATIVE_LM_WEIGHT,
            acoustic_scale: 1.0,
        }
    }
}

impl DecoderSettings {
    /// The configuration that measures 72/74 on our corpus instead of 68/74,
    /// with the char-class sign the disassembly proves is backwards. Kept
    /// because the gap is real and unexplained (SPEC.md section 19).
    pub fn empirical() -> Self {
        DecoderSettings {
            lm_weight: 1.0,
            char_class_weight: -1.0,
            ..Default::default()
        }
    }
}
