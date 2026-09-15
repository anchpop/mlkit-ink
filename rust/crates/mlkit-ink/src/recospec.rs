//! The recospec protobuf: charset, preprocessing pipeline, decoder parameters.
//!
//! See `proto/recospec.proto` for the schema recovered from all 391 shipped
//! recospecs plus `libdigitalink.so`, and SPEC.md section 4 for how.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::Result;
use crate::proto::for_each_field;
use crate::settings::{CurveSettings, PreprocessingStep};

/// Decoder configuration, as far as it has been recovered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecoderConfig {
    /// Search field 1.
    pub beam_width: Option<usize>,
    /// Field 4.6.7: scales both the word-FST cost and the char-class term.
    pub lm_weight: Option<f64>,
    /// Field 4.6.1.7, as stored: a negative *cost* per emitted label.
    pub per_label_cost: Option<f64>,
    /// Field 4.6.10: synthetic initial-space penalty. Parsed, not applied — it
    /// only bites on continuation contexts, which single-shot recognition
    /// never supplies.
    pub initial_space_penalty: Option<f64>,
    /// `char_class_table`, one `name<space>members` line per class.
    pub char_class_table: Option<String>,
    /// Class name -> weight, in wire order.
    pub char_class_weights: Vec<(String, f64)>,
    /// FST symbol id -> symbol text, from the word LM's symbol table.
    pub symbol_table: BTreeMap<u32, String>,
}

impl DecoderConfig {
    /// Class name -> member characters, preserving explicitly empty classes:
    /// a bare line is how en-US makes `upper_`/`lower_` inert, so dropping it
    /// would silently change scoring.
    pub fn char_classes(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for line in text_lines(self.char_class_table.as_deref().unwrap_or("")) {
            let (name, members) = match line.find(' ') {
                Some(i) => (&line[..i], &line[i + 1..]),
                None => (line, ""),
            };
            out.push((String::from(name), String::from(members)));
        }
        out
    }
}

/// A parsed `research_handwriting.RecognizerSpec`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecoSpec {
    pub languages: Vec<String>,
    /// Network labels excluding the CTC blank (313 for Latin, not 314).
    pub charset: Vec<String>,
    pub num_features: Option<usize>,
    pub curve_settings: CurveSettings,
    pub pipeline: Vec<PreprocessingStep>,
    pub decoder: DecoderConfig,
    /// True when the spec carries an FST decoder, i.e. it is a CTC recognizer
    /// rather than a gesture/emoji classifier.
    pub has_fst_decoder: bool,
}

/// How network output indices map onto FST labels.
///
/// Blank is the LAST network output, and normal index `k` maps to FST label
/// `k + 2`. Native evidence: `NetworkScoreCache` ctor 0x4f1330, score
/// 0x50a224..261, diagnostic 0x4dfa4c..aae. (An early hypothesis had blank at
/// 0 and `k + 1`; both halves of it were wrong.)
#[derive(Debug, Clone, PartialEq)]
pub struct CtcMapping {
    /// FST label for each network index.
    pub net_to_fst: Vec<u32>,
    /// Symbol text for each network index; `[[space]]` stays verbatim here.
    pub net_symbols: Vec<String>,
    pub blank_index: usize,
    /// Whether the spec shipped a symbol table we could check the charset against.
    pub symbol_table_verified: bool,
}

impl RecoSpec {
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut spec = Self::default();
        let (mut has_tf, mut has_processor, mut has_decoder) = (false, false, false);
        for_each_field(data, |field, value| {
            match field {
                2 => spec.languages.push(String::from(value.as_str()?)),
                158518157 => {
                    has_tf = true;
                    for_each_field(value.as_bytes()?, |field, value| {
                        match field {
                            2 => {
                                has_processor = true;
                                spec.parse_processor(value.as_bytes()?)?;
                            }
                            4 => {
                                has_decoder = true;
                                for_each_field(value.as_bytes()?, |field, value| {
                                    if field == 6 {
                                        spec.has_fst_decoder = true;
                                        spec.decoder.parse_fst(value.as_bytes()?)?;
                                    }
                                    Ok(())
                                })?;
                            }
                            _ => {}
                        }
                        Ok(())
                    })?;
                }
                _ => {}
            }
            Ok(())
        })?;
        ensure!(
            has_tf,
            Format,
            "RecognizerSpec has no TfRecognizerSpec extension"
        );
        ensure!(
            has_processor && has_decoder,
            Format,
            "TfRecognizerSpec requires processor and decoder configs"
        );
        Ok(spec)
    }

    fn parse_processor(&mut self, data: &[u8]) -> Result<()> {
        for_each_field(data, |field, value| {
            match field {
                1 => self.charset.push(String::from(value.as_str()?)),
                6 => parse_curve_settings(value.as_bytes()?, &mut self.curve_settings)?,
                7 => {
                    let n = value.as_u64()?;
                    ensure!(n <= i32::MAX as u64, Format, "invalid num_features {n}");
                    self.num_features = Some(n as usize);
                }
                12 => for_each_field(value.as_bytes()?, |field, value| {
                    if field == 1 {
                        self.pipeline.push(parse_step(value.as_bytes()?)?);
                    }
                    Ok(())
                })?,
                _ => {}
            }
            Ok(())
        })
    }

    /// Build the CTC mapping, validating the net's output dimension against
    /// the charset. Returns `None` for non-CTC gesture classifiers.
    pub fn ctc_mapping(&self, num_classes: usize) -> Result<Option<CtcMapping>> {
        if !self.has_fst_decoder {
            return Ok(None);
        }
        let count = self.charset.len();
        ensure!(
            num_classes == count + 1,
            Invalid,
            "expected {} CTC outputs for {count} characters, got {num_classes}",
            count + 1
        );
        let count = u32::try_from(count)
            .ok()
            .filter(|n| *n <= u32::MAX - 3)
            .ok_or_else(|| err!(Format, "charset is too large for FST labels"))?;
        let mut expected = BTreeMap::from([
            (0, String::from("<epsilon>")),
            (1, String::from("<reserved>")),
            (count + 2, String::from("<S>")),
            (count + 3, String::from("</S>")),
        ]);
        expected.extend(self.charset.iter().enumerate().map(|(index, symbol)| {
            (
                index as u32 + 2,
                String::from(if symbol == " " { "[[space]]" } else { symbol }),
            )
        }));
        let symbol_table_verified = !self.decoder.symbol_table.is_empty();
        ensure!(
            !symbol_table_verified || self.decoder.symbol_table == expected,
            Format,
            "charset does not match the expected contiguous FST symbol table"
        );
        // Reserved label 1 is the blank, not the first character. Epsilon and
        // sentence boundaries are FST-only and never become network outputs.
        let mut net_to_fst: Vec<_> = (2..count + 2).collect();
        net_to_fst.push(1);
        let net_symbols = net_to_fst.iter().map(|id| expected[id].clone()).collect();
        Ok(Some(CtcMapping {
            net_to_fst,
            net_symbols,
            blank_index: num_classes - 1,
            symbol_table_verified,
        }))
    }
}

fn parse_curve_settings(data: &[u8], settings: &mut CurveSettings) -> Result<()> {
    for_each_field(data, |field, value| {
        match field {
            1 => settings.tol1 = value.as_f64()?,
            2 => settings.tol2 = value.as_f64()?,
            3 => settings.tol3 = value.as_f64()?,
            4 => settings.max_arc_ratio = value.as_f64()?,
            5 => settings.split_cos_threshold = value.as_f64()?,
            6 => settings.generate_second_order_features = value.as_bool()?,
            7 => settings.use_angles_ratios = value.as_bool()?,
            9 => settings.interpolate_time = value.as_bool()?,
            10 => settings.normalize_outputs_to_zero_one = value.as_bool()?,
            _ => {}
        }
        Ok(())
    })
}

fn parse_step(data: &[u8]) -> Result<PreprocessingStep> {
    let mut step = PreprocessingStep::Unsupported {
        field: 0,
        name: String::from("UNKNOWN"),
    };
    for_each_field(data, |field, value| {
        let data = value.as_bytes()?;
        step = match field {
            5 => PreprocessingStep::NormalizeTime,
            11 => PreprocessingStep::AddPenUpStrokes,
            // Proto2 scalar defaults apply even though handwriting packs
            // explicitly set interval=20 and first_point_origin=true.
            2 | 6 | 8 => {
                // Repeated occurrences of the same message-valued oneof
                // branch merge; a different branch resets its defaults.
                let (mut amount, mut flag) = match (&step, field) {
                    (PreprocessingStep::HallucinateTime { interval, force }, 6) => {
                        (*interval, *force)
                    }
                    (
                        PreprocessingStep::NormalizeSize {
                            margin,
                            first_point_origin,
                        },
                        2,
                    )
                    | (
                        PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
                            margin,
                            first_point_origin,
                        },
                        8,
                    ) => (*margin, *first_point_origin),
                    _ => (0.0, false),
                };
                for_each_field(data, |field, value| {
                    match field {
                        1 => amount = value.as_f32()?,
                        2 => flag = value.as_bool()?,
                        _ => {}
                    }
                    Ok(())
                })?;
                match field {
                    2 => PreprocessingStep::NormalizeSize {
                        margin: amount,
                        first_point_origin: flag,
                    },
                    6 => PreprocessingStep::HallucinateTime {
                        interval: amount,
                        force: flag,
                    },
                    _ => PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
                        margin: amount,
                        first_point_origin: flag,
                    },
                }
            }
            _ => PreprocessingStep::Unsupported {
                field,
                name: String::from(step_name(field)),
            },
        };
        Ok(())
    })?;
    Ok(step)
}

fn step_name(field: u32) -> &'static str {
    // These names come from the native oneof registry, not guesses based on
    // settings values. In particular field 7 is NOT NormalizeSize (field 2).
    match field {
        1 => "resampling",
        2 => "normalize_size",
        3 => "sort_strokes",
        4 => "sanitize_time",
        5 => "normalize_time",
        6 => "hallucinate_time",
        7 => "ink_based_slope_correction",
        8 => "normalize_size_writing_guide_first_stroke",
        9 => "detect_and_rearrange_multi_line",
        10 => "remove_pressure",
        11 => "add_pen_up_strokes",
        12 => "time_ms_to_s",
        13 => "ramer_resampling",
        14 => "resampling_by_time",
        15 => "normalize_time_by_size",
        16 => "normalize_multiline_size",
        17 => "filter_strokes",
        18 => "remove_time",
        19 => "remove_guide",
        20 => "normalize_size_for_scribe",
        21 => "normalize_size_by_time",
        22 => "start_at_origin",
        23 => "resample_penup_strokes",
        24 => "remove_time_order",
        25 => "scale_shift_for_rendering",
        26 => "shift_ink_based_on_writing_guide",
        27 => "writing_guide_from_bounding_box",
        28 => "set_writing_guide",
        29 => "randomize_ink_position_within_writing_guide",
        30 => "normalize_size_first_stroke",
        _ => "UNKNOWN",
    }
}

impl DecoderConfig {
    fn parse_fst(&mut self, data: &[u8]) -> Result<()> {
        for_each_field(data, |field, value| {
            match field {
                1 => for_each_field(value.as_bytes()?, |field, value| {
                    match field {
                        1 => {
                            self.beam_width = Some(
                                usize::try_from(value.as_u64()?)
                                    .map_err(|_| err!(Format, "beam width does not fit usize"))?,
                            )
                        }
                        7 => self.per_label_cost = Some(value.as_f64()?),
                        _ => {}
                    }
                    Ok(())
                })?,
                3 => for_each_field(value.as_bytes()?, |field, value| {
                    if field == 4 {
                        self.symbol_table = parse_symbol_table(value.as_str()?)?;
                    }
                    Ok(())
                })?,
                7 => self.lm_weight = Some(value.as_f64()?),
                9 => self.parse_char_classes(value.as_bytes()?)?,
                10 => self.initial_space_penalty = Some(value.as_f64()?),
                _ => {}
            }
            Ok(())
        })
    }

    fn parse_char_classes(&mut self, data: &[u8]) -> Result<()> {
        for_each_field(data, |field, value| {
            match field {
                2 => for_each_field(value.as_bytes()?, |field, value| {
                    if field == 1 {
                        let (mut name, mut weight) = (String::new(), 0.0);
                        for_each_field(value.as_bytes()?, |field, value| {
                            match field {
                                1 => name = String::from(value.as_str()?),
                                2 => weight = value.as_f64()?,
                                _ => {}
                            }
                            Ok(())
                        })?;
                        self.char_class_weights.push((name, weight));
                    }
                    Ok(())
                })?,
                4 => self.char_class_table = Some(String::from(value.as_str()?)),
                _ => {}
            }
            Ok(())
        })
    }
}

fn parse_symbol_table(text: &str) -> Result<BTreeMap<u32, String>> {
    let mut symbols = BTreeMap::new();
    for line in text_lines(text) {
        let (symbol, id) = line
            .rsplit_once('\t')
            .ok_or_else(|| err!(Format, "FST symbol line has no tab: {line:?}"))?;
        let id = id
            .trim()
            .parse::<u32>()
            .map_err(|_| err!(Format, "invalid FST symbol id {id:?}"))?;
        ensure!(
            symbols.insert(id, String::from(symbol)).is_none(),
            Format,
            "duplicate FST symbol id {id}"
        );
    }
    Ok(symbols)
}

/// Python's splitlines also recognizes bare CR and Unicode line boundaries.
/// Keep empty interior lines and bare class names, but not a trailing line.
fn text_lines(mut text: &str) -> impl Iterator<Item = &str> {
    core::iter::from_fn(move || {
        if text.is_empty() {
            return None;
        }
        if let Some((index, separator)) = text.char_indices().find(|(_, c)| {
            matches!(
                c,
                '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'
                    ..='\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
            )
        }) {
            let line = &text[..index];
            text = &text[index + separator.len_utf8()..];
            if separator == '\r' {
                text = text.strip_prefix('\n').unwrap_or(text);
            }
            Some(line)
        } else {
            let line = text;
            text = "";
            Some(line)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn varint(mut n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        while n >= 128 {
            out.push(n as u8 | 128);
            n >>= 7;
        }
        out.push(n as u8);
        out
    }

    fn message(field: u32, body: &[u8]) -> Vec<u8> {
        let mut out = varint((u64::from(field) << 3) | 2);
        out.extend(varint(body.len() as u64));
        out.extend(body);
        out
    }

    fn spec_with_processor(body: &[u8]) -> Vec<u8> {
        let mut tf = message(2, body);
        tf.extend(message(4, &[]));
        message(158518157, &tf)
    }

    #[test]
    fn requires_extension_processor_and_decoder() {
        assert!(RecoSpec::parse(&[]).is_err());
        assert!(RecoSpec::parse(&message(158518157, &[])).is_err());
        assert!(RecoSpec::parse(&message(158518157, &message(2, &[]))).is_err());
        assert!(RecoSpec::parse(&message(158518157, &message(4, &[]))).is_err());
        let spec = RecoSpec::parse(&spec_with_processor(&[])).unwrap();
        assert_eq!(spec.curve_settings, CurveSettings::default());
        assert_eq!(spec.num_features, None);
        assert!(!spec.has_fst_decoder);
        assert_eq!(spec.ctc_mapping(123).unwrap(), None);
    }

    #[test]
    fn curve_settings_use_double_precision_and_proto_defaults() {
        let mut curve = vec![0x21]; // field 4, fixed64
        curve.extend(3.1234567890123f64.to_le_bytes());
        curve.extend([0x30, 1, 0x38, 1, 0x48, 1, 0x50, 1]);
        let spec = RecoSpec::parse(&spec_with_processor(&message(6, &curve))).unwrap();
        assert_eq!(
            spec.curve_settings,
            CurveSettings {
                max_arc_ratio: 3.1234567890123,
                generate_second_order_features: true,
                use_angles_ratios: true,
                interpolate_time: true,
                normalize_outputs_to_zero_one: true,
                ..CurveSettings::default()
            }
        );
    }

    #[test]
    fn pipeline_preserves_named_and_future_unsupported_steps() {
        let mut pipeline = Vec::new();
        for field in [5, 6, 2, 8, 7, 17, 31, 11] {
            pipeline.extend(message(1, &message(field, &[])));
        }
        let spec = RecoSpec::parse(&spec_with_processor(&message(12, &pipeline))).unwrap();
        assert_eq!(
            spec.pipeline,
            vec![
                PreprocessingStep::NormalizeTime,
                PreprocessingStep::HallucinateTime {
                    interval: 0.0,
                    force: false
                },
                PreprocessingStep::NormalizeSize {
                    margin: 0.0,
                    first_point_origin: false
                },
                PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
                    margin: 0.0,
                    first_point_origin: false
                },
                PreprocessingStep::Unsupported {
                    field: 7,
                    name: String::from("ink_based_slope_correction")
                },
                PreprocessingStep::Unsupported {
                    field: 17,
                    name: String::from("filter_strokes")
                },
                PreprocessingStep::Unsupported {
                    field: 31,
                    name: String::from("UNKNOWN")
                },
                PreprocessingStep::AddPenUpStrokes,
            ]
        );
        assert_eq!(
            parse_step(&[]).unwrap(),
            PreprocessingStep::Unsupported {
                field: 0,
                name: String::from("UNKNOWN"),
            }
        );
    }

    #[test]
    fn step_parameters_and_last_oneof_branch() {
        let mut params = vec![0x0d];
        params.extend(12.5f32.to_le_bytes());
        params.extend([0x10, 1]);
        assert_eq!(
            parse_step(&message(6, &params)).unwrap(),
            PreprocessingStep::HallucinateTime {
                interval: 12.5,
                force: true
            }
        );
        assert_eq!(
            parse_step(&message(2, &params)).unwrap(),
            PreprocessingStep::NormalizeSize {
                margin: 12.5,
                first_point_origin: true
            }
        );
        let mut branches = message(6, &params);
        branches.extend(message(6, &[0x10, 0]));
        assert_eq!(
            parse_step(&branches).unwrap(),
            PreprocessingStep::HallucinateTime {
                interval: 12.5,
                force: false
            }
        );
        branches.extend(message(2, &[]));
        assert_eq!(
            parse_step(&branches).unwrap(),
            PreprocessingStep::NormalizeSize {
                margin: 0.0,
                first_point_origin: false
            }
        );
        branches.extend(message(5, &[]));
        assert_eq!(
            parse_step(&branches).unwrap(),
            PreprocessingStep::NormalizeTime
        );
    }

    #[test]
    fn class_lines_preserve_bare_empty_and_spaced_members() {
        let config = DecoderConfig {
            char_class_table: Some(String::from(
                "upper_\nlower_\r\nletters abc def\rblank \n\n space\n",
            )),
            ..DecoderConfig::default()
        };
        assert_eq!(
            config.char_classes(),
            vec![
                (String::from("upper_"), String::new()),
                (String::from("lower_"), String::new()),
                (String::from("letters"), String::from("abc def")),
                (String::from("blank"), String::new()),
                (String::new(), String::new()),
                (String::new(), String::from("space")),
            ]
        );
        assert!(DecoderConfig::default().char_classes().is_empty());
        assert_eq!(
            text_lines("a\u{85}b\u{2028}c\u{2029}d\x0be\x0cf\x1cg\x1dh\x1e").collect::<Vec<_>>(),
            vec!["a", "b", "c", "d", "e", "f", "g", "h"]
        );
    }

    #[test]
    fn symbol_table_splits_at_last_tab_and_rejects_duplicate_ids() {
        let symbols = parse_symbol_table("a\tb\t2\r\n<reserved>\t1\n").unwrap();
        assert_eq!(symbols[&2], "a\tb");
        assert!(parse_symbol_table("a\t2\nb\t2\n").is_err());
        assert!(parse_symbol_table("a 2").is_err());
        assert!(parse_symbol_table("a\t-1").is_err());
        assert!(parse_symbol_table("a\tx").is_err());
    }

    #[test]
    fn ctc_blank_last_with_or_without_word_lm() {
        let mut spec = RecoSpec {
            charset: vec![String::from(" "), String::from("𠮟")],
            has_fst_decoder: true,
            ..RecoSpec::default()
        };
        let mapping = spec.ctc_mapping(3).unwrap().unwrap();
        assert_eq!(mapping.net_to_fst, vec![2, 3, 1]);
        assert_eq!(mapping.net_symbols, vec!["[[space]]", "𠮟", "<reserved>"]);
        assert_eq!(mapping.blank_index, 2);
        assert!(!mapping.symbol_table_verified);
        assert!(spec.ctc_mapping(2).is_err());
        spec.decoder.symbol_table = parse_symbol_table(
            "<epsilon>\t0\n<reserved>\t1\n[[space]]\t2\n𠮟\t3\n<S>\t4\n</S>\t5\n",
        )
        .unwrap();
        assert!(spec.ctc_mapping(3).unwrap().unwrap().symbol_table_verified);
        spec.decoder.symbol_table.insert(2, String::from(" "));
        assert!(spec.ctc_mapping(3).is_err());
        spec.has_fst_decoder = false;
        assert_eq!(spec.ctc_mapping(0).unwrap(), None);
    }

    #[test]
    fn duplicate_symbols_are_rejected_during_parse() {
        let word_lm = message(4, b"a\t2\nb\t2\n");
        let decoder = message(6, &message(3, &word_lm));
        let mut tf = message(2, &[]);
        tf.extend(message(4, &decoder));
        assert!(RecoSpec::parse(&message(158518157, &tf)).is_err());
    }
}
