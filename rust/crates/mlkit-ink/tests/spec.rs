mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use mlkit_ink::Error;
use mlkit_ink::packs::{PackMapping, PackNames};
use mlkit_ink::proto::{Reader, Value as WireValue};
use mlkit_ink::recospec::RecoSpec;
use mlkit_ink::settings::{CurveSettings, PreprocessingStep, handwriting_pipeline};
use serde_json::{Value, json};

fn catalog() -> Option<PackMapping> {
    let Some(path) = common::model_path("packmapping.pb") else {
        eprintln!("skipping catalog tests: packmapping.pb is absent");
        return None;
    };
    Some(PackMapping::parse(&std::fs::read(path).unwrap()).unwrap())
}

fn names_json(names: &PackNames) -> Value {
    json!({"recospec": names.recospec, "tflite": names.tflite, "fst": names.fst})
}

#[test]
fn every_catalog_tag_and_fallback_matches_python() {
    let Some(mapping) = catalog() else { return };
    let Some(golden) = common::spec_goldens() else {
        return;
    };
    let resolution = golden["resolution"].as_object().unwrap();
    assert_eq!(resolution.len(), 725);
    let tags: BTreeSet<_> = mapping.tags().collect();
    assert_eq!(tags, resolution.keys().map(String::as_str).collect());
    for (tag, expected) in resolution {
        assert_eq!(
            names_json(mapping.resolve(tag).unwrap()),
            *expected,
            "{tag}"
        );
    }
    for (query, expected) in golden["fallback_resolution"].as_object().unwrap() {
        if let Some(tag) = expected.as_str() {
            // Aliases can have identical pack names. Pointer identity checks
            // that the exact catalog entry won, not merely an equivalent pack.
            assert!(
                std::ptr::eq(
                    mapping.resolve(query).unwrap(),
                    mapping.resolve(tag).unwrap()
                ),
                "{query}: expected resolved tag {tag}"
            );
        } else {
            assert!(expected["error"].is_string());
            assert!(
                matches!(mapping.resolve(query), Err(Error::NoSuchLanguage(_))),
                "{query}"
            );
        }
    }
}

#[test]
fn catalog_statistics_match_python() {
    let Some(mapping) = catalog() else { return };
    let mut languages = BTreeSet::new();
    let mut scripts = BTreeSet::new();
    let mut nets = BTreeSet::new();
    let mut text_nets = BTreeSet::new();
    let mut recospecs = BTreeSet::new();
    for tag in mapping.tags() {
        let lower = tag.to_lowercase();
        let core = lower.split("-x-").next().unwrap();
        languages.insert(core.split('-').next().unwrap().to_owned());
        if let Some(script) = core
            .split('-')
            .skip(1)
            .find(|p| p.len() == 4 && p.chars().all(char::is_alphabetic))
        {
            scripts.insert(script.to_owned());
        }
        let names = mapping.resolve(tag).unwrap();
        nets.insert(names.tflite.as_str());
        if names.fst.is_some() {
            text_nets.insert(names.tflite.as_str());
        }
        recospecs.insert(names.recospec.as_str());
    }
    let families: BTreeSet<_> = nets
        .iter()
        .map(|name| {
            let mut name = *name;
            for prefix in ["indy_lstm_", "scribe_", "lstm_"] {
                name = name.strip_prefix(prefix).unwrap_or(name);
            }
            name.split('_').next().unwrap()
        })
        .collect();
    let script_families = families
        .iter()
        .filter(|name| !["autodraw", "emoji", "shapes"].contains(*name))
        .count();
    assert_eq!(
        json!({
            "language_tags": mapping.tags().count(), "languages": languages.len(),
            "explicit_script_subtags": scripts.len(), "script_families": script_families,
            "net_families": families.len(), "nets": nets.len(), "text_nets": text_nets.len(),
            "recospecs": recospecs.len(),
        }),
        common::spec_goldens().expect("checked above")["counts"]
    );
}

fn recospec_paths() -> Vec<PathBuf> {
    fn visit(dir: &Path, paths: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                visit(&path, paths);
            } else if path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".recospec.local")
            {
                paths.push(path);
            }
        }
    }
    let Some(root) = common::model_path("models") else {
        eprintln!("skipping recospec tests: models/ is absent");
        return Vec::new();
    };
    let mut paths = Vec::new();
    visit(&root, &mut paths);
    paths.sort();
    paths
}

fn fields(data: &[u8]) -> Vec<(u32, WireValue<'_>)> {
    let mut reader = Reader::new(data);
    let mut out = Vec::new();
    while let Some(field) = reader.next_field().unwrap() {
        out.push(field);
    }
    out
}

fn child<'a>(data: &'a [u8], path: &[u32]) -> &'a [u8] {
    let mut data = data;
    for number in path {
        data = fields(data)
            .into_iter()
            .rev()
            .find(|(field, _)| field == number)
            .map(|(_, value)| value.as_bytes().unwrap())
            .unwrap_or(&[]);
    }
    data
}

fn scalars(data: &[u8]) -> Value {
    Value::Object(
        fields(data)
            .into_iter()
            .filter_map(|(field, value)| {
                let value = match value {
                    WireValue::Varint(n) => json!(n),
                    WireValue::Fixed32(n) => json!(f32::from_bits(n) as f64),
                    WireValue::Fixed64(n) => json!(f64::from_bits(n)),
                    WireValue::Bytes(_) => return None,
                };
                Some((field.to_string(), value))
            })
            .collect(),
    )
}

fn step_identity(step: &PreprocessingStep) -> (u32, &str) {
    match step {
        PreprocessingStep::NormalizeTime => (5, "normalize_time"),
        PreprocessingStep::HallucinateTime { .. } => (6, "hallucinate_time"),
        PreprocessingStep::NormalizeSize { .. } => (2, "normalize_size"),
        PreprocessingStep::NormalizeSizeWritingGuideFirstStroke { .. } => {
            (8, "normalize_size_writing_guide_first_stroke")
        }
        PreprocessingStep::AddPenUpStrokes => (11, "add_pen_up_strokes"),
        PreprocessingStep::Unsupported { field, name } => (*field, name),
    }
}

fn assert_step_parameters(step: &PreprocessingStep, parameters: &Value) {
    let amount = parameters["1"].as_f64().unwrap_or(0.0) as f32;
    let flag = parameters["2"].as_u64().unwrap_or(0) != 0;
    match step {
        PreprocessingStep::HallucinateTime { interval, force } => {
            assert_eq!(*interval, amount);
            assert_eq!(*force, flag);
        }
        PreprocessingStep::NormalizeSize {
            margin,
            first_point_origin,
        }
        | PreprocessingStep::NormalizeSizeWritingGuideFirstStroke {
            margin,
            first_point_origin,
        } => {
            assert_eq!(*margin, amount);
            assert_eq!(*first_point_origin, flag);
        }
        _ => {}
    }
}

fn curve_json(settings: CurveSettings) -> Value {
    json!({
        "1": settings.tol1, "2": settings.tol2, "3": settings.tol3,
        "4": settings.max_arc_ratio, "5": settings.split_cos_threshold,
        "6": settings.generate_second_order_features, "7": settings.use_angles_ratios,
        "9": settings.interpolate_time, "10": settings.normalize_outputs_to_zero_one,
    })
}

// serde_json without its optional float_roundtrip feature can move Python's
// shortest-round-trip double literals by one ULP. Keep exact assertions: quote
// numeric tokens before JSON parsing, then use Rust's correctly rounded f64
// parser. This needs no change to the fixed dev-dependency configuration.
fn exact_spec_goldens() -> Option<Value> {
    // git-crypt encrypted, like the model archives it is derived from: see
    // common::encrypted_fixture. Absent key means skip, not fail.
    common::spec_goldens()?;
    let text = std::fs::read_to_string(common::goldens_dir().join("spec_goldens.json")).ok()?;
    let marker = "__golden_number__";
    assert!(!text.contains(marker));
    let mut quoted = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            quoted.push(c);
            while let Some(c) = chars.next() {
                quoted.push(c);
                if c == '\\' {
                    quoted.push(chars.next().unwrap());
                } else if c == '"' {
                    break;
                }
            }
        } else if c == '-' || c.is_ascii_digit() {
            let mut token = String::from(c);
            while let Some(c) =
                chars.next_if(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
            {
                token.push(c);
            }
            if token.contains(['.', 'e', 'E']) {
                quoted.push_str(&format!("\"{marker}{token}\""));
            } else {
                quoted.push_str(&token);
            }
        } else {
            quoted.push(c);
        }
    }
    fn restore(value: &mut Value, marker: &str) {
        match value {
            Value::String(text) => {
                if let Some(number) = text.strip_prefix(marker) {
                    *value = json!(number.parse::<f64>().unwrap());
                }
            }
            Value::Array(items) => items.iter_mut().for_each(|v| restore(v, marker)),
            Value::Object(items) => items.values_mut().for_each(|v| restore(v, marker)),
            _ => {}
        }
    }
    let mut value: Value = serde_json::from_str(&quoted).unwrap();
    restore(&mut value, marker);
    Some(value)
}

#[test]
fn sampled_spec_facts_match_python() {
    let paths = recospec_paths();
    let mapping = catalog();
    let mut checked = 0;
    let Some(goldens) = exact_spec_goldens() else {
        return;
    };
    for expected in goldens["specs"].as_array().unwrap() {
        let tag = expected["tag"].as_str().unwrap();
        if expected.get("error").is_some() {
            // The frozen "emoji" sample is a resolver error, not a parsed
            // spec: the real classifier's tag is zxx-Zsye-x-emoji.
            if let Some(mapping) = &mapping {
                assert!(
                    matches!(mapping.resolve(tag), Err(Error::NoSuchLanguage(_))),
                    "{tag}"
                );
            }
            continue;
        }
        let filename = expected["recospec_file"].as_str().unwrap();
        let Some(path) = paths.iter().find(|p| p.file_name().unwrap() == filename) else {
            eprintln!("skipping {tag}: pack containing {filename} is absent");
            continue;
        };
        let data = std::fs::read(path).unwrap();
        let spec = RecoSpec::parse(&data).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(json!(spec.languages), expected["languages"], "{tag}");
        assert_eq!(json!(spec.charset.len()), expected["charset_size"], "{tag}");
        assert_eq!(
            json!(&spec.charset[..16]),
            expected["charset_head"],
            "{tag}"
        );
        assert_eq!(
            json!(&spec.charset[spec.charset.len() - 8..]),
            expected["charset_tail"],
            "{tag}"
        );
        assert_eq!(json!(spec.num_features), expected["num_features"], "{tag}");
        assert_eq!(
            json!(spec.has_fst_decoder),
            expected["has_fst_decoder"],
            "{tag}"
        );
        let mut expected_curve = curve_json(CurveSettings::default());
        for (field, value) in expected["curve_settings"].as_object().unwrap() {
            assert!(
                expected_curve.get(field).is_some(),
                "unrepresented curve field {field}"
            );
            expected_curve[field] = value.clone();
        }
        assert_eq!(curve_json(spec.curve_settings), expected_curve, "{tag}");
        let pipeline = expected["pipeline"].as_array().unwrap();
        assert_eq!(spec.pipeline.len(), pipeline.len(), "{tag}");
        let raw_steps: Vec<_> = fields(child(&data, &[158518157, 2, 12]))
            .into_iter()
            .filter(|(field, _)| *field == 1)
            .collect();
        assert_eq!(raw_steps.len(), pipeline.len());
        for ((step, expected), (_, raw)) in spec.pipeline.iter().zip(pipeline).zip(raw_steps) {
            let (field, name) = step_identity(step);
            assert_eq!(json!(field), expected["field"], "{tag}");
            assert_eq!(json!(name), expected["kind"], "{tag}");
            assert_step_parameters(step, &expected["parameters"]);
            // The fixed enum has no writing-guide parameter 3, nor settings
            // for unsupported steps. Pin those fixture fields at the wire
            // boundary too, without pretending the public API exposes them.
            assert_eq!(
                scalars(child(raw.as_bytes().unwrap(), &[field])),
                expected["parameters"],
                "{tag}"
            );
        }
        let decoder = &spec.decoder;
        assert_eq!(json!(decoder.beam_width), expected["beam_width"], "{tag}");
        let weights = &expected["lm_weights_by_field"];
        assert_eq!(json!(decoder.lm_weight), weights["7"], "{tag}");
        assert_eq!(json!(decoder.per_label_cost), weights["1.7"], "{tag}");
        assert_eq!(json!(decoder.initial_space_penalty), weights["10"], "{tag}");
        // Search field 5 is deliberately absent from DecoderConfig's fixed
        // contract. Verify the Chinese golden at its wire path, not by aliasing
        // it to a different score with a known meaning.
        let search = scalars(child(&data, &[158518157, 4, 6, 1]));
        assert_eq!(search["5"], weights["1.5"], "{tag}");
        let actual_weights: Vec<_> = decoder
            .char_class_weights
            .iter()
            .map(|(name, value)| json!({"name": name, "value": value}))
            .collect();
        assert_eq!(
            json!(actual_weights),
            expected["char_class_weights"],
            "{tag}"
        );
        let classes: BTreeMap<_, _> = decoder.char_classes().into_iter().collect();
        assert_eq!(
            json!(classes.len()),
            expected["char_class_line_count"],
            "{tag}"
        );
        let empty: Vec<_> = classes
            .iter()
            .filter(|(_, members)| members.is_empty())
            .map(|(name, _)| name)
            .collect();
        assert_eq!(json!(empty), expected["char_class_empty"], "{tag}");
        assert_eq!(
            json!(decoder.symbol_table.len()),
            expected["symbol_table_size"],
            "{tag}"
        );
        assert_ctc_mapping(&spec);
        checked += 1;
    }
    eprintln!("checked {checked} available parsed-spec goldens");
}

fn assert_ctc_mapping(spec: &RecoSpec) {
    let count = spec.charset.len();
    let mapping = spec.ctc_mapping(count + 1).unwrap();
    assert_eq!(mapping.is_some(), spec.has_fst_decoder);
    if let Some(mapping) = mapping {
        assert_eq!(mapping.blank_index, count);
        assert_eq!(mapping.net_to_fst.len(), count + 1);
        assert_eq!(mapping.net_symbols.len(), count + 1);
        assert_eq!(mapping.net_to_fst[count], 1);
        assert_eq!(mapping.net_symbols[count], "<reserved>");
        assert_eq!(
            mapping.symbol_table_verified,
            !spec.decoder.symbol_table.is_empty()
        );
        for (index, symbol) in spec.charset.iter().enumerate() {
            assert_eq!(mapping.net_to_fst[index], index as u32 + 2);
            assert_eq!(
                mapping.net_symbols[index],
                if symbol == " " { "[[space]]" } else { symbol }
            );
        }
        assert!(spec.ctc_mapping(count).is_err());
    }
}

#[test]
fn every_available_recospec_parses() {
    let paths = recospec_paths();
    if paths.is_empty() {
        eprintln!("skipping recospec sweep: no unpacked recospecs are present");
        return;
    }
    let mut ctc = 0;
    let mut classifiers = 0;
    for path in &paths {
        let data = std::fs::read(path).unwrap();
        let spec = RecoSpec::parse(&data).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            !spec.charset.is_empty(),
            "{}: empty charset",
            path.display()
        );
        assert!(
            !spec.pipeline.is_empty(),
            "{}: empty pipeline",
            path.display()
        );
        assert!(
            spec.charset.iter().all(|s| !s.is_empty()),
            "{}: empty symbol",
            path.display()
        );
        let processor = child(&data, &[158518157, 2]);
        let raw_charset: Vec<_> = fields(processor)
            .into_iter()
            .filter(|(field, _)| *field == 1)
            .map(|(_, value)| value.as_str().unwrap())
            .collect();
        assert_eq!(spec.charset, raw_charset, "{}", path.display());
        let raw_pipeline: Vec<_> = fields(child(processor, &[12]))
            .into_iter()
            .filter(|(field, _)| *field == 1)
            .collect();
        assert_eq!(
            spec.pipeline.len(),
            raw_pipeline.len(),
            "{}",
            path.display()
        );
        for (step, (_, raw)) in spec.pipeline.iter().zip(raw_pipeline) {
            let branch = fields(raw.as_bytes().unwrap());
            let (field, settings) = branch.last().unwrap();
            assert_eq!(step_identity(step).0, *field, "{}", path.display());
            assert_step_parameters(step, &scalars(settings.as_bytes().unwrap()));
        }
        if spec.has_fst_decoder {
            assert_eq!(spec.num_features, Some(10), "{}", path.display());
            assert_eq!(
                spec.curve_settings,
                CurveSettings::handwriting(),
                "{}",
                path.display()
            );
            assert_eq!(spec.pipeline, handwriting_pipeline(), "{}", path.display());
            ctc += 1;
        } else {
            // Autodraw/emoji use RawSettings and omit this field; their
            // five-column input width lives in the network, not the recospec.
            assert!(
                matches!(spec.num_features, None | Some(10)),
                "{}",
                path.display()
            );
            classifiers += 1;
        }
        assert_ctc_mapping(&spec);
    }
    eprintln!(
        "parsed {} recospecs: {ctc} CTC, {classifiers} classifiers",
        paths.len()
    );
}
