//! Resolve a BCP-47 tag to the model packs that serve it.
//!
//! `packmapping.pb` is the SDK's own tag -> (recospec, tflite, fst) table.
//! Fetching and unzipping the packs is a host concern and lives in the CLI;
//! resolution is pure logic and lives here so every target shares it.

use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::error::{Error, Result};
use crate::proto::for_each_field;

/// The three pack names serving one language tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackNames {
    pub recospec: String,
    pub tflite: String,
    /// Absent for gesture models, and for the CTC configs that ship no word LM.
    pub fst: Option<String>,
}

/// The parsed `packmapping.pb` catalog.
#[derive(Debug, Clone, Default)]
pub struct PackMapping {
    entries: Vec<(String, PackNames)>,
}

impl PackMapping {
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut mapping = Self::default();
        for_each_field(data, |field, value| {
            if field != 1 {
                return Ok(());
            }
            let mut tag = String::new();
            let mut names = PackNames {
                recospec: String::new(),
                tflite: String::new(),
                fst: None,
            };
            for_each_field(value.as_bytes()?, |field, value| {
                match field {
                    1 => tag = String::from(value.as_str()?),
                    5 => names.recospec = String::from(value.as_str()?),
                    6 => names.tflite = String::from(value.as_str()?),
                    7 => names.fst = Some(String::from(value.as_str()?)),
                    _ => {}
                }
                Ok(())
            })?;
            ensure!(
                !tag.is_empty() && !names.recospec.is_empty() && !names.tflite.is_empty(),
                Format,
                "incomplete pack mapping entry"
            );
            ensure!(
                !mapping.entries.iter().any(|(existing, _)| existing == &tag),
                Format,
                "duplicate language tag {tag:?}"
            );
            mapping.entries.push((tag, names));
            Ok(())
        })?;
        Ok(mapping)
    }

    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(tag, _)| tag.as_str())
    }

    /// Exact match wins. Otherwise keep the language and private-use variant,
    /// prefer a matching script and region, then the least specific entry.
    /// An unknown language never silently becomes English.
    pub fn resolve(&self, tag: &str) -> Result<&PackNames> {
        let normalized = tag.trim().replace('_', "-").to_lowercase();
        // Python's lowercase-tag dictionary retains the last case-only alias.
        if let Some((_, names)) = self
            .entries
            .iter()
            .rev()
            .find(|(candidate, _)| candidate.to_lowercase() == normalized)
        {
            return Ok(names);
        }
        let (language, script, region, private) = parts(&normalized);
        let preferred_region = region.as_deref().or(match script.as_deref() {
            Some("hans") => Some("cn"),
            Some("hant") => Some("tw"),
            _ => None,
        });
        self.entries
            .iter()
            .filter_map(|(candidate, names)| {
                let (lang, scr, reg, variant) = parts(candidate);
                if lang != language || variant != private {
                    return None;
                }
                if let (Some(script), Some(scr)) = (script.as_deref(), scr.as_deref())
                    && script != scr
                    && !(matches!(script, "hans" | "hant") && scr == "hani")
                {
                    return None;
                }
                let score = (
                    preferred_region.is_some() && reg.as_deref() == preferred_region,
                    script.is_some() && scr == script,
                    Reverse(candidate.split('-').count()),
                    candidate.as_str(),
                );
                Some((score, names))
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, names)| names)
            .ok_or_else(|| Error::NoSuchLanguage(String::from(tag)))
    }
}

/// Split the same subtags as the Python resolver, without interpreting variants
/// or extensions other than private use. Normalization here deliberately does
/// not trim: only the requested tag is stripped by `resolve`.
fn parts(tag: &str) -> (String, Option<String>, Option<String>, String) {
    let normalized = tag.replace('_', "-").to_lowercase();
    let subtags: Vec<_> = normalized.split('-').collect();
    let private = subtags
        .iter()
        .position(|part| *part == "x")
        .unwrap_or(subtags.len());
    let core = &subtags[..private];
    let script = core
        .iter()
        .skip(1)
        .find(|part| part.chars().count() == 4 && part.chars().all(char::is_alphabetic));
    let region = core.iter().skip(1).find(|part| {
        (part.chars().count() == 2 && part.chars().all(char::is_alphabetic))
            || (part.chars().count() == 3 && part.chars().all(|c| c.is_ascii_digit()))
    });
    (
        String::from(core.first().copied().unwrap_or("")),
        script.map(|part| String::from(*part)),
        region.map(|part| String::from(*part)),
        subtags[private..].join("-"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_field(field: u8, bytes: &[u8]) -> Vec<u8> {
        let mut out = alloc::vec![(field << 3) | 2];
        let mut len = bytes.len();
        while len >= 128 {
            out.push((len as u8 & 0x7f) | 0x80);
            len >>= 7;
        }
        out.push(len as u8);
        out.extend_from_slice(bytes);
        out
    }

    fn entry(tag: &str, fst: Option<&str>) -> Vec<u8> {
        let mut out = bytes_field(1, tag.as_bytes());
        out.extend(bytes_field(5, tag.as_bytes()));
        out.extend(bytes_field(6, b"net"));
        if let Some(fst) = fst {
            out.extend(bytes_field(7, fst.as_bytes()));
        }
        out
    }

    fn catalog(tags: &[&str]) -> PackMapping {
        let data: Vec<_> = tags
            .iter()
            .flat_map(|tag| bytes_field(1, &entry(tag, None)))
            .collect();
        PackMapping::parse(&data).unwrap()
    }

    fn resolves(mapping: &PackMapping, requested: &str, expected: &str) {
        assert_eq!(mapping.resolve(requested).unwrap().recospec, expected);
    }

    #[test]
    fn parses_names_presence_unknown_fields_and_wire_order() {
        let mut first = entry("en-US", Some("words"));
        first.extend(bytes_field(9, b"ignored"));
        // Singular protobuf fields use their last occurrence.
        first.extend(bytes_field(6, b"last-net"));
        let mut data = bytes_field(1, &first);
        data.extend(bytes_field(2, b"ignored"));
        data.extend(bytes_field(1, &entry("und-x-gesture", None)));
        data.extend(bytes_field(1, &entry("fr", Some(""))));
        let mapping = PackMapping::parse(&data).unwrap();
        assert_eq!(
            mapping.tags().collect::<Vec<_>>(),
            ["en-US", "und-x-gesture", "fr"]
        );
        assert_eq!(
            mapping.resolve("en-US").unwrap(),
            &PackNames {
                recospec: String::from("en-US"),
                tflite: String::from("last-net"),
                fst: Some(String::from("words")),
            }
        );
        assert_eq!(mapping.resolve("und-x-gesture").unwrap().fst, None);
        assert_eq!(mapping.resolve("fr").unwrap().fst.as_deref(), Some(""));
        assert_eq!(PackMapping::parse(&[]).unwrap().tags().count(), 0);
    }

    #[test]
    fn rejects_incomplete_duplicate_and_malformed_entries() {
        let duplicate = bytes_field(1, &entry("en", None)).repeat(2);
        let mut missing_net = bytes_field(1, b"en");
        missing_net.extend(bytes_field(5, b"spec"));
        let mut empty_spec = entry("en", None);
        empty_spec.extend(bytes_field(5, b""));
        for data in [
            duplicate,
            bytes_field(1, &entry("", None)),
            bytes_field(1, &missing_net),
            bytes_field(1, &empty_spec),
            bytes_field(1, &[]),
            alloc::vec![0x0a, 0x02, 0x0a],
            bytes_field(1, &bytes_field(1, &[0xff])),
            alloc::vec![0x08, 0x01],
        ] {
            assert!(matches!(PackMapping::parse(&data), Err(Error::Format(_))));
        }
    }

    #[test]
    fn parts_normalize_and_take_first_script_and_region_before_private_use() {
        for (tag, language, script, region, private) in [
            (
                "SR_latn_RS_X_Gesture",
                "sr",
                Some("latn"),
                Some("rs"),
                "x-gesture",
            ),
            ("es-419", "es", None, Some("419"), ""),
            ("en-US-GB-Latn-Cyrl", "en", Some("latn"), Some("us"), ""),
            ("en-x-Latn-US", "en", None, None, "x-latn-us"),
            ("en-abc-1234-1a", "en", None, None, ""),
            ("en-x", "en", None, None, "x"),
            ("en", "en", None, None, ""),
            (" en ", " en ", None, None, ""),
            ("", "", None, None, ""),
        ] {
            let actual = parts(tag);
            assert_eq!(
                (
                    actual.0.as_str(),
                    actual.1.as_deref(),
                    actual.2.as_deref(),
                    actual.3.as_str()
                ),
                (language, script, region, private),
                "{tag}"
            );
        }
    }

    #[test]
    fn exact_matches_normalize_requests_and_last_case_alias_wins() {
        let mapping = catalog(&["en", "en-US", "EN-us", "en-GB"]);
        resolves(&mapping, "  eN_uS\n", "EN-us");
        resolves(&mapping, "en", "en");
    }

    #[test]
    fn private_use_never_crosses_variants_or_becomes_text() {
        let mapping = catalog(&["en", "en-x-gesture", "en-x-shapes", "und-x-emoji"]);
        resolves(&mapping, "EN_us_X_GESTURE", "en-x-gesture");
        for requested in ["en-x-emoji", "en-x", "und", "fr-x-gesture", "x-gesture", ""] {
            assert_eq!(
                mapping.resolve(requested),
                Err(Error::NoSuchLanguage(String::from(requested)))
            );
        }
    }

    #[test]
    fn conflicting_scripts_are_excluded_but_unspecified_scripts_are_allowed() {
        let mapping = catalog(&["sr-Cyrl", "sr-Latn", "sr"]);
        resolves(&mapping, "sr-Latn-BA", "sr-Latn");
        resolves(&mapping, "sr-Arab", "sr");
        let mapping = catalog(&["sr-Cyrl"]);
        assert!(matches!(
            mapping.resolve("sr-Latn"),
            Err(Error::NoSuchLanguage(_))
        ));
    }

    #[test]
    fn han_umbrella_uses_implicit_regions_and_explicit_regions_override_them() {
        let mapping = catalog(&["zh-Hani-CN", "zh-Hani-TW", "zh-Hani-HK"]);
        resolves(&mapping, "zh-Hans", "zh-Hani-CN");
        resolves(&mapping, "zh-Hant", "zh-Hani-TW");
        resolves(&mapping, "zh-Hans-HK", "zh-Hani-HK");
        resolves(&mapping, "zh-Hant-CN", "zh-Hani-CN");
        let competing = catalog(&["zh-Hans-SG", "zh-Hani-CN"]);
        resolves(&competing, "zh-Hans", "zh-Hani-CN");
        let directional = catalog(&["zh-Hans", "zh-Hant"]);
        assert!(matches!(
            directional.resolve("zh-Hani"),
            Err(Error::NoSuchLanguage(_))
        ));
    }

    #[test]
    fn score_orders_region_before_script_then_specificity() {
        let mapping = catalog(&["en-Latn-GB", "en-US", "en-Latn", "en"]);
        resolves(&mapping, "en-Latn-US", "en-US");
        resolves(&mapping, "en-Latn-AU", "en-Latn");
        resolves(&mapping, "en-AU", "en");
        let han = catalog(&["zh-Hans", "zh-Hani-CN"]);
        resolves(&han, "zh-Hans-SG", "zh-Hans");
    }

    #[test]
    fn score_ties_choose_lexicographically_largest_original_tag() {
        for tags in [["en-GB", "en-US"], ["en-US", "en-GB"]] {
            resolves(&catalog(&tags), "en-AU", "en-US");
        }
        // Original spelling, not normalized spelling, breaks the tie.
        resolves(&catalog(&["en-us", "en-zz", "en_ZZ"]), "en-AU", "en_ZZ");
        resolves(&catalog(&["en-US", "en-gb"]), "en-AU", "en-gb");
    }

    #[test]
    fn unknown_language_keeps_original_request_in_error() {
        let requested = " ZZ_unknown ";
        assert_eq!(
            catalog(&["en"]).resolve(requested),
            Err(Error::NoSuchLanguage(String::from(requested)))
        );
    }
}
