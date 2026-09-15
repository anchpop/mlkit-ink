//! Shared fixture loading for the golden tests.
//!
//! The goldens are the Python reference implementation's output, frozen stage
//! by stage (`tools/dump_goldens.py`, `tools/dump_spec_goldens.py`). That
//! implementation is itself validated 74/74 against the real ML Kit SDK on
//! device, so matching it is the port's correctness criterion.
//!
//! Stage-by-stage matters: a features bug and a decoder bug produce the same
//! symptom at the end of the pipeline, so each stage is pinned separately.
//!
//! Two of the fixtures — `spec_goldens.json` and `decoder_goldens.json` —
//! reproduce Google's own tables verbatim (the pack catalog, the charset, the
//! character-class table), so they are git-crypt encrypted alongside the model
//! archives. Loaders for those return `None` on a clone without the key, and
//! their tests skip rather than fail. The rest are outputs of our own code and
//! stay plaintext.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mlkit_ink::ink::Stroke;

/// Repo root. Overridable so the tests can run from a copy of the workspace.
pub fn repo_root() -> PathBuf {
    match std::env::var_os("MLKIT_INK_ROOT") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap(),
    }
}

pub fn goldens_dir() -> PathBuf {
    repo_root().join("testdata/goldens")
}

/// Skip rather than fail when the model packs are absent: the goldens are
/// checked in, but the packs are large, encrypted, and fetched on demand.
pub fn model_path(relative: &str) -> Option<PathBuf> {
    let path = repo_root().join(relative);
    path.exists().then_some(path)
}

pub const EN_US_RECOSPEC: &str = "models/qrnn_en_us_reco_20200318_fst_20191208_recospec_zip/\
     qrnn.en_us.reco_20200318.fst_20191208.recospec.local";
pub const EN_US_TFLITE: &str =
    "models/indy_lstm_latin_6x216_tflite_20191208_zip/latin_indy_lstm_6x216_20191208.tflite";
pub const EN_US_FST: &str = "models/en_us_20191208_compact_fst_zip/en_us.compact.fst.local";

pub fn goldens() -> &'static serde_json::Value {
    static CACHE: OnceLock<serde_json::Value> = OnceLock::new();
    CACHE.get_or_init(|| {
        let text = std::fs::read_to_string(goldens_dir().join("goldens.json"))
            .expect("run tools/dump_goldens.py to regenerate the fixtures");
        serde_json::from_str(&text).expect("goldens.json is not valid JSON")
    })
}

pub fn spec_goldens() -> Option<&'static serde_json::Value> {
    static CACHE: OnceLock<Option<serde_json::Value>> = OnceLock::new();
    CACHE
        .get_or_init(|| encrypted_fixture("spec_goldens.json"))
        .as_ref()
}

/// Load a git-crypt encrypted fixture, or `None` on a clone without the key.
///
/// The check is for git-crypt's magic specifically, not "did parsing fail".
/// Treating every error as "no key" would turn a deleted or truncated fixture
/// into a silently skipped test, which is the one failure mode a golden test
/// exists to prevent.
pub fn encrypted_fixture(name: &str) -> Option<serde_json::Value> {
    const MAGIC: &[u8] = b"\x00GITCRYPT\x00";
    let path = goldens_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    if bytes.starts_with(MAGIC) {
        eprintln!("skipping: {name} is git-crypt encrypted; run `git-crypt unlock` to include it");
        return None;
    }
    Some(serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "{} is neither valid JSON nor git-crypt encrypted: {e}",
            path.display()
        )
    }))
}

pub fn trials() -> &'static [serde_json::Value] {
    goldens()["trials"]
        .as_array()
        .expect("goldens.trials")
        .as_slice()
}

/// Load one trial's ink in the oracle harness's JSON schema.
pub fn load_ink(relative: &str) -> Vec<Stroke> {
    let text = std::fs::read_to_string(repo_root().join(relative)).expect("trial ink");
    let value: serde_json::Value = serde_json::from_str(&text).expect("ink JSON");
    value["strokes"]
        .as_array()
        .expect("strokes")
        .iter()
        .map(|s| Stroke {
            x: numbers(&s["x"]),
            y: numbers(&s["y"]),
            t: s.get("t").map(numbers).unwrap_or_default(),
            pen_up: false,
        })
        .collect()
}

fn numbers(value: &serde_json::Value) -> Vec<f64> {
    match value.as_array() {
        Some(items) => items.iter().map(|v| v.as_f64().expect("number")).collect(),
        None => Vec::new(),
    }
}

/// The reference `[T, num_classes]` logits for one trial, as raw little-endian f32.
pub fn golden_logits(trial_id: &str) -> Vec<f32> {
    let path = goldens_dir().join("logits").join(format!("{trial_id}.f32"));
    let bytes = std::fs::read(path).expect("golden logits");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect()
}

/// Assert two float sequences agree, reporting the worst element rather than
/// the first: the size of the largest deviation is what says whether a
/// mismatch is a rounding difference or a real divergence.
pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let mut worst = (0usize, 0.0f32);
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let delta = (a - e).abs();
        if delta > worst.1 {
            worst = (i, delta);
        }
    }
    assert!(
        worst.1 <= tolerance,
        "{what}: worst difference {} at index {} (got {}, want {}), tolerance {tolerance}",
        worst.1,
        worst.0,
        actual[worst.0],
        expected[worst.0]
    );
}
