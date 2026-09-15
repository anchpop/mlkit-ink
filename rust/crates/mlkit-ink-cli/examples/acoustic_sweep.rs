//! Sweep `acoustic_scale` against the oracle corpus.
//!
//! Mirrors `tools/sweep_acoustic_scale.py`, which is the reference generator;
//! this one exists because it is roughly a hundred times faster, so a sweep
//! that takes an hour in Python takes seconds here. Running both and comparing
//! the overlapping scales also checks the Rust decoder's `acoustic_scale`
//! against the Python's independently of the frozen goldens.
//!
//! Decodes the cached golden logits, so only the decoder is re-evaluated.
//!
//!     cargo run --release -p mlkit-ink-cli --example acoustic_sweep -- [SCALE...]

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use mlkit_ink::mat::Mat;
use mlkit_ink::settings::DecoderSettings;

const SCALES: &[f64] = &[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

fn main() -> Result<()> {
    let root = PathBuf::from(
        std::env::var("MLKIT_INK_ROOT")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../..").to_string()),
    );
    let goldens: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("testdata/goldens/goldens.json"),
    )?)?;
    let trials = goldens["trials"].as_array().context("trials")?;

    let loaded = mlkit_ink_paths(&root)?;
    let recognizer = mlkit_ink::Recognizer::load(&loaded.0, &loaded.1, Some(&loaded.2))
        .map_err(|e| anyhow!("{e}"))?;

    let cases: Vec<_> = trials
        .iter()
        .map(|trial| {
            let id = trial["id"].as_str().unwrap();
            let truth = trial["oracle_top"].as_str().unwrap_or("");
            let shape = trial["logits_shape"].as_array().unwrap();
            let (rows, cols) = (
                shape[0].as_u64().unwrap() as usize,
                shape[1].as_u64().unwrap() as usize,
            );
            let bytes = std::fs::read(
                root.join("testdata/goldens/logits")
                    .join(format!("{id}.f32")),
            )
            .expect("golden logits");
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect();
            (truth.to_string(), Mat::from_vec(rows, cols, values))
        })
        .collect();
    let singles = cases.iter().filter(|(t, _)| t.chars().count() == 1).count();

    println!("scale  total  singles  words  ranking-misses  absent-from-nbest");
    let requested: Vec<f64> = std::env::args()
        .skip(1)
        .map(|a| a.parse().context("scales must be numbers"))
        .collect::<Result<_>>()?;
    let scales = if requested.is_empty() {
        SCALES.to_vec()
    } else {
        requested
    };

    for scale in scales {
        let settings = DecoderSettings {
            acoustic_scale: scale,
            ..DecoderSettings::default()
        };
        let (mut correct, mut single_hits, mut ranking, mut absent) = (0, 0, 0, 0);
        for (truth, logits) in &cases {
            let candidates = recognizer
                .decode_logits(logits, 3, &settings)
                .map_err(|e| anyhow!("{e}"))?;
            let top = candidates.first().map(|c| c.text.as_str()).unwrap_or("");
            if top == truth {
                correct += 1;
                if truth.chars().count() == 1 {
                    single_hits += 1;
                }
            } else if candidates.iter().any(|c| c.text == *truth) {
                // The truth was offered and ranked below something else.
                ranking += 1;
            } else {
                // The truth never appeared at all.
                absent += 1;
            }
        }
        println!(
            "{scale:<6} {correct:>2}/{}  {single_hits:>2}/{singles}   {:>2}/{}   {ranking:>3}            {absent:>3}",
            cases.len(),
            correct - single_hits,
            cases.len() - singles,
        );
    }
    Ok(())
}

fn mlkit_ink_paths(root: &std::path::Path) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let read = |relative: &str| -> Result<Vec<u8>> {
        std::fs::read(root.join(relative)).with_context(|| format!("reading {relative}"))
    };
    Ok((
        read(
            "models/qrnn_en_us_reco_20200318_fst_20191208_recospec_zip/\
             qrnn.en_us.reco_20200318.fst_20191208.recospec.local",
        )?,
        read(
            "models/indy_lstm_latin_6x216_tflite_20191208_zip/latin_indy_lstm_6x216_20191208.tflite",
        )?,
        read("models/en_us_20191208_compact_fst_zip/en_us.compact.fst.local")?,
    ))
}
