mod common;

use mlkit_ink::decoder::{
    Alphabet, CharClassRescoringLm, CharClassScorer, LanguageModel, greedy_decode,
    prefix_beam_search,
};
use mlkit_ink::fst::{BackoffLm, CompactFst};
use mlkit_ink::mat::Mat;
use mlkit_ink::settings::DecoderSettings;
use serde_json::Value;
use std::sync::OnceLock;

fn decoder_goldens() -> Option<&'static Value> {
    static CACHE: OnceLock<Option<Value>> = OnceLock::new();
    CACHE
        .get_or_init(|| common::encrypted_fixture("decoder_goldens.json"))
        .as_ref()
}

fn alphabet() -> Option<Alphabet> {
    let a = &decoder_goldens()?["alphabet"];
    Some(Alphabet {
        blank: a["blank"].as_u64().unwrap() as usize,
        labels: a["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect(),
        texts: a["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect(),
    })
}

fn scorer(a: &Alphabet) -> CharClassScorer {
    let d = decoder_goldens().expect("callers check first");
    let classes: Vec<_> = d["char_class_table"]
        .as_str()
        .unwrap()
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (name, members) = line.split_once(' ').unwrap_or((line, ""));
            (name.to_owned(), members.to_owned())
        })
        .collect();
    let weights: Vec<_> = d["char_class_weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            (
                w["name"].as_str().unwrap().to_owned(),
                w["value"].as_f64().unwrap(),
            )
        })
        .collect();
    CharClassScorer::from_tables(&classes, &weights, a).unwrap()
}

fn model_bytes() -> Option<&'static [u8]> {
    static CACHE: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let Some(path) = common::model_path(common::EN_US_FST) else {
            eprintln!("skipping FST-dependent decode tests: {} is absent (model packs are fetched on demand)", common::EN_US_FST);
            return None;
        };
        Some(std::fs::read(path).expect("English FST"))
    }).as_deref()
}

fn logits(trial: &Value) -> Mat {
    let shape = trial["logits_shape"].as_array().unwrap();
    Mat::from_vec(
        shape[0].as_u64().unwrap() as usize,
        shape[1].as_u64().unwrap() as usize,
        common::golden_logits(trial["id"].as_str().unwrap()),
    )
}

#[test]
fn character_class_goldens() {
    let (Some(alphabet), Some(d)) = (alphabet(), decoder_goldens()) else {
        return;
    };
    let scorer = scorer(&alphabet);
    for (label, value) in d["char_class_scores"].as_object().unwrap() {
        assert_eq!(
            scorer.score_for_label(label.parse().unwrap()),
            value.as_f64().unwrap(),
            "label {label}"
        );
    }
    for (actual, key) in [
        (&scorer.empty_classes, "char_class_empty"),
        (&scorer.unmapped_weights, "char_class_unmapped_weights"),
    ] {
        let expected: Vec<_> = d[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert_eq!(actual, &expected, "{key}");
    }
}

#[test]
fn fst_metadata_and_stepwise_goldens() {
    let Some(bytes) = model_bytes() else {
        return;
    };
    let fst = CompactFst::parse(bytes).unwrap();
    let Some(d) = decoder_goldens() else { return };
    let expected = &d["fst"];
    assert_eq!(
        u64::from(fst.num_states()),
        expected["num_states"].as_u64().unwrap()
    );
    assert_eq!(fst.num_arcs(), expected["num_arcs"].as_u64().unwrap());
    assert_eq!(
        u64::from(fst.num_labels()),
        expected["num_labels"].as_u64().unwrap()
    );
    assert_eq!(
        f64::from(fst.quantum()),
        expected["quantum"].as_f64().unwrap()
    );
    assert_eq!(
        u64::from(fst.start()),
        expected["start_state"].as_u64().unwrap()
    );
    assert_eq!(
        fst.use_final_weights(),
        expected["use_final_weights"].as_bool().unwrap()
    );
    let lm = BackoffLm::new(fst);
    for sequence in expected["sequences"].as_array().unwrap() {
        let mut state = lm.start();
        let mut total = 0.0;
        for step in sequence["steps"].as_array().unwrap() {
            let label = step["label"].as_u64().unwrap() as u32;
            let (next, cost) = lm.advance(state, label).unwrap();
            assert_eq!(
                u64::from(next),
                step["state"].as_u64().unwrap(),
                "{} label {label}",
                sequence["text"]
            );
            assert_eq!(
                cost,
                step["cost"].as_f64().unwrap(),
                "{} label {label}",
                sequence["text"]
            );
            state = next;
            total += cost;
        }
        let final_cost = lm.finish(state);
        assert_eq!(
            final_cost,
            sequence["final"].as_f64().unwrap(),
            "{} final",
            sequence["text"]
        );
        assert_eq!(
            total + final_cost,
            sequence["total"].as_f64().unwrap(),
            "{} total",
            sequence["text"]
        );
    }
}

#[test]
fn greedy_all_74_goldens() {
    let Some(a) = alphabet() else { return };
    let trials = common::trials();
    assert_eq!(trials.len(), 74);
    let mut matched = 0;
    let mut worst = 0.0_f64;
    for trial in trials {
        let candidate = greedy_decode(&logits(trial), &a).unwrap();
        assert_eq!(
            candidate.text,
            trial["greedy"]["text"].as_str().unwrap(),
            "{}",
            trial["id"]
        );
        let delta = (candidate.score - trial["greedy"]["score"].as_f64().unwrap()).abs();
        worst = worst.max(delta);
        assert!(
            delta <= 1e-6,
            "{} greedy score deviation {delta}",
            trial["id"]
        );
        matched += 1;
    }
    eprintln!("greedy: {matched}/74 exact texts; worst score deviation {worst:.17e}");
}

fn beam_goldens(settings: DecoderSettings, key: &str) {
    let Some(bytes) = model_bytes() else {
        return;
    };
    let word_lm = BackoffLm::new(CompactFst::parse(bytes).unwrap());
    let Some(a) = alphabet() else { return };
    let scorer = scorer(&a);
    let lm = CharClassRescoringLm::new(&word_lm, &scorer, &settings).unwrap();
    // Exactly as in Python: the wrapper applies both independent weights,
    // leaving the search's LM multiplier neutral rather than scaling twice.
    let search = DecoderSettings {
        lm_weight: 1.0,
        ..settings
    };
    let trials = common::trials();
    assert_eq!(trials.len(), 74);
    let mut matched = 0;
    let mut worst = 0.0_f64;
    for trial in trials {
        let candidates = prefix_beam_search(&logits(trial), &a, Some(&lm), &search, 5).unwrap();
        let expected = trial[key].as_array().unwrap();
        assert_eq!(candidates.len(), 5, "{} {key}", trial["id"]);
        assert_eq!(candidates.len(), expected.len(), "{} {key}", trial["id"]);
        for (rank, (actual, expected)) in candidates.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.text,
                expected["text"].as_str().unwrap(),
                "{} {key} rank {rank}",
                trial["id"]
            );
            let delta = (actual.score - expected["score"].as_f64().unwrap()).abs();
            worst = worst.max(delta);
            assert!(
                delta <= 1e-6,
                "{} {key} rank {rank} score deviation {delta}: got {}, expected {}",
                trial["id"],
                actual.score,
                expected["score"]
            );
        }
        matched += 1;
    }
    eprintln!("{key}: {matched}/74 exact five-best lists; worst score deviation {worst:.17e}");
}

#[test]
fn native_beam_all_74_goldens() {
    beam_goldens(DecoderSettings::default(), "lm_native");
}

#[test]
fn empirical_beam_all_74_goldens() {
    beam_goldens(DecoderSettings::empirical(), "lm_empirical");
}
