//! CTC decoding, optionally shallow-fused with a tropical-cost language model.
//!
//! Scores are maximised log-probabilities; LM weights are tropical *costs* and
//! are therefore subtracted. Getting that convention backwards is exactly the
//! bug SPEC.md section 17 documents, so the sign lives in one place here.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::ctc::log_add_exp;
use crate::error::Result;
use crate::mat::Mat;
use crate::settings::DecoderSettings;

/// A deterministic LM over FST labels. The implementation owns
/// epsilon/backoff/boundary handling; the decoder just asks for transitions.
pub trait LanguageModel {
    type State: Copy;

    fn start(&self) -> Self::State;
    /// `None` rejects the extension outright; `Some(_, cost)` with an infinite
    /// cost is also a rejection but keeps the state meaningful.
    fn advance(&self, state: Self::State, ilabel: u32) -> Option<(Self::State, f64)>;
    fn finish(&self, state: Self::State) -> f64;
}

/// The network's output alphabet, indexed by network output index.
#[derive(Debug, Clone, PartialEq)]
pub struct Alphabet {
    pub blank: usize,
    /// FST label per network index; the blank entry is unused.
    pub labels: Vec<u32>,
    /// Rendered text per network index; `[[space]]` is already a real space.
    pub texts: Vec<String>,
}

/// One hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub score: f64,
}

/// Per-FST-label class scores built from the recospec's explicit membership
/// table. There is no Unicode upper/lower heuristic: lookup is by whole symbol,
/// and a symbol that misses falls back to `no_char_class`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CharClassScorer {
    /// Score per FST label, densely indexed. Beam search hits this millions of
    /// times per decode, so it is a flat array rather than a map.
    pub scores: Vec<f64>,
    /// Classes whose line was bare, so they contributed no members.
    pub empty_classes: Vec<String>,
    /// Weights naming a class no line ever populated.
    pub unmapped_weights: Vec<String>,
}

impl CharClassScorer {
    /// 0.0 for any label outside the table, matching the native default.
    pub fn score_for_label(&self, label: u32) -> f64 {
        self.scores.get(label as usize).copied().unwrap_or(0.0)
    }

    /// Mirror the native parser: a bare line is skipped entirely; a suffixed
    /// name is truncated to the prefix *including* the underscore (`upper_be`
    /// -> `upper_`); overlapping membership is last-write-wins, not an error;
    /// a class with no weight scores 0.0.
    pub fn from_tables(
        classes: &[(String, String)],
        weights: &[(String, f64)],
        alphabet: &Alphabet,
    ) -> Result<Self> {
        ensure!(
            alphabet.labels.len() == alphabet.texts.len(),
            Invalid,
            "alphabet labels and texts have different lengths"
        );
        let weights: BTreeMap<_, _> = weights.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        ensure!(
            weights.values().all(|v| v.is_finite()),
            Invalid,
            "character-class weights must be finite"
        );
        let mut membership = BTreeMap::new();
        let mut contributing = BTreeSet::new();
        let mut empty_classes = Vec::new();
        for (name, characters) in classes {
            // Empty suffixed lines must not activate their prefix's weight.
            if characters.is_empty() {
                empty_classes.push(name.clone());
                continue;
            }
            let name = name.find('_').map_or(name.as_str(), |i| &name[..=i]);
            contributing.insert(name);
            for member in expand_class_members(characters) {
                membership.insert(member.to_string(), name);
            }
        }
        let len = alphabet
            .labels
            .iter()
            .copied()
            .max()
            .map_or(Ok(0), |label| {
                (label as usize)
                    .checked_add(1)
                    .ok_or_else(|| err!(Invalid, "FST label is too large"))
            })?;
        let mut scores = vec![0.0; len];
        for (&label, text) in alphabet.labels.iter().zip(&alphabet.texts) {
            let text = rendered(text);
            let class = membership.get(text).copied().unwrap_or("no_char_class");
            scores[label as usize] = weights.get(class).copied().unwrap_or(0.0);
        }
        let unmapped_weights = weights
            .keys()
            .filter(|&&name| name != "no_char_class" && !contributing.contains(name))
            .map(|&name| name.to_string())
            .collect();
        Ok(Self {
            scores,
            empty_classes,
            unmapped_weights,
        })
    }
}

// Invalid bracket expressions stay literal, including a valid expression nested
// inside one. Advancing one codepoint on failure matches the reference parser.
fn expand_class_members(text: &str) -> impl Iterator<Item = char> + '_ {
    let mut rest = text;
    let mut range = 0..0;
    core::iter::from_fn(move || {
        loop {
            if let Some(cp) = range.next() {
                // Rust strings cannot represent Python's surrogate codepoints. They
                // cannot match any UTF-8 symbol either, so omit them from membership.
                if let Some(ch) = char::from_u32(cp) {
                    return Some(ch);
                }
                continue;
            }
            if let Some(body) = rest.strip_prefix("[[")
                && let Some(close) = body.find("]]")
            {
                let body_text = &body[..close];
                let (lo, hi) = body_text.split_once('-').unwrap_or((body_text, ""));
                let start = parse_hex(lo);
                let stop = if hi.is_empty() { start } else { parse_hex(hi) };
                if let (Some(start), Some(stop)) = (start, stop)
                    && start <= stop
                    && stop <= 0x10ffff
                {
                    range = start..stop + 1;
                    rest = &body[close + 2..];
                    continue;
                }
            }
            let ch = rest.chars().next()?;
            rest = &rest[ch.len_utf8()..];
            return Some(ch);
        }
    })
}

fn parse_hex(text: &str) -> Option<u32> {
    let text = text.trim();
    let text = text.strip_prefix('+').unwrap_or(text);
    let prefixed = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X"));
    let text = prefixed.map_or(text, |text| text.strip_prefix('_').unwrap_or(text));
    let mut value = 0_u32;
    let mut digit = false;
    for ch in text.chars() {
        if ch == '_' && digit {
            digit = false;
        } else {
            value = value.checked_mul(16)?.checked_add(ch.to_digit(16)?)?;
            digit = true;
        }
    }
    digit.then_some(value)
}

/// Python's `CharClassRescoringLM`: combine the two independent multipliers
/// before feeding costs to beam search. Use `lm_weight: 1.0` in the search
/// settings with this wrapper; it already applies the requested word weight.
#[derive(Debug)]
pub struct CharClassRescoringLm<'a, L> {
    word_lm: &'a L,
    scorer: &'a CharClassScorer,
    word_weight: f64,
    class_weight: f64,
}

impl<'a, L: LanguageModel> CharClassRescoringLm<'a, L> {
    pub fn new(
        word_lm: &'a L,
        scorer: &'a CharClassScorer,
        settings: &DecoderSettings,
    ) -> Result<Self> {
        ensure!(
            settings.lm_weight.is_finite() && settings.char_class_weight.is_finite(),
            Invalid,
            "word and character-class weights must be finite"
        );
        Ok(Self {
            word_lm,
            scorer,
            word_weight: settings.lm_weight,
            class_weight: settings.char_class_weight,
        })
    }
}

impl<L: LanguageModel> LanguageModel for CharClassRescoringLm<'_, L> {
    type State = L::State;

    fn start(&self) -> Self::State {
        self.word_lm.start()
    }

    fn advance(&self, state: Self::State, ilabel: u32) -> Option<(Self::State, f64)> {
        let (state, cost) = self.word_lm.advance(state, ilabel)?;
        // Preserve rejection (and malformed costs) even when the multiplier is zero.
        let cost = if cost.is_finite() {
            self.word_weight * cost - self.class_weight * self.scorer.score_for_label(ilabel)
        } else {
            cost
        };
        Some((state, cost))
    }

    fn finish(&self, state: Self::State) -> f64 {
        let cost = self.word_lm.finish(state);
        if cost.is_finite() {
            self.word_weight * cost
        } else {
            cost
        }
    }
}

/// Normalize `[T, C]` logits into log probabilities, in f64.
pub fn log_softmax(logits: &Mat) -> Result<Vec<f64>> {
    ensure!(logits.cols() > 0, Invalid, "logits must have C > 0");
    for row in logits.iter_rows() {
        ensure!(
            row.iter().all(|&v| !v.is_nan() && v != f32::INFINITY),
            Invalid,
            "logits cannot contain NaN or +inf"
        );
        ensure!(
            row.iter().any(|v| v.is_finite()),
            Invalid,
            "each frame must have at least one finite logit"
        );
    }
    // Loss and decoding must normalize identical logits identically; only the
    // public decoder's input validation differs from the internal loss helper.
    Ok(crate::ctc::log_softmax(logits))
}

fn validate_alphabet(alphabet: &Alphabet, cols: usize) -> Result<()> {
    ensure!(
        alphabet.blank < cols,
        Invalid,
        "blank index is outside the network alphabet"
    );
    ensure!(
        alphabet.labels.len() == cols && alphabet.texts.len() == cols,
        Invalid,
        "alphabet width does not match logits"
    );
    ensure!(
        alphabet
            .texts
            .iter()
            .enumerate()
            .all(|(i, text)| i == alphabet.blank || !text.is_empty()),
        Invalid,
        "nonblank symbols must be nonempty strings"
    );
    Ok(())
}

fn rendered(text: &str) -> &str {
    if text == "[[space]]" { " " } else { text }
}

/// Best-path decode. The score is that path's log probability, not the sum over
/// every path producing the same text.
pub fn greedy_decode(logits: &Mat, alphabet: &Alphabet) -> Result<Candidate> {
    let probs = log_softmax(logits)?;
    validate_alphabet(alphabet, logits.cols())?;
    let mut text = String::new();
    let mut score = 0.0;
    let mut previous = None;
    for frame in probs.chunks_exact(logits.cols()) {
        let mut index = 0;
        for i in 1..frame.len() {
            if frame[i] > frame[index] {
                index = i;
            }
        }
        if index != alphabet.blank && previous != Some(index) {
            text.push_str(rendered(&alphabet.texts[index]));
        }
        previous = Some(index);
        score += frame[index];
    }
    Ok(Candidate { text, score })
}

fn validate_cost(cost: f64) -> Result<f64> {
    ensure!(
        !cost.is_nan() && cost != f64::NEG_INFINITY,
        Invalid,
        "LM costs must be finite or +inf (rejection)"
    );
    Ok(cost)
}

#[derive(Clone)]
struct Prefix<S> {
    tokens: Vec<usize>,
    state: Option<S>,
    offset: f64,
    p_blank: f64,
    p_nonblank: f64,
}

impl<S> Prefix<S> {
    fn acoustic(&self) -> f64 {
        log_add_exp(self.p_blank, self.p_nonblank)
    }
    fn score(&self) -> f64 {
        self.acoustic() + self.offset
    }
}

/// CTC prefix beam search with optional shallow fusion.
///
/// ```text
/// score = log(sum over retained paths of exp(acoustic_scale * path log prob))
///       - lm_weight * (incremental LM costs + final cost)
///       + insertion_bonus * emitted nonblank tokens
/// ```
///
/// `acoustic_scale` multiplies the per-frame log probabilities *before* the
/// alignment sum, which is where native applies it (`0x42ed88`). It is
/// tempting to fold it into the LM weights instead, since it only ever appears
/// as a ratio against them — but that is wrong here, because this search sums
/// over alignments: `a * logsumexp(x)` is not `logsumexp(a * x)`. The two agree
/// only when a single alignment dominates, so the shortcut silently changes
/// which transcript wins on exactly the ambiguous inputs that matter.
///
/// `char_class_weight` belongs to [`CharClassRescoringLm`], not this search:
/// an arbitrary `LanguageModel` need not have a character-class table.
pub fn prefix_beam_search<L: LanguageModel>(
    logits: &Mat,
    alphabet: &Alphabet,
    lm: Option<&L>,
    settings: &DecoderSettings,
    nbest: usize,
) -> Result<Vec<Candidate>> {
    ensure!(
        settings.beam_width > 0 && nbest > 0,
        Invalid,
        "beam_width and nbest must be positive"
    );
    ensure!(
        settings.lm_weight.is_finite() && settings.insertion_bonus.is_finite(),
        Invalid,
        "lm_weight and insertion_bonus must be finite"
    );
    ensure!(
        settings.acoustic_scale.is_finite() && settings.acoustic_scale > 0.0,
        Invalid,
        "acoustic_scale must be finite and positive, got {}",
        settings.acoustic_scale
    );
    let mut probs = log_softmax(logits)?;
    if settings.acoustic_scale != 1.0 {
        // Multiplying rather than renormalizing: the resulting rows are no
        // longer a distribution, which is fine and is what native does. Every
        // CTC path has exactly T frames, so the missing normalizer is a shared
        // constant that cancels between hypotheses.
        for value in &mut probs {
            *value *= settings.acoustic_scale;
        }
    }
    let probs = probs;
    validate_alphabet(alphabet, logits.cols())?;
    let mut beam = vec![Prefix {
        tokens: Vec::new(),
        state: lm.map(LanguageModel::start),
        offset: 0.0,
        p_blank: 0.0,
        p_nonblank: f64::NEG_INFINITY,
    }];
    for frame in probs.chunks_exact(logits.cols()) {
        // A tree is only the lookup index. Iteration and stable pruning must
        // follow dict insertion order, not lexicographic prefix order.
        let mut next: Vec<Prefix<L::State>> = Vec::new();
        let mut slots = BTreeMap::<Vec<usize>, usize>::new();
        let active: Vec<_> = frame
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, p)| i != alphabet.blank && p != f64::NEG_INFINITY)
            .collect();
        for hypothesis in &beam {
            let acoustic = hypothesis.acoustic();
            let blank_mass = acoustic + frame[alphabet.blank];
            let repeat_mass = hypothesis
                .tokens
                .last()
                .map_or(f64::NEG_INFINITY, |&i| hypothesis.p_nonblank + frame[i]);
            if blank_mass != f64::NEG_INFINITY || repeat_mass != f64::NEG_INFINITY {
                let slot = *slots.entry(hypothesis.tokens.clone()).or_insert_with(|| {
                    let slot = next.len();
                    next.push(Prefix {
                        p_blank: f64::NEG_INFINITY,
                        p_nonblank: f64::NEG_INFINITY,
                        ..hypothesis.clone()
                    });
                    slot
                });
                next[slot].p_blank = log_add_exp(next[slot].p_blank, blank_mass);
                next[slot].p_nonblank = log_add_exp(next[slot].p_nonblank, repeat_mass);
            }
            let mut extended = hypothesis.tokens.clone();
            extended.push(0);
            for &(index, emission) in &active {
                let source = if hypothesis.tokens.last() == Some(&index) {
                    hypothesis.p_blank
                } else {
                    acoustic
                };
                if source == f64::NEG_INFINITY {
                    continue;
                }
                *extended.last_mut().unwrap() = index;
                let slot = if let Some(&slot) = slots.get(&extended) {
                    slot
                } else {
                    let (state, cost) = if let Some(lm) = lm {
                        let Some((state, cost)) =
                            lm.advance(hypothesis.state.unwrap(), alphabet.labels[index])
                        else {
                            continue;
                        };
                        let cost = validate_cost(cost)?;
                        if cost == f64::INFINITY {
                            continue;
                        }
                        (Some(state), cost)
                    } else {
                        (hypothesis.state, 0.0)
                    };
                    let slot = next.len();
                    slots.insert(extended.clone(), slot);
                    next.push(Prefix {
                        tokens: extended.clone(),
                        state,
                        offset: hypothesis.offset - settings.lm_weight * cost
                            + settings.insertion_bonus,
                        p_blank: f64::NEG_INFINITY,
                        p_nonblank: f64::NEG_INFINITY,
                    });
                    slot
                };
                next[slot].p_nonblank = log_add_exp(next[slot].p_nonblank, source + emission);
            }
        }
        // Precompute scores: comparison sorting must not redo logaddexp.
        let mut ranked: Vec<_> = next.into_iter().map(|p| (p.score(), p)).collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        ranked.truncate(settings.beam_width);
        beam = ranked.into_iter().map(|(_, p)| p).collect();
        if beam.is_empty() {
            return Ok(Vec::new());
        }
    }
    let mut results = BTreeMap::<String, f64>::new();
    for hypothesis in beam {
        let cost = validate_cost(lm.map_or(0.0, |lm| lm.finish(hypothesis.state.unwrap())))?;
        if cost == f64::INFINITY {
            continue;
        }
        let mut text = String::new();
        for &index in &hypothesis.tokens {
            text.push_str(rendered(&alphabet.texts[index]));
        }
        let score = hypothesis.score() - settings.lm_weight * cost;
        let previous = results.entry(text).or_insert(f64::NEG_INFINITY);
        *previous = log_add_exp(*previous, score);
    }
    let mut results: Vec<_> = results
        .into_iter()
        .map(|(text, score)| Candidate { text, score })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap()
            .then_with(|| a.text.cmp(&b.text))
    });
    results.truncate(nbest);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alphabet(texts: &[&str]) -> Alphabet {
        Alphabet {
            blank: texts.len() - 1,
            labels: (2..texts.len() as u32 + 2).collect(),
            texts: texts.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn neutral() -> DecoderSettings {
        DecoderSettings {
            beam_width: 1000,
            lm_weight: 1.0,
            insertion_bonus: 0.0,
            char_class_weight: 0.0,
            acoustic_scale: 1.0,
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "got {a}, expected {b}");
    }

    #[test]
    fn softmax_is_f64_and_validates_frames() {
        let m = Mat::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1000.0, 1001.0, f32::NEG_INFINITY]);
        let p = log_softmax(&m).unwrap();
        for &v in &p[..3] {
            close(v, -3.0_f64.ln());
        }
        close(p[3], -1.0 - (-1.0_f64).exp().ln_1p());
        close(p[4], -(-1.0_f64).exp().ln_1p());
        assert_eq!(p[5], f64::NEG_INFINITY);
        assert!(log_softmax(&Mat::zeros(0, 3)).unwrap().is_empty());
        assert!(log_softmax(&Mat::zeros(0, 0)).is_err());
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(log_softmax(&Mat::from_vec(1, 1, vec![v])).is_err());
        }
    }

    #[test]
    fn class_ranges_and_malformed_literals() {
        for (input, expected) in [
            ("a[[0042]][[43-45]]é", "aBCDEé"),
            ("[[0x1f642-0x1f643]]", "\u{1f642}\u{1f643}"),
            ("[[41-]]", "A"),
            ("[[ +0X41 ]]", "A"),
            ("[[0x_4_1]]", "A"),
            ("[[++41]]", "[[++41]]"),
            ("[[4__1]]", "[[4__1]]"),
            ("[[41_]]", "[[41_]]"),
            ("[[D7FF-E000]]", "\u{d7ff}\u{e000}"),
            ("[[GG]]", "[[GG]]"),
            ("[[42-41]]", "[[42-41]]"),
            ("[[110000]]", "[[110000]]"),
            ("[[-1]]", "[[-1]]"),
            ("[[41", "[[41"),
            ("[[41-42-43]]", "[[41-42-43]]"),
            ("[[x[[41]]", "[[xA"),
        ] {
            assert_eq!(
                expand_class_members(input).collect::<String>(),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn classes_skip_bare_lines_truncate_suffixes_and_override() {
        let a = alphabet(&["A", "B", "C", "a", "é", "aA", "[[space]]", "\u{1f642}", ""]);
        let classes = [
            ("upper", "ABC"),
            ("upper_be", "B[[1f642]]"),
            ("lower_en_us", ""),
            ("unweighted", "C"),
            ("last", "A"),
        ]
        .map(|(name, chars)| (name.to_string(), chars.to_string()));
        let weights = [
            ("upper", 1.0),
            ("upper_", 2.0),
            ("lower_", 9.0),
            ("last", 3.0),
            ("no_char_class", -4.0),
        ]
        .map(|(name, value)| (name.to_string(), value));
        let s = CharClassScorer::from_tables(&classes, &weights, &a).unwrap();
        assert_eq!(
            &s.scores[2..],
            &[3.0, 2.0, 0.0, -4.0, -4.0, -4.0, -4.0, 2.0, -4.0]
        );
        assert_eq!(s.empty_classes, ["lower_en_us"]);
        assert_eq!(s.unmapped_weights, ["lower_"]);
        assert_eq!(s.score_for_label(100), 0.0);
        let empty = CharClassScorer::from_tables(&[], &[], &a).unwrap();
        assert!(empty.scores.iter().all(|&v| v == 0.0));
        assert!(CharClassScorer::from_tables(&[], &[("x".to_string(), f64::NAN)], &a).is_err());
    }

    struct TinyLm;
    impl LanguageModel for TinyLm {
        type State = u32;
        fn start(&self) -> u32 {
            0
        }
        fn advance(&self, state: u32, label: u32) -> Option<(u32, f64)> {
            match (state, label) {
                (0, 2) => Some((1, 0.5)),
                (1, 2) => Some((2, 0.25)),
                // +inf is rejection even at a zero LM multiplier.
                (_, 3) => Some((state, f64::INFINITY)),
                _ => None,
            }
        }
        fn finish(&self, state: u32) -> f64 {
            match state {
                1 => 0.75,
                2 => 1.0,
                _ => f64::INFINITY,
            }
        }
    }

    #[test]
    fn beam_matches_enumerated_ctc_paths_and_lm_costs() {
        let a = alphabet(&["a", "b", ""]);
        let m = Mat::zeros(3, 3);
        let result = prefix_beam_search(&m, &a, Some(&TinyLm), &neutral(), 5).unwrap();
        assert_eq!(
            result.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            ["a", "aa"]
        );
        // Of 27 equiprobable paths, six collapse to a and only a-blank-a to aa.
        close(result[0].score, (6.0_f64 / 27.0).ln() - 1.25);
        close(result[1].score, (1.0_f64 / 27.0).ln() - 1.75);
        let settings = DecoderSettings {
            lm_weight: 0.0,
            insertion_bonus: 0.3,
            ..neutral()
        };
        let result = prefix_beam_search(&m, &a, Some(&TinyLm), &settings, 5).unwrap();
        assert_eq!(result.len(), 2);
        close(result[0].score, (6.0_f64 / 27.0).ln() + 0.3);
        close(result[1].score, (1.0_f64 / 27.0).ln() + 0.6);
    }

    #[test]
    fn beam_pruning_ties_follow_insertion_order_not_prefix_order() {
        let a = alphabet(&["a", "b", ""]);
        let m = Mat::from_vec(2, 3, vec![0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0]);
        let s = DecoderSettings {
            beam_width: 1,
            insertion_bonus: 2.0_f64.ln(),
            ..neutral()
        };
        let out = prefix_beam_search::<TinyLm>(&m, &a, None, &s, 5).unwrap();
        // At frame 2, continuing a (two paths) ties extending to ab (one path
        // plus ln(2) bonus). The existing prefix was inserted first and wins.
        assert_eq!(out[0].text, "a");
        struct TiedLm;
        impl LanguageModel for TiedLm {
            type State = bool;
            fn start(&self) -> bool {
                true
            }
            fn advance(&self, first: bool, label: u32) -> Option<(bool, f64)> {
                Some((
                    false,
                    if first && label == 2 {
                        2.0_f64.ln()
                    } else {
                        0.0
                    },
                ))
            }
            fn finish(&self, _: bool) -> f64 {
                0.0
            }
        }
        let m = Mat::from_vec(
            2,
            3,
            vec![0.0, 0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, 0.0],
        );
        let s = DecoderSettings {
            beam_width: 2,
            ..neutral()
        };
        let out = prefix_beam_search(&m, &a, Some(&TiedLm), &s, 5).unwrap();
        // Frame 1 ranks b before a. Frame 2 inserts b, ba, a, all tied;
        // lexicographic prefix traversal would retain a instead of ba.
        assert_eq!(
            out.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            ["b", "ba"]
        );
    }

    #[test]
    fn beam_merges_rendered_text_and_applies_finals_after_pruning() {
        let a = alphabet(&["a", "a", ""]);
        let result =
            prefix_beam_search::<TinyLm>(&Mat::zeros(1, 3), &a, None, &neutral(), 5).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "a");
        close(result[0].score, (2.0_f64 / 3.0).ln());
        let settings = DecoderSettings {
            beam_width: 1,
            ..neutral()
        };
        // Blank is inserted first; final rejection must not rescue a pruned a.
        assert!(
            prefix_beam_search(&Mat::zeros(1, 3), &a, Some(&TinyLm), &settings, 5)
                .unwrap()
                .is_empty()
        );
        assert_eq!(greedy_decode(&Mat::zeros(1, 3), &a).unwrap().text, "a");
        let m = Mat::from_vec(
            4,
            3,
            vec![
                0.0, -10.0, -10.0, 0.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.0, -10.0, -10.0,
            ],
        );
        assert_eq!(greedy_decode(&m, &a).unwrap().text, "aa");
    }

    #[test]
    fn class_wrapper_scales_only_true_extensions_and_word_final() {
        let a = alphabet(&["a", "b", ""]);
        let scorer = CharClassScorer {
            scores: vec![0.0, 0.0, 3.0, 9.0],
            ..Default::default()
        };
        let settings = DecoderSettings {
            lm_weight: 2.0,
            char_class_weight: -0.5,
            ..neutral()
        };
        let lm = CharClassRescoringLm::new(&TinyLm, &scorer, &settings).unwrap();
        let result = prefix_beam_search(&Mat::zeros(3, 3), &a, Some(&lm), &neutral(), 5).unwrap();
        close(result[0].score, (6.0_f64 / 27.0).ln() - 2.0 * 1.25 - 1.5);
        close(result[1].score, (1.0_f64 / 27.0).ln() - 2.0 * 1.75 - 3.0);
        let zero = DecoderSettings {
            lm_weight: 0.0,
            ..settings
        };
        let lm = CharClassRescoringLm::new(&TinyLm, &scorer, &zero).unwrap();
        assert_eq!(lm.advance(0, 3), Some((0, f64::INFINITY)));
        assert_eq!(lm.finish(0), f64::INFINITY);
    }

    #[test]
    fn empty_input_and_invalid_parameters() {
        let a = alphabet(&["a", "b", ""]);
        let m = Mat::zeros(0, 3);
        assert_eq!(
            greedy_decode(&m, &a).unwrap(),
            Candidate {
                text: String::new(),
                score: 0.0
            }
        );
        assert_eq!(
            prefix_beam_search::<TinyLm>(&m, &a, None, &neutral(), 1).unwrap(),
            vec![Candidate {
                text: String::new(),
                score: 0.0
            }]
        );
        assert!(prefix_beam_search::<TinyLm>(&m, &a, None, &neutral(), 0).is_err());
        for settings in [
            DecoderSettings {
                beam_width: 0,
                ..neutral()
            },
            DecoderSettings {
                lm_weight: f64::NAN,
                ..neutral()
            },
            DecoderSettings {
                insertion_bonus: f64::INFINITY,
                ..neutral()
            },
        ] {
            assert!(prefix_beam_search::<TinyLm>(&m, &a, None, &settings, 1).is_err());
        }
        assert!(validate_cost(f64::NAN).is_err());
        assert!(validate_cost(f64::NEG_INFINITY).is_err());
        let mut invalid = a.clone();
        invalid.blank = 3;
        assert!(greedy_decode(&m, &invalid).is_err());
        invalid = a.clone();
        invalid.texts[0].clear();
        assert!(greedy_decode(&m, &invalid).is_err());
        invalid = a;
        invalid.labels.pop();
        assert!(greedy_decode(&m, &invalid).is_err());
    }
}
