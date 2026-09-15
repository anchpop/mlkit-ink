#!/usr/bin/env python3
"""End-to-end handwriting recognition, entirely outside the ML Kit SDK.

    ink strokes
      -> features.extract_features      preprocessing + Bezier fit + 10 features
      -> indylstm.forward               6x216 bidirectional IndyLSTM + FC
      -> decoder.prefix_beam_search     CTC beam search, shallow-fused with the FST LM
      -> n-best candidates

Every artifact is Google's own, fetched from the public dl.google.com catalog; only the
inference code is ours. See SPEC.md for how each piece was recovered.

    from src.recognize import Recognizer
    r = Recognizer.for_language("en-US")
    print(r.recognize(strokes))
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import decoder, features, packs, recospec, tflite_weights
from .fst import BackoffLanguageModel, CompactFst
from .indylstm import forward


# Fixed-sign experiment on the saved en-US corpus: treating explicit class
# values as costs improved full-LM discovery from 55/61 to 59/61, with greedy
# unchanged. This sign is empirical; native rescoring semantics remain unverified.
# Both the word-FST cost and the char-class term are scaled by the SAME native multiplier,
# decoder field f7. The combiner (SumWeightCombiner, 0x593898) forms an aggregate whose
# rescoring contribution is  +f7 * (word_FST_cost - class_weight), with the word component
# being the plain FST cost (arc weights ADDED, no negation, 0x592ae2/0x592b4a) and the en-US
# graph-arc cost being 0. So the faithful setting is word_weight == char_class_weight == f7,
# and the class term is SUBTRACTED from the word cost.
#
# Note this is the opposite sign from what scored best on our corpus (-1 gave 59/61 vs 53/61
# for the positive direction). We follow the binary: a mechanism that produces better numbers
# for the wrong reason is exactly the trap that SPEC section 13's time-scaling bug set.
NATIVE_LM_WEIGHT = 0.615223527          # recospec decoder field 4.6.7
# Per-emitted-label cost, exempted on blanks/collapsed repeats (field 4.6.1.7; serializer
# 0x59ed26, addition 0x506f43, zero-output exemption 0x4e8954). Native stores it as a negative
# COST, which is a positive bonus in our maximised-score convention.
NATIVE_INSERTION_BONUS = 1.901311993598938
DEFAULT_CHAR_CLASS_WEIGHT = NATIVE_LM_WEIGHT


@dataclass(frozen=True)
class Candidate:
    """One recognition hypothesis. Mirrors ML Kit's RecognitionCandidate."""

    text: str
    score: float

    def __repr__(self) -> str:
        return f"Candidate({self.text!r}, {self.score:.4f})"


class Recognizer:
    """A loaded, ready-to-run recognizer for one language."""

    def __init__(self, spec, weights, mapping, curve_settings, pipeline, lm=None, fst=None):
        self.spec = spec
        self.weights = weights
        self.mapping = mapping
        self.curve_settings = curve_settings
        self.pipeline = pipeline
        self.lm = lm
        self.fst = fst
        self._char_class_scorer = None

    @classmethod
    def for_language(cls, language_tag: str, *, use_lm: bool = True) -> "Recognizer":
        """Resolve, download (if needed) and load everything for a BCP-47 tag."""
        paths = packs.ensure(language_tag)
        spec = recospec.load(paths["recospec"])
        weights = tflite_weights.load_weights(paths["tflite"])

        mapping = spec.ctc_mapping_for(num_classes=weights.num_classes)
        if mapping is None:
            raise ValueError(
                f"{language_tag} resolves to a gesture//classifier model, not a CTC recognizer"
            )

        cs = spec.curve_settings
        curve_settings = (
            features.CurveSettings.from_proto(cs) if cs is not None else features.CurveSettings()
        )
        if spec.num_features not in (None, features.NUM_FEATURES):
            raise ValueError(
                f"{language_tag} expects {spec.num_features} input features; "
                f"the Bezier encoder produces {features.NUM_FEATURES}"
            )

        pipeline = tuple(step.kind for step in spec.pipeline)

        lm = fst = None
        if use_lm and paths["fst"] is not None:
            fst = CompactFst(paths["fst"])
            lm = BackoffLanguageModel(fst)

        return cls(spec, weights, mapping, curve_settings, pipeline, lm, fst)

    @property
    def char_class_scorer(self) -> decoder.CharClassScorer | None:
        """Lazily build the explicit-membership scorer; no language classes are guessed."""
        if self._char_class_scorer is None:
            settings = self.spec.decoder
            weights = {item.name: item.value for item in settings.char_class_weights}
            if not weights or not settings.char_classes:
                return None
            symbols = dict(zip(self.mapping.net_to_fst, self.mapping.net_symbols))
            self._char_class_scorer = decoder.CharClassScorer.from_tables(
                settings.char_classes, weights, symbols)
        return self._char_class_scorer

    # -- pipeline stages, exposed individually so each can be diffed against the oracle ----

    def features_for(self, strokes: list[features.Stroke]) -> np.ndarray:
        """Ink -> [T, 10] Bezier curve features."""
        return features.extract_features(strokes, self.curve_settings, pipeline=self.pipeline)

    def logits_for(self, strokes: list[features.Stroke]) -> np.ndarray:
        """Ink -> [T, num_classes] unnormalized logits."""
        return forward(self.weights, self.features_for(strokes))

    def recognize(
        self,
        strokes: list[features.Stroke],
        *,
        nbest: int = 5,
        beam_width: int = 1000,
        lm_weight: float = NATIVE_LM_WEIGHT,
        insertion_bonus: float = NATIVE_INSERTION_BONUS,
        greedy: bool = False,
        char_class_weight: float = DEFAULT_CHAR_CLASS_WEIGHT,
    ) -> list[Candidate]:
        """Ink -> n-best candidates, best first.

        beam_width defaults to 1000, which is the value carried in the recospec's decoder
        config. lm_weight/insertion_bonus default to neutral values -- the corresponding raw
        floats in the recospec are present but their meanings are not yet established, so we
        do not pretend to have recovered Google's tuned weights.

        char_class_weight is independent of lm_weight. The default subtracts
        explicit class values as costs, the better of two fixed-sign corpus
        tests; the native additive interpretation is still unverified. Set 0 to
        disable, or a positive value to add class values to scores instead.
        Empty language-specific memberships are not guessed. Greedy never uses
        this scorer, and absent class tables leave the word LM unchanged.
        """
        logits = self.logits_for(strokes)
        if len(logits) == 0:
            return []

        # decoder keys `symbols` by FST label, not by network index, so zip the mapping's two
        # parallel tuples rather than enumerating either one.
        symbols = dict(zip(self.mapping.net_to_fst, self.mapping.net_symbols))
        net_to_label = dict(enumerate(self.mapping.net_to_fst))
        if greedy or self.lm is None:
            results = decoder.greedy_decode(
                logits, symbols, net_to_label=net_to_label, blank=self.mapping.blank_index
            )
        else:
            lm, word_weight = self.lm, lm_weight
            if char_class_weight != 0.0 and self.char_class_scorer is not None:
                lm = decoder.CharClassRescoringLM(
                    lm, self.char_class_scorer, word_weight=lm_weight,
                    class_weight=char_class_weight)
                word_weight = 1.0
            results = decoder.prefix_beam_search(
                logits,
                symbols,
                blank=self.mapping.blank_index,
                net_to_label=net_to_label,
                beam_width=beam_width,
                nbest=nbest,
                lm=lm,
                lm_weight=word_weight,
                insertion_bonus=insertion_bonus,
            )
        return [Candidate(text, score) for text, score in results[:nbest]]


def load_ink(path: str | Path) -> list[features.Stroke]:
    """Read the oracle harness's ink JSON schema (see android-oracle/)."""
    return features.strokes_from_json(json.loads(Path(path).read_text()))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ink_json", help="ink file in the oracle harness schema")
    ap.add_argument("-l", "--language", default="en-US")
    ap.add_argument("-n", "--nbest", type=int, default=5)
    ap.add_argument("--beam-width", type=int, default=1000)
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--insertion-bonus", type=float, default=0.0)
    ap.add_argument("--char-class-weight", type=float, default=DEFAULT_CHAR_CLASS_WEIGHT,
                    help="Experimental per-token class score weight; 0 disables, negative subtracts costs")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--show-features", action="store_true")
    args = ap.parse_args()

    strokes = load_ink(args.ink_json)
    r = Recognizer.for_language(args.language, use_lm=not args.no_lm)
    feats = r.features_for(strokes)
    print(f"{len(strokes)} strokes -> {len(feats)} curves ({feats.shape[1]} features each)")
    if args.show_features:
        np.set_printoptions(precision=3, suppress=True, linewidth=140)
        print(feats)
    for i, c in enumerate(
        r.recognize(
            strokes,
            nbest=args.nbest,
            beam_width=args.beam_width,
            lm_weight=args.lm_weight,
            insertion_bonus=args.insertion_bonus,
            greedy=args.greedy,
            char_class_weight=args.char_class_weight,
        )
    ):
        print(f"  {i + 1}. {c.text!r}  ({c.score:.4f})")


if __name__ == "__main__":
    main()
