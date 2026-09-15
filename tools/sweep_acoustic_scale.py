#!/usr/bin/env python3
"""Sweep native's unknown `acoustic_scale` against the oracle corpus.

The disassembly (evidence/acoustic_scale_investigation.md) proves every acoustic
score the native decoder sees is `-acoustic_scale * raw_net_logit`, but not what
that scale is. The recovered LM-side weights of SPEC.md section 17 are held at
their native values; only this one scalar varies.

It must be applied to the per-frame log probabilities BEFORE the alignment sum,
not folded into the LM weights. Those look interchangeable -- the scale only
ever appears as a ratio against them -- but CTC sums over alignments, and
`a * logsumexp(x)` is not `logsumexp(a * x)`. The shortcut agrees only when one
alignment dominates, so it silently differs on exactly the ambiguous inputs
that decide the corpus.

`prefix_beam_search` applies its own `log_softmax`, which renormalizes the
scaled rows. That subtracts a per-frame constant, and every CTC path has exactly
T frames, so it cancels between hypotheses and leaves the ranking correct while
shifting the absolute scores.

Runs off the cached golden logits, so only the decoder is re-evaluated.

    .venv/bin/python tools/sweep_acoustic_scale.py --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import decoder, packs, recospec, tflite_weights  # noqa: E402
from src.fst import BackoffLanguageModel, CompactFst  # noqa: E402
from src.recognize import (  # noqa: E402
    DEFAULT_CHAR_CLASS_WEIGHT,
    NATIVE_INSERTION_BONUS,
    NATIVE_LM_WEIGHT,
)

GOLDENS = ROOT / "testdata" / "goldens"

_STATE: dict = {}


def initialize():
    paths = packs.ensure("en-US")
    spec = recospec.load(paths["recospec"])
    weights = tflite_weights.load_weights(paths["tflite"])
    mapping = spec.ctc_mapping_for(num_classes=weights.num_classes)
    symbols = dict(zip(mapping.net_to_fst, mapping.net_symbols))
    settings = spec.decoder
    scorer = decoder.CharClassScorer.from_tables(
        settings.char_classes,
        {item.name: item.value for item in settings.char_class_weights},
        symbols,
    )
    fst = CompactFst(paths["fst"])
    _STATE.update(
        symbols=symbols,
        net_to_label=dict(enumerate(mapping.net_to_fst)),
        blank=mapping.blank_index,
        scorer=scorer,
        lm=BackoffLanguageModel(fst),
        num_classes=weights.num_classes,
    )


def decode(job):
    trial_id, shape, scale = job
    logits = np.fromfile(GOLDENS / "logits" / f"{trial_id}.f32", dtype="<f4").reshape(shape)
    scaled = scale * decoder.log_softmax(logits)
    lm = decoder.CharClassRescoringLM(
        _STATE["lm"],
        _STATE["scorer"],
        word_weight=NATIVE_LM_WEIGHT,
        class_weight=DEFAULT_CHAR_CLASS_WEIGHT,
    )
    results = decoder.prefix_beam_search(
        scaled,
        _STATE["symbols"],
        blank=_STATE["blank"],
        net_to_label=_STATE["net_to_label"],
        beam_width=1000,
        nbest=3,
        lm=lm,
        lm_weight=1.0,
        insertion_bonus=NATIVE_INSERTION_BONUS,
    )
    return trial_id, [text for text, _ in results]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="*", default=[
        0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    ])
    ap.add_argument("--output", type=Path, default=ROOT / "evidence" / "acoustic_scale_sweep.json")
    args = ap.parse_args()

    goldens = json.loads((GOLDENS / "goldens.json").read_text())
    trials = goldens["trials"]
    truth = {t["id"]: t["oracle_top"] for t in trials}
    shapes = {t["id"]: tuple(t["logits_shape"]) for t in trials}
    singles = {t["id"] for t in trials if len(t["oracle_top"] or "") == 1}

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        for scale in args.scales:
            jobs = [(t["id"], shapes[t["id"]], scale) for t in trials]
            nbest = dict(pool.map(decode, jobs))
            correct = [i for i, p in nbest.items() if p and p[0] == truth[i]]
            # A miss where the truth is still in the n-best is a ranking
            # failure; a miss where it is absent means the FST never offered it
            # at all. The two have completely different remedies.
            misses = []
            for trial_id, candidates in sorted(nbest.items()):
                if candidates and candidates[0] == truth[trial_id]:
                    continue
                misses.append({
                    "truth": truth[trial_id],
                    "predicted": candidates[0] if candidates else "",
                    "truth_in_nbest": truth[trial_id] in candidates,
                    "nbest": candidates,
                })
            row = {
                "acoustic_scale": scale,
                "total": len(correct),
                "singles": sum(i in singles for i in correct),
                "words": sum(i not in singles for i in correct),
                "ranking_misses": sum(m["truth_in_nbest"] for m in misses),
                "absent_from_nbest": sum(not m["truth_in_nbest"] for m in misses),
                "misses": misses,
            }
            rows.append(row)
            print(
                f"a={scale:<6.4g} total={row['total']}/{len(trials)} "
                f"singles={row['singles']}/{len(singles)} "
                f"words={row['words']}/{len(trials) - len(singles)} "
                f"(ranking {row['ranking_misses']}, absent {row['absent_from_nbest']})",
                flush=True,
            )

    best = max(rows, key=lambda r: r["total"])
    args.output.write_text(json.dumps({
        "generator": "tools/sweep_acoustic_scale.py",
        "premise": "0x42ed74..0x42ed88 scales every net log-posterior by -acoustic_scale, "
                   "applied before the CTC alignment sum. LM-side weights held at their "
                   "recovered native values.",
        "native_lm_weight": NATIVE_LM_WEIGHT,
        "native_char_class_weight": DEFAULT_CHAR_CLASS_WEIGHT,
        "native_insertion_bonus": NATIVE_INSERTION_BONUS,
        "trials": len(trials),
        "rows": rows,
        "best": best,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"\nbest: acoustic_scale={best['acoustic_scale']}, {best['total']}/{len(trials)}")


if __name__ == "__main__":
    main()
