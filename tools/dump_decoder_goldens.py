#!/usr/bin/env python3
"""Freeze the decoder's inputs so the decoder port can be tested on its own.

The decoder needs an alphabet and the character-class tables, which normally
come from the recospec parser. Pinning them here decouples the two ports: a
decoder test failing should mean the decoder is wrong, not that the recospec
parser is not written yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import decoder, packs, recospec, tflite_weights  # noqa: E402

OUT = ROOT / "testdata" / "goldens" / "decoder_goldens.json"


def main() -> None:
    paths = packs.ensure("en-US")
    spec = recospec.load(paths["recospec"])
    weights = tflite_weights.load_weights(paths["tflite"])
    mapping = spec.ctc_mapping_for(num_classes=weights.num_classes)

    symbols = dict(zip(mapping.net_to_fst, mapping.net_symbols))
    settings = spec.decoder
    class_weights = {item.name: item.value for item in settings.char_class_weights}
    scorer = decoder.CharClassScorer.from_tables(settings.char_classes, class_weights, symbols)

    payload = {
        "generator": "tools/dump_decoder_goldens.py",
        "language": "en-US",
        "fst_file": paths["fst"].name,
        "alphabet": {
            "blank": mapping.blank_index,
            "labels": [int(v) for v in mapping.net_to_fst],
            "texts": [" " if s == "[[space]]" else s for s in mapping.net_symbols],
        },
        "char_class_table": settings.char_class_table,
        "char_class_weights": [{"name": w.name, "value": w.value}
                               for w in settings.char_class_weights],
        "char_class_scores": {str(label): value for label, value in sorted(scorer.scores.items())},
        "char_class_empty": list(scorer.empty_classes),
        "char_class_unmapped_weights": list(scorer.unmapped_weights),
        "char_class_override_count": len(scorer.overrides),
    }

    # A handful of label sequences scored end to end, so the port can check the
    # backoff walk itself rather than only its effect on a whole decode.
    fst = __import__("src.fst", fromlist=["CompactFst"]).CompactFst(paths["fst"])
    lm = __import__("src.fst", fromlist=["BackoffLanguageModel"]).BackoffLanguageModel(fst)
    text_to_label = {t: l for l, t in symbols.items()}
    sequences = ["hi", "cat", "the", "q", "zzq", "Hello", "a b"]
    scored = []
    for text in sequences:
        labels = [text_to_label.get(" " if c == " " else c) for c in text]
        if any(l is None for l in labels):
            continue
        state, steps = lm.start(), []
        cost = 0.0
        ok = True
        for label in labels:
            step = lm.advance(state, label)
            if step is None:
                ok = False
                break
            state, weight = step
            cost += weight
            steps.append({"label": label, "state": int(state), "cost": weight})
        scored.append({
            "text": text,
            "labels": labels,
            "accepted": ok,
            "steps": steps,
            "final": (lm.finish(state) if ok else None),
            "total": (cost + lm.finish(state)) if ok else None,
        })
    payload["fst"] = {
        "num_states": int(fst.num_states),
        "num_arcs": int(fst.num_arcs),
        "num_labels": int(fst.num_labels),
        "quantum": float(fst.quantum),
        "start_state": int(fst.start_state),
        "use_final_weights": bool(fst.use_final_weights),
        "sequences": scored,
    }
    fst.close()

    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
