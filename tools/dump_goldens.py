#!/usr/bin/env python3
"""Freeze the Python reference pipeline's per-stage output as Rust test fixtures.

The Rust port is verified stage by stage against these, not just end to end: a
features mismatch and a decoder mismatch look identical in the final string, so
each stage gets its own golden.

    .venv/bin/python tools/dump_goldens.py

Writes testdata/goldens/goldens.json plus one raw little-endian float32 blob of
shape [T, num_classes] per trial under testdata/goldens/logits/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import recognize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "testdata" / "corpus"
OUT = ROOT / "testdata" / "goldens"


def main() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    entries = manifest["entries"]
    r = recognize.Recognizer.for_language("en-US")

    (OUT / "logits").mkdir(parents=True, exist_ok=True)
    trials = []
    for entry in entries:
        ink_path = CORPUS / entry["ink"]
        strokes = recognize.load_ink(ink_path)
        feats = r.features_for(strokes)
        logits = r.logits_for(strokes)
        greedy = r.recognize(strokes, greedy=True)
        native = r.recognize(strokes, nbest=5)
        empirical = r.recognize(strokes, nbest=5, lm_weight=1.0, char_class_weight=-1.0)

        (OUT / "logits" / f"{entry['id']}.f32").write_bytes(
            np.ascontiguousarray(logits, dtype="<f4").tobytes())

        oracle = entry.get("oracle_response", {})
        trials.append({
            "id": entry["id"],
            "intended": entry["intended"],
            "ink": str(ink_path.relative_to(ROOT)),
            "oracle_top": (oracle.get("candidates") or [{}])[0].get("text"),
            "num_strokes": len(strokes),
            "features": [[float(v) for v in row] for row in feats],
            "logits_shape": list(logits.shape),
            "greedy": {"text": greedy[0].text, "score": greedy[0].score} if greedy else None,
            "lm_native": [{"text": c.text, "score": c.score} for c in native],
            "lm_empirical": [{"text": c.text, "score": c.score} for c in empirical],
        })

    mapping = r.mapping
    goldens = {
        "language": "en-US",
        "generator": "tools/dump_goldens.py",
        "note": "Reference output of the Python implementation, for the Rust port to match.",
        "weights": {
            "lm_weight": recognize.NATIVE_LM_WEIGHT,
            "insertion_bonus": recognize.NATIVE_INSERTION_BONUS,
            "char_class_weight": recognize.DEFAULT_CHAR_CLASS_WEIGHT,
            "beam_width": 1000,
        },
        "model": {
            "num_classes": r.weights.num_classes,
            "input_size": r.weights.input_size,
            "layers": len(r.weights.layers),
            "hidden_sizes": [layer.forward.hidden_size for layer in r.weights.layers],
            "cell_clips": [layer.cell_clip for layer in r.weights.layers],
            "blank_index": mapping.blank_index,
            "charset_size": len(r.spec.charset),
            "pipeline": list(r.pipeline),
        },
        "trials": trials,
    }
    (OUT / "goldens.json").write_text(json.dumps(goldens, indent=1) + "\n")
    print(f"{len(trials)} trials -> {OUT / 'goldens.json'}")
    print(f"greedy matches oracle: {sum(t['greedy']['text'] == t['oracle_top'] for t in trials)}/{len(trials)}")
    print(f"lm_native matches oracle: {sum(t['lm_native'] and t['lm_native'][0]['text'] == t['oracle_top'] for t in trials)}/{len(trials)}")


if __name__ == "__main__":
    main()
