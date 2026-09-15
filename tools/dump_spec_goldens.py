#!/usr/bin/env python3
"""Freeze recospec parsing and pack resolution as Rust test fixtures.

Separate from dump_goldens.py because this needs no model inference and so runs
in seconds: it covers every language tag in the catalog, not just the 74 inks we
have oracle labels for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import packs, recospec  # noqa: E402

OUT = ROOT / "testdata" / "goldens"

# One representative per distinct shape we care about: the reference language,
# a non-Latin script, a right-to-left script, and one with no FST at all.
SPEC_SAMPLES = ["en-US", "zh-Hani-CN", "ar", "ko", "ja", "hi", "ru", "th", "ta", "emoji"]


def spec_facts(tag: str) -> dict:
    paths = packs.ensure(tag)
    spec = recospec.load(paths["recospec"])
    decoder = spec.decoder
    classes = decoder.char_classes
    return {
        "tag": tag,
        "recospec_file": paths["recospec"].name,
        "languages": list(spec.languages),
        "charset_size": len(spec.charset),
        "charset_head": list(spec.charset[:16]),
        "charset_tail": list(spec.charset[-8:]),
        "num_features": spec.num_features,
        "pipeline": [
            {"field": step.field_number, "kind": step.kind,
             "parameters": {str(k): v for k, v in step.parameters.items()}}
            for step in spec.pipeline
        ],
        "curve_settings": {str(k): v for k, v in
                           (recospec._scalars(spec.curve_settings) if spec.curve_settings else {}).items()},
        "beam_width": decoder.beam_width,
        "lm_weights_by_field": decoder.lm_weights,
        "char_class_weights": [{"name": w.name, "value": w.value} for w in decoder.char_class_weights],
        "char_class_line_count": len(classes),
        "char_class_empty": sorted(n for n, members in classes.items() if not members),
        "symbol_table_size": len(spec.symbol_table),
        "has_fst_decoder": decoder.fst is not None,
    }


def main() -> None:
    resolver = packs.PackResolver()
    resolution = {tag: resolver.resolve(tag) for tag in resolver.languages}
    # Fallback behaviour is the part most likely to drift in a port, so pin the
    # inexact lookups too, not just the exact hits.
    fallbacks = ["en", "en-GB", "en-Latn-US", "zh", "zh-Hans", "zh-Hant", "zh-TW",
                 "pt-BR", "pt", "sr-Latn", "fr_CA", "EN-us", "es-419", "und"]
    fallback_results = {}
    for tag in fallbacks:
        try:
            fallback_results[tag] = resolver.resolve_tag(tag)
        except KeyError as exc:
            fallback_results[tag] = {"error": str(exc)}

    specs = []
    for tag in SPEC_SAMPLES:
        try:
            specs.append(spec_facts(tag))
        except Exception as exc:  # a sample we cannot fetch is not fatal
            specs.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})

    (OUT / "spec_goldens.json").write_text(json.dumps({
        "generator": "tools/dump_spec_goldens.py",
        "counts": resolver.counts(),
        "resolution": resolution,
        "fallback_resolution": fallback_results,
        "specs": specs,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(resolution)} tags, {len(specs)} specs -> {OUT / 'spec_goldens.json'}")


if __name__ == "__main__":
    main()
