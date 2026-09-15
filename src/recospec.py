#!/usr/bin/env python3
"""Typed, lossless access to ML Kit's recovered recospec protobuf.

The checked-in .proto is compiled with protoc into a temporary descriptor set,
then loaded by protobuf's message factory. No generated source needs checking in.
Unknown protobuf fields are retained; serialize() actually serializes the parsed
message, rather than returning a saved copy of the input bytes.

Exact recovered names coexist with explicitly numbered unknowns. In particular,
beam/LM semantic aliases below are provisional, not recovered accessor names.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "proto" / "recospec.proto"
Scalar = bool | int | float | str | bytes


@cache
def schema_pool() -> descriptor_pool.DescriptorPool:
    protoc = shutil.which("protoc")
    if protoc is None:
        raise RuntimeError("Loading recospec.proto requires protoc on PATH")
    with tempfile.TemporaryDirectory(prefix="mlkit-recospec-") as directory:
        output = Path(directory) / "recospec.desc"
        subprocess.run([protoc, f"--proto_path={PROTO.parent}", "--include_imports",
                        f"--descriptor_set_out={output}", str(PROTO)],
                       check=True, capture_output=True)
        descriptors = descriptor_pb2.FileDescriptorSet.FromString(output.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for descriptor in descriptors.file:  # protoc emits dependencies first.
        pool.Add(descriptor)
    return pool


def message_type(name: str) -> type[Message]:
    """Return a protobuf class by its exact recovered fully-qualified name."""
    return message_factory.GetMessageClass(schema_pool().FindMessageTypeByName(name))


RecognizerSpec = message_type("research_handwriting.RecognizerSpec")
CurveSettings = message_type("research_handwriting.CurveSettings")
InkPreprocessingStepSpec = message_type("research_handwriting.InkPreprocessingStepSpec")
TF_RECOGNIZER_SPEC = schema_pool().FindExtensionByName("research_handwriting.tf_recognizer_spec")


def _scalars(message: Message) -> dict[int, Scalar]:
    return {field.number: value for field, value in message.ListFields()
            if field.message_type is None and not field.is_repeated}


@dataclass(frozen=True)
class PreprocessingStep:
    """A pipeline branch and its concrete protobuf settings, in execution order."""
    field_number: int | None
    kind: str
    settings: Message

    @property
    def settings_type(self) -> str:
        return self.settings.DESCRIPTOR.full_name

    @property
    def parameters(self) -> dict[int, Scalar]:
        """Present scalar parameters keyed by original wire field number."""
        return _scalars(self.settings)


@dataclass(frozen=True)
class CharClassWeight:
    name: str
    value: float


@dataclass(frozen=True)
class DecoderSettings:
    proto: Message

    @property
    def fst(self) -> Message | None:
        return self.proto.fst if self.proto.HasField("fst") else None

    @property
    def search_parameters(self) -> dict[int, Scalar]:
        fst = self.fst
        return _scalars(fst.search) if fst is not None and fst.HasField("search") else {}

    @property
    def beam_width(self) -> int | None:
        """Provisional alias for search field 1 (1000, hypothesis-count-shaped).

        The original accessor and exact pruning semantics are UNKNOWN; this is
        not the score threshold in search field 2.
        """
        value = self.search_parameters.get(1)
        return int(value) if value is not None else None

    @property
    def beam_threshold(self) -> float | None:
        """Provisional alias for search field 2 (10.0 in the catalog)."""
        value = self.search_parameters.get(2)
        return float(value) if value is not None else None

    @property
    def lm_weights(self) -> dict[str, float]:
        """Language-dependent score/weight fields, without invented semantics.

        Keys are protobuf numeric paths relative to FstDecoderConfig. Whether
        each is an LM scale, insertion penalty, etc. remains UNKNOWN.
        """
        fst = self.fst
        if fst is None:
            return {}
        weights = {str(number): float(getattr(fst, f"unknown_{number}"))
                   for number in (7, 10) if fst.HasField(f"unknown_{number}")}
        weights.update({f"1.{number}": float(self.search_parameters[number])
                        for number in (5, 7) if number in self.search_parameters})
        return weights

    @property
    def char_class_weights(self) -> tuple[CharClassWeight, ...]:
        fst = self.fst
        if fst is None or not fst.HasField("char_classes"):
            return ()
        return tuple(CharClassWeight(weight.name, weight.value)
                     for weight in fst.char_classes.weights.weights)

    @property
    def char_class_table(self) -> str | None:
        fst = self.fst
        if fst is None or not fst.HasField("char_classes"):
            return None
        classes = fst.char_classes
        return classes.char_class_table if classes.HasField("char_class_table") else None

    @property
    def char_classes(self) -> dict[str, str]:
        """Class name -> characters; preserve explicitly empty language classes."""
        return {name: characters for line in (self.char_class_table or "").splitlines()
                for name, _, characters in (line.partition(" "),)}


@dataclass(frozen=True)
class CtcMapping:
    """Blank-last mapping from native NetworkScoreCache; oracle not yet run.

    Normal net index k is charset[k] and FST ID k+2. The last output is blank,
    represented here by reserved FST ID 1. Epsilon/sentence boundaries are not
    network outputs. Native evidence: ctor 0x4f132b..32, score 0x50a224..261,
    diagnostic 0x4dfa4c..aae. An explicit override can test other blank positions.
    """
    net_to_fst: tuple[int, ...]
    net_symbols: tuple[str, ...]
    blank_index: int
    symbol_table_verified: bool
    oracle_verified: bool = False


@dataclass(frozen=True)
class RecoSpec:
    proto: Message
    source: Path | None = None

    @classmethod
    def from_bytes(cls, data: bytes, source: Path | None = None) -> RecoSpec:
        message = RecognizerSpec.FromString(data)
        if not message.HasExtension(TF_RECOGNIZER_SPEC):
            raise ValueError("RecognizerSpec has no TfRecognizerSpec extension")
        tf = message.Extensions[TF_RECOGNIZER_SPEC]
        if not tf.HasField("processor") or not tf.HasField("decoder"):
            raise ValueError("TfRecognizerSpec requires processor and decoder configs")
        return cls(message, source)

    @property
    def tf(self) -> Message:
        return self.proto.Extensions[TF_RECOGNIZER_SPEC]

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self.proto.language)

    @property
    def charset(self) -> tuple[str, ...]:
        """Network labels excluding CTC blank (313 for English; not 314)."""
        return tuple(self.tf.processor.charset)

    @property
    def curve_settings(self) -> CurveSettings | None:
        processor = self.tf.processor
        return processor.curve_settings if processor.HasField("curve_settings") else None

    @property
    def num_features(self) -> int | None:
        processor = self.tf.processor
        return processor.num_features if processor.HasField("num_features") else None

    @property
    def pipeline(self) -> tuple[PreprocessingStep, ...]:
        steps = []
        for step in self.tf.processor.preprocessing.steps:
            name = step.WhichOneof("step")
            if name is None:
                # Future unknown branches are preserved on the original message.
                steps.append(PreprocessingStep(None, "UNKNOWN", step))
            else:
                number = step.DESCRIPTOR.fields_by_name[name].number
                steps.append(PreprocessingStep(number, name, getattr(step, name)))
        return tuple(steps)

    @property
    def decoder(self) -> DecoderSettings:
        return DecoderSettings(self.tf.decoder)

    @property
    def symbol_table(self) -> dict[int, str]:
        fst = self.decoder.fst
        if fst is None or not fst.HasField("word_lm") or not fst.word_lm.HasField("symbol_table"):
            return {}
        symbols = {}
        for line in fst.word_lm.symbol_table.splitlines():
            symbol, index = line.rsplit("\t", 1)
            number = int(index)
            if number in symbols:
                raise ValueError(f"Duplicate FST symbol id {number}")
            symbols[number] = symbol
        return symbols

    @property
    def ctc_mapping(self) -> CtcMapping | None:
        return self.ctc_mapping_for()

    def ctc_mapping_for(self, *, num_classes: int | None = None,
                        blank_index: int | None = None) -> CtcMapping | None:
        """Build a script-independent mapping, optionally overriding blank position.

        Pass the real network output dimension as num_classes to validate it
        against the charset. Default: len(charset)+1, blank=num_classes-1.
        blank_index=0 recreates the old hypothesis for oracle comparisons.
        Non-CTC gesture classifiers return None. Chinese's FST-less CTC config
        still has a mapping, with symbol_table_verified=False.
        """
        if self.decoder.fst is None:
            return None
        count = len(self.charset)
        if num_classes is None:
            num_classes = count + 1
        if num_classes != count + 1:
            raise ValueError(f"Expected {count + 1} CTC outputs for {count} characters, got {num_classes}")
        if blank_index is None:
            blank_index = num_classes - 1
        if not 0 <= blank_index < num_classes:
            raise ValueError(f"Blank index {blank_index} is outside {num_classes} CTC outputs")
        expected = {0: "<epsilon>", 1: "<reserved>", count + 2: "<S>", count + 3: "</S>"}
        expected.update({index + 2: "[[space]]" if char == " " else char
                         for index, char in enumerate(self.charset)})
        symbols = self.symbol_table
        if symbols and symbols != expected:
            raise ValueError("Charset does not match the expected contiguous FST symbol table")
        ids = list(range(2, count + 2))
        ids.insert(blank_index, 1)
        return CtcMapping(tuple(ids), tuple(expected[index] for index in ids),
                          blank_index, symbol_table_verified=bool(symbols))

    def serialize(self) -> bytes:
        return self.proto.SerializeToString()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.serialize())


def load(path: str | Path) -> RecoSpec:
    source = Path(path)
    return RecoSpec.from_bytes(source.read_bytes(), source)


load_recospec = load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    spec = load(args.path)
    print(json.dumps({"languages": spec.languages, "charset_size": len(spec.charset),
                      "fst_symbols": len(spec.symbol_table), "num_features": spec.num_features,
                      "pipeline": [{"field": step.field_number, "kind": step.kind,
                                    "settings_type": step.settings_type,
                                    "parameters": step.parameters} for step in spec.pipeline],
                      "beam_width_provisional": spec.decoder.beam_width,
                      "lm_weights_by_field": spec.decoder.lm_weights}, indent=2))


if __name__ == "__main__":
    main()
