"""Real-corpus regression tests (offline unless explicitly opted into downloads).

Set MLKIT_DOWNLOAD_RECOSPECS=1 to download all manifest recospecs through the
SHA1-verifying fetcher. Otherwise the corpus must already exist under models/.
"""
from collections import Counter
import os
from pathlib import Path
import unittest

from google.protobuf.message import DecodeError

from src import recospec
from src.fetch_pack import fetch
from src.packs import PackResolver

ROOT = Path(__file__).resolve().parent.parent


class RecoSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = PackResolver()
        if os.environ.get("MLKIT_DOWNLOAD_RECOSPECS") == "1":
            names = sorted({entry["recospec"] for entry in cls.resolver.entries.values()})
            for name in names:
                fetch(name, ROOT / "models" / name)
        cls.paths = sorted((ROOT / "models").glob("*/*.recospec.local"))
        if len(cls.paths) < 40:
            raise RuntimeError("Need at least 40 recospecs; run with MLKIT_DOWNLOAD_RECOSPECS=1")

    def spec(self, language):
        name = self.resolver.resolve(language)["recospec"]
        paths = list((ROOT / "models" / name).glob("*.recospec.local"))
        self.assertEqual(len(paths), 1, f"Missing corpus pack {name}")
        return recospec.load(paths[0])

    def test_byte_identical_roundtrip_entire_corpus(self):
        self.assertEqual(len(self.paths), 391)
        for path in self.paths:
            with self.subTest(pack=path.parent.name):
                original = path.read_bytes()
                spec = recospec.load(path)
                self.assertEqual(spec.serialize(), original)
                # Known structural fields, not a top-level opaque blob. The
                # explicitly opaque beam extension is documented in the proto.
                spec.proto.DiscardUnknownFields()
                self.assertEqual(spec.serialize(), original)
        print(f"\nByte-identical protobuf round-trips: {len(self.paths)}/{len(self.paths)}")

    def test_diverse_languages(self):
        languages = (
            "en-US", "fr", "de", "es", "it", "pt", "af", "vi", "tr", "ru", "uk", "bg",
            "ar", "fa", "ur", "zh-Hani-CN", "zh-Hani-TW", "zh-Hani-HK", "ja", "ko", "hi",
            "mr", "ne", "bn", "gu", "pa", "ta", "te", "kn", "ml", "th", "he", "am",
            "hy", "ka", "el", "my", "si", "bo", "lo", "km", "zxx-Zsye-x-emoji",
            "zxx-Zsym-x-autodraw", "zxx-Zsym-x-shapes", "en-x-gesture", "ar-x-gesture",
            "ja-x-gesture", "ko-x-gesture", "hi-x-gesture", "th-x-gesture", "he-x-gesture",
        )
        for language in languages:
            with self.subTest(language=language):
                spec = self.spec(language)
                self.assertEqual(spec.serialize(), spec.source.read_bytes())
                self.assertTrue(spec.charset)
                self.assertTrue(spec.pipeline)
        self.assertGreaterEqual(len({self.resolver.resolve(tag)["recospec"] for tag in languages}), 40)

    def test_pipeline_inventory(self):
        counts = Counter(tuple(step.field_number for step in recospec.load(path).pipeline)
                         for path in self.paths)
        self.assertEqual(counts, {(5, 6, 8, 11): 362, (10, 4, 12, 2, 11, 1): 2,
                                  (17, 10, 18, 19, 4, 2): 27})
        english = self.spec("en-US")
        self.assertEqual([step.kind for step in english.pipeline],
                         ["normalize_time", "hallucinate_time",
                          "normalize_size_writing_guide_first_stroke", "add_pen_up_strokes"])
        self.assertEqual(english.pipeline[1].parameters, {1: 20.0, 2: 0})
        self.assertTrue(english.pipeline[1].settings_type.endswith(".HallucinateTimeSettings"))
        self.assertTrue(self.spec("zxx-Zsye-x-emoji").pipeline[1].settings_type.endswith(".SanitizeTimeSettings"))
        descriptor = recospec.InkPreprocessingStepSpec.DESCRIPTOR
        self.assertEqual({f.number for f in descriptor.oneofs_by_name["step"].fields}, set(range(1, 31)))
        self.assertEqual(descriptor.fields_by_number[3].message_type.full_name,
                         "research_handwriting.StrokeOrderSettings")

    def test_curve_settings(self):
        curve = self.spec("en-US").curve_settings
        self.assertIsInstance(curve, recospec.CurveSettings)
        self.assertEqual([getattr(curve, f"unknown_{i}") for i in range(1, 6)],
                         [.05, .02, .01, 3., -.8])
        self.assertFalse(curve.generate_second_order_features)
        self.assertTrue(curve.unknown_7)
        self.assertTrue(curve.interpolate_time)
        self.assertFalse(curve.HasField("normalize_outputs_to_zero_one"))
        self.assertFalse(curve.normalize_outputs_to_zero_one)
        self.assertEqual(recospec.CurveSettings().unknown_4, 4.)
        self.assertIsNone(self.spec("zxx-Zsye-x-emoji").curve_settings)
        self.assertIsNone(self.spec("zxx-Zsym-x-autodraw").num_features)

    def test_english_charset_and_actual_tflite_output(self):
        spec = self.spec("en-US")
        self.assertEqual(spec.languages, ("en", "en_us"))
        self.assertEqual(len(spec.charset), 313)
        self.assertEqual(len(spec.symbol_table), 317)
        mapping = spec.ctc_mapping
        self.assertEqual(len(mapping.net_to_fst), 314)
        self.assertEqual(mapping.net_to_fst, tuple(range(2, 315)) + (1,))
        self.assertEqual(mapping.blank_index, len(spec.charset))
        self.assertEqual(mapping.net_symbols[:3], ("[[space]]", "!", '"'))
        self.assertEqual(mapping.net_symbols[-2:], ("✡", "<reserved>"))
        self.assertTrue(mapping.symbol_table_verified)
        self.assertFalse(mapping.oracle_verified)
        import tflite
        paths = list((ROOT / "models").rglob("latin_indy_lstm_6x216_20191208.tflite"))
        self.assertTrue(paths, "Download the English tflite with python src/packs.py en-US")
        model = tflite.Model.GetRootAsModel(paths[0].read_bytes(), 0)
        graph = model.Subgraphs(0)
        output = graph.Tensors(graph.Outputs(0))
        num_classes = output.Shape(output.ShapeLength() - 1)
        self.assertEqual(num_classes, len(mapping.net_to_fst))
        self.assertEqual(spec.ctc_mapping_for(num_classes=num_classes).blank_index, num_classes - 1)

    def test_all_symbol_tables_match_charsets(self):
        verified = 0
        for path in self.paths:
            spec = recospec.load(path)
            if spec.symbol_table:
                with self.subTest(pack=path.parent.name):
                    mapping = spec.ctc_mapping
                    self.assertEqual(len(mapping.net_to_fst), len(spec.charset) + 1)
                    self.assertEqual(mapping.blank_index, len(spec.charset))
                    self.assertEqual(mapping.net_to_fst[:-1], tuple(range(2, len(spec.charset) + 2)))
                    self.assertEqual(mapping.net_to_fst[-1], 1)
                    verified += 1
        self.assertEqual(verified, 360)
        print(f"Charset/FST ordering verified: {verified}/{verified} symbol tables")
        self.assertIsNone(self.spec("zxx-Zsye-x-emoji").ctc_mapping)
        chinese = self.spec("zh-Hani")
        self.assertEqual(chinese.ctc_mapping.blank_index, len(chinese.charset))
        self.assertFalse(chinese.ctc_mapping.symbol_table_verified)

    def test_ctc_override_and_script_dependent_blank(self):
        sizes = set()
        for language in ("en-US", "ru", "ar", "zh-Hani-CN", "ja", "ko", "hi", "th", "he"):
            spec = self.spec(language)
            sizes.add(len(spec.charset))
            default = spec.ctc_mapping
            self.assertEqual(default.blank_index, len(spec.charset))
            self.assertEqual(default.net_to_fst[0], 2)
            first = spec.ctc_mapping_for(blank_index=0)
            self.assertEqual(first.blank_index, 0)
            self.assertEqual(first.net_to_fst, tuple(range(1, len(spec.charset) + 2)))
            self.assertEqual(first.net_symbols[0], "<reserved>")
            self.assertFalse(first.oracle_verified)
            for invalid in (-1, len(spec.charset) + 1):
                with self.assertRaises(ValueError):
                    spec.ctc_mapping_for(blank_index=invalid)
            with self.assertRaises(ValueError):
                spec.ctc_mapping_for(num_classes=len(spec.charset))
        self.assertGreater(len(sizes), 5)

    def test_decoder_config_and_optional_fields(self):
        decoder = self.spec("en-US").decoder
        self.assertEqual(decoder.beam_width, 1000)
        self.assertEqual(decoder.beam_threshold, 10.)
        self.assertAlmostEqual(decoder.lm_weights["7"], .6152235269546509)
        self.assertEqual(len(decoder.char_class_weights), 10)
        self.assertEqual(decoder.char_classes["number"], "0123456789")
        self.assertEqual(decoder.char_classes["upper_en_us"], "")
        self.assertEqual(decoder.fst.search.DESCRIPTOR.full_name, "speech_decoder.FstSearchParams")
        self.assertEqual(decoder.fst.char_classes.weights.DESCRIPTOR.full_name, "aksara.DecoderWeights")
        gesture = self.spec("en-x-gesture")
        self.assertIsNone(gesture.decoder.beam_width)
        self.assertEqual(gesture.decoder.lm_weights, {})
        self.assertEqual(gesture.decoder.char_classes, {})
        self.assertTrue(gesture.proto.HasField("unknown_54"))
        self.assertFalse(self.spec("en-US").proto.HasField("unknown_54"))
        chinese = self.spec("zh-Hani-CN").decoder
        self.assertIn(5, chinese.search_parameters)
        self.assertNotIn(5, decoder.search_parameters)

    def test_changes_are_serialized_not_cached_bytes(self):
        spec = self.spec("en-US")
        original = spec.serialize()
        spec.curve_settings.unknown_1 = .125
        updated = spec.serialize()
        self.assertNotEqual(updated, original)
        self.assertEqual(recospec.RecoSpec.from_bytes(updated).curve_settings.unknown_1, .125)

    def test_unknown_fields_survive(self):
        spec = self.spec("en-US")
        # Canonical unknown top-level field 100, varint 123.
        payload = spec.serialize() + b"\xa0\x06\x7b"
        self.assertEqual(recospec.RecoSpec.from_bytes(payload).serialize(), payload)
        # Unknown future preprocessing branch 31, empty embedded message.
        step = spec.tf.processor.preprocessing.steps.add()
        step.ParseFromString(b"\xfa\x01\x00")
        self.assertEqual(spec.pipeline[-1].kind, "UNKNOWN")
        updated = spec.serialize()
        self.assertEqual(recospec.RecoSpec.from_bytes(updated).serialize(), updated)

    def test_ambiguous_varints_do_not_collapse_to_boolean(self):
        spec = self.spec("en-US")
        spec.pipeline[1].settings.unknown_2 = 2
        payload = spec.serialize()
        parsed = recospec.RecoSpec.from_bytes(payload)
        self.assertEqual(parsed.pipeline[1].parameters[2], 2)
        self.assertEqual(parsed.serialize(), payload)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            recospec.RecoSpec.from_bytes(b"")
        with self.assertRaises(DecodeError):
            recospec.RecoSpec.from_bytes(b"\x80")
        spec = self.spec("en-US")
        spec.tf.decoder.fst.word_lm.symbol_table = "bad\t0\n"
        with self.assertRaises(ValueError):
            _ = spec.ctc_mapping


if __name__ == "__main__":
    unittest.main()
