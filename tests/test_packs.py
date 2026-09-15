"""Catalog tests are offline; downloading is tested through the fetch boundary."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.packs import PackResolver, resolve


class PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = PackResolver()

    def test_catalog(self):
        self.assertEqual(self.resolver.counts()["language_tags"], 725)
        self.assertEqual(self.resolver.counts()["recospecs"], 391)
        for tag in self.resolver.languages:
            self.assertEqual(self.resolver.resolve_tag(tag), tag)

    def test_english_fallback(self):
        self.assertEqual(self.resolver.resolve_tag("en-US"), "en-US")
        self.assertEqual(resolve("EN_us"), resolve("en-US"))
        self.assertEqual(resolve("en-NZ"), resolve("en"))

    def test_scripts_and_regions(self):
        self.assertEqual(self.resolver.resolve_tag("zh-Hani-CN"), "zh-Hani-CN")
        self.assertEqual(resolve("zh-CN"), resolve("zh-Hani-CN"))
        self.assertEqual(resolve("zh-Hant"), resolve("zh-Hani-TW"))
        self.assertEqual(resolve("zh-Hans"), resolve("zh-Hani-CN"))
        self.assertEqual(resolve("sr-Latn-RS"), resolve("sr-Latn"))
        with self.assertRaises(KeyError):
            resolve("sr-Arab")

    def test_gestures_and_specials(self):
        for tag in ("en-US-x-gesture", "en-NZ-x-gesture", "zh-Hani-CN-x-gesture"):
            names = resolve(tag)
            self.assertIsNone(names["fst"])
            self.assertTrue(names["recospec"].startswith("scribe_"))
        for tag in ("zxx-Zsye-x-emoji", "zxx-Zsym-x-autodraw", "zxx-Zsym-x-shapes"):
            self.assertIsNone(resolve(tag)["fst"])
        for tag in ("not-a-language", "en-x-emoji", "", "zxx"):
            with self.assertRaises(KeyError):
                resolve(tag)

    def test_ensure(self):
        calls = []

        def fetch(name, dest):
            calls.append(name)
            dest.mkdir(parents=True)
            path = dest / ("model.recospec.local" if "recospec" in name else "model.tflite")
            path.write_bytes(b"fixture")
            return [path]

        with tempfile.TemporaryDirectory() as tmp, patch("src.packs.fetch_pack.fetch", fetch):
            paths = self.resolver.ensure("en-US-x-gesture", Path(tmp))
            self.assertIsNone(paths["fst"])
            self.assertTrue(paths["recospec"].is_file())
            self.assertTrue(paths["tflite"].is_file())
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
