"""Independent path enumeration and raw-artifact checks; no ML Kit oracle claim."""
import ast
from collections import defaultdict
import itertools
import math
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decoder import greedy_decode, log_softmax, prefix_beam_search


SYMBOLS = {0: "<epsilon>", 1: "<reserved>", 2: "a", 3: "b"}


def brute_force(probabilities, blank=None):
    """Sum ordinary probabilities over EVERY alignment (no decoder recurrence)."""
    blank = probabilities.shape[1] - 1 if blank is None else blank
    masses = defaultdict(float)
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        collapsed = tuple(index for i, index in enumerate(path)
                          if index != blank and (i == 0 or index != path[i - 1]))
        probability = math.prod(float(probabilities[t, index]) for t, index in enumerate(path))
        if probability:
            masses[collapsed] += probability
    return masses


class ToyLM:
    """Deterministic prefix LM; costs deliberately include negative increments."""

    def __init__(self, words, symbols, arc_costs=None):
        self.words = words  # Complete text -> final tropical cost.
        self.symbols = symbols
        self.arc_costs = arc_costs or {}
        self.calls = []

    def start(self):
        return ""

    def advance(self, state, ilabel):
        self.calls.append((state, ilabel))
        text = state + self.symbols[ilabel]
        if not any(word.startswith(text) for word in self.words):
            return None
        return text, self.arc_costs.get(text, 0.0)

    def finish(self, state):
        return self.words.get(state, math.inf)


class DecoderTests(unittest.TestCase):
    def test_exhaustive_path_probabilities(self):
        rng = np.random.default_rng(2026)
        for timesteps in range(7):
            for _ in range(4):
                probabilities = rng.dirichlet(np.ones(3), size=timesteps)
                expected = brute_force(probabilities)
                actual = dict(prefix_beam_search(np.log(probabilities), SYMBOLS, nbest=1000))
                rendered = {"".join(SYMBOLS[index + 2] for index in path): math.log(mass)
                            for path, mass in expected.items()}
                self.assertEqual(actual.keys(), rendered.keys())
                for text, score in rendered.items():
                    self.assertAlmostEqual(actual[text], score, places=12)
                self.assertAlmostEqual(sum(math.exp(score) for score in actual.values()), 1.0, places=12)

    def test_greedy_repeat_and_blank_order(self):
        for path, expected in [([0, 0, 2, 0, 1, 1], "aab"), ([2, 2], ""), ([1, 1], "b")]:
            logits = np.full((len(path), 3), -math.inf)
            logits[np.arange(len(path)), path] = 0.0
            self.assertEqual(greedy_decode(logits, SYMBOLS), [(expected, 0.0)])
            self.assertEqual(prefix_beam_search(logits, SYMBOLS), [(expected, 0.0)])

    def test_repeated_character_acoustic_mass(self):
        # aa can only arise from a/blank/a in three frames, while a merges six paths.
        actual = dict(prefix_beam_search(np.zeros((3, 2)), {2: "a"}))
        self.assertAlmostEqual(math.exp(actual["aa"]), 1 / 8)
        self.assertAlmostEqual(math.exp(actual["a"]), 6 / 8)
        self.assertAlmostEqual(math.exp(actual[""]), 1 / 8)

    def test_synthetic_hello_logits(self):
        symbols = {2: "h", 3: "e", 4: "l", 5: "o"}
        path = [4, 0, 0, 1, 1, 2, 2, 4, 2, 2, 3, 3, 4]
        logits = np.full((1, len(path), 5), -9.0)
        logits[0, np.arange(len(path)), path] = 9.0
        greedy = greedy_decode(logits, symbols)
        beam = prefix_beam_search(logits, symbols, nbest=3)
        self.assertEqual(greedy[0][0], "hello")
        self.assertEqual(beam[0][0], "hello")
        self.assertGreater(beam[0][1], greedy[0][1])
        self.assertEqual(len(beam), 3)
        self.assertEqual(beam, sorted(beam, key=lambda result: (-result[1], result[0])))

    def test_lm_resolves_real_word_nonword_ambiguity(self):
        symbols = {2: "c", 3: "a", 4: "t", 5: "x"}
        logits = np.full((3, 5), -math.inf)
        logits[0, 0] = logits[1, 1] = 0.0
        logits[2, 2:4] = np.log([0.4, 0.6])
        lm = ToyLM({"cat": 0.0, "cax": 2.0}, symbols)
        self.assertEqual(prefix_beam_search(logits, symbols)[0][0], "cax")
        self.assertEqual(prefix_beam_search(logits, symbols, lm=lm, lm_weight=0)[0][0], "cax")
        ranked = prefix_beam_search(logits, symbols, lm=lm, lm_weight=1)
        self.assertEqual(ranked[0][0], "cat")
        self.assertAlmostEqual(dict(ranked)["cat"], math.log(0.4))
        self.assertAlmostEqual(dict(ranked)["cax"], math.log(0.6) - 2)
        lexicon = ToyLM({"cat": 0.0}, symbols)
        self.assertEqual(prefix_beam_search(logits, symbols, lm=lexicon, lm_weight=0), [("cat", math.log(0.4))])

    def test_lm_and_bonus_once_per_prefix_not_alignment(self):
        probabilities = np.array([[.2, .5, .3], [.3, .4, .3], [.5, .2, .3], [.1, .7, .2]])
        lm = ToyLM({"a": 0.7, "aa": -0.3, "ab": 1.2, "b": 0.2}, SYMBOLS,
                   {"a": 0.4, "aa": -0.5, "ab": 0.9, "b": 0.1})
        expected = {}
        for path, mass in brute_force(probabilities).items():
            text = "".join(SYMBOLS[index + 2] for index in path)
            if text in lm.words:
                cost = lm.words[text] + sum(lm.arc_costs.get(text[:i], 0) for i in range(1, len(text) + 1))
                expected[text] = math.log(mass) - 1.7 * cost + 0.6 * len(text)
        actual = dict(prefix_beam_search(np.log(probabilities), SYMBOLS, lm=lm,
                                         lm_weight=1.7, insertion_bonus=0.6))
        self.assertEqual(actual.keys(), expected.keys())
        for text, score in expected.items():
            self.assertAlmostEqual(actual[text], score, places=12)
        self.assertTrue(all(label in (2, 3) for _, label in lm.calls))

    def test_bonus_is_per_token_and_affects_pruning(self):
        logits = np.log([[0.4, 0.6]])
        self.assertEqual(prefix_beam_search(logits, {2: "word"}, beam_width=1)[0][0], "")
        result = prefix_beam_search(logits, {2: "word"}, beam_width=1, insertion_bonus=1)
        self.assertEqual(result[0][0], "word")
        self.assertAlmostEqual(result[0][1], math.log(.4) + 1)

    def test_custom_mapping_blank_and_space(self):
        symbols = {42: "[[space]]", 19: "z"}
        logits = np.full((5, 3), -math.inf)
        logits[np.arange(5), [0, 0, 2, 0, 1]] = 0
        for mapping in ({0: 19, 1: 42}, [19, 42, 999]):
            kwargs = dict(blank=2, net_to_label=mapping)
            self.assertEqual(greedy_decode(logits, symbols, **kwargs), [("zz ", 0.0)])
            lm = ToyLM({"zz[[space]]": 0}, symbols)
            self.assertEqual(prefix_beam_search(logits, symbols, lm=lm, **kwargs), [("zz ", 0.0)])

    def test_blank_and_mapping_overrides_are_independent(self):
        logits = np.full((4, 3), -math.inf)
        logits[np.arange(4), [1, 0, 1, 2]] = 0
        # Explicit first blank leaves the default k+2 label mapping unchanged.
        for decoder in (greedy_decode, prefix_beam_search):
            self.assertEqual(decoder(logits, {3: "a", 4: "b"}, blank=0), [("aab", 0.0)])
            # Both overrides also support the original blank-zero toy convention.
            self.assertEqual(decoder(logits, SYMBOLS, blank=0, net_to_label=[1, 2, 3]),
                             [("aab", 0.0)])
            # Explicit labels leave the default last blank unchanged.
            self.assertEqual(decoder(logits, {8: "a", 9: "b"}, net_to_label={0: 8, 1: 9}),
                             [("bab", 0.0)])

    def test_empty_input_and_final_rejection(self):
        self.assertEqual(greedy_decode(np.empty((0, 3)), SYMBOLS), [("", 0.0)])
        self.assertEqual(prefix_beam_search(np.empty((0, 3)), SYMBOLS), [("", 0.0)])
        self.assertEqual(prefix_beam_search(np.empty((0, 3)), SYMBOLS,
                                           lm=ToyLM({"a": 0}, SYMBOLS)), [])
        self.assertEqual(prefix_beam_search(np.empty((0, 3)), SYMBOLS,
                                           lm=ToyLM({"": 3}, SYMBOLS), lm_weight=.5), [("", -1.5)])
        self.assertEqual(prefix_beam_search([[-math.inf, -math.inf, 0]], SYMBOLS,
                                           lm=ToyLM({"a": 0}, SYMBOLS)), [])

    def test_infinite_increment_rejects_even_with_zero_weight(self):
        lm = ToyLM({"a": 0}, SYMBOLS, {"a": math.inf})
        self.assertEqual(prefix_beam_search([[0, -math.inf, -math.inf]], SYMBOLS,
                                           lm=lm, lm_weight=0), [])

    def test_duplicate_renderings_merge_nbest(self):
        result = prefix_beam_search(np.log([[.4, .5, .1]]), {2: "a", 3: "a"})
        self.assertEqual([text for text, _ in result], ["a", ""])
        self.assertAlmostEqual(result[0][1], math.log(.9))
        remapped = prefix_beam_search(np.log([[.4, .5, .1]]), {7: "a"},
                                      net_to_label={0: 7, 1: 7})
        self.assertEqual(remapped, result)

    def test_stable_normalization(self):
        logits = np.array([[10000, 10001, -math.inf], [-10000, -10001, -10002]])
        probabilities = np.exp(log_softmax(logits))
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])
        np.testing.assert_allclose(log_softmax(logits), log_softmax(logits + 20000))
        np.testing.assert_allclose(log_softmax(log_softmax(logits)), log_softmax(logits))

    def test_invalid_inputs(self):
        for logits in ([], np.zeros((2, 1, 3)), np.zeros((3,)), np.empty((2, 0)),
                       [[-math.inf] * 3], [[0, math.nan, 0]], [[0, math.inf, 0]]):
            for decoder in (greedy_decode, prefix_beam_search):
                with self.subTest(logits=logits, decoder=decoder.__name__):
                    with self.assertRaises(ValueError):
                        decoder(logits, SYMBOLS)
        for kwargs in ({"blank": 3}, {"blank": -1}, {"net_to_label": [1]},
                       {"net_to_label": {0: 999, 1: 3}}):
            with self.assertRaises(ValueError):
                prefix_beam_search(np.zeros((2, 3)), SYMBOLS, **kwargs)
        for kwargs in ({"beam_width": 0}, {"nbest": 0}, {"lm_weight": math.nan},
                       {"insertion_bonus": math.inf}):
            with self.assertRaises(ValueError):
                prefix_beam_search(np.zeros((2, 3)), SYMBOLS, **kwargs)
        for cost in (math.nan, -math.inf):
            with self.assertRaises(ValueError):
                prefix_beam_search([[0, -math.inf, -math.inf]], SYMBOLS,
                                   lm=ToyLM({"a": cost}, SYMBOLS))


def raw_charset_and_symbols():
    raw = (ROOT / "evidence/en_us.recospec.raw.txt").read_text()
    # protoc's C-style UTF-8 byte escapes; deliberately independent of recospec.py.
    def decode_string(value):
        return ast.literal_eval("b" + value).decode("utf-8")
    charset = [decode_string(value) for value in re.findall(r'^    1: (".*")$', raw, re.M)]
    table = decode_string(re.search(r'^        4: ("<epsilon>.*")$', raw, re.M)[1])
    symbols = {int(label): text for text, label in
               (line.rsplit("\t", 1) for line in table.splitlines())}
    return charset, symbols


class RawCharsetEvidenceTests(unittest.TestCase):
    def test_full_processor_charset_matches_fst_table(self):
        charset, symbols = raw_charset_and_symbols()
        self.assertEqual(len(charset), 313)
        self.assertEqual(len(symbols), 317)
        self.assertEqual({label: symbols[label] for label in (0, 1, 315, 316)},
                         {0: "<epsilon>", 1: "<reserved>", 315: "<S>", 316: "</S>"})
        self.assertEqual([" " if symbols[i + 2] == "[[space]]" else symbols[i + 2]
                          for i in range(len(charset))], charset)
        # Exercise every native-evidenced character index and both end blanks.
        logits = np.full((len(charset) + 2, 314), -math.inf)
        logits[[0, -1], 313] = 0
        logits[np.arange(1, len(charset) + 1), np.arange(313)] = 0
        self.assertEqual(greedy_decode(logits, symbols), [("".join(charset), 0.0)])
        self.assertEqual(prefix_beam_search(logits, symbols), [("".join(charset), 0.0)])


@unittest.skipUnless((ROOT / "models/en_us.compact.fst.local").is_file(),
                     "shipped en-US FST is not downloaded")
class ShippedFstIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fst import BackoffLanguageModel, CompactFst
        _, cls.symbols = raw_charset_and_symbols()
        cls.labels = {text: label for label, text in cls.symbols.items()}
        cls.fst = CompactFst(ROOT / "models/en_us.compact.fst.local")
        cls.addClassCleanup(cls.fst.close)
        cls.lm = BackoffLanguageModel(cls.fst)

    def test_hello_with_shipped_lm(self):
        blank = 313
        h, e, l, o = [self.labels[char] - 2 for char in "helo"]
        path = [blank, h, h, e, e, l, l, blank, l, l, o, o, blank]
        logits = np.full((1, len(path), 314), -math.inf)
        # Finite alternatives test acoustic path merging, rather than a sole path.
        logits[0, :, [h, e, l, o, blank]] = -40.0
        logits[0, np.arange(len(path)), path] = 0.0
        self.assertEqual(greedy_decode(logits, self.symbols)[0][0], "hello")
        self.assertEqual(prefix_beam_search(logits, self.symbols)[0][0], "hello")
        ranked = prefix_beam_search(logits, self.symbols, lm=self.lm)
        self.assertEqual(ranked[0][0], "hello")
        self.assertTrue(math.isfinite(ranked[0][1]))
        acoustic = dict(prefix_beam_search(logits, self.symbols))["hello"]
        expected = acoustic - self.lm.score(self.labels[char] for char in "hello")
        self.assertAlmostEqual(ranked[0][1], expected, places=10)

    def test_shipped_lm_prefers_cat_over_acoustic_cax(self):
        logits = np.full((7, 314), -math.inf)
        logits[[0, 2, 4, 6], 313] = 0
        logits[1, self.labels["c"] - 2] = 0
        logits[3, self.labels["a"] - 2] = 0
        logits[5, self.labels["t"] - 2] = math.log(.4)
        logits[5, self.labels["x"] - 2] = math.log(.6)
        self.assertEqual(greedy_decode(logits, self.symbols)[0][0], "cax")
        acoustic = prefix_beam_search(logits, self.symbols)
        self.assertEqual(acoustic[0][0], "cax")
        ranked = prefix_beam_search(logits, self.symbols, lm=self.lm)
        self.assertEqual(ranked[0][0], "cat")
        # Backoff accepts nonwords too; it penalizes cax rather than banning it.
        self.assertEqual({text for text, _ in ranked}, {"cat", "cax"})
        for text, score in ranked:
            cost = self.lm.score(self.labels[char] for char in text)
            self.assertTrue(math.isfinite(cost))
            self.assertAlmostEqual(score, dict(acoustic)[text] - cost, places=10)


if __name__ == "__main__":
    unittest.main()
