"""Portable tiny binary fixtures and opt-in exhaustive shipped-model validation."""
import ast
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fst import Arc, BackoffLanguageModel, CompactFst, FstFormatError

MODEL = ROOT / 'models/en_us.compact.fst.local'


def bitmap(values):
    bits = np.asarray(values, dtype=np.uint8)
    packed = np.packbits(bits, bitorder='little').tobytes()
    return packed + bytes((-len(packed)) % 8)


def tiny_fst(use_final=True):
    """Four states: root, BOS, a, b. External labels 0,2,4 are present.

    Root a:1, b:2; BOS a:0.5; a b:0.25. Missing external1 is forbidden,
    missing3 costs5. Epsilon backoffs BOS:1.5, a:0.75, b:0.25. Only root
    (cost1) and b (cost0.5) have literal finals. Compact labels are0,1,2.
    """
    body = struct.pack('<QQQ', 4, 4, 2)
    body += bitmap([1, 0, 1, 1, 1, 0, 0, 0, 0])
    body += bitmap([0, 1, 1, 0, 1, 0, 1, 0, 0])
    body += bitmap([1, 0, 0, 1])
    body += struct.pack('<5H', 65535, 0, 1, 2, 0)
    body += struct.pack('<4H', 1, 2, 1, 2)
    body += bytes([254, 6, 3, 1, 0])  # backoffs
    body += bytes([4, 2])            # finals
    body += bytes([4, 8, 2, 1, 0])   # futures + guard
    header = struct.pack('<I', 0x7EB2FDD6)
    for value in (b'compact_lm', b'standard'):
        header += struct.pack('<i', len(value)) + value
    # Writer's count includes the trailing fallback guard (10 vs actual9).
    header += struct.pack('<iiQqqq', 2, 4, 0, 1, 4, 10)
    header += struct.pack('<IIBfQ', 5, 3, use_final, .25, len(body))
    header += bytes((-len(header)) % 16)
    result = header + body
    result += bytes((-len(result)) % 8)
    result += bitmap([1, 0, 1, 0, 1])
    result += bytes([254, 20, 254])
    return result


def english_symbols():
    raw = (ROOT / 'evidence/en_us.recospec.raw.txt').read_text()
    quoted = re.search(r'^        4: ("<epsilon>.*")$', raw, re.M)[1]
    text = ast.literal_eval('b' + quoted).decode('utf-8')
    return {int(label): symbol for symbol, label in
            (line.rsplit('\t', 1) for line in text.splitlines())}


class CompactFstTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'tiny.fst'
        self.path.write_bytes(tiny_fst())

    def load(self, data=None):
        if data is not None:
            self.path.write_bytes(data)
        fst = CompactFst(self.path)
        self.addCleanup(fst.close)
        return fst

    def test_header_and_literal_arcs(self):
        fst = self.load()
        self.assertEqual(fst.header.byte_size, 70)
        self.assertEqual(fst.storage_offset, 96)
        self.assertEqual((fst.start_state, fst.num_states, fst.num_arcs), (1, 4, 9))
        self.assertEqual(list(fst.arcs(0)), [Arc(1, 1, math.inf, 0), Arc(2, 2, 1., 2),
                                            Arc(3, 3, 5., 0), Arc(4, 4, 2., 3)])
        self.assertEqual(list(fst.arcs(1)), [Arc(0, 0, 1.5, 0), Arc(2, 2, .5, 2)])
        self.assertEqual(list(fst.arcs(2)), [Arc(0, 0, .75, 0), Arc(4, 4, .25, 3)])
        self.assertEqual(list(fst.arcs(3)), [Arc(0, 0, .25, 0)])
        self.assertEqual([fst.final(s) for s in range(4)], [1., math.inf, math.inf, .5])

    def test_failure_backoff_and_final_scoring(self):
        lm = BackoffLanguageModel(self.load())
        self.assertEqual(lm.advance(lm.start(), 2), (2, .5))
        self.assertEqual(lm.advance(lm.start(), 4), (3, 3.5))
        self.assertEqual(lm.advance(2, 2), (2, 1.75))
        self.assertEqual(lm.advance(1, 3), (0, 6.5))
        self.assertIsNone(lm.advance(1, 1))
        self.assertEqual(lm.score([2, 4]), 1.25)
        self.assertEqual(lm.score([2]), 2.25)
        self.assertEqual(lm.score([]), 2.5)
        for labels in ([1], [0], [5], [-1]):
            self.assertEqual(lm.score(labels), math.inf)

    def test_final_flag_and_quantization_sentinels(self):
        fst = self.load(tiny_fst(False))
        self.assertEqual([fst.final(s) for s in range(4)], [0.] * 4)
        self.assertEqual(fst._weights[253], 63.25)
        self.assertEqual(fst._weights[254], math.inf)
        self.assertEqual(fst._weights[255], 63.75)

    def test_full_validation(self):
        report = self.load().validate(5)
        self.assertEqual(report, dict(states=4, arcs=9, header_arcs=10, header_arc_surplus=1,
                                     finals=2, finite_arcs=8, min_label=0, max_label=4,
                                     reachable_states=4, reachable_fraction=1.0))
        with self.assertRaisesRegex(FstFormatError, 'external label out of range'):
            self.load().validate(4)

    def test_bad_headers_and_truncations(self):
        original = tiny_fst()
        cases = [b'', original[:3], original[:69], original[:96], original[:-1], original + b'x']
        for position, value in [(0, 0), (8, ord('x')), (30, 3), (34, 0),
                                (54, 5), (62, 9), (78, 2), (79, 0), (83, 0)]:
            bad = bytearray(original)
            if position in (54, 62, 83):
                struct.pack_into('<Q', bad, position, value)
            elif position == 79:
                struct.pack_into('<f', bad, position, value)
            else:
                bad[position] = value
            cases.append(bytes(bad))
        for data in cases:
            with self.subTest(length=len(data), head=data[:10]):
                with self.assertRaises(FstFormatError):
                    self.load(data)

    def test_bad_topology_and_label_data(self):
        original = tiny_fst()
        # Each mutation breaks a separate count, ordering or range invariant.
        for position, value in [(120, 0), (127, 1), (128, 0), (136, 0),
                                (144, 0), (146, 1), (150, 1), (154, 0), (156, 9)]:
            bad = bytearray(original)
            bad[position] = value
            with self.subTest(position=position):
                with self.assertRaises(FstFormatError):
                    self.load(bytes(bad))

    def test_state_bounds_and_close(self):
        fst = self.load()
        for state in (-1, 4, .5):
            with self.assertRaises(IndexError):
                list(fst.arcs(state))
            with self.assertRaises(IndexError):
                fst.final(state)
        fst.close()
        fst.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            fst.final(0)


@unittest.skipUnless(MODEL.exists(), 'fetch en_us_20191208_compact_fst_zip first')
class EnglishFstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fst = CompactFst(MODEL)
        cls.lm = BackoffLanguageModel(cls.fst)
        cls.labels = {text: label for label, text in english_symbols().items()}

    @classmethod
    def tearDownClass(cls):
        cls.fst.close()

    def test_shipped_header_and_root(self):
        f = self.fst
        self.assertEqual((f.header.fst_type, f.header.arc_type, f.header.version),
                         ('compact_lm', 'standard', 2))
        self.assertEqual((f.num_states, f.num_futures, f.num_finals),
                         (1921076, 4787084, 115291))
        self.assertEqual(f.header.num_arcs, 6708179)
        self.assertEqual(f.num_arcs, 6708178)
        root = list(f.arcs(0))
        self.assertEqual([a.ilabel for a in root], list(range(1, 315)))
        self.assertEqual(root[0], Arc(1, 1, math.inf, 0))
        self.assertEqual(root[283], Arc(284, 284, 22.223331451416016, 0))

    def test_common_words_have_lower_cost_than_nonword(self):
        expected = {'the': 10.628549791872501, 'and': 8.608246713876724,
                    'hello': 13.615084320306778, 'qxzqxz': 43.216912508010864}
        costs = {word: self.lm.score(self.labels[c] for c in word) for word in expected}
        for word, cost in costs.items():
            self.assertTrue(math.isfinite(cost))
            self.assertAlmostEqual(cost, expected[word], places=6)
        self.assertGreater(costs['qxzqxz'], max(costs[w] for w in ('the', 'and', 'hello')) + 20)

    @unittest.skipUnless(os.environ.get('MLKIT_VALIDATE_FST') == '1', 'slow exhaustive graph validation')
    def test_every_arc_and_reachability(self):
        result = self.fst.validate(317)
        self.assertEqual(result['arcs'], 6708178)
        self.assertEqual(result['header_arc_surplus'], 1)
        self.assertEqual(result['reachable_states'], self.fst.num_states)


if __name__ == '__main__':
    unittest.main()
