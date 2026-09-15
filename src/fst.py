"""Read Google's ``compact_lm`` v2 OpenFst acceptors without native libraries.

This is NOT OpenFst's generic CompactFst format. It wraps an NGramFst LOUDS
trie with uint16 labels, uint8 quantized tropical weights and a label bitmap.
The LOUDS algorithms follow the format documented in OpenFst's
src/include/fst/extensions/ngram/ngram-fst.h (Google, Apache-2.0). The outer
layout and quantization were checked against libdigitalink's CompactLmFst.

All multibyte fields are little endian. The variable-length OpenFst header
is followed by <IIBfQ: label bitmap length, fallback weight count, use-final flag,
quantum, embedded storage bytes. Storage begins on a 16-byte boundary:
  uint64 states, futures (non-backoff arcs), finals;
  context/future/final bitmaps (uint64 padded);
  uint16 context labels[states+1], future labels[futures];
  uint8 backoffs[states+1], finals[finals], futures[futures+1].
After alignment to 8 bytes: label bitmap, then uint8 unigram weights for
missing labels plus a sentinel. Bitmap select1 maps compact labels to external
FST labels.
254 means tropical infinity; other bytes multiply the float32 quantum.
The writer includes the unused fallback sentinel in header.num_arcs; num_arcs
on this reader is the actual arc count (header.num_arcs - 1).

The file stays mmap-backed. Three uint32 LOUDS indices plus a sparse final
index take about 24 MB for English; arcs are never Python-object-cached.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import mmap
from pathlib import Path
import struct
from typing import Iterable, Iterator, NamedTuple

import numpy as np


class FstFormatError(ValueError):
    """Malformed or unsupported binary FST."""


class Arc(NamedTuple):
    ilabel: int
    olabel: int
    weight: float
    nextstate: int


@dataclass(frozen=True)
class FstHeader:
    fst_type: str
    arc_type: str
    version: int
    flags: int
    properties: int
    start: int
    num_states: int
    num_arcs: int
    byte_size: int


class CompactFst:
    """Lazy arc access to a shipped compact_lm v2 tropical acceptor.

    ``arcs`` exposes literal epsilon backoff arcs. For language-model scoring,
    use ``BackoffLanguageModel``: backoff is a failure transition, taken only
    when the requested label (or final weight) is absent at the current state.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mapped_names: list[str] = []
        self._map = None
        with self.path.open('rb') as stream:
            if stream.seek(0, 2) == 0:
                raise FstFormatError('empty FST')
            self._map = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._read()
        except Exception:
            self.close()
            raise

    def _unpack(self, fmt: str, offset: int):
        if offset + struct.calcsize(fmt) > len(self._map):
            raise FstFormatError(f'truncated FST at byte {offset}')
        return struct.unpack_from(fmt, self._map, offset)

    def _array(self, name: str, dtype: str, count: int, offset: int):
        size = np.dtype(dtype).itemsize * count
        if count < 0 or offset + size > len(self._map):
            raise FstFormatError(f'truncated {name} at byte {offset}')
        array = np.frombuffer(self._map, dtype=dtype, count=count, offset=offset)
        setattr(self, name, array)
        self._mapped_names.append(name)
        return offset + size

    def _bitmap(self, offset: int, count: int):
        size = ((count + 63) // 64) * 8
        if offset + size > len(self._map):
            raise FstFormatError(f'truncated bitmap at byte {offset}')
        bits = np.unpackbits(np.frombuffer(self._map, 'u1', size, offset),
                             bitorder='little')
        if bits[count:].any():
            raise FstFormatError('nonzero bitmap padding')
        return bits[:count], offset + size

    def _read(self):
        if self._unpack('<I', 0)[0] != 0x7EB2FDD6:
            raise FstFormatError('not an OpenFst binary (bad magic)')
        offset = 4
        strings = []
        for _ in range(2):
            size, = self._unpack('<i', offset)
            offset += 4
            if size < 0 or offset + size > len(self._map):
                raise FstFormatError('invalid FST type string length')
            try:
                strings.append(self._map[offset:offset + size].decode('ascii'))
            except UnicodeDecodeError as error:
                raise FstFormatError('non-ASCII FST type') from error
            offset += size
        fields = self._unpack('<iiQqqq', offset)
        offset += 40
        self.header = FstHeader(*strings, *fields, offset)
        h = self.header
        if (h.fst_type, h.arc_type, h.version, h.flags) != (
                'compact_lm', 'standard', 2, 4):
            raise FstFormatError(f'unsupported FST header: {h}')
        if not 1 < h.num_states < 2**32 or h.start != 1:
            raise FstFormatError('invalid compact LM states/start')
        self.start_state = h.start
        self.num_states = h.num_states
        self.num_arcs = h.num_arcs - 1
        # Google's writer counts the unused fallback sentinel as an arc.
        # Runtime CompactLmFst::NumArcs(0) is num_labels - 1.
        (self.num_labels, missing, use_final, self.quantum,
         storage_size) = self._unpack('<IIBfQ', offset)
        if use_final not in (0, 1):
            raise FstFormatError('invalid use-final flag')
        self.use_final_weights = bool(use_final)
        if not 0 < self.num_labels <= 65536 or missing > self.num_labels:
            raise FstFormatError('invalid label counts')
        if not math.isfinite(self.quantum) or self.quantum <= 0:
            raise FstFormatError('invalid weight quantum')
        offset = (offset + 21 + 15) & ~15
        self.storage_offset = offset
        n, m, f = self._unpack('<QQQ', offset)
        if (n != h.num_states or m + n - 1 + missing != h.num_arcs or
                f > n or m + n + 1 >= 2**32):
            raise FstFormatError('header and embedded state/arc counts disagree')
        self.num_futures, self.num_finals = m, f
        offset += 24
        context, offset = self._bitmap(offset, 2 * n + 1)
        ones = np.flatnonzero(context).astype(np.uint32)
        zeros = np.flatnonzero(context == 0).astype(np.uint32)
        if len(ones) != n or len(zeros) != n + 1 or zeros[0] != 1:
            raise FstFormatError('invalid context LOUDS bitmap')
        self._child_starts = zeros - np.arange(n + 1, dtype=np.uint32)
        self._parents = ones - np.arange(n, dtype=np.uint32) - np.uint32(1)
        self._parents[0] = 0
        if np.any(self._parents[1:] >= np.arange(1, n, dtype=np.uint32)):
            raise FstFormatError('cyclic or disconnected context trie')
        del context, ones, zeros
        future, offset = self._bitmap(offset, m + n + 1)
        zeros = np.flatnonzero(future == 0).astype(np.uint32)
        if len(zeros) != n + 1 or zeros[0] != 0:
            raise FstFormatError('invalid future LOUDS bitmap')
        self._future_starts = zeros - np.arange(n + 1, dtype=np.uint32)
        if self._future_starts[-1] != m:
            raise FstFormatError('future count mismatch')
        del future, zeros
        final, offset = self._bitmap(offset, n)
        self._final_states = np.flatnonzero(final).astype(np.uint32)
        if len(self._final_states) != f:
            raise FstFormatError('final count mismatch')
        del final
        for name, dtype, count in (
            ('_context_words', '<u2', n + 1), ('_future_words', '<u2', m),
            ('_backoffs', 'u1', n + 1), ('_final_weights', 'u1', f),
            ('_future_weights', 'u1', m + 1),
        ):
            offset = self._array(name, dtype, count, offset)
        if offset != self.storage_offset + storage_size:
            raise FstFormatError('embedded storage size mismatch')
        offset = (offset + 7) & ~7
        labels, offset = self._bitmap(offset, self.num_labels)
        self._external = np.flatnonzero(labels).astype(np.uint16)
        self._missing_labels = np.flatnonzero(labels == 0).astype(np.uint16)
        if len(self._missing_labels) + 1 != missing or not labels[0]:
            raise FstFormatError('label bitmap count mismatch')
        self._internal = np.full(self.num_labels, -1, dtype=np.int32)
        self._internal[self._external] = np.arange(len(self._external))
        offset = self._array('_missing_weights', 'u1', missing, offset)
        if offset != len(self._map):
            raise FstFormatError('unexpected trailing bytes')
        # Cache only the 256 possible float32 dequantizations, not arc objects.
        self._weights = (np.arange(256, dtype=np.float32) *
                         np.float32(self.quantum)).astype(float)
        self._weights[254] = math.inf
        label_error = self._label_error()
        if label_error is not None:
            raise FstFormatError(label_error)

    def _label_error(self):
        # Return errors so traceback frames cannot keep mmap slices exported
        # while the constructor closes a malformed file.
        if self._context_words[0] != 65535 or self._context_words[1] != 0:
            return 'invalid root/BOS context labels'
        for words, starts in ((self._context_words[1:self.num_states], self._child_starts - 1),
                              (self._future_words, self._future_starts)):
            if len(words) and int(words.max()) >= len(self._external):
                return 'compact label out of range'
            bad = np.flatnonzero(words[1:] <= words[:-1]) + 1
            if not np.isin(bad, starts).all():
                return 'arc/context labels are not strictly sorted'
        if np.any(self._future_words == 0):
            return 'unexpected explicit epsilon future'

    def _state(self, state: int):
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.num_states:
            raise IndexError(f'FST state out of range: {state}')
        if self._map is None:
            raise ValueError('FST is closed')

    def _context(self, state: int) -> list[int]:
        result = []
        while state:
            result.append(int(self._context_words[state]))
            state = int(self._parents[state])
        return result

    def _transition(self, context: list[int], label: int) -> int:
        state = 0
        for word in (label, *reversed(context)):
            lo, hi = map(int, self._child_starts[state:state + 2])
            i = bisect_left(self._context_words, word, lo, hi)
            if i == hi or self._context_words[i] != word:
                break
            state = i
        return state

    def final(self, state: int) -> float:
        """Literal tropical final weight (infinity means nonfinal)."""
        self._state(state)
        if not self.use_final_weights:
            return 0.0
        i = bisect_left(self._final_states, state)
        if i == len(self._final_states) or self._final_states[i] != state:
            return math.inf
        return float(self._weights[self._final_weights[i]])

    def arcs(self, state: int) -> Iterator[Arc]:
        """Iterate sorted external-label arcs, including epsilon backoff."""
        self._state(state)
        if state:
            yield Arc(0, 0, float(self._weights[self._backoffs[state]]),
                      int(self._parents[state]))
        context = self._context(state)
        lo, hi = map(int, self._future_starts[state:state + 2])
        extra = 0
        extra_count = len(self._missing_labels) if state == 0 else 0
        for i in range(lo, hi):
            compact = int(self._future_words[i])
            label = int(self._external[compact])
            while extra < extra_count and self._missing_labels[extra] < label:
                missing = int(self._missing_labels[extra])
                yield Arc(missing, missing, float(self._weights[self._missing_weights[extra]]), 0)
                extra += 1
            yield Arc(label, label, float(self._weights[self._future_weights[i]]),
                      self._transition(context, compact))
        while extra < extra_count:
            missing = int(self._missing_labels[extra])
            yield Arc(missing, missing, float(self._weights[self._missing_weights[extra]]), 0)
            extra += 1

    def _direct(self, state: int, label: int) -> tuple[int, float] | None:
        if not 0 < label < self.num_labels:
            return None
        compact = int(self._internal[label])
        if compact < 0:
            if state != 0:
                return None
            i = bisect_left(self._missing_labels, label)
            return 0, float(self._weights[self._missing_weights[i]])
        lo, hi = map(int, self._future_starts[state:state + 2])
        i = bisect_left(self._future_words, compact, lo, hi)
        if i == hi or self._future_words[i] != compact:
            return None
        return (self._transition(self._context(state), compact),
                float(self._weights[self._future_weights[i]]))

    def validate(self, symbol_count: int = 317) -> dict[str, int | float]:
        """Enumerate every arc and traverse the start component (slow, opt-in).

        Temporary packed adjacency arrays cost ~42 MB for English. They are
        discarded on return; normal decoding only needs lazy arc access.
        """
        targets = np.empty(self.num_arcs, dtype=np.uint32)
        offsets = np.empty(self.num_states + 1, dtype=np.uint32)
        count = 0
        finite_arcs = 0
        min_label, max_label = symbol_count, 0
        for state in range(self.num_states):
            offsets[state] = count
            for arc in self.arcs(state):
                if not 0 <= arc.nextstate < self.num_states:
                    raise FstFormatError('arc target out of range')
                if not (0 <= arc.ilabel < symbol_count and 0 <= arc.olabel < symbol_count):
                    raise FstFormatError('external label out of range')
                if math.isnan(arc.weight) or arc.weight < 0:
                    raise FstFormatError('invalid arc weight')
                if count >= self.num_arcs:
                    raise FstFormatError('too many arcs')
                targets[count] = arc.nextstate
                count += 1
                finite_arcs += math.isfinite(arc.weight)
                min_label = min(min_label, arc.ilabel)
                max_label = max(max_label, arc.ilabel)
        offsets[-1] = count
        if count != self.num_arcs:
            raise FstFormatError('arc count mismatch')
        seen = np.zeros(self.num_states, dtype=bool)
        queue = np.empty(self.num_states, dtype=np.uint32)
        queue[0] = self.start_state
        seen[self.start_state] = True
        head, tail = 0, 1
        while head < tail:
            state = int(queue[head])
            head += 1
            for target in targets[offsets[state]:offsets[state + 1]]:
                if not seen[target]:
                    seen[target] = True
                    queue[tail] = target
                    tail += 1
        return dict(states=self.num_states, arcs=count, header_arcs=self.header.num_arcs,
                    header_arc_surplus=self.header.num_arcs - count, finals=self.num_finals,
                    finite_arcs=finite_arcs, min_label=min_label, max_label=max_label,
                    reachable_states=tail, reachable_fraction=tail / self.num_states)

    def close(self):
        """Release all mmap views; copied topology indices remain ordinary arrays."""
        for name in self._mapped_names:
            if hasattr(self, name):
                delattr(self, name)
        self._mapped_names.clear()
        if self._map is not None:
            self._map.close()
            self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BackoffLanguageModel:
    """Deterministic ngram failure matcher for CTC shallow fusion.

    State 1 already encodes BOS; no acoustic <S> label is consumed. Final
    weights encode EOS. Epsilon backoff arcs are taken only on a missing label,
    avoiding inappropriate lower-order alternatives to explicit ngram scores.
    """

    def __init__(self, fst: CompactFst):
        self.fst = fst

    def start(self) -> int:
        return self.fst.start_state

    def advance(self, state: int, ilabel: int) -> tuple[int, float] | None:
        self.fst._state(state)
        cost = 0.0
        while True:
            match = self.fst._direct(state, ilabel)
            if match is not None:
                target, weight = match
                cost += weight
                return (target, cost) if math.isfinite(cost) else None
            if state == 0:
                return None
            cost += float(self.fst._weights[self.fst._backoffs[state]])
            state = int(self.fst._parents[state])

    def finish(self, state: int) -> float:
        self.fst._state(state)
        cost = 0.0
        while True:
            final = self.fst.final(state)
            if math.isfinite(final) or state == 0:
                return cost + final
            cost += float(self.fst._weights[self.fst._backoffs[state]])
            state = int(self.fst._parents[state])

    def score(self, labels: Iterable[int]) -> float:
        """Negative log LM weight for a complete external-label sequence."""
        state, cost = self.start(), 0.0
        for label in labels:
            match = self.advance(state, label)
            if match is None:
                return math.inf
            state, weight = match
            cost += weight
        return cost + self.finish(state)
