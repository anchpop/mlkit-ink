//! Google's `compact_lm` v2 n-gram acceptors.
//!
//! This is NOT OpenFst's generic `CompactFst`. It wraps an `NGramFst` LOUDS
//! trie with uint16 labels, uint8-quantized tropical weights and a label
//! bitmap. The LOUDS algorithms follow the format documented in OpenFst's
//! `src/include/fst/extensions/ngram/ngram-fst.h` (Google, Apache-2.0); the
//! outer layout and quantization were checked against `libdigitalink`'s
//! `CompactLmFst`. English is 1,921,076 states and 6,708,178 arcs.
//!
//! Payload arrays borrow the input bytes; only compact topology/label indices
//! and the 256 dequantized weights are owned. No alignment, native endianness,
//! or filesystem support is required.

use alloc::vec::Vec;

use crate::decoder::LanguageModel;
use crate::error::Result;

/// 254 is the quantized tropical infinity; other bytes multiply the quantum.
pub const QUANTIZED_INFINITY: u8 = 254;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    pub ilabel: u32,
    pub olabel: u32,
    pub weight: f64,
    pub nextstate: u32,
}

/// A checked, little-endian view, including on unaligned input buffers.
#[derive(Debug, Clone, Copy)]
struct Words<'a>(&'a [u8]);

impl Words<'_> {
    fn get(self, index: usize) -> u32 {
        let i = index * 2;
        u16::from_le_bytes([self.0[i], self.0[i + 1]]) as u32
    }

    fn find(self, label: u32, mut lo: usize, hi: usize) -> Option<usize> {
        let end = hi;
        let mut hi = hi;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(mid) < label {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < end && self.get(lo) == label).then_some(lo)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| err!(Format, "FST byte range overflows at byte {}", self.offset))?;
        let bytes = self.data.get(self.offset..end).ok_or_else(|| {
            err!(
                Format,
                "truncated FST at byte {} (need {count} bytes)",
                self.offset
            )
        })?;
        self.offset = end;
        Ok(bytes)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        bytes.copy_from_slice(self.take(N)?);
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn string(&mut self) -> Result<&'a [u8]> {
        let size = i32::from_le_bytes(self.fixed()?);
        ensure!(size >= 0, Format, "invalid FST type string length {size}");
        let bytes = self.take(size as usize)?;
        ensure!(bytes.is_ascii(), Format, "non-ASCII FST type");
        Ok(bytes)
    }

    fn align(&mut self, alignment: usize) -> Result<()> {
        self.take((alignment - self.offset % alignment) % alignment)?;
        Ok(())
    }

    fn words(&mut self, count: usize) -> Result<Words<'a>> {
        let bytes = count.checked_mul(2).ok_or_else(|| {
            err!(
                Format,
                "FST label array size overflows at byte {}",
                self.offset
            )
        })?;
        Ok(Words(self.take(bytes)?))
    }

    fn bitmap(&mut self, count: u64) -> Result<Bitmap<'a>> {
        let size = usize::try_from(count.div_ceil(64) * 8)
            .map_err(|_| err!(Format, "FST bitmap size exceeds address space"))?;
        let bytes = self.take(size)?;
        // Keep padded bit positions in u64: the last padding bit can exceed
        // usize::MAX on a 32-bit target even when the byte slice fits.
        for bit in count..count.div_ceil(64) * 64 {
            ensure!(
                bytes[(bit / 8) as usize] & (1 << (bit % 8)) == 0,
                Format,
                "nonzero bitmap padding"
            );
        }
        let count = usize::try_from(count)
            .map_err(|_| err!(Format, "FST bitmap bit count exceeds address space"))?;
        Ok(Bitmap { bytes, count })
    }
}

struct Bitmap<'a> {
    bytes: &'a [u8],
    count: usize,
}

impl Bitmap<'_> {
    fn get(&self, bit: usize) -> bool {
        self.bytes[bit / 8] & (1 << (bit % 8)) != 0
    }
}

/// Lazy arc access to a shipped acceptor.
///
/// `arcs` exposes the literal epsilon backoff arcs. For scoring use
/// [`BackoffLm`], where backoff is a *failure* transition taken only when the
/// requested label is genuinely absent.
#[derive(Debug)]
pub struct CompactFst<'a> {
    num_arcs: u64,
    num_labels: u32,
    quantum: f32,
    use_final_weights: bool,
    child_starts: Vec<u32>,
    parents: Vec<u32>,
    future_starts: Vec<u32>,
    final_states: Vec<u32>,
    context_words: Words<'a>,
    future_words: Words<'a>,
    backoffs: &'a [u8],
    final_weights: &'a [u8],
    future_weights: &'a [u8],
    missing_weights: &'a [u8],
    external: Vec<u32>,
    internal: Vec<u32>,
    missing_labels: Vec<u32>,
    weights: [f64; 256],
}

impl<'a> CompactFst<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut reader = Reader { data, offset: 0 };
        ensure!(
            reader.u32()? == 0x7eb2_fdd6,
            Format,
            "not an OpenFst binary (bad magic)"
        );
        let fst_type = reader.string()?;
        let arc_type = reader.string()?;
        let version = reader.u32()?;
        let flags = reader.u32()?;
        let _properties = reader.u64()?;
        let start = reader.u64()?;
        let num_states = reader.u64()?;
        let header_arcs = reader.u64()?;
        ensure!(
            (fst_type, arc_type, version, flags)
                == (b"compact_lm".as_slice(), b"standard".as_slice(), 2, 4),
            Format,
            "unsupported FST header: type {fst_type:?}, arcs {arc_type:?}, version {version}, flags {flags}"
        );
        ensure!(
            num_states > 1 && num_states < 1 << 32 && start == 1,
            Format,
            "invalid compact LM states/start: {num_states}/{start}"
        );
        let num_labels = reader.u32()?;
        let missing = reader.u32()?;
        let use_final = reader.take(1)?[0];
        let quantum = f32::from_le_bytes(reader.fixed()?);
        let storage_size = reader.u64()?;
        ensure!(use_final <= 1, Format, "invalid use-final flag {use_final}");
        ensure!(
            num_labels > 0 && num_labels <= 65536 && missing <= num_labels,
            Format,
            "invalid label counts: {num_labels} labels, {missing} fallback weights"
        );
        ensure!(
            quantum.is_finite() && quantum > 0.0,
            Format,
            "invalid weight quantum {quantum}"
        );
        reader.align(16)?;
        let storage_offset = reader.offset;
        let n = reader.u64()?;
        let m = reader.u64()?;
        let f = reader.u64()?;
        // Bound before adding, including hostile u64 counts. This also keeps
        // every derived state/arc index representable by u32.
        ensure!(
            n == num_states
                && m < 1 << 32
                && f <= n
                && m + n + 1 < 1 << 32
                && m + n - 1 + u64::from(missing) == header_arcs,
            Format,
            "header and embedded state/arc counts disagree"
        );
        let context = reader.bitmap(2 * n + 1)?;
        let future = reader.bitmap(m + n + 1)?;
        let final_bits = reader.bitmap(n)?;
        let n = n as usize;
        let m = m as usize;
        let f = f as usize;
        // Check all payload bounds before allocating indices from file counts.
        let context_words = reader.words(n + 1)?;
        let future_words = reader.words(m)?;
        let backoffs = reader.take(n + 1)?;
        let final_weights = reader.take(f)?;
        let future_weights = reader.take(m + 1)?;
        ensure!(
            (reader.offset - storage_offset) as u64 == storage_size,
            Format,
            "embedded storage size mismatch"
        );
        reader.align(8)?;
        let labels = reader.bitmap(u64::from(num_labels))?;
        let missing_weights = reader.take(missing as usize)?;
        ensure!(
            reader.offset == data.len(),
            Format,
            "unexpected trailing bytes at byte {}",
            reader.offset
        );

        let mut child_starts = Vec::with_capacity(n + 1);
        let mut parents = Vec::with_capacity(n);
        for bit in 0..context.count {
            if context.get(bit) {
                let state = parents.len();
                ensure!(state < n, Format, "invalid context LOUDS bitmap");
                if state == 0 {
                    ensure!(bit == 0, Format, "invalid context LOUDS bitmap");
                    parents.push(0);
                } else {
                    let parent = bit
                        .checked_sub(state + 1)
                        .ok_or_else(|| err!(Format, "invalid context LOUDS bitmap"))?;
                    ensure!(
                        parent < state,
                        Format,
                        "cyclic or disconnected context trie"
                    );
                    parents.push(parent as u32);
                }
            } else {
                ensure!(
                    child_starts.len() <= n && (!child_starts.is_empty() || bit == 1),
                    Format,
                    "invalid context LOUDS bitmap"
                );
                child_starts.push((bit - child_starts.len()) as u32);
            }
        }
        ensure!(
            parents.len() == n && child_starts.len() == n + 1,
            Format,
            "invalid context LOUDS bitmap"
        );
        let mut future_starts = Vec::with_capacity(n + 1);
        for bit in 0..future.count {
            if !future.get(bit) {
                ensure!(
                    future_starts.len() <= n && (!future_starts.is_empty() || bit == 0),
                    Format,
                    "invalid future LOUDS bitmap"
                );
                future_starts.push((bit - future_starts.len()) as u32);
            }
        }
        ensure!(
            future_starts.len() == n + 1,
            Format,
            "invalid future LOUDS bitmap"
        );
        ensure!(
            future_starts[n] as usize == m,
            Format,
            "future count mismatch"
        );
        let mut final_states = Vec::with_capacity(f);
        for state in 0..n {
            if final_bits.get(state) {
                final_states.push(state as u32);
            }
        }
        ensure!(final_states.len() == f, Format, "final count mismatch");
        let mut external = Vec::new();
        let mut internal = Vec::with_capacity(num_labels as usize);
        let mut missing_labels = Vec::new();
        for label in 0..num_labels {
            if labels.get(label as usize) {
                internal.push(external.len() as u32);
                external.push(label);
            } else {
                internal.push(u32::MAX);
                missing_labels.push(label);
            }
        }
        ensure!(
            labels.get(0) && missing_labels.len() + 1 == missing as usize,
            Format,
            "label bitmap count mismatch"
        );
        let weights = core::array::from_fn(|i| {
            if i == QUANTIZED_INFINITY as usize {
                f64::INFINITY
            } else {
                // Native dequantization multiplies in f32, THEN widens.
                f64::from(i as f32 * quantum)
            }
        });
        let fst = CompactFst {
            num_arcs: header_arcs - 1,
            num_labels,
            quantum,
            use_final_weights: use_final != 0,
            child_starts,
            parents,
            future_starts,
            final_states,
            context_words,
            future_words,
            backoffs,
            final_weights,
            future_weights,
            missing_weights,
            external,
            internal,
            missing_labels,
            weights,
        };
        fst.validate_labels()?;
        Ok(fst)
    }

    fn validate_labels(&self) -> Result<()> {
        ensure!(
            self.context_words.get(0) == 65535 && self.context_words.get(1) == 0,
            Format,
            "invalid root/BOS context labels"
        );
        for (words, starts) in [
            (self.context_words, &self.child_starts),
            (self.future_words, &self.future_starts),
        ] {
            for range in starts.windows(2) {
                let lo = range[0] as usize;
                let hi = range[1] as usize;
                for i in lo..hi {
                    let label = words.get(i);
                    ensure!(
                        (label as usize) < self.external.len(),
                        Format,
                        "compact label out of range"
                    );
                    ensure!(
                        i == lo || words.get(i - 1) < label,
                        Format,
                        "arc/context labels are not strictly sorted"
                    );
                }
            }
        }
        for i in 0..self.num_futures() as usize {
            ensure!(
                self.future_words.get(i) != 0,
                Format,
                "unexpected explicit epsilon future"
            );
        }
        Ok(())
    }

    pub fn start(&self) -> u32 {
        1
    }

    pub fn num_states(&self) -> u32 {
        self.parents.len() as u32
    }

    /// Actual arcs, excluding the unused fallback sentinel counted by the writer.
    pub fn num_arcs(&self) -> u64 {
        self.num_arcs
    }

    pub fn num_labels(&self) -> u32 {
        self.num_labels
    }

    pub fn num_futures(&self) -> u32 {
        self.future_starts[self.parents.len()]
    }

    pub fn num_finals(&self) -> u32 {
        self.final_states.len() as u32
    }

    pub fn quantum(&self) -> f32 {
        self.quantum
    }

    pub fn use_final_weights(&self) -> bool {
        self.use_final_weights
    }

    fn state(&self, state: u32) -> Result<()> {
        ensure!(
            state < self.num_states(),
            Invalid,
            "FST state out of range: {state}"
        );
        Ok(())
    }

    fn context(&self, mut state: u32) -> Vec<u32> {
        let mut context = Vec::new();
        while state != 0 {
            context.push(self.context_words.get(state as usize));
            state = self.parents[state as usize];
        }
        context
    }

    fn transition(&self, context: &[u32], label: u32) -> u32 {
        let mut state = 0;
        for word in core::iter::once(label).chain(context.iter().rev().copied()) {
            let lo = self.child_starts[state] as usize;
            let hi = self.child_starts[state + 1] as usize;
            let Some(child) = self.context_words.find(word, lo, hi) else {
                break;
            };
            state = child;
        }
        state as u32
    }

    fn transition_from_state(&self, state: u32, label: u32) -> u32 {
        // Decoding hits this path millions of times. Shipped n-gram histories
        // fit on the stack; deeper valid tries still work via the Vec path.
        let mut context = [0; 16];
        let mut depth = 0;
        let mut ancestor = state;
        while ancestor != 0 {
            if depth == context.len() {
                return self.transition(&self.context(state), label);
            }
            context[depth] = self.context_words.get(ancestor as usize);
            depth += 1;
            ancestor = self.parents[ancestor as usize];
        }
        self.transition(&context[..depth], label)
    }

    /// Literal tropical final weight; infinity means nonfinal.
    pub fn final_weight(&self, state: u32) -> Result<f64> {
        self.state(state)?;
        Ok(self.literal_final(state))
    }

    fn literal_final(&self, state: u32) -> f64 {
        if !self.use_final_weights {
            return 0.0;
        }
        match self.final_states.binary_search(&state) {
            Ok(i) => self.weights[self.final_weights[i] as usize],
            Err(_) => f64::INFINITY,
        }
    }

    /// Iterate sorted external-label arcs, including literal epsilon backoff.
    /// State bounds are checked before returning the lazy iterator.
    pub fn arcs(&self, state: u32) -> Result<impl Iterator<Item = Arc> + '_> {
        self.state(state)?;
        Ok(Arcs {
            fst: self,
            backoff: (state != 0).then_some(state),
            context: self.context(state),
            future: self.future_starts[state as usize] as usize,
            future_end: self.future_starts[state as usize + 1] as usize,
            missing: 0,
            missing_end: if state == 0 {
                self.missing_labels.len()
            } else {
                0
            },
        })
    }

    fn direct(&self, state: u32, label: u32) -> Option<(u32, f64)> {
        if label == 0 || label >= self.num_labels {
            return None;
        }
        let compact = self.internal[label as usize];
        if compact == u32::MAX {
            if state != 0 {
                return None;
            }
            let i = self.missing_labels.binary_search(&label).ok()?;
            return Some((0, self.weights[self.missing_weights[i] as usize]));
        }
        let lo = self.future_starts[state as usize] as usize;
        let hi = self.future_starts[state as usize + 1] as usize;
        let i = self.future_words.find(compact, lo, hi)?;
        Some((
            self.transition_from_state(state, compact),
            self.weights[self.future_weights[i] as usize],
        ))
    }
}

struct Arcs<'f, 'a> {
    fst: &'f CompactFst<'a>,
    backoff: Option<u32>,
    context: Vec<u32>,
    future: usize,
    future_end: usize,
    missing: usize,
    missing_end: usize,
}

impl Iterator for Arcs<'_, '_> {
    type Item = Arc;

    fn next(&mut self) -> Option<Arc> {
        let fst = self.fst;
        if let Some(state) = self.backoff.take() {
            return Some(Arc {
                ilabel: 0,
                olabel: 0,
                weight: fst.weights[fst.backoffs[state as usize] as usize],
                nextstate: fst.parents[state as usize],
            });
        }
        let future_label = (self.future < self.future_end)
            .then(|| fst.external[fst.future_words.get(self.future) as usize]);
        if self.missing < self.missing_end
            && future_label.is_none_or(|label| fst.missing_labels[self.missing] < label)
        {
            let label = fst.missing_labels[self.missing];
            let weight = fst.weights[fst.missing_weights[self.missing] as usize];
            self.missing += 1;
            return Some(Arc {
                ilabel: label,
                olabel: label,
                weight,
                nextstate: 0,
            });
        }
        let label = future_label?;
        let compact = fst.future_words.get(self.future);
        let arc = Arc {
            ilabel: label,
            olabel: label,
            weight: fst.weights[fst.future_weights[self.future] as usize],
            nextstate: fst.transition(&self.context, compact),
        };
        self.future += 1;
        Some(arc)
    }
}

/// Deterministic n-gram failure matcher for CTC shallow fusion.
///
/// State 1 already encodes BOS, so no `<S>` label is ever consumed; final
/// weights encode EOS. Backoff arcs are taken only on a missing label, so an
/// explicit n-gram score is never undercut by a lower-order alternative.
#[derive(Debug)]
pub struct BackoffLm<'a> {
    pub fst: CompactFst<'a>,
}

impl<'a> BackoffLm<'a> {
    pub fn new(fst: CompactFst<'a>) -> Self {
        BackoffLm { fst }
    }
}

impl LanguageModel for BackoffLm<'_> {
    type State = u32;

    fn start(&self) -> u32 {
        self.fst.start()
    }

    /// Walk backoff arcs until the label matches or the unigram state rejects it.
    /// Invalid caller-supplied states reject rather than indexing the payload.
    fn advance(&self, mut state: u32, ilabel: u32) -> Option<(u32, f64)> {
        if state >= self.fst.num_states() {
            return None;
        }
        let mut cost = 0.0;
        loop {
            if let Some((target, weight)) = self.fst.direct(state, ilabel) {
                cost += weight;
                return cost.is_finite().then_some((target, cost));
            }
            if state == 0 {
                return None;
            }
            cost += self.fst.weights[self.fst.backoffs[state as usize] as usize];
            state = self.fst.parents[state as usize];
        }
    }

    /// Final weights encode EOS, also reached through backoff.
    fn finish(&self, mut state: u32) -> f64 {
        if state >= self.fst.num_states() {
            return f64::INFINITY;
        }
        let mut cost = 0.0;
        loop {
            let final_weight = self.fst.literal_final(state);
            if final_weight.is_finite() || state == 0 {
                return cost + final_weight;
            }
            cost += self.fst.weights[self.fst.backoffs[state as usize] as usize];
            state = self.fst.parents[state as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use alloc::vec;

    // Byte-for-byte equivalent to tests/test_fst.py's portable fixture. Four
    // states: root, BOS, a, b; present external labels 0, 2, 4, missing 1, 3.
    fn tiny_fst(use_final: bool) -> Vec<u8> {
        let mut body = Vec::new();
        for count in [4u64, 4, 2] {
            body.extend(count.to_le_bytes());
        }
        for bitmap in [0b0001_1101u64, 0b0101_0110, 0b1001] {
            body.extend(bitmap.to_le_bytes());
        }
        for word in [65535u16, 0, 1, 2, 0, 1, 2, 1, 2] {
            body.extend(word.to_le_bytes());
        }
        body.extend([254, 6, 3, 1, 0]); // backoffs and guard
        body.extend([4, 2]); // finals
        body.extend([4, 8, 2, 1, 0]); // futures and guard
        let mut data = 0x7eb2_fdd6u32.to_le_bytes().to_vec();
        for text in [b"compact_lm".as_slice(), b"standard".as_slice()] {
            data.extend((text.len() as i32).to_le_bytes());
            data.extend(text);
        }
        data.extend(2i32.to_le_bytes());
        data.extend(4i32.to_le_bytes());
        for field in [0u64, 1, 4, 10] {
            data.extend(field.to_le_bytes());
        }
        data.extend(5u32.to_le_bytes());
        data.extend(3u32.to_le_bytes());
        data.push(u8::from(use_final));
        data.extend(0.25f32.to_le_bytes());
        data.extend((body.len() as u64).to_le_bytes());
        data.resize(data.len().next_multiple_of(16), 0);
        data.extend(body);
        data.resize(data.len().next_multiple_of(8), 0);
        data.extend(0b10101u64.to_le_bytes());
        data.extend([254, 20, 254]); // missing labels and unused sentinel
        data
    }

    fn arc(label: u32, weight: f64, nextstate: u32) -> Arc {
        Arc {
            ilabel: label,
            olabel: label,
            weight,
            nextstate,
        }
    }

    #[test]
    fn metadata_and_literal_arcs() {
        let bytes = tiny_fst(true);
        let fst = CompactFst::parse(&bytes).unwrap();
        assert_eq!((fst.start(), fst.num_states(), fst.num_arcs()), (1, 4, 9));
        assert_eq!(
            (fst.num_labels(), fst.num_futures(), fst.num_finals()),
            (5, 4, 2)
        );
        assert_eq!(fst.quantum(), 0.25);
        assert!(fst.use_final_weights());
        let expected = [
            vec![
                arc(1, f64::INFINITY, 0),
                arc(2, 1.0, 2),
                arc(3, 5.0, 0),
                arc(4, 2.0, 3),
            ],
            vec![arc(0, 1.5, 0), arc(2, 0.5, 2)],
            vec![arc(0, 0.75, 0), arc(4, 0.25, 3)],
            vec![arc(0, 0.25, 0)],
        ];
        for (state, arcs) in expected.iter().enumerate() {
            assert_eq!(&fst.arcs(state as u32).unwrap().collect::<Vec<_>>(), arcs);
        }
        assert_eq!(
            (0..4)
                .map(|s| fst.final_weight(s).unwrap())
                .collect::<Vec<_>>(),
            [1.0, f64::INFINITY, f64::INFINITY, 0.5]
        );
        // The borrowed data need not begin at an aligned address.
        let mut unaligned = vec![0];
        unaligned.extend(bytes);
        assert_eq!(CompactFst::parse(&unaligned[1..]).unwrap().num_arcs(), 9);
    }

    #[test]
    fn failure_matching_and_eos() {
        let bytes = tiny_fst(true);
        let lm = BackoffLm::new(CompactFst::parse(&bytes).unwrap());
        assert_eq!(lm.advance(lm.start(), 2), Some((2, 0.5)));
        assert_eq!(lm.advance(1, 4), Some((3, 3.5)));
        assert_eq!(lm.advance(2, 2), Some((2, 1.75)));
        assert_eq!(lm.advance(1, 3), Some((0, 6.5)));
        for label in [0, 1, 5, u32::MAX] {
            assert_eq!(lm.advance(1, label), None);
        }
        assert_eq!(lm.finish(1), 2.5);
        assert_eq!(lm.finish(2), 1.75);
        assert_eq!(lm.finish(3), 0.5);

        // An explicit expensive/infinite n-gram must never be replaced by a
        // cheaper lower-order arc. EOS differs: infinity means back off.
        let mut bytes = tiny_fst(true);
        bytes[171] = 100; // BOS -> a costs 25, root -> a costs 1
        let lm = BackoffLm::new(CompactFst::parse(&bytes).unwrap());
        assert_eq!(lm.advance(1, 2), Some((2, 25.0)));
        bytes[171] = 254;
        let lm = BackoffLm::new(CompactFst::parse(&bytes).unwrap());
        assert_eq!(lm.advance(1, 2), None);
        bytes[171] = 2;
        bytes[163] = 254; // infinite BOS backoff forbids missing labels/EOS
        let lm = BackoffLm::new(CompactFst::parse(&bytes).unwrap());
        assert_eq!(lm.advance(1, 4), None);
        assert_eq!(lm.advance(1, 2), Some((2, 0.5)));
        assert_eq!(lm.finish(1), f64::INFINITY);
    }

    #[test]
    fn final_flag_and_float32_dequantization() {
        let mut bytes = tiny_fst(false);
        bytes[79..83].copy_from_slice(&0.1f32.to_le_bytes());
        let fst = CompactFst::parse(&bytes).unwrap();
        assert!(!fst.use_final_weights());
        for state in 0..4 {
            assert_eq!(fst.final_weight(state).unwrap(), 0.0);
        }
        assert_eq!(fst.weights[253], f64::from(253.0f32 * 0.1f32));
        assert_eq!(fst.weights[254], f64::INFINITY);
        assert_eq!(fst.weights[255], f64::from(255.0f32 * 0.1f32));
        assert_ne!(fst.weights[253], 253.0 * f64::from(0.1f32));
        assert_eq!(BackoffLm::new(fst).finish(1), 0.0);
    }

    #[test]
    fn context_order_and_longest_suffix_transition() {
        let bytes = tiny_fst(true);
        let mut fst = CompactFst::parse(&bytes).unwrap();
        let words: Vec<_> = [65535u16, 0, 1, 2, 0, 2, 1, 1, 0]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        fst.context_words = Words(&words);
        fst.parents = vec![0, 0, 0, 0, 2, 2, 3, 5];
        fst.child_starts = vec![1, 4, 4, 6, 7, 7, 8, 8, 8];
        assert_eq!(fst.context(0), []);
        assert_eq!(fst.context(6), [1, 2]);
        assert_eq!(fst.context(7), [1, 2, 1]);
        assert_eq!(fst.transition(&fst.context(6), 1), 7);
        assert_eq!(fst.transition(&fst.context(7), 2), 6);
        assert_eq!(fst.transition(&fst.context(4), 1), 2);
        assert_eq!(fst.transition(&fst.context(6), 3), 0);
        for state in 0..fst.num_states() {
            for label in 0..4 {
                assert_eq!(
                    fst.transition_from_state(state, label),
                    fst.transition(&fst.context(state), label)
                );
            }
        }
        // No n-gram order limit or recursive traversal for unusually deep
        // (but valid) tries; exercise the stack-buffer fallback explicitly.
        let words: Vec<_> = core::iter::once(65535u16)
            .chain(core::iter::once(0))
            .chain(core::iter::repeat_n(1, 32))
            .chain(core::iter::once(0))
            .flat_map(u16::to_le_bytes)
            .collect();
        fst.context_words = Words(&words);
        fst.parents = vec![0, 0];
        fst.parents.extend(1..33);
        fst.parents[2] = 0;
        fst.child_starts = vec![1, 3];
        fst.child_starts.extend(3..35);
        fst.child_starts.push(34);
        assert_eq!(fst.context(33).len(), 32);
        assert_eq!(fst.transition_from_state(33, 1), 33);
    }

    #[test]
    fn invalid_states_reject_without_panicking() {
        let bytes = tiny_fst(true);
        let lm = BackoffLm::new(CompactFst::parse(&bytes).unwrap());
        for state in [4, u32::MAX] {
            assert!(matches!(lm.fst.final_weight(state), Err(Error::Invalid(_))));
            assert!(matches!(lm.fst.arcs(state), Err(Error::Invalid(_))));
            assert_eq!(lm.advance(state, 2), None);
            assert_eq!(lm.finish(state), f64::INFINITY);
        }
    }

    fn malformed(bytes: &[u8]) {
        assert!(matches!(CompactFst::parse(bytes), Err(Error::Format(_))));
    }

    #[test]
    fn all_truncations_and_trailing_bytes_are_rejected() {
        let mut bytes = tiny_fst(true);
        for end in 0..bytes.len() {
            malformed(&bytes[..end]);
        }
        bytes.push(0);
        malformed(&bytes);
    }

    #[test]
    fn single_byte_mutations_never_panic() {
        let original = tiny_fst(true);
        for position in 0..original.len() {
            let mut bytes = original.clone();
            for value in 0..=255 {
                bytes[position] = value;
                if let Ok(fst) = CompactFst::parse(&bytes) {
                    let mut count = 0;
                    for state in 0..fst.num_states() {
                        assert!(!fst.final_weight(state).unwrap().is_nan());
                        for arc in fst.arcs(state).unwrap() {
                            assert!(arc.nextstate < fst.num_states());
                            assert!(arc.ilabel < fst.num_labels());
                            assert_eq!(arc.ilabel, arc.olabel);
                            assert!(!arc.weight.is_nan() && arc.weight >= 0.0);
                            count += 1;
                        }
                    }
                    assert_eq!(count, fst.num_arcs());
                }
            }
        }
    }

    #[test]
    fn malformed_headers_counts_topology_and_labels() {
        let original = tiny_fst(true);
        for (position, value) in [
            (0, 0),
            (8, b'x'),
            (8, 255),
            (30, 3),
            (34, 0),
            (78, 2),
            (120, 0),
            (120, 0b1101_0001),
            (120, 0b0001_1011),
            (127, 1),
            (128, 0),
            (128, 1),
            (135, 128),
            (136, 0),
            (143, 1),
            (144, 0),
            (146, 1),
            (150, 1),
            (154, 0),
            (156, 9),
            (176, 0),
            (183, 1),
        ] {
            let mut bytes = original.clone();
            bytes[position] = value;
            malformed(&bytes);
        }
        for (position, value) in [
            (54, 5u64),
            (62, 9),
            (83, 0),
            (96, u64::MAX),
            (104, u64::MAX),
            (112, 5),
            (104, 1 << 32),
        ] {
            let mut bytes = original.clone();
            bytes[position..position + 8].copy_from_slice(&value.to_le_bytes());
            malformed(&bytes);
        }
        for (position, value) in [
            (4, u32::MAX),
            (4, i32::MAX as u32),
            (70, 0),
            (70, 65537),
            (74, 0),
            (74, 6),
        ] {
            let mut bytes = original.clone();
            bytes[position..position + 4].copy_from_slice(&value.to_le_bytes());
            malformed(&bytes);
        }
        for quantum in [0.0f32, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bytes = original.clone();
            bytes[79..83].copy_from_slice(&quantum.to_le_bytes());
            malformed(&bytes);
        }
    }
}
