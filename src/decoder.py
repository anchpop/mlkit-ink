"""NumPy CTC decoding, optionally scored by a deterministic tropical-cost LM.

The default CTC blank is the final network output (C-1); other network indices
k map to FST label k+2. This corrects the initial blank-zero hypothesis. Static
native evidence: NetworkScoreCache constructor 0x4f1330 stores dimension-1 as
the special index; Score 0x50a224 excludes that index, 0x50a22a maps normal keys
by +2, and 0x50a249..0x50a261 indexes the raw tensor without reordering. The
0x4dfa4c..0x4dfaae diagnostic requires outputdim=charset.size()+1 (blank).
The raw en-US processor's 313 characters match FST labels 2..314 in order
(317 symbols altogether). This is strong native/charset evidence, NOT an
Android-oracle validation. ``[[space]]`` is rendered as a literal space.

Both blank and network-to-label mapping are independently overridable. An
explicit blank does not change the default k+2 mapping; a custom mapping never
changes the default last blank. The blank itself is not sent to the LM.
Raw decoder block 158518157.4.6.1 contains field 1=1000, used here as the requested
beam default. Other fields include 2=10.0, 4=20.0, 7=-1.9013119936, 20=20.0;
158518157.4.6 has 7=0.615223527 and 10=-1.228597641 (float32 interpretations).
Their meanings are unknown: none is assumed to be an LM weight or insertion
bonus. This implements standard CTC, not Google's unverified decoder/scorers.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from heapq import nlargest
import math
import operator
from typing import Iterator, Protocol

import numpy as np


class LanguageModel(Protocol):
    """The wrapper owns epsilon/backoff/boundary handling and determinization."""

    def start(self) -> Hashable: ...

    def advance(self, state: Hashable, ilabel: int) -> tuple[Hashable, float] | None: ...

    def finish(self, state: Hashable) -> float: ...


def _expand_class_members(text: str) -> "Iterator[str]":
    """Yield each member character of a char_class_table line.

    Plain text is a list of single codepoints, but the table also supports
    ``[[HEX]]`` and ``[[HEX-HEX]]`` inclusive Unicode scalar ranges (native parser at
    0x50b438, hex conversion base 16 via 0x6395fc, inclusive expansion 0x50b697..0x50b7d3,
    UTF-8 encode with a 4-byte/non-BMP path at 0x639956).

    75 of the 361 shipped tables use this -- every non-Latin script (Devanagari, Arabic,
    Ethiopic, Bengali, ...). Without expansion their members would be read as the literal
    characters '[', '0', 'x', ... and the whole class table would be meaningless.
    """
    i = 0
    while i < len(text):
        if text.startswith("[[", i):
            close = text.find("]]", i + 2)
            if close != -1:
                body = text[i + 2:close]
                lo, _, hi = body.partition("-")
                try:
                    start = int(lo, 16)
                    stop = int(hi, 16) if hi else start
                except ValueError:
                    start = None
                if start is not None and 0 <= start <= stop <= 0x10FFFF:
                    for cp in range(start, stop + 1):
                        yield chr(cp)
                    i = close + 2
                    continue
        yield text[i]
        i += 1


@dataclass(frozen=True)
class CharClassScorer:
    """Experimental per-token scores from explicit recospec class membership.

    The table/weights are Google's, but additive per-token interpretation and
    sign are NOT native-verified. Empty language-specific classes and weights
    such as ``lower_`` are recorded, never assigned guessed membership.
    """

    scores: Mapping[int, float]
    empty_classes: tuple[str, ...]
    unmapped_weights: tuple[str, ...]
    overrides: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def from_tables(cls, classes: Mapping[str, str], weights: Mapping[str, float],
                    symbols: Mapping[int, str]) -> "CharClassScorer":
        """Build the per-label score table, mirroring the native parser's behaviour.

        Native details this reproduces (all read out of libdigitalink.so):
          * A class line with no members is SKIPPED entirely -- that is why en-US's bare
            `upper_en_us` / `lower_en_us` lines contribute nothing, and why `upper_`/`lower_`
            are inert for that language despite carrying weights.
          * A suffixed class name is truncated to the prefix INCLUDING the underscore, so
            `upper_be <chars>` contributes its characters to class `upper_`. 163 of the 361
            shipped tables rely on this; only en-US has the suffixed lines bare.
          * Overlapping membership is LAST-WRITE-WINS, not an error. The native code logs
            "overriding character class from ... to ...". 78 of the 361 tables overlap, so
            raising here would break those languages outright.
          * A class with no weight scores 0.0 rather than failing.
          * Lookup is by the WHOLE symbol string, not a per-codepoint sum, so multi-codepoint
            symbols simply miss and fall back to `no_char_class` -- there is no Unicode
            upper/lower heuristic.
        """
        weights = {name: float(value) for name, value in weights.items()}
        if not all(math.isfinite(value) for value in weights.values()):
            raise ValueError("character-class weights must be finite")

        membership: dict[str, str] = {}
        overrides: list[tuple[str, str, str]] = []
        contributing: set[str] = set()      # classes that actually received members
        for name, characters in classes.items():
            if not characters:
                continue                      # bare line: native skips before suffix handling
            if "_" in name:
                name = name[: name.index("_") + 1]   # upper_be -> upper_
            contributing.add(name)
            for char in _expand_class_members(characters):
                if char in membership and membership[char] != name:
                    overrides.append((char, membership[char], name))
                membership[char] = name       # last write wins
        # A class with no weight contributes 0.0; so does an unlisted character whose
        # no_char_class weight is itself absent.
        scores = {
            label: weights.get(
                membership.get(" " if text == "[[space]]" else text, "no_char_class"), 0.0
            )
            for label, text in symbols.items()
        }
        # A weight is "unmapped" only if no line actually gave its class any members --
        # a bare line contributes nothing, so its weight really is inert.
        return cls(scores,
                   tuple(name for name, characters in classes.items() if not characters),
                   tuple(sorted(set(weights) - contributing - {"no_char_class"})),
                   tuple(overrides))


class CharClassRescoringLM:
    """Add explicit per-token class scores alongside a word LM, never on blanks.

    Combined cost = word_weight * word_cost - class_weight * class_value.
    Positive class_weight ADDS the listed value to beam scores; negative treats
    it as a cost. Feed this wrapper to prefix_beam_search with lm_weight=1 so
    word and class weights remain independent. Final cost is word-LM-only.
    """

    def __init__(self, word_lm: LanguageModel, scorer: CharClassScorer, *,
                 word_weight: float = 1.0, class_weight: float = 1.0):
        if not math.isfinite(word_weight) or not math.isfinite(class_weight):
            raise ValueError("word and character-class weights must be finite")
        self.word_lm = word_lm
        self.scorer = scorer
        self.word_weight = word_weight
        self.class_weight = class_weight

    def start(self):
        return self.word_lm.start()

    def advance(self, state, ilabel):
        transition = self.word_lm.advance(state, ilabel)
        if transition is None:
            return None
        next_state, cost = transition
        cost = _cost(cost)
        if cost == math.inf:
            return next_state, cost  # Rejection must survive even word_weight=0.
        return (next_state, self.word_weight * cost -
                self.class_weight * self.scorer.scores[ilabel])

    def finish(self, state):
        cost = _cost(self.word_lm.finish(state))
        return cost if cost == math.inf else self.word_weight * cost


NetToLabel = Mapping[int, int] | Sequence[int]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Normalize [T,C] or single-batch [1,T,C] logits in float64.

    Normalized log probabilities are also valid input. Raw probabilities are
    not: take their logarithm first. -inf denotes an impossible emission; each
    frame must have at least one finite entry. Empty time axes are supported.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("logits must have shape [T,C] or [1,T,C], with C > 0")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logits cannot contain NaN or +inf")
    if not np.isfinite(values).any(axis=1).all():
        raise ValueError("each frame must have at least one finite logit")
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _alphabet(class_count, symbols, net_to_label, blank):
    blank = class_count - 1 if blank is None else operator.index(blank)
    if not 0 <= blank < class_count:
        raise ValueError("blank index is outside the network alphabet")
    labels, texts = {}, {}
    for index in range(class_count):
        if index == blank:
            continue
        try:
            label = operator.index(index + 2 if net_to_label is None else net_to_label[index])
            text = symbols[label]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"missing or invalid symbol mapping for network index {index}") from exc
        if not isinstance(text, str) or not text:
            raise ValueError("nonblank symbols must be nonempty strings")
        labels[index] = label
        texts[index] = " " if text == "[[space]]" else text
    return blank, labels, texts


def greedy_decode(
    logits: np.ndarray,
    symbols: Mapping[int, str],
    *,
    net_to_label: NetToLabel | None = None,
    blank: int | None = None,
) -> list[tuple[str, float]]:
    """Return one (text, best-path log probability) after CTC collapse.

    Unlike beam search, the score is NOT the sum over every path for this text.
    Repeats are collapsed before blanks are removed, so a blank separates two
    identical characters. Ties choose the lowest network index (NumPy argmax).
    """
    log_probs = log_softmax(logits)
    blank, _, texts = _alphabet(log_probs.shape[1], symbols, net_to_label, blank)
    path = log_probs.argmax(axis=1)
    output = []
    previous = None
    for index in path:
        if index != blank and index != previous:
            output.append(texts[index])
        previous = index
    score = float(log_probs[np.arange(len(path)), path].sum())
    return [("".join(output), score)]


@dataclass
class _Prefix:
    state: Hashable
    offset: float = 0.0  # LM and insertion terms, never part of acoustic sums.
    p_blank: float = -math.inf
    p_nonblank: float = -math.inf

    @property
    def acoustic(self):
        return float(np.logaddexp(self.p_blank, self.p_nonblank))

    @property
    def score(self):
        return self.acoustic + self.offset


def _cost(value):
    value = float(value)
    if math.isnan(value) or value == -math.inf:
        raise ValueError("LM costs must be finite or +inf (rejection)")
    return value


def prefix_beam_search(
    logits: np.ndarray,
    symbols: Mapping[int, str],
    *,
    net_to_label: NetToLabel | None = None,
    blank: int | None = None,
    beam_width: int = 1000,
    nbest: int = 10,
    lm: LanguageModel | None = None,
    lm_weight: float = 1.0,
    insertion_bonus: float = 0.0,
) -> list[tuple[str, float]]:
    """Return n-best (text, score), best first, using CTC prefix beam search.

    score = log(sum of retained CTC path probabilities)
            - lm_weight * (incremental LM costs + final cost)
            + insertion_bonus * number of emitted nonblank tokens

    LM weight defaults to 1 and insertion bonus to 0; these are neutral choices,
    not recovered Google parameters. The bonus is per token, NOT per word or
    frame. LM rejection applies even when lm_weight=0; use lm=None to disable it.
    The LM state/cost must be deterministic for a collapsed label prefix.

    Acoustic blank/nonblank masses merge with logaddexp independently of the LM.
    Only true prefix extensions consume labels/add costs; repeated frames and
    blanks do not. Pruning uses combined scores; LM final costs are applied
    after the last frame's pruning. Thus scores are approximate with a finite
    beam, and rejected finalists can yield fewer than nbest results (even []).
    No extra token/top-k pruning is performed. Distinct label sequences that
    render identically are merged for the returned text n-best.
    """
    beam_width, nbest = operator.index(beam_width), operator.index(nbest)
    if beam_width <= 0 or nbest <= 0:
        raise ValueError("beam_width and nbest must be positive")
    if not math.isfinite(lm_weight) or not math.isfinite(insertion_bonus):
        raise ValueError("lm_weight and insertion_bonus must be finite")
    log_probs = log_softmax(logits)
    blank, labels, texts = _alphabet(log_probs.shape[1], symbols, net_to_label, blank)
    start = lm.start() if lm is not None else None
    beam = {(): _Prefix(start, p_blank=0.0)}

    for frame in log_probs:
        next_beam = {}
        active = [(index, float(frame[index])) for index in labels if frame[index] != -math.inf]
        for prefix, hypothesis in beam.items():
            acoustic = hypothesis.acoustic
            # Same prefix: a blank, or another frame of its last character.
            blank_mass = acoustic + float(frame[blank])
            repeat_mass = (hypothesis.p_nonblank + float(frame[prefix[-1]])) if prefix else -math.inf
            if blank_mass != -math.inf or repeat_mass != -math.inf:
                same = next_beam.get(prefix)
                if same is None:
                    same = next_beam[prefix] = _Prefix(hypothesis.state, hypothesis.offset)
                same.p_blank = float(np.logaddexp(same.p_blank, blank_mass))
                same.p_nonblank = float(np.logaddexp(same.p_nonblank, repeat_mass))

            for index, emission in active:
                # Repeated characters can extend only from a blank-ending path.
                source = hypothesis.p_blank if prefix and index == prefix[-1] else acoustic
                if source == -math.inf:
                    continue
                extended = prefix + (index,)
                child = next_beam.get(extended)
                if child is None:
                    state, cost = hypothesis.state, 0.0
                    if lm is not None:
                        transition = lm.advance(state, labels[index])
                        if transition is None:
                            continue
                        state, cost = transition
                        cost = _cost(cost)
                        if cost == math.inf:
                            continue
                    child = next_beam[extended] = _Prefix(
                        state, hypothesis.offset - lm_weight * cost + insertion_bonus
                    )
                child.p_nonblank = float(np.logaddexp(child.p_nonblank, source + emission))

        beam = dict(nlargest(beam_width, next_beam.items(), key=lambda item: item[1].score))
        if not beam:
            return []

    results = {}
    for prefix, hypothesis in beam.items():
        final_cost = _cost(lm.finish(hypothesis.state)) if lm is not None else 0.0
        if final_cost == math.inf:
            continue
        text = "".join(texts[index] for index in prefix)
        score = hypothesis.score - lm_weight * final_cost
        results[text] = float(np.logaddexp(results.get(text, -math.inf), score))
    return sorted(results.items(), key=lambda item: (-item[1], item[0]))[:nbest]
