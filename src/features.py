#!/usr/bin/env python3
"""Ink -> native ML Kit Bezier-curve features.

The recovered CoeffsToFeaturesAnglesRatios layout is NOT the paper's order:

    (pen_down, dx, dy, a1, d1, a2, d2, dt, start_time_leg, end_time_leg)

Angles are signed radians; ratios use the fixed corresponding endpoint. Time
features are (T3-T0, T1-T0, T2-T3), calculated directly from float32 power-basis
coefficients to avoid cancellation against the absolute timestamp. When time
interpolation is disabled the last three features are omitted.

Curve acceptance uses spatial max/RMS residuals relative to the WHOLE stroke's
bounding-box diagonal, control-polygon length, and adjacent control-leg cosines.
These definitions come from the driver's native disassembly, not corpus tuning.
The named preprocessing pipeline, thinning, time rescale and split gates follow
native disassembly. Least-squares/Newton optimization remains a numerical
reference implementation, not a claim of bit-identical native fitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------


@dataclass
class CurveSettings:
    """research_handwriting.CurveSettings.

    Field numbers are the recospec's; names for 4/6/7/9/10 were recovered from the native
    library (see SPEC.md section 8). Defaults are the real en-US values.
    """

    tol1: float = 0.05           # field 1  - corner-neighbour distance / stroke bbox diagonal
    tol2: float = 0.02           # field 2  - maximum spatial residual / bbox diagonal
    tol3: float = 0.01           # field 3  - RMS spatial residual / bbox diagonal
    max_arc_ratio: float = 3.0   # field 4  - CONTROL-POLYGON length / endpoint distance
    split_cos_threshold: float = -0.8  # field 5  - minimum adjacent control-leg cosine
    generate_second_order_features: bool = False  # field 6
    use_angles_ratios: bool = True                # field 7  - selects this encoder
    interpolate_time: bool = True                 # field 9
    normalize_outputs_to_zero_one: bool = False   # field 10 (absent in en-US => false)

    @classmethod
    def from_proto(cls, proto) -> "CurveSettings":
        """Build from a parsed recospec CurveSettings message.

        Fields 1-5 and 7 have no recovered accessor names, so the loader exposes them as
        unknown_N; we read them by that wire-field identity rather than inventing names.
        """
        def g(name, default):
            return getattr(proto, name) if proto.HasField(name) else default

        return cls(
            tol1=g("unknown_1", 0.05),
            tol2=g("unknown_2", 0.02),
            tol3=g("unknown_3", 0.01),
            max_arc_ratio=g("unknown_4", 4.0),
            split_cos_threshold=g("unknown_5", -0.9),
            generate_second_order_features=g("generate_second_order_features", False),
            use_angles_ratios=g("unknown_7", False),
            interpolate_time=g("interpolate_time", False),
            normalize_outputs_to_zero_one=g("normalize_outputs_to_zero_one", False),
        )


NUM_FEATURES = 10


@dataclass
class Stroke:
    """One pen trace; missing/incomplete t is preserved until HallucinateTime.

    x/y must agree in length. A t-size mismatch is the native trigger to
    regenerate timestamps for the entire ink, not an input-validation error.
    """

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray | None = None
    pen_up: bool = False  # True for synthetic strokes bridging a gap between real strokes

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.t = np.asarray([] if self.t is None else self.t, dtype=np.float64)
        if len(self.x) != len(self.y):
            raise ValueError("stroke x/y must be the same length")

    def __len__(self):
        return len(self.x)

    @property
    def points(self) -> np.ndarray:
        """[N, 3] array of (x, y, t)."""
        if len(self.t) != len(self.x):
            raise ValueError("stroke timestamps must be completed before fitting")
        return np.stack([self.x, self.y, self.t], axis=1)


# --------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------


# The recospec's InkPreprocessorSpec is an ordered list of steps, so we model it the same
# way: a registry of named steps plus a driver. Adding any of the other ~26 steps the native
# library implements is then just another entry here, and recospec.py can drive this directly
# from the parsed config rather than us hardcoding a per-language order.

PREPROCESSING_STEPS: dict[str, callable] = {}


def _step(name):
    def deco(fn):
        PREPROCESSING_STEPS[name] = fn
        return fn

    return deco


@_step("normalize_time")
def normalize_time(strokes: list[Stroke], settings=None) -> list[Stroke]:
    """Subtract the first stroke's first timestamp once, in float32, across ALL strokes.

    Native preprocessing neither scales time nor repairs its ordering. The
    spatial-path-length rescale lives in the per-stroke curve fitter instead.
    """
    if not strokes or not len(strokes[0].t):
        return strokes
    origin = np.float32(strokes[0].t[0])
    return [Stroke(s.x, s.y, np.asarray(s.t, dtype=np.float32) - origin, s.pen_up)
            for s in strokes]


@_step("hallucinate_time")
def hallucinate_time(strokes: list[Stroke], settings=None) -> list[Stroke]:
    """If any timestamp length mismatches, regenerate ALL times at a global 20ms step.

    Matching arrays are untouched, even when times are constant, decreasing,
    or nonfinite. Native field 2 forces regeneration; it is false for en-US.
    """
    interval = np.float32(getattr(settings, "unknown_1", 20.0))
    force = bool(getattr(settings, "unknown_2", False))
    if not force and all(len(s.x) == len(s.t) for s in strokes):
        return strokes
    out, index = [], 0
    for s in strokes:
        t = np.arange(index, index + len(s), dtype=np.float32) * interval
        out.append(Stroke(s.x, s.y, t, s.pen_up))
        index += len(s)
    return out


@_step("normalize_size")
def normalize_size(strokes: list[Stroke], settings=None) -> list[Stroke]:
    """Native no-guide normalization: x/y only, divisor max(height, width/100).

    en-US margin fraction is zero and x origin is the first point, not xmin.
    The float32 epsilon guard also gives a finite scale to stationary ink.
    """
    nonempty = [s for s in strokes if len(s)]
    if not nonempty:
        return strokes
    ys = np.concatenate([s.y for s in nonempty]).astype(np.float32)
    xs = np.concatenate([s.x for s in nonempty]).astype(np.float32)
    height = max(ys.max() - ys.min(), (xs.max() - xs.min()) / np.float32(100))
    if height < np.float32(2**-23):
        height = np.float32(1)
    margin = np.float32(getattr(settings, "unknown_1", 0.0)) * height
    denominator = height + np.float32(2) * margin
    y0 = ys.min() - margin
    first_point_origin = bool(getattr(settings, "unknown_2", True))
    x0 = np.float32(nonempty[0].x[0]) if first_point_origin else xs.min() - margin
    return [Stroke((np.asarray(s.x, dtype=np.float32) - x0) / denominator,
                   (np.asarray(s.y, dtype=np.float32) - y0) / denominator,
                   s.t, s.pen_up) for s in strokes]


@_step("normalize_size_writing_guide_first_stroke")
def normalize_size_writing_guide_first_stroke(strokes: list[Stroke], settings=None) -> list[Stroke]:
    """Normalise against the writing guide, using the first stroke to establish scale.

    We build Ink from raw stroke JSON with no writing guide attached, and the native library
    is explicit about that case -- it carries the string "Ink doesn't have writing guide. Use
    NormalizeSize." So for our inputs this degrades to NormalizeSize. If a writing guide is
    ever plumbed through, this is where it goes.
    """
    return normalize_size(strokes, settings)


@_step("add_pen_up_strokes")
def add_penup_strokes(strokes: list[Stroke], settings=None) -> list[Stroke]:
    """Insert synthetic strokes bridging each stroke's end to the next stroke's start.

    Their native pen-down feature is 0.0; real pen-down traces have feature 1.0.
    """
    real = [Stroke(s.x, s.y, s.t, pen_up=False) for s in strokes if len(s)]
    out: list[Stroke] = []
    for i, s in enumerate(real):
        out.append(s)
        if i + 1 < len(real):
            nxt = real[i + 1]
            times = [s.t[-1], nxt.t[0]] if len(s.t) and len(nxt.t) else []
            out.append(Stroke([s.x[-1], nxt.x[0]], [s.y[-1], nxt.y[0]], times, pen_up=True))
    return out


# The real en-US pipeline, recovered by disassembling the preprocessing-step factory in
# libdigitalink.so (InkPreprocessingStepSpec oneof field -> step class):
#   field 5  -> NormalizeTime                            (no settings)
#   field 6  -> HallucinateTime                          {1: 20.0, 2: 0}
#   field 8  -> NormalizeSizeWritingGuideFirstStroke     {1: 0.0, 2: 1, 3: 0.5}
#   field 11 -> AddPenUpStrokes                          (no settings)
# Step keys are the recospec's own oneof field names, so a parsed RecoSpec's pipeline can
# drive this registry directly (recognize.py just passes step.kind through).
# 362 of the 391 shipped recospecs use exactly this pipeline -- it is the universal
# handwriting one, not an en-US special case. The other 29 are emoji/autodraw and scribe
# gesture models, which are not CTC recognizers.
HANDWRITING_PIPELINE = (
    "normalize_time",
    "hallucinate_time",
    "normalize_size_writing_guide_first_stroke",
    "add_pen_up_strokes",
)
EN_US_PIPELINE = HANDWRITING_PIPELINE  # backwards-compatible alias


def run_pipeline(strokes: list[Stroke], steps=HANDWRITING_PIPELINE) -> list[Stroke]:
    """Apply an ordered preprocessing pipeline by name."""
    for name in steps:
        fn = PREPROCESSING_STEPS.get(name)
        if fn is None:
            raise KeyError(f"unimplemented preprocessing step: {name}")
        strokes = fn(strokes)
    return strokes


# --------------------------------------------------------------------------------------
# cubic fitting  (paper eq. 3-6)
# --------------------------------------------------------------------------------------


def _vandermonde(s: np.ndarray) -> np.ndarray:
    """[N, 4] matrix V of eq. 4: rows (1, s, s^2, s^3)."""
    return np.stack([np.ones_like(s), s, s * s, s * s * s], axis=1)


def _solve_coeffs(points: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Least-squares solve of eq. 5 (V^T Z = V^T V Omega) -> Omega, shape [4, 3].

    With fewer than 4 points the cubic system is underdetermined, and plain lstsq returns a
    minimum-norm solution whose implied control points are arbitrary -- a 2-point pen-up
    segment came out with d1=0.111, d2=0.667 instead of the 1/3, 1/3 a straight line must
    give. So for N < 4 we fit the highest exactly-determined degree (N-1) and zero-pad, which
    makes a 2-point segment a true straight line.
    """
    n = len(points)
    if n == 1:
        omega = np.zeros((4, points.shape[1]))
        omega[0, :2] = points[0, :2]
        return omega  # Native singleton path leaves the entire time polynomial zero.
    V = _vandermonde(s)
    if n >= 4:
        omega, *_ = np.linalg.lstsq(V, points, rcond=None)
        return omega
    degree = max(n - 1, 0)
    omega = np.zeros((4, points.shape[1]))
    sub, *_ = np.linalg.lstsq(V[:, : degree + 1], points, rcond=None)
    omega[: degree + 1] = sub
    return omega


def _eval(omega: np.ndarray, s: np.ndarray) -> np.ndarray:
    return _vandermonde(s) @ omega


def _eval_d1(omega: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = np.stack([np.zeros_like(s), np.ones_like(s), 2 * s, 3 * s * s], axis=1)
    return d @ omega


def _eval_d2(omega: np.ndarray, s: np.ndarray) -> np.ndarray:
    d = np.stack([np.zeros_like(s), np.zeros_like(s), 2 * np.ones_like(s), 6 * s], axis=1)
    return d @ omega


def _newton_update_s(points: np.ndarray, omega: np.ndarray, s: np.ndarray) -> np.ndarray:
    """One Newton step of eq. 6, projecting each point onto the curve.

    Solves x'(s)(xi - x(s)) + y'(s)(yi - y(s)) = 0, i.e. the residual is orthogonal to the
    curve direction. Only x and y participate, per the paper.
    """
    p = _eval(omega, s)[:, :2]
    d1 = _eval_d1(omega, s)[:, :2]
    d2 = _eval_d2(omega, s)[:, :2]
    r = points[:, :2] - p
    f = np.sum(d1 * r, axis=1)
    fp = np.sum(d2 * r, axis=1) - np.sum(d1 * d1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(np.abs(fp) > 1e-12, f / fp, 0.0)
    return np.clip(s - step, 0.0, 1.0)


def _sse(points: np.ndarray, omega: np.ndarray, s: np.ndarray) -> float:
    """Sum of squared errors, eq. 3 (includes the time term)."""
    return float(np.sum((points - _eval(omega, s)) ** 2))


def _initial_s(points: np.ndarray) -> np.ndarray:
    """Chord-length parameterisation, normalised to [0, 1].

    Deliberate safety divergence: the native all-identical-point path divides
    by zero; upstream handling is unknown. Retain finite uniform parameters
    for zero (or numerically tiny) spatial extent rather than propagate NaNs.
    """
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    return cum / total if total > 1e-12 else np.linspace(0.0, 1.0, len(points))


def fit_cubic(points: np.ndarray, iters: int = 8) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit one cubic to [N, 3] points, alternating least squares with Newton reprojection.

    Returns (omega [4, 3], sse, s).
    """
    if len(points) == 1:
        s = np.zeros(1)
        omega = _solve_coeffs(points, s)
        return omega, _sse(points, omega, s), s
    s = _initial_s(points)
    omega = _solve_coeffs(points, s)
    sse = _sse(points, omega, s)
    for _ in range(iters):
        s_new = _newton_update_s(points, omega, s)
        omega_new = _solve_coeffs(points, s_new)
        sse_new = _sse(points, omega_new, s_new)
        # Alternating minimisation should descend; if a Newton step overshoots, stop rather
        # than let the fit wander.
        if sse_new > sse - 1e-12:
            break
        s, omega, sse = s_new, omega_new, sse_new
    return omega, sse, s


def _arc_length(omega: np.ndarray, n: int = 32) -> float:
    s = np.linspace(0.0, 1.0, n)
    d = _eval_d1(omega, s)[:, :2]
    return float(np.trapezoid(np.hypot(d[:, 0], d[:, 1]), s))


def _endpoint_distance(omega: np.ndarray) -> float:
    a, b = _eval(omega, np.array([0.0, 1.0]))[:, :2]
    return float(np.hypot(*(b - a)))


def _split_at_min_angle(points: np.ndarray, neighbour_distance: float = 0.0) -> int | None:
    """Native earliest argmin corner, or None when no score is strictly below 1.

    Neighbours must reach the Euclidean radius from the candidate on BOTH
    sides. Exhausting either side skips that candidate (sentinel cosine 1).
    """
    p = points[:, :2]
    best_score, best_index = 1.0, None
    for i in range(1, len(points) - 1):
        left, right = i - 1, i + 1
        while left >= 0 and np.linalg.norm(p[i] - p[left]) < neighbour_distance:
            left -= 1
        while right < len(points) and np.linalg.norm(p[right] - p[i]) < neighbour_distance:
            right += 1
        if left < 0 or right >= len(points):
            continue
        a, b = p[i] - p[left], p[right] - p[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.dot(a, b) / denom
        if score < best_score:
            best_score, best_index = score, i
    return best_index


def _split_at_max_curvature(points: np.ndarray, omega: np.ndarray, s: np.ndarray) -> int:
    """Native 100-point interior-parameter grid, then earliest nearest input parameter.

    Geometry rejection always schedules a split, even if every curvature is zero.
    The zero-denominator guard is a deliberate safety divergence from the native
    unguarded division; it gives degenerate samples zero curvature, not NaNs.
    """
    s = np.asarray(s, dtype=np.float32)
    a, b = s[1], s[-2]
    j = np.arange(100, dtype=np.float32)
    h = (b - a) / np.float32(99)
    if abs(a) <= abs(b):
        ss = a + j * h
        ss[-1] = b
    else:
        ss = b + (j - np.float32(99)) * h
        ss[0] = a
    omega = np.asarray(omega, dtype=np.float32)
    d1 = _eval_d1(omega, ss)[:, :2]
    d2 = _eval_d2(omega, ss)[:, :2]
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    speed2 = np.sum(d1 * d1, axis=1)
    den = np.sqrt((speed2 * speed2) * speed2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(den > 0, num / den, np.float32(0))
    s_star = ss[int(np.argmax(kappa))]
    return int(np.argmin(np.abs(s[1:-1] - s_star))) + 1


def _bbox_diagonal(points: np.ndarray) -> float:
    """Spatial scale of the original whole stroke, reused by every split and merge."""
    return float(np.linalg.norm(np.ptp(points[:, :2], axis=0))) if len(points) else 0.0


def _fit_failures(points: np.ndarray, omega: np.ndarray, s: np.ndarray,
                  cfg: CurveSettings, stroke_diagonal: float) -> tuple[bool, bool]:
    """Return (spatial residual failure, control-polygon failure).

    Time participates in the existing fit optimizer, but never in these native
    acceptance gates. Comparisons are strict: equality to a limit is accepted.
    """
    residuals = points[:, :2] - _eval(omega, s)[:, :2]
    squared = np.sum(residuals * residuals, axis=1)
    max_error = float(np.sqrt(np.max(squared)))
    rms_error = float(np.sqrt(np.mean(squared)))
    wrong = (max_error > cfg.tol2 * stroke_diagonal or
             rms_error > cfg.tol3 * stroke_diagonal)

    # The native geometric gate is bypassed for fewer than four samples.
    if len(points) < 4:
        return bool(wrong), False
    p0, p1, p2, p3 = power_to_control_points(np.asarray(omega, dtype=np.float32))[:, :2]
    v, middle, w = p1 - p0, p2 - p1, p2 - p3
    lv, lm, lw = np.linalg.norm(v), np.linalg.norm(middle), np.linalg.norm(w)
    chord = np.linalg.norm(p3 - p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ((lv + lm) + lw) / chord
        a = np.dot(v, middle) / lv
        b = -np.dot(w, middle) / lw
        # x86 MINSS chooses its source operand a when either operand is NaN.
        # np.minimum and np.fmin both have different NaN behavior here.
        minimum = a if not (b < a) else b
        score = minimum / lm
    # Ordered comparisons reject zero/NaN chords and unordered ratios/cosines.
    bend = not (chord > 0 and ratio <= cfg.max_arc_ratio and score >= cfg.split_cos_threshold)
    return bool(wrong), bool(bend)


def _thin_points(points: np.ndarray) -> np.ndarray:
    """Native maximum-cardinality minimum-spacing chain, including DP tie order.

    Distances and cumulative lengths are spatial float32. Equal-length
    predecessor chains prefer the earliest predecessor; final-chain ties
    prefer the latest endpoint, but no valid transition retains index zero.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n < 2:
        return points.copy()
    delta = np.diff(points[:, :2], axis=0)
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    cumulative = np.concatenate([np.zeros(1, dtype=np.float32),
                                 np.cumsum(distances, dtype=np.float32)])
    e = np.float32(_bbox_diagonal(points)) * np.float32(0.0003452669770922512)
    e2 = e * e
    length = np.ones(n, dtype=np.int64)
    pred = np.zeros(n, dtype=np.int64)
    prefixmax = np.ones(n, dtype=np.int64)
    best = 0
    for i in range(1, n):
        pred[i] = i
        for j in range(i - 1, -1, -1):
            if length[i] > prefixmax[j] + 1:
                break
            candidate = length[j] + 1
            if candidate < length[i]:
                continue
            if not (cumulative[i] - cumulative[j] > e):
                continue
            dx, dy = points[i, :2] - points[j, :2]
            if not (dx * dx + dy * dy > e2):
                continue
            length[i] = candidate
            pred[i] = j
            if length[i] >= length[best]:
                best = i
        prefixmax[i] = length[best]
    indices = []
    index = best
    for _ in range(int(length[best])):
        indices.append(index)
        index = int(pred[index])
    return points[indices[::-1]].copy()


def _normalize_curve_time(points: np.ndarray) -> np.ndarray:
    """Native per-stroke fitter rescale, after point selection and before splitting.

    All timestamps are multiplied; no local origin is subtracted. Both real
    strokes and synthetic bridges take this path. Nonpositive/NaN duration
    zeroes the entire time column. Call after _thin_points: length and duration
    must both be computed over the retained points, not the original samples.
    """
    points = np.asarray(points, dtype=np.float32).copy()
    if not len(points):
        return points
    duration = points[-1, 2] - points[0, 2]
    if not (duration > 0):
        points[:, 2] = 0
    else:
        delta = np.diff(points[:, :2], axis=0)
        length = np.sum(np.sqrt(np.sum(delta * delta, axis=1)))
        points[:, 2] *= length / duration
    return points


def fit_beziers(points: np.ndarray, cfg: CurveSettings) -> list[np.ndarray]:
    """Fit a stroke with the same point segmentation used by extraction and merging."""
    diagonal = _bbox_diagonal(np.asarray(points, dtype=np.float32))
    points = _normalize_curve_time(_thin_points(points))
    return [fit_cubic(seg)[0] for seg in _split_points(points, cfg, diagonal)]


def merge_beziers(points_per_curve: list[np.ndarray], cfg: CurveSettings,
                  stroke_diagonal: float | None = None) -> list[np.ndarray]:
    """Merge adjacent segments only if the same whole-stroke-scaled gates pass."""
    segs = list(points_per_curve)
    if not segs:
        return []
    if stroke_diagonal is None:
        stroke_diagonal = _bbox_diagonal(np.concatenate(segs))
    changed = True
    while changed and len(segs) > 1:
        changed = False
        out: list[np.ndarray] = []
        i = 0
        while i < len(segs):
            if i + 1 < len(segs):
                combined = np.concatenate([segs[i], segs[i + 1][1:]], axis=0)
                omega, _, s = fit_cubic(combined)
                if not any(_fit_failures(combined, omega, s, cfg, stroke_diagonal)):
                    out.append(combined)
                    i += 2
                    changed = True
                    continue
            out.append(segs[i])
            i += 1
        segs = out
    return segs


# --------------------------------------------------------------------------------------
# the 10 features
# --------------------------------------------------------------------------------------


def power_to_control_points(omega: np.ndarray) -> np.ndarray:
    """Power-basis coefficients [4, D] -> Bezier control points [4, D].

    For x(s) = a0 + a1 s + a2 s^2 + a3 s^3:
        P0 = a0
        P1 = a0 + a1/3
        P2 = a0 + 2 a1/3 + a2/3
        P3 = a0 + a1 + a2 + a3
    """
    a0, a1, a2, a3 = omega
    return np.stack([a0, a0 + a1 / 3.0, a0 + 2.0 * a1 / 3.0 + a2 / 3.0, a0 + a1 + a2 + a3])


def curve_features(omega: np.ndarray, pen_up: bool, cfg: CurveSettings) -> np.ndarray:
    """Native layout: (pen_down, dx, dy, a1, d1, a2, d2[, dt, t1-t0, t2-t3])."""
    omega = np.asarray(omega, dtype=np.float32)
    p0, p1, p2, p3 = power_to_control_points(omega)[:, :2]
    dx, dy = p3 - p0
    vx, vy = p1 - p0
    wx, wy = p2 - p3
    length = np.sqrt(dx * dx + dy * dy)
    # The native ratio guard is strictly L > 0, not an epsilon comparison.
    d1 = np.sqrt(vx * vx + vy * vy) / length if length > 0 else np.float32(0)
    d2 = np.sqrt(wx * wx + wy * wy) / length if length > 0 else np.float32(0)
    # atan2f is unconditional, including signed zeros and degenerate chords.
    a1 = np.arctan2(dx * vy - dy * vx, dx * vx + dy * vy)
    a2 = np.arctan2(dy * wx - dx * wy, -dx * wx - dy * wy)
    values = [0.0 if pen_up else 1.0, dx, dy, a1, d1, a2, d2]
    if cfg.interpolate_time:
        g1, g2, g3 = omega[1:, 2]
        three = np.float32(3)
        dt = (g1 + g2) + g3
        start_leg = g1 / three
        end_leg = (-g1 / three - (np.float32(2) * g2) / three) - g3
        values.extend([dt, start_leg, end_leg])
    feats = np.asarray(values, dtype=np.float32)
    if cfg.normalize_outputs_to_zero_one:
        affine = [1, 2, 7, 8, 9] if cfg.interpolate_time else [1, 2]
        feats[affine] = np.clip((feats[affine] + np.float32(1)) / np.float32(2), 0, 1)
        pi = np.float32(np.pi)
        feats[[3, 5]] = np.clip((feats[[3, 5]] + pi) / (np.float32(2) * pi), 0, 1)
        feats[[4, 6]] = np.clip(feats[[4, 6]], 0, 1)
    return feats


# --------------------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------------------


def extract_features(
    strokes: list[Stroke],
    cfg: CurveSettings | None = None,
    include_penup: bool = True,
    pipeline: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Ink -> [T, 10] float32 features (or [T, 7] without time interpolation).

    T is the number of Bezier curves, roughly 4x fewer than the raw point count.
    """
    cfg = cfg or CurveSettings()
    num_features = NUM_FEATURES if cfg.interpolate_time else 7
    # Preserve empty/mismatched strokes until preprocessing: the original first
    # timestamp controls NormalizeTime and ANY mismatch triggers HallucinateTime.
    # AddPenUpStrokes is the native step that drops empty strokes.
    if not strokes:
        return np.zeros((0, num_features), dtype=np.float32)

    steps = tuple(pipeline if pipeline is not None else HANDWRITING_PIPELINE)
    if not include_penup:
        steps = tuple(st for st in steps if st != "add_pen_up_strokes")
    strokes = run_pipeline(strokes, steps)

    rows: list[np.ndarray] = []
    for index, stroke in enumerate(strokes):
        if not (len(stroke.x) == len(stroke.y) == len(stroke.t)):
            raise ValueError(f"Malformed input (x.size != t.size). stroke={index}: "
                             f"x={len(stroke.x)}, y={len(stroke.y)}, t={len(stroke.t)}")
        raw = np.asarray(stroke.points, dtype=np.float32)
        if len(raw) == 0:
            continue
        diagonal = _bbox_diagonal(raw)
        pts = _normalize_curve_time(_thin_points(raw))
        # Fit, then merge, then re-fit each surviving segment so the coefficients correspond
        # exactly to the merged point sets.
        segments = _split_points(pts, cfg, diagonal)
        for seg in merge_beziers(segments, cfg, diagonal):
            omega, _, _ = fit_cubic(seg)
            rows.append(curve_features(omega, stroke.pen_up, cfg))

    if not rows:
        return np.zeros((0, num_features), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _split_points(points: np.ndarray, cfg: CurveSettings,
                  stroke_diagonal: float | None = None) -> list[np.ndarray]:
    """Split point sets once, carrying the original stroke scale down the recursion."""
    if len(points) == 0:
        return []
    if stroke_diagonal is None:
        stroke_diagonal = _bbox_diagonal(points)
    # The native N<4 fit is exactly determined, with zero-padded coefficients;
    # it bypasses the geometric gate (including non-monotone quadratics).
    if len(points) < 4:
        return [points]
    omega, _, s = fit_cubic(points)
    wrong, bend = _fit_failures(points, omega, s, cfg, stroke_diagonal)
    if not (wrong or bend):
        return [points]
    idx = _split_at_min_angle(points, cfg.tol1 * stroke_diagonal) if wrong else None
    if idx is None:
        # No eligible residual corner is not sufficient to accept: native falls
        # through to the independent geometric gate, then always splits on rejection.
        if not bend:
            return [points]
        idx = _split_at_max_curvature(points, omega, s)
    idx = int(np.clip(idx, 1, len(points) - 2))
    return (_split_points(points[: idx + 1], cfg, stroke_diagonal) +
            _split_points(points[idx:], cfg, stroke_diagonal))


def strokes_from_json(obj: dict) -> list[Stroke]:
    """Build strokes from the oracle harness's JSON schema (see android-oracle/)."""
    out = []
    for s in obj["strokes"]:
        out.append(Stroke(s["x"], s["y"], s.get("t")))
    return out
