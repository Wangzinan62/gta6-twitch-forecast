"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch viewers.

Reads every case file (<id>.in) in $MAKEFUN_INPUT_DIR (else /input), in name
order, and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output): for each of
hours 2..24 of the release day, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of
that hour's concurrent-viewer count in the game's Twitch category.

Model (v3, improved in v4-v6): each past launch in the case's `comparables`
gives a release-day curve; hour h's log-ratio r_h = ln(v_h / v_1) measures
how the day moved relative to its own first hour. How a day evolves depends
strongly on how big its first hour already is (ln v_1 correlates about -0.9
with the late hours' ratios: a small first hour means the unlock is still
to come, a big one means the day is front-loaded). So for each hour this
program fits a kernel-weighted local line of r_h against ln v_1 over the
comparables (Gaussian kernel, wider early in the day, slope shrunk toward
flat), reads the line at the case's own ln v_1, and takes that hour's
quantiles from the weighted distribution of the fit's residuals, blended
half-and-half with a normal fit of the same residuals, and finally averaged
with the neighbouring hours' quantiles. Two refinements sit on top:
comparables whose unlock pattern (rolling midnight-local vs one global
instant, read from their `unlock` text) matches the target's count four
times as much in the kernel, and hours 2-6 get a small bounded shift from
the target's pre-release ramp level relative to the pool's. Leave-one-out
validation over the 37 usable practice comparables scores this at a mean
log pinball loss of 0.365 overall, 0.201 on the largest launches and 0.225
on rolling-unlock launches, against 0.93 for the house baseline's flat
forecast on the same pool (and 0.370 / 0.208 / 0.239 for this model
before the v4-v6 refinements).

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]


def parse_ts(text):
    """Epoch seconds for an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        try:
            return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def day_curve(comp):
    """{hour: viewers} for hours 1..24 of a comparable's release day (UTC)."""
    start = parse_ts(comp.get("day_start"))
    if start is None:
        return {}
    by_ts = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        ts = parse_ts(pt.get("from"))
        if v is not None and ts is not None:
            try:
                by_ts[ts] = float(v)
            except (TypeError, ValueError):
                pass
    curve = {}
    for h in range(1, 25):
        v = by_ts.get(start + (h - 1) * 3600)
        if v is not None:
            curve[h] = v
    return curve


def log_ratios(curve):
    """r_h = ln(v_h / v_1) for h = 2..24, where both prints exist and are > 0."""
    v1 = curve.get(1)
    if not v1 or v1 <= 0:
        return {}
    return {h: math.log(curve[h] / v1) for h in range(2, 25) if curve.get(h, 0) > 0}


def pre_day_viewers(comp):
    """A comparable's hourly viewers for the 24 hours before its release
    day, oldest first (None where the source has no point)."""
    start = parse_ts(comp.get("day_start"))
    if start is None:
        return []
    by_ts = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        ts = parse_ts(pt.get("from"))
        if v is not None and ts is not None:
            try:
                by_ts[ts] = float(v)
            except (TypeError, ValueError):
                pass
    return [by_ts.get(start - k * 3600) for k in range(24, 0, -1)]


def ramp_f1(pre, v1):
    """ln(last pre-day hour's viewers / v_1): how much of the launch wave
    had already arrived before the UTC day opened. High means the day
    starts mature (early zones live for hours); low means the first hour
    itself is the wave's leading edge."""
    if not v1 or v1 <= 0 or not pre:
        return None
    last = pre[-1]
    if last is None:
        return None
    return math.log(max(last, 1.0) / v1)


def case_ramp_f1(case, p):
    """The target's ramp_f1, from the case's own history_hourly."""
    start = parse_ts(case.get("day_start"))
    if start is None:
        return None
    by_ts = {}
    for row in case.get("history_hourly") or []:
        ts = parse_ts(row.get("from"))
        if ts is not None and row.get("viewers") is not None:
            try:
                by_ts[ts] = float(row["viewers"])
            except (TypeError, ValueError):
                pass
    pre = [by_ts.get(start - k * 3600) for k in range(24, 0, -1)]
    return ramp_f1(pre, p)


def weighted_quantile(pairs, tau):
    """Weighted empirical quantile of (value, weight) pairs.

    Each point is plotted at the midpoint of its own weight slice of the
    cumulative distribution and the quantile is read off by linear
    interpolation between those midpoints.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    xs, ps = [], []
    cum = 0.0
    for v, w in pairs:
        cum += w
        xs.append(v)
        ps.append((cum - 0.5 * w) / total)
    if tau <= ps[0]:
        return xs[0]
    if tau >= ps[-1]:
        return xs[-1]
    for i in range(1, len(ps)):
        if ps[i] >= tau:
            frac = (tau - ps[i - 1]) / (ps[i] - ps[i - 1])
            return xs[i - 1] * (1 - frac) + xs[i] * frac
    return xs[-1]


def classify_unlock(text):
    """'rolling' | 'global' | 'other' from an unlock/notes free-text.

    A rolling launch unlocks at midnight local time zone by zone, so its
    UTC day opens with later zones still to come; a global launch unlocks
    everywhere at one instant. The two shapes evolve differently enough
    that the pool is weighted toward the target's own kind (below).
    """
    t = (text or "").lower()
    if "midnight local" in t or "midnight-local" in t or ("rolling" in t and "midnight" in t):
        return "rolling"
    if ("single global" in t or "simultaneous global" in t or "single simultaneous" in t
            or "simultaneous across" in t or "single time" in t or "one global" in t
            or "global unlock" in t or "simultaneous worldwide" in t):
        return "global"
    return "other"


def target_unlock_class(case):
    """The target's unlock pattern: from the case notes if they say, else
    inferred from the pre-release ramp in history_hourly — a rolling
    launch's last pre-day hours are already elevated and rising because
    the earliest time zones are live, while a global unlock still to come
    leaves them flat."""
    cls = classify_unlock(case.get("notes"))
    if cls != "other":
        return cls
    start = parse_ts(case.get("day_start"))
    if start is None:
        return "other"
    by_ts = {}
    for row in case.get("history_hourly") or []:
        ts = parse_ts(row.get("from"))
        if ts is not None and row.get("viewers") is not None:
            try:
                by_ts[ts] = float(row["viewers"])
            except (TypeError, ValueError):
                pass
    pre = [by_ts.get(start - k * 3600) for k in range(13, 0, -1)]
    pre = [v for v in pre if v is not None]
    if len(pre) >= 8:
        early = sum(pre[:5]) / 5.0
        late = sum(pre[-3:]) / 3.0
        if late > 3.0 * max(early, 1.0) and late > 50:
            return "rolling"
    return "other"


# Kernel bandwidth on ln v_1: the day's shape tracks the first hour's size
# loosely in the morning (wide kernel) and tightly by the evening (narrow).
BW_EARLY = 2.5
BW_LATE = 1.5
BW_SPLIT = 12
# A comparable whose unlock pattern matches the target's (both read as
# rolling or both as one global instant) counts UNLOCK_MATCH times as much
# in the kernel: leave-one-out over the practice pool, this is worth about
# 0.003 of loss overall and about 0.010 on rolling-unlock targets.
UNLOCK_MATCH = 4.0
# The local line's slope is shrunk toward flat: with ~40 comparables the raw
# weighted-least-squares slope over-extrapolates at the edges of the range.
# (0.8, retuned in v6: with the pool concentrated by the unlock weighting,
# the slope is estimated on more homogeneous members and needs less shrink.)
SLOPE_SHRINK = 0.8
# Each quantile blends the weighted empirical residual quantile with the
# normal fit of the same residuals, which steadies the 5% and 95% tails.
NORMAL_BLEND = 0.5
# Adjacent hours' true shapes are close, so each hour's log-quantiles are
# averaged with its neighbours' (weights SMOOTH : 1 : SMOOTH) to trade a
# little bias for less per-hour sampling noise.
SMOOTH = 0.5
# Early hours (2..6) also get a bounded shift from the target's ramp
# level: RAMP_ADJ * (the target's ramp_f1 minus the pool's mean ramp_f1),
# clipped to +/- RAMP_ADJ_CAP. A target whose pre-day level is high for
# its first hour has less wave left to arrive, so its early hours sit a
# little lower than the ln v_1 fit alone says (the fitted sign is
# negative). Leave-one-out: worth about 0.002 overall, 0.004 on the
# largest launches.
RAMP_ADJ = -0.2
RAMP_ADJ_CAP = 0.5
RAMP_ADJ_HOURS = (2, 6)


def hour_log_quantiles(items, xt, h):
    """The five log-ratio quantiles for hour h.

    items: (r_h, weight, ln v_1) for every comparable covering hour h;
    xt: the case's own ln v_1.
    """
    sw = sum(w for _, w, _ in items)
    mx = sum(w * x for _, w, x in items) / sw
    my = sum(w * v for v, w, _ in items) / sw
    sxx = sum(w * (x - mx) ** 2 for _, w, x in items)
    sxy = sum(w * (x - mx) * (v - my) for v, w, x in items)
    slope = (sxy / sxx if sxx > 1e-9 else 0.0) * SLOPE_SHRINK
    center = my + slope * (xt - mx)
    resid = [(v - (my + slope * (x - mx)), w) for v, w, x in items]
    rsd = math.sqrt(max(sum(w * r * r for r, w in resid) / sw, 1e-9))
    row = []
    for tau, z in zip(LEVELS, Z):
        emp = center + weighted_quantile(resid, tau)
        row.append((1 - NORMAL_BLEND) * emp + NORMAL_BLEND * (center + z * rsd))
    return row


def hour1_print(case):
    """The case's hour-1 viewer print: `known` first, history as a fallback."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(float(row["viewers"]), 1.0)
    hist = case.get("history_hourly") or []
    if hist and hist[-1].get("viewers"):
        return max(float(hist[-1]["viewers"]), 1.0)
    return 1.0


def forecast_case(case):
    p = hour1_print(case)
    xt = math.log(p)
    tcls = target_unlock_class(case)
    # The pool: each comparable's log-ratios, tagged with its own ln v_1
    # and its unlock pattern. A comparable with no release-day first hour
    # (the target category's own pre-release history is one) yields no
    # ratios and drops out on its own.
    pool = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        ratios = log_ratios(curve)
        if ratios:
            pool.append((ratios, math.log(curve[1]),
                         classify_unlock(comp.get("unlock")),
                         ramp_f1(pre_day_viewers(comp), curve[1])))
    logq = {}
    wanted = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        wanted.append(h)
        bw = BW_EARLY if h <= BW_SPLIT else BW_LATE
        items = []
        for ratios, x, cls, _f1 in pool:
            if h not in ratios:
                continue
            wgt = math.exp(-0.5 * ((x - xt) / bw) ** 2)
            if tcls != "other" and cls == tcls:
                wgt *= UNLOCK_MATCH
            items.append((ratios[h], wgt, x))
        if items:
            logq[h] = hour_log_quantiles(items, xt, h)
        else:
            # No comparable covers this hour: fall back to the house baseline's
            # flat-with-widening-spread shape so the row stays sane.
            logq[h] = [z * 0.15 * math.sqrt(h - 1) for z in Z]
    # Bounded early-hour shift from the target's pre-release ramp level
    # relative to the pool's (see RAMP_ADJ above).
    tf1 = case_ramp_f1(case, p)
    pool_f1 = [f1 for _r, _x, _c, f1 in pool if f1 is not None]
    if tf1 is not None and pool_f1:
        shift = RAMP_ADJ * (tf1 - sum(pool_f1) / len(pool_f1))
        shift = min(max(shift, -RAMP_ADJ_CAP), RAMP_ADJ_CAP)
        lo_h, hi_h = RAMP_ADJ_HOURS
        for h in wanted:
            if lo_h <= h <= hi_h and h in logq:
                logq[h] = [v + shift for v in logq[h]]
    hours = []
    for h in wanted:
        row = []
        for k in range(len(LEVELS)):
            acc, wsum = 0.0, 0.0
            for dh, wgt in ((-1, SMOOTH), (0, 1.0), (1, SMOOTH)):
                if h + dh in logq:
                    acc += wgt * logq[h + dh][k]
                    wsum += wgt
            row.append(acc / wsum)
        qs = sorted(min(max(p * math.exp(v), 0.01), 1e9) for v in row)
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
