#!/usr/bin/env python3
"""Precise re-timing pass: Gemini watches short windows of the video and reports when each
known transcript line starts. Output is just numbers, so it is cheap and avoids the
recitation filter; short windows keep the timestamps accurate.

Usage: python3 pipeline/retime.py data/episodes/001.json [...]
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process as P  # noqa: E402

WINDOW = 180  # seconds of video per request
MARGIN = 45   # look for lines whose rough time is within this distance of the window
RETIME_MODELS = ["gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash"]  # lite models tested too imprecise

SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "start": {"type": "number", "description": "seconds from the start of THIS CLIP; -1 if not in the clip"}},
        "required": ["id", "start"]}}},
    "required": ["lines"],
}


def line_preview(s):
    if s["kind"] == "speech":
        words = P.MARK.sub(r"\1", s.get("text", "")).split()
        return " ".join(words[:14])
    return "(Arabic recitation) " + " ".join((s.get("translit") or "").split()[:8])


LEGACY_OK = set()  # videos whose line list has not changed since their old (unsigned) cache was written


def _legacy_ok(vid):
    return vid in LEGACY_OK


def retime_window(vid, w0, w1, items):
    sig = hashlib.sha1("\n".join(f"{i}:{line_preview(s)}" for i, s in items).encode()).hexdigest()[:10]
    cache = os.path.join(P.CACHE, f"retime2_{vid}_{int(w0)}_{int(w1)}.json")
    if os.path.exists(cache) and json.load(open(cache)).get("sig") not in (None, sig):
        cache = os.path.join(P.CACHE, f"retime2_{vid}_{int(w0)}_{int(w1)}_{sig}.json")
    elif os.path.exists(cache) and json.load(open(cache)).get("sig") is None and not _legacy_ok(vid):
        cache = os.path.join(P.CACHE, f"retime2_{vid}_{int(w0)}_{int(w1)}_{sig}.json")
    if os.path.exists(cache):
        c = json.load(open(cache))
        return {int(k): v for k, v in c["res"].items()}, c["model"]
    res, model = _retime_window(vid, w0, w1, items)
    json.dump({"res": res, "model": model, "sig": sig}, open(cache, "w"))
    return res, model


def _retime_window(vid, w0, w1, items):
    listing = "\n".join(f"{i}: {line_preview(s)}" for i, s in items)
    prompt = f"""You are given a {int(w1 - w0)}-second clip of a lecture video and a list of transcript lines, some of which are spoken in this clip.
For each line id, give "start": the moment (in seconds, with one decimal, measured from the beginning of THIS CLIP) when the speaker starts saying that line. Listen carefully and be precise to within half a second. If a line is not spoken in this clip, give -1.

Lines:
{listing}"""
    body = {
        "contents": [{"parts": [P.video_part(vid, w0, w1), {"text": prompt}]}],
        "generationConfig": {"mediaResolution": "MEDIA_RESOLUTION_LOW",
                             "responseMimeType": "application/json", "responseSchema": SCHEMA},
    }
    resp, model = P.call("generateContent", body, pool=RETIME_MODELS, timeout=240)
    out = json.loads(P.text_of(resp))["lines"]
    return {o["id"]: w0 + o["start"] for o in out if 0 <= o.get("start", -1) <= (w1 - w0) + 1}, model


STEP = 90      # windows overlap by half, so (almost) every line is timed twice
AGREE = 3.0    # two independent estimates must agree within this many seconds


def windows(dur):
    """Overlapping 3-minute windows, plus short extra windows so the first and last lines are also covered twice."""
    ws = []
    w0 = 0
    while w0 < dur:
        ws.append((w0, min(w0 + WINDOW, dur)))
        if w0 + WINDOW >= dur:
            break
        w0 += STEP
    ws.append((0, min(120, dur)))
    ws.append((max(0, dur - 120), dur))
    return sorted(set(ws))


def collect(vid, segs, wins, ref, est, first_wid):
    """Ask Gemini, window by window, when each line starts; lines are picked by their position in `ref`."""
    for k, (w0, w1) in enumerate(wins):
        items = [(i, s) for i, s in enumerate(segs) if w0 - MARGIN <= ref[i] < w1 + MARGIN]
        if not items:
            continue
        res, model = retime_window(vid, w0, w1, items)
        for i, t in res.items():
            if 0 <= i < len(segs):
                est.setdefault(i, []).append((t, first_wid + k))
        P.log(f"  retime {P.fmt(w0)}-{P.fmt(w1)}: {len(res)}/{len(items)} lines via {model}")


def combine(est, old, n):
    """Turn per-window answers into one start time per line (see comments)."""
    # 1. trust an answer if another window agrees with it, or if its own window's answers for the
    #    nearby lines agree with other windows (a window that goes wrong is off for a whole run).
    def agrees(j, t, w):
        return any(abs(u - t) <= AGREE for u, v in est.get(j, []) if v != w)
    own = {(i, w): t for i, ts in est.items() for t, w in ts}
    def local_score(i, w):
        return sum(1 for j in range(i - 4, i + 5) if j != i and (j, w) in own and agrees(j, own[(j, w)], w))
    times = [None] * n
    for i, ts in est.items():
        pair = min(((abs(a - b), (a + b) / 2) for k, (a, wa) in enumerate(ts) for b, wb in ts[k + 1:] if wa != wb), default=None)
        if pair and pair[0] <= AGREE:
            times[i] = pair[1]
            continue
        scored = sorted(((local_score(i, w), t) for t, w in ts), reverse=True)
        if scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            times[i] = scored[0][1]
    plain = {i: [t for t, w in ts] for i, ts in est.items()}

    def in_order():
        last = -1
        for i, t in enumerate(times):
            if t is not None and t < last - 0.5:
                times[i] = None
            elif t is not None:
                last = t
        return [i for i, t in enumerate(times) if t is not None]

    def interp(i, anchors):
        p = max((a for a in anchors if a < i), default=None)
        q = min((a for a in anchors if a > i), default=None)
        if p is not None and q is not None and old[q] > old[p]:
            return times[p] + (old[i] - old[p]) * (times[q] - times[p]) / (old[q] - old[p])
        if p is not None:
            return times[p] + (old[i] - old[p])
        if q is not None:
            return max(0, times[q] - (old[q] - old[i]))
        return old[i]

    anchors = in_order()
    # 2. lines with a single or conflicting answer: accept an answer that falls in order between
    #    the confirmed lines on either side
    strict = list(times)
    for i in range(n):
        ts = plain.get(i) or []
        if times[i] is not None or not ts:
            continue
        lo = max((times[j] for j in range(i) if times[j] is not None), default=-1)
        hi = min((strict[j] for j in range(i + 1, n) if strict[j] is not None), default=float("inf"))
        fits = [t for t in ts if lo + 0.5 < t < hi - 0.5]
        if fits:
            guess = interp(i, anchors)
            times[i] = min(fits, key=lambda t: abs(t - guess))
    anchors = in_order()
    # 3. interpolate the rest
    for i in range(n):
        if times[i] is None:
            times[i] = interp(i, anchors)
    for i in range(1, n):
        times[i] = max(times[i], times[i - 1])
    return times, len(anchors)


def words(s):
    return len((s.get("text") or s.get("en") or s.get("translit") or "").split())


def implausible(segs, times, dur):
    """Spots where lines would have to be spoken impossibly fast, or are followed by a long silence."""
    bad = []
    for k in range(len(segs)):
        nxt = times[k + 1] if k + 1 < len(segs) else dur
        dt, w = nxt - times[k], words(segs[k])
        if (w >= 8 and dt < w / 3.4 * 0.7) or dt > max(45, w / 2.5 * 2 + 15):
            bad.append(k)
    return bad


def retime_episode(ep):
    segs, dur, vid = ep["segments"], ep["duration"], ep["id"]
    old = [s["t"] for s in segs]
    est = {}
    base = windows(dur)
    collect(vid, segs, base, old, est, 0)
    times, n_anchor = combine(est, old, len(segs))
    # second opinion: re-check implausible spots with windows centred on them, choosing the lines
    # by their improved positions (the first pass may not have shown them to the right window)
    bad = implausible(segs, times, dur)
    extra = []
    for k in bad:
        a = max(0, int((times[k] - 60) // 10 * 10) + 5)   # offset by 5 s so these never reuse a base window
        b = min(dur, a + WINDOW)
        if extra and a < extra[-1][1] - 60:
            extra[-1] = (extra[-1][0], min(dur, max(extra[-1][1], b), extra[-1][0] + 240))
        else:
            extra.append((a, b))
    if extra:
        P.log(f"  second opinion for {len(bad)} implausible spots: {len(extra)} extra windows")
        collect(vid, segs, extra, times, est, len(base))
        times, n_anchor = combine(est, old, len(segs))
    for s, t in zip(segs, times):
        s["t"] = round(t, 1)
    for c in ep["chapters"][1:]:
        idx = min(range(len(old)), key=lambda k: abs(old[k] - c["t"]))
        c["t"] = segs[idx]["t"]
    ep["timing"] = "gemini-retime"
    return n_anchor / max(len(segs), 1)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ep = json.load(open(path))
        cov = retime_episode(ep)
        json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"{path}: re-timed, {cov:.0%} of lines anchored")
