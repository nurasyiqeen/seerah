#!/usr/bin/env python3
"""Precise re-timing pass: Gemini watches short windows of the video and reports when each
known transcript line starts. Output is just numbers, so it is cheap and avoids the
recitation filter; short windows keep the timestamps accurate.

Usage: python3 pipeline/retime.py data/episodes/001.json [...]
"""
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


def retime_window(vid, w0, w1, items):
    cache = os.path.join(P.CACHE, f"retime2_{vid}_{int(w0)}_{int(w1)}.json")
    if os.path.exists(cache):
        c = json.load(open(cache))
        return {int(k): v for k, v in c["res"].items()}, c["model"]
    res, model = _retime_window(vid, w0, w1, items)
    json.dump({"res": res, "model": model}, open(cache, "w"))
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


def retime_episode(ep):
    segs, dur, vid = ep["segments"], ep["duration"], ep["id"]
    old = [s["t"] for s in segs]
    est = {}  # line index -> list of estimates
    for (w0, w1) in windows(dur):
        items = [(i, s) for i, s in enumerate(segs) if w0 - MARGIN <= old[i] < w1 + MARGIN]
        if not items:
            continue
        res, model = retime_window(vid, w0, w1, items)
        for i, t in res.items():
            if 0 <= i < len(segs) and abs(t - old[i]) <= 120:
                est.setdefault(i, []).append(t)
        P.log(f"  retime {P.fmt(w0)}-{P.fmt(w1)}: {len(res)}/{len(items)} lines via {model}")
    times = [None] * len(segs)
    for i, ts in est.items():
        # anchor only where two estimates agree; a single or conflicting answer is not trusted
        best = None
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                if abs(ts[a] - ts[b]) <= AGREE and (best is None or abs(ts[a] - ts[b]) < best[0]):
                    best = (abs(ts[a] - ts[b]), (ts[a] + ts[b]) / 2)
        if best:
            times[i] = best[1]
    # keep anchors in order: drop any anchor that goes backwards
    last = -1
    for i, t in enumerate(times):
        if t is not None and t < last - 0.5:
            times[i] = None
        elif t is not None:
            last = t
    anchors = [i for i, t in enumerate(times) if t is not None]
    # tie-break: where the two answers disagreed, keep the one that fits smoothly between the
    # trusted lines on either side (a window that went wrong is usually off by a whole block)
    def interp(i):
        p = max((a for a in anchors if a < i), default=None)
        n = min((a for a in anchors if a > i), default=None)
        if p is not None and n is not None and old[n] > old[p]:
            return times[p] + (old[i] - old[p]) * (times[n] - times[p]) / (old[n] - old[p])
        if p is not None:
            return times[p] + (old[i] - old[p])
        if n is not None:
            return times[n] - (old[n] - old[i])
        return old[i]
    soft = {}
    for i, ts in est.items():
        if times[i] is None and ts:
            guess = interp(i)
            pick = min(ts, key=lambda t: abs(t - guess))
            if abs(pick - guess) <= 10:
                soft[i] = pick
    for i, t in soft.items():
        times[i] = t
    last = -1
    for i, t in enumerate(times):
        if t is not None and t < last - 0.5:
            times[i] = None
        elif t is not None:
            last = t
    anchors = [i for i, t in enumerate(times) if t is not None]
    # fill the rest by interpolating along the original spacing between trusted anchors
    for i in range(len(segs)):
        if times[i] is not None:
            continue
        p = max((a for a in anchors if a < i), default=None)
        n = min((a for a in anchors if a > i), default=None)
        if p is not None and n is not None and old[n] > old[p]:
            times[i] = times[p] + (old[i] - old[p]) * (times[n] - times[p]) / (old[n] - old[p])
        elif p is not None:
            times[i] = times[p] + (old[i] - old[p])
        elif n is not None:
            times[i] = max(0, times[n] - (old[n] - old[i]))
        else:
            times[i] = old[i]
    for i in range(1, len(times)):
        times[i] = max(times[i], times[i - 1])
    for s, t in zip(segs, times):
        s["t"] = round(t, 1)
    for c in ep["chapters"][1:]:
        idx = min(range(len(old)), key=lambda k: abs(old[k] - c["t"]))
        c["t"] = segs[idx]["t"]
    ep["timing"] = "gemini-retime"
    return len(anchors) / max(len(segs), 1)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ep = json.load(open(path))
        cov = retime_episode(ep)
        json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"{path}: re-timed, {cov:.0%} of lines anchored")
