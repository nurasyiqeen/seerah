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
    cache = os.path.join(P.CACHE, f"retime_{vid}_{int(w0)}.json")
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


def retime_episode(ep):
    segs, dur, vid = ep["segments"], ep["duration"], ep["id"]
    old = [s["t"] for s in segs]
    found = {}
    w0 = 0
    while w0 < dur:
        w1 = min(w0 + WINDOW, dur)
        items = [(i, s) for i, s in enumerate(segs) if w0 - MARGIN <= old[i] < w1 + MARGIN]
        if items:
            res, model = retime_window(vid, w0, w1, items)
            # outlier guard: a line's shift must agree with the window's typical shift
            shifts = sorted(t - old[i] for i, t in res.items() if 0 <= i < len(segs))
            med = shifts[len(shifts) // 2] if shifts else 0
            res = {i: t for i, t in res.items() if 0 <= i < len(segs) and abs((t - old[i]) - med) <= 12}
            for i, t in res.items():
                if 0 <= i < len(segs) and abs(t - old[i]) <= 120:
                    # prefer the window where the line sits furthest from the edges
                    edge = min(t - w0, w1 - t)
                    if i not in found or edge > found[i][1]:
                        found[i] = (t, edge)
            P.log(f"  retime {P.fmt(w0)}-{P.fmt(w1)}: {len(res)}/{len(items)} lines via {model}")
        w0 = w1
    times = [found[i][0] if i in found else None for i in range(len(segs))]
    # drop anchors that break the order (keep the longest increasing run, greedy)
    last = -1
    for i, t in enumerate(times):
        if t is not None and t < last - 0.5:
            times[i] = None
        elif t is not None:
            last = t
    # fill gaps by interpolating along the old timings
    anchors = [i for i, t in enumerate(times) if t is not None]
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
