#!/usr/bin/env python3
"""Re-time transcript segments using YouTube's automatic-caption word timings.

Gemini's transcript has accurate words but only approximate timestamps. YouTube's
auto-captions have inaccurate words but precise per-word timings. We align the two
word sequences and give each segment the caption time of its first matched word.

Usage (re-time existing episode files in place):
    python3 pipeline/align.py data/episodes/001.json [...]
"""
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTIONS = os.path.join(HERE, "captions")
MARK = re.compile(r"\[\[([^|\]]+)\|[^\]]+\]\]")
WORD = re.compile(r"[a-z0-9']+")


def norm_words(text):
    text = MARK.sub(r"\1", text or "").lower().replace("’", "'")
    return [w.strip("'") for w in WORD.findall(text) if w.strip("'")]


def load_captions(video_id):
    path = os.path.join(CAPTIONS, f"{video_id}.json")
    if not os.path.exists(path):
        return None
    out = []
    for ms, w in json.load(open(path)):
        for nw in norm_words(w):
            out.append((ms / 1000.0, nw))
    return out


def align(segments, captions, max_shift=90.0):
    """Return a list of new start times (seconds), one per segment."""
    seg_words, owner = [], []
    for i, s in enumerate(segments):
        text = s.get("text") if s.get("kind") == "speech" else s.get("en")
        if s.get("kind") != "speech":
            text = ""  # recitations: captions rarely match; interpolate instead
        for w in norm_words(text):
            seg_words.append(w)
            owner.append(i)
    cap_words = [w for _, w in captions]
    sm = difflib.SequenceMatcher(None, seg_words, cap_words, autojunk=False)
    first_match = {}  # segment index -> caption time of its earliest matched word
    matched_cnt = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            i = owner[a + k]
            matched_cnt[i] = matched_cnt.get(i, 0) + 1
            if i not in first_match:
                # if the match isn't the segment's first word, back off by the word gap
                first_word_idx = owner.index(i)
                back = (a + k - first_word_idx) * 0.33  # ~0.33 s per word
                first_match[i] = max(0.0, captions[b + k][0] - back)
    seg_count = {i: owner.count(i) for i in set(owner)}
    times = [None] * len(segments)
    for i, t in first_match.items():
        # trust only segments where a reasonable share of words matched, and the shift is sane
        if matched_cnt[i] >= max(2, 0.3 * seg_count[i]) and abs(t - segments[i]["t"]) <= max_shift:
            times[i] = t
    # enforce monotonic order: drop anchors that go backwards
    last = -1.0
    for i, t in enumerate(times):
        if t is not None:
            if t < last:
                times[i] = None
            else:
                last = t
    # interpolate gaps between anchors using the original relative spacing
    anchors = [i for i, t in enumerate(times) if t is not None]
    if not anchors:
        return [s["t"] for s in segments], 0.0
    for i in range(len(segments)):
        if times[i] is not None:
            continue
        prev = max((a for a in anchors if a < i), default=None)
        nxt = min((a for a in anchors if a > i), default=None)
        o = segments[i]["t"]
        if prev is not None and nxt is not None:
            o0, o1 = segments[prev]["t"], segments[nxt]["t"]
            f = (o - o0) / (o1 - o0) if o1 > o0 else (i - prev) / (nxt - prev)
            times[i] = times[prev] + min(max(f, 0), 1) * (times[nxt] - times[prev])
        elif prev is not None:
            times[i] = times[prev] + (o - segments[prev]["t"])
        else:
            times[i] = max(0.0, times[nxt] - (segments[nxt]["t"] - o))
    for i in range(1, len(times)):
        times[i] = max(times[i], times[i - 1])
    return [round(t, 1) for t in times], len(anchors) / len(segments)


def retime_episode(ep):
    caps = load_captions(ep["id"])
    if not caps:
        return None
    old = [s["t"] for s in ep["segments"]]
    new, coverage = align(ep["segments"], caps)
    for s, t in zip(ep["segments"], new):
        s["t"] = t
    # move chapter starts with their segment
    for c in ep["chapters"]:
        idx = min(range(len(old)), key=lambda i: abs(old[i] - c["t"]))
        c["t"] = new[idx] if c is not ep["chapters"][0] else 0
    ep["timing"] = "youtube-captions"
    return coverage


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ep = json.load(open(path))
        if ep.get("timing") == "youtube-captions":
            print(path, "already aligned")
            continue
        cov = retime_episode(ep)
        if cov is None:
            print(path, "no captions file")
            continue
        json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"{path}: aligned, {cov:.0%} of segments anchored to caption timings")
