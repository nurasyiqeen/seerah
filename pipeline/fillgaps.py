#!/usr/bin/env python3
"""Find stretches where the transcript skipped speech and transcribe just those minutes.

A gap is suspicious when the time between two lines is far longer than the first line needs
(this speaker talks at ~150 words/min). Run AFTER retime.py, so the timestamps are trustworthy.

Usage: python3 pipeline/fillgaps.py data/episodes/005.json [...]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process as P  # noqa: E402

WPS = 2.5          # speaking rate, words per second
MIN_GAP = 45       # only look at silences longer than this (seconds)


def words(s):
    return len((s.get("text") or s.get("en") or s.get("translit") or "").split())


def norm(t):
    return set(re.findall(r"[a-z']+", P.MARK.sub(r"\1", t or "").lower()))


def find_gaps(segs, duration):
    gaps = []
    for k in range(len(segs)):
        start = segs[k]["t"]
        end = segs[k + 1]["t"] if k + 1 < len(segs) else duration
        need = words(segs[k]) / WPS
        if end - start > max(MIN_GAP, need * 2 + 15):
            gaps.append((k, start + need, end))
    return gaps


def fill_episode(ep):
    gpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "glossary.json")
    glossary = json.load(open(gpath)) if os.path.exists(gpath) else []
    segs = ep["segments"]
    added = 0
    for k, a, b in reversed(find_gaps(segs, ep["duration"])):  # reversed so indices stay valid
        a, b = max(0, a - 5), min(ep["duration"], b + 2)
        if b - a < 20:
            continue
        P.log(f"  gap after line {k}: transcribing {P.fmt(a)}-{P.fmt(b)}")
        try:
            new, _ = P.transcribe_chunk(ep["id"], a, b, glossary)
        except P.QuotaExhausted:
            raise
        except Exception as e:  # leave the gap rather than fail the whole run
            P.log(f"  gap fill failed: {e!r:.120}")
            continue
        new = [P.fill_quran(s) for s in new if (s.get("text") or s.get("ar") or s.get("translit"))]
        # drop lines that repeat what is already there on either side of the gap
        near = norm(segs[k].get("text") or segs[k].get("en")) | (norm(segs[k + 1].get("text") or segs[k + 1].get("en")) if k + 1 < len(segs) else set())
        keep = []
        for s in new:
            ws = norm(s.get("text") or s.get("en"))
            if ws and len(ws & near) / len(ws) > 0.6:
                continue
            s["t"] = round(min(max(s["t"], segs[k]["t"] + 0.5), (segs[k + 1]["t"] - 0.5) if k + 1 < len(segs) else ep["duration"]), 1)
            keep.append(s)
        keep.sort(key=lambda s: s["t"])
        segs[k + 1:k + 1] = keep
        added += len(keep)
        P.log(f"  inserted {len(keep)} lines")
    P.normalise(segs, glossary + ep.get("glossary", []))
    for i in range(1, len(segs)):
        segs[i]["t"] = max(segs[i]["t"], segs[i - 1]["t"])
    return added


def refresh_summary(ep):
    """Re-write title, overview, chapters, lessons and glossary from the (now complete) transcript."""
    gpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "glossary.json")
    glossary = json.load(open(gpath)) if os.path.exists(gpath) else []
    segs = ep["segments"]
    analysis = P.analyse(ep["n"], segs, glossary)
    for c in analysis.get("corrections") or []:
        if 0 <= c["index"] < len(segs) and segs[c["index"]]["kind"] == "speech" and c["text"].strip():
            fixed = P.clean_text(c["text"])
            if not P.is_garbled(fixed):
                segs[c["index"]]["text"] = fixed
    P.normalise(segs, glossary + analysis["glossary"])
    chapters = []
    for c in sorted(analysis["chapters"], key=lambda c: c["start_index"]):
        i = min(max(c["start_index"], 0), len(segs) - 1)
        chapters.append({"t": segs[i]["t"] if chapters else 0, "title": c["title"],
                         "summary": c["summary"], "key_points": c["key_points"]})
    ep.update({"title": analysis["title"], "overview": analysis["overview"], "lessons": analysis["lessons"],
               "chapters": chapters, "glossary": sorted(analysis["glossary"], key=lambda e: e["translit"].lower())})
    ep.pop("review", None)  # flashcards follow the chapters; the daily job regenerates them


if __name__ == "__main__":
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        ep = json.load(open(path))
        n = fill_episode(ep)
        if n >= 8 or "--refresh" in sys.argv:
            refresh_summary(ep)
        json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
        print(path, f"added {n} lines")
