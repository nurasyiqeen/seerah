#!/usr/bin/env python3
"""Lecture pipeline: YouTube URL -> accurate transcript + chapters + glossary (JSON).

Uses the Gemini API, which watches the public YouTube video directly (nothing downloaded).
The API key is injected by the environment's network proxy (x-goog-api-key header).

Usage: python3 process.py <out_dir> <episode_number>:<video_id> [...]
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"
MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-3.7-flash"]  # each has its own free daily quota; quality checks guard output
TEXT_MODELS = MODELS + ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]  # for text-only summaries
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spend.json")) and \
        json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spend.json"))).get("active"):
    MODELS = ["gemini-3.5-flash"]  # paid: most accurate transcription model only
    TEXT_MODELS = ["gemini-3-flash-preview", "gemini-3.5-flash"]  # paid: cheaper model for summaries/flashcards
MIN_WPM = 110
CHUNK = 600  # seconds per transcription request
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
SERIES_CONTEXT = (
    "This is a lecture series in English on the Seerah (biography) of the Prophet Muhammad ﷺ. "
    "The speaker frequently uses Arabic: names of people, tribes and places, Islamic terminology, "
    "honorifics, Qur'anic verses, hadith and du'a."
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class QuotaExhausted(Exception):
    pass


# ---------------------------------------------------------------- spend cap (paid mode)
# When pipeline/spend.json has "active": true, every request's cost is estimated from Google's own
# token counts and the run stops before the budget is reached. Prices are USD per 1M tokens and
# deliberately rounded UP; output includes "thinking" tokens, which Google bills as output.
PRICES = {
    "gemini-3.5-flash": (1.50, 9.00), "gemini-3.6-flash": (1.50, 9.00),
    "gemini-3-flash-preview": (0.50, 4.00),
    "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.30, 2.50),
}
DEFAULT_PRICE = (2.00, 12.00)  # unknown models: assume expensive
SPEND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spend.json")
SAFETY_MARGIN = 0.30  # stop this many USD before the budget
CACHED_RATE = 0.25    # cached input tokens: assume 25% of the normal price (Google discounts them by 75-90%)


def spend_state():
    try:
        return json.load(open(SPEND_FILE))
    except (OSError, ValueError):
        return {"active": False}


def paid_mode():
    return spend_state().get("active", False)


def record_spend(model, usage):
    st = spend_state()
    if not st.get("active"):
        return
    pin, pout = PRICES.get(model, DEFAULT_PRICE)
    tin = usage.get("promptTokenCount", 0)
    cached = usage.get("cachedContentTokenCount", 0)  # Google bills cached input at a large discount
    tout = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    cost = (tin - cached) / 1e6 * pin + cached / 1e6 * pin * CACHED_RATE + tout / 1e6 * pout
    st["spent_usd"] = round(st.get("spent_usd", 0) + cost, 4)
    st["requests"] = st.get("requests", 0) + 1
    json.dump(st, open(SPEND_FILE, "w"), indent=1)


def check_budget():
    st = spend_state()
    if st.get("active") and st.get("spent_usd", 0) >= st["budget_usd"] - SAFETY_MARGIN:
        raise QuotaExhausted(f"paid budget reached: ${st.get('spent_usd', 0):.2f} of ${st['budget_usd']:.2f} spent")


EXHAUSTED = set()  # models whose free daily quota is used up in this run


def call(method, body, model=None, timeout=300, pool=None):
    """POST with retries; skips models whose daily quota is exhausted."""
    pool = [model] if model else (pool or MODELS)
    last = None
    for attempt in range(20):
        live = [m for m in pool if m not in EXHAUSTED]
        if not live:
            raise QuotaExhausted("daily free quota used up for: " + ", ".join(pool))
        m = live[attempt % len(live)]
        if method == "generateContent":
            check_budget()
        try:
            r = requests.post(API.format(model=m, method=method), json=body, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            if method == "generateContent":
                record_spend(m, data.get("usageMetadata", {}))
            return data, m
        last = f"{r.status_code} {r.text[:300]}"
        if r.status_code == 429 and "per day" in r.text:
            log(f"  {m}: daily quota used up; switching model")
            EXHAUSTED.add(m)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            log(f"  retry ({m}): {last[:60]}")
            time.sleep(min(10 * (attempt + 1), 60))
            continue
        raise RuntimeError(last)
    raise RuntimeError(f"gave up: {last}")


def video_part(vid, start=None, end=None):
    p = {"fileData": {"fileUri": f"https://www.youtube.com/watch?v={vid}"}}
    if start is not None:
        p["videoMetadata"] = {"startOffset": f"{int(start)}s", "endOffset": f"{int(end)}s"}
    return p


def duration_seconds(vid):
    d, _ = call("countTokens", {"contents": [{"parts": [video_part(vid)]}]}, timeout=120)
    audio = next(x["tokenCount"] for x in d["promptTokensDetails"] if x["modality"] == "AUDIO")
    return audio / 32.0  # Gemini bills audio at 32 tokens/second


def text_of(resp):
    parts = resp["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


# ---------------------------------------------------------------- transcription
SEG_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "number", "description": "start time, seconds from the start of the FULL video"},
                    "kind": {"type": "string", "enum": ["speech", "arabic", "quran"]},
                    "surah": {"type": "integer", "description": "for quran: surah number 1-114"},
                    "ayah_from": {"type": "integer"},
                    "ayah_to": {"type": "integer"},
                    "text": {"type": "string", "description": "for speech: the English sentence(s) with [[translit|arabic]] markup"},
                    "ar": {"type": "string", "description": "for arabic: the Arabic text with full tashkeel where known"},
                    "translit": {"type": "string", "description": "for arabic: transliteration"},
                    "en": {"type": "string", "description": "for arabic: English meaning"},
                },
                "required": ["t", "kind"],
            },
        }
    },
    "required": ["segments"],
}


def transcribe_prompt(start, end, glossary):
    g = "\n".join(f"- {e['translit']} = {e['arabic']}" for e in glossary[:250]) or "(none yet)"
    return f"""{SERIES_CONTEXT}

Transcribe ONLY the portion of the video from {fmt(start)} to {fmt(end)} (the clip you are given), verbatim and complete. Do not summarise or skip anything.

Output rules:
1. Split into segments of one or two sentences each (roughly 5-15 seconds). "t" = when the segment starts, in seconds counted from the start of the FULL video (so values lie between {int(start)} and {int(end)}).
2. kind "speech": English speech. Remove filler ("um", "uh") and false starts but keep every meaningful word.
   Wrap EVERY Arabic word or phrase (names of people, tribes, places, Islamic terms, short phrases like insha'Allah, honorific phrases) in markup [[transliteration|Arabic script]], e.g. [[Abu Talib|أبو طالب]], [[Quraysh|قريش]], [[tawakkul|توكل]], [[insha'Allah|إن شاء الله]].
   Use a clear, conventional English transliteration (no diacritic dots), and spell each name/term the same way every time. Honorifics after the Prophet's name: write [[ṣallallāhu ʿalayhi wa sallam|ﷺ]] as just ﷺ. Common words "Allah" and "Prophet" may stay plain.
3. kind "quran": when the speaker recites a Qur'an verse, do NOT write the verse text. Only identify it: "surah" (number), "ayah_from", "ayah_to". The exact text is filled in later from a verified Mushaf.
   kind "arabic": other Arabic recited at length (hadith wording, du'a, poetry, opening khutbah). Do NOT write it in Arabic script: give only "translit" (transliteration) and "en" (English meaning). Leave "ar" empty.
   If the speaker then translates or explains it himself, transcribe that as separate speech segments.
4. Be extremely careful with Arabic names and terms: use your knowledge of the Seerah to identify the correct person/place even if the pronunciation is unclear.
5. Preferred spellings already used in this series (keep these consistent):
{g}
"""


def fmt(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


GARBLE = re.compile(r"[\u1000-\u109F\u0250-\u02AF]|#\d{4,};|&#\d+;|\ufffd")
LEAKED_TS = re.compile(r"(?:^\s*\d+\s*)?\[\d{1,2}:\d{2}(?::\d{2})?\]\s*")   # "[05:00]" anywhere, "1 [00:13]" at start
LOOP = re.compile(r"(.{2,30}?)\1{4,}")                                         # same fragment 5+ times in a row
FILE_EXT = re.compile(r"\.(?:json|csv|txt|xml|docx?|pdf|zip|tar|gz|bz2|7z|rar)\b", re.I)
EN_TAG = re.compile(r"\s*\.?\s*\ben:\s*", re.I)


def clean_text(v):
    v = re.sub(r"&?#6501[08];", "ﷺ", v)
    v = LEAKED_TS.sub("", v)
    return v.strip()


def is_garbled(v):
    return bool(GARBLE.search(v) or LOOP.search(v) or len(FILE_EXT.findall(v)) >= 2)


def clean_segments(segs):
    """Repair small artefacts; reject output that is visibly garbled so it gets retried."""
    bad = 0
    for s in segs:
        # a recitation whose transliteration swallowed the English ("... al-Quran.en: His character ...")
        if s.get("kind") == "arabic" and isinstance(s.get("translit"), str) and EN_TAG.search(s["translit"]):
            tl, en = EN_TAG.split(s["translit"], 1)
            s["translit"] = tl.rstrip(" .")
            if not s.get("en"):
                s["en"] = en
        for k in ("text", "translit", "en", "ar"):
            v = s.get(k)
            if not isinstance(v, str):
                continue
            v = clean_text(v)
            if is_garbled(v):
                bad += 1
            s[k] = v
    if bad:
        raise ValueError(f"garbled output ({bad} bad fields)")
    return segs


def transcribe_chunk(vid, start, end, glossary):
    cache = os.path.join(CACHE, f"{vid}_{int(start)}.json")
    if os.path.exists(cache):
        return json.load(open(cache)), {}
    # Completeness check: a full verbatim transcript runs ~140-170 words/min for this speaker.
    best, best_wpm = None, -1
    for attempt in range(4):
        segs, usage = _transcribe_chunk(vid, start, end, glossary)
        words = sum(len((s.get("text") or s.get("en") or "").split()) for s in segs)
        wpm = words / max((end - start) / 60, 0.5)
        if wpm > best_wpm:
            best, best_wpm = segs, wpm
        if wpm >= MIN_WPM:
            break
        log(f"  chunk {fmt(start)}: only {wpm:.0f} words/min, looks incomplete; retrying")
    json.dump(best, open(cache, "w"), ensure_ascii=False)
    return best, usage


def _transcribe_chunk(vid, start, end, glossary):
    body = {
        "contents": [{"parts": [video_part(vid, start, end), {"text": transcribe_prompt(start, end, glossary)}]}],
        "generationConfig": {
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "responseMimeType": "application/json",
            "responseSchema": SEG_SCHEMA,
            "maxOutputTokens": 20000,
        },
    }
    for attempt in range(2):
        resp, model = call("generateContent", body)
        try:
            segs = clean_segments(json.loads(text_of(resp))["segments"])
            break
        except Exception as e:  # malformed / truncated JSON (e.g. the model looped until MAX_TOKENS)
            fr = (resp.get("candidates") or [{}])[0].get("finishReason")
            log(f"  chunk {fmt(start)}-{fmt(end)} unusable response (finish={fr}, {e!r:.60}); retrying")
    else:
        if end - start <= 150:
            for alt in [m for m in (MODELS[1:] or ["gemini-3-flash-preview"]) if m not in EXHAUSTED]:
                try:
                    resp, model = call("generateContent", body, model=alt)
                    segs = clean_segments(json.loads(text_of(resp))["segments"])
                    log(f"  chunk {fmt(start)}-{fmt(end)}: recovered with {alt}")
                    break
                except QuotaExhausted:
                    raise
                except Exception as e:
                    log(f"  chunk {fmt(start)}-{fmt(end)}: {alt} also failed ({e!r:.60})")
            else:
                log(f"  chunk {fmt(start)}-{fmt(end)}: GAP - marking as unavailable")
                return [{"t": float(start), "kind": "speech",
                         "text": "[Transcript unavailable for this part. Please listen to the video.]", "gap": True}], {}
        else:
            mid = start + (end - start) / 2
            log(f"  chunk {fmt(start)}-{fmt(end)}: splitting into two halves")
            a, _ = _transcribe_chunk(vid, start, mid, glossary)
            b, u = _transcribe_chunk(vid, mid, end, glossary)
            return a + b, u
    # Timestamp sanity: if the model counted from the clip start instead of the video start, shift.
    ts = [s["t"] for s in segs if isinstance(s.get("t"), (int, float))]
    if start > 0 and ts and max(ts) <= (end - start) + 5 and min(ts) < start - 30:
        for s in segs:
            s["t"] = s.get("t", 0) + start
    # Drift fix: if timestamps overrun the clip window, rescale them linearly into it.
    ts = [float(s.get("t", start)) for s in segs]
    if ts and (max(ts) > end + 15 or min(ts) < start - 15):
        lo, hi = min(ts), max(ts)
        span = (end - start) * len(ts) / (len(ts) + 1)  # leave room for the last segment
        log(f"  chunk {fmt(start)}: rescaling timestamps {fmt(lo)}-{fmt(hi)} into {fmt(start)}-{fmt(end)}")
        for s, t in zip(segs, ts):
            s["t"] = start + (t - lo) * span / max(hi - lo, 1)
    for s in segs:
        s["t"] = round(min(max(float(s.get("t", start)), start), end), 1)
    segs.sort(key=lambda s: s["t"])
    log(f"  chunk {fmt(start)}-{fmt(end)}: {len(segs)} segments via {model}")
    return segs, resp.get("usageMetadata", {})


# ---------------------------------------------------------------- analysis
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_index": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_index", "title", "summary", "key_points"],
            },
        },
        "lessons": {"type": "array", "items": {"type": "string"}},
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "translit": {"type": "string"},
                    "arabic": {"type": "string"},
                    "category": {"type": "string", "enum": ["person", "tribe", "place", "term", "event", "phrase", "other"]},
                    "meaning": {"type": "string"},
                    "variants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["translit", "arabic", "category", "meaning"],
            },
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["index", "text"],
            },
        },
    },
    "required": ["title", "overview", "chapters", "lessons", "glossary"],
}


def analyse(ep_no, segs, glossary):
    lines = []
    for i, s in enumerate(segs):
        if s["kind"] == "arabic":
            lines.append(f"{i} [{fmt(s['t'])}] (ARABIC) {s.get('ar','')} | {s.get('translit','')} | {s.get('en','')}")
        else:
            lines.append(f"{i} [{fmt(s['t'])}] {s.get('text','')}")
    g = "\n".join(f"- {e['translit']} = {e['arabic']}" for e in glossary[:250]) or "(none yet)"
    prompt = f"""{SERIES_CONTEXT}

Below is the machine transcript of episode {ep_no}, one numbered segment per line. Arabic words are marked [[transliteration|Arabic]].

Tasks:
1. "title": a short descriptive episode title (e.g. "The Year of the Elephant and the Birth of the Prophet ﷺ").
2. "overview": 2-3 sentence summary of the whole episode.
3. "chapters": split the lecture into 4-10 logical chapters in order. start_index = the segment number where each chapter begins (first chapter starts at 0). For each: a clear title, a summary of 3-6 sentences that a student can revise from, and 3-6 key_points (facts, names, dates, verses mentioned).
4. "lessons": 3-6 practical lessons/reflections the speaker draws.
5. "glossary": every distinct Arabic name, tribe, place, term, event and phrase used in this episode. Use ONE canonical transliteration (prefer the series spellings below), correct Arabic script, a short meaning/identification (who the person was, what the term means), and "variants": any other spellings of it that appear in the transcript.
6. "corrections": only for segments that contain a clear transcription mistake (a misheard name, wrong person/place given the historical context, garbled phrase, wrong Arabic script). Give the segment index and the full corrected text with the same markup. Do not rewrite style. Leave empty if none.

Series spellings so far:
{g}

TRANSCRIPT:
""" + "\n".join(lines)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": ANALYSIS_SCHEMA,
            "maxOutputTokens": 32000,
        },
    }
    for attempt in range(3):
        resp, model = call("generateContent", body, pool=TEXT_MODELS)
        try:
            out = json.loads(text_of(resp))
            log(f"  analysis via {model}: {len(out['chapters'])} chapters, {len(out['glossary'])} terms, "
                f"{len(out.get('corrections', []))} corrections")
            return out
        except Exception as e:
            log(f"  analysis parse error ({e}); retrying")
    raise RuntimeError("analysis failed")


MARK = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")


def normalise(segs, glossary_terms):
    """Map spelling variants to the canonical transliteration inside [[...]] markup."""
    canon = {}
    for e in glossary_terms:
        canon[e["translit"].lower()] = e["translit"]
        for v in e.get("variants") or []:
            canon[v.lower()] = e["translit"]
    def fix(m):
        t = canon.get(m.group(1).strip().lower(), m.group(1).strip())
        return f"[[{t}|{m.group(2).strip()}]]"
    for s in segs:
        if s.get("text"):
            s["text"] = MARK.sub(fix, s["text"])


def merge_glossary(series, terms, ep_no):
    by = {e["translit"].lower(): e for e in series}
    for t in terms:
        k = t["translit"].lower()
        if k in by:
            by[k].setdefault("episodes", [])
            if ep_no not in by[k]["episodes"]:
                by[k]["episodes"].append(ep_no)
        else:
            e = {k2: t[k2] for k2 in ("translit", "arabic", "category", "meaning") if k2 in t}
            e["episodes"] = [ep_no]
            series.append(e)
            by[k] = e
    series.sort(key=lambda e: e["translit"].lower())
    return series


QURAN = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quran_data.json")))
ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def fill_quran(s):
    """Replace a {kind: quran, surah, ayah_from, ayah_to} reference with verified text."""
    if s.get("kind") != "quran":
        return s
    sur = QURAN.get(str(s.get("surah")))
    a, b = s.get("ayah_from") or 0, s.get("ayah_to") or s.get("ayah_from") or 0
    if not sur or not (1 <= a <= b <= len(sur["v"])) or b - a > 15:
        return {"t": s["t"], "kind": "speech", "text": ""}  # unidentifiable: drop
    vs = sur["v"][a - 1:b]
    return {
        "t": s["t"], "kind": "arabic",
        "ar": " ".join(f"{v[0]} ﴿{str(a + i).translate(ARABIC_DIGITS)}﴾" for i, v in enumerate(vs)),
        "translit": " ".join(v[1] for v in vs),
        "en": " ".join(v[2] for v in vs),
        "ref": f"Qur'an, {sur['name']} {s['surah']}:{a}" + (f"-{b}" if b > a else ""),
    }


# ---------------------------------------------------------------- driver
def process(out_dir, ep_no, vid):
    path = os.path.join(out_dir, "episodes", f"{ep_no:03d}.json")
    if os.path.exists(path):
        log(f"ep {ep_no}: already done")
        return
    gpath = os.path.join(out_dir, "glossary.json")
    glossary = json.load(open(gpath)) if os.path.exists(gpath) else []
    dur = duration_seconds(vid)
    log(f"ep {ep_no} ({vid}): {fmt(dur)}")
    windows = [(s, min(s + CHUNK, dur)) for s in range(0, int(dur), CHUNK)]
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < 60:  # fold a tiny tail into the previous chunk
        windows[-2] = (windows[-2][0], windows[-1][1]); windows.pop()
    with cf.ThreadPoolExecutor(2) as ex:
        results = list(ex.map(lambda w: transcribe_chunk(vid, w[0], w[1], glossary), windows))
    segs = [fill_quran(s) for r, _ in results for s in r]
    segs = [s for s in segs if (s.get("text") or s.get("ar") or s.get("translit"))]
    analysis = analyse(ep_no, segs, glossary)
    for c in analysis.get("corrections") or []:
        if 0 <= c["index"] < len(segs) and segs[c["index"]]["kind"] == "speech" and c["text"].strip():
            segs[c["index"]]["orig"] = segs[c["index"]]["text"]
            segs[c["index"]]["text"] = c["text"]
    normalise(segs, glossary + analysis["glossary"])
    chapters = []
    for c in sorted(analysis["chapters"], key=lambda c: c["start_index"]):
        i = min(max(c["start_index"], 0), len(segs) - 1)
        chapters.append({"t": segs[i]["t"] if chapters else 0, "title": c["title"],
                         "summary": c["summary"], "key_points": c["key_points"]})
    ep = {
        "n": ep_no, "id": vid, "duration": round(dur), "title": analysis["title"],
        "overview": analysis["overview"], "lessons": analysis["lessons"], "chapters": chapters,
        "glossary": sorted(analysis["glossary"], key=lambda e: e["translit"].lower()),
        "segments": [{k: v for k, v in s.items() if k != "orig"} for s in segs],
    }
    cov = align.retime_episode(ep)
    log(f"  timing: " + (f"aligned to YouTube captions ({cov:.0%} anchored)" if cov is not None else "no captions file; using Gemini timestamps"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
    json.dump(merge_glossary(glossary, analysis["glossary"], ep_no), open(gpath, "w"), ensure_ascii=False, indent=1)
    log(f"ep {ep_no}: saved ({len(segs)} segments, {len(chapters)} chapters)")


def build_index(out_dir):
    eps = []
    d = os.path.join(out_dir, "episodes")
    for f in sorted(os.listdir(d)):
        e = json.load(open(os.path.join(d, f)))
        eps.append({"n": e["n"], "id": e["id"], "title": e["title"], "duration": e["duration"],
                    "overview": e["overview"], "chapters": [c["title"] for c in e["chapters"]]})
    json.dump(eps, open(os.path.join(out_dir, "index.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for arg in sys.argv[2:]:
        n, vid = arg.split(":", 1)
        try:
            process(out, int(n), vid)  # sequential so the series glossary grows episode by episode
        except QuotaExhausted as e:
            log(f"STOPPED: {e}. Finished chunks are cached; re-run tomorrow to continue.")
            break
    if os.path.isdir(os.path.join(out, "episodes")):
        build_index(out)
