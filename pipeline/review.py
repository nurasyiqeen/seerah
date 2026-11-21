#!/usr/bin/env python3
"""Generate quick-review flashcards for an episode from its summary, key points and glossary.
Text-only, so it can use the lighter models.

Usage: python3 pipeline/review.py data/episodes/001.json [...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process as P  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"cards": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "question"},
            "a": {"type": "string", "description": "short answer, 1-2 sentences"},
            "chapter": {"type": "integer", "description": "0-based index of the chapter that covers it"},
        },
        "required": ["q", "a", "chapter"]}}},
    "required": ["cards"],
}


def make_cards(ep):
    chapters = "\n\n".join(
        f"Chapter {i}: {c['title']}\n{c['summary']}\nKey points:\n" + "\n".join(f"- {k}" for k in c["key_points"])
        for i, c in enumerate(ep["chapters"]))
    gloss = "\n".join(f"- {g['translit']} ({g['arabic']}): {g['meaning']}" for g in ep["glossary"])
    prompt = f"""{P.SERIES_CONTEXT}

Write 6-8 flashcards to help a student review episode {ep['n']}: "{ep['title']}".
Mix question types: who/what (people, tribes, places, terms), why something happened, and what lesson the speaker drew.
Use only facts stated in the material below. Keep each answer to 1-2 short sentences.
Write Arabic names and terms with [[transliteration|Arabic]] markup exactly as in the glossary, e.g. [[Jurhum|جرهم]].
"chapter" = the 0-based chapter index the card comes from.

CHAPTERS:
{chapters}

GLOSSARY:
{gloss}

LESSONS:
""" + "\n".join(f"- {l}" for l in ep.get("lessons", []))
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA}}
    resp, model = P.call("generateContent", body, pool=P.TEXT_MODELS)
    cards = json.loads(P.text_of(resp))["cards"]
    for c in cards:
        c["chapter"] = min(max(int(c.get("chapter", 0)), 0), len(ep["chapters"]) - 1)
    P.log(f"  review: {len(cards)} cards via {model}")
    return cards


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ep = json.load(open(path))
        ep["review"] = make_cards(ep)
        json.dump(ep, open(path, "w"), ensure_ascii=False, indent=1)
        print(path, "ok")
