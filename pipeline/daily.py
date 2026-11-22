#!/usr/bin/env python3
"""Daily job: spend today's free Gemini quota on the series, in this order:
  1. precise re-timing of episodes that only have approximate timing,
  2. transcribing the next unprocessed episodes from playlist.tsv (in order),
then rebuild data/index.json. Safe to re-run: all partial work is cached.

Usage (from the repo root): python3 pipeline/daily.py [max_new_episodes]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import process as P  # noqa: E402
import retime as R  # noqa: E402
import review as RV  # noqa: E402
import fillgaps as FG  # noqa: E402

DATA = os.path.join(ROOT, "data")
EPS = os.path.join(DATA, "episodes")


def playlist():
    rows = []
    for line in open(os.path.join(HERE, "playlist.tsv"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            rows.append({"n": int(parts[0]), "id": parts[1], "duration": int(parts[2]), "yt_title": parts[3]})
    return rows


def ep_path(n):
    return os.path.join(EPS, f"{n:03d}.json")


def build_index():
    out = []
    for row in playlist():
        p = ep_path(row["n"])
        if os.path.exists(p):
            e = json.load(open(p))
            out.append({"n": e["n"], "id": e["id"], "title": e["title"], "duration": e["duration"],
                        "overview": e["overview"], "chapters": [c["title"] for c in e["chapters"]],
                        "ready": True, "synced": e.get("timing") == "gemini-retime"})
        else:
            title = row["yt_title"].split(" - ", 1)[-1].strip()
            out.append({"n": row["n"], "id": row["id"], "title": title, "duration": row["duration"],
                        "overview": "", "chapters": [], "ready": False})
    json.dump(out, open(os.path.join(DATA, "index.json"), "w"), ensure_ascii=False, indent=1)
    return out


def main(max_new):
    os.makedirs(EPS, exist_ok=True)
    done_new = 0
    try:
        # 0. episodes flagged for a full redo (keep the old file live until the new one succeeds)
        for row in playlist():
            p = ep_path(row["n"])
            if os.path.exists(p) and json.load(open(p)).get("needs_redo"):
                P.log(f"ep {row['n']}: redoing transcript")
                os.replace(p, p + ".bak")
                try:
                    P.process(DATA, row["n"], row["id"])
                finally:
                    if not os.path.exists(p):
                        os.replace(p + ".bak", p)
                    elif os.path.exists(p + ".bak"):
                        os.remove(p + ".bak")
        # 1. precise timing for existing episodes
        for row in playlist():
            p = ep_path(row["n"])
            if os.path.exists(p):
                ep = json.load(open(p))
                if ep.get("timing") != "gemini-retime" and not ep.get("needs_redo"):
                    P.log(f"ep {row['n']}: precise timing pass")
                    R.retime_episode(ep)
                    if FG.fill_episode(ep) >= 8:
                        FG.refresh_summary(ep)
                    json.dump(ep, open(p, "w"), ensure_ascii=False, indent=1)
        # 2. new episodes, in order
        for row in playlist():
            if done_new >= max_new:
                break
            if os.path.exists(ep_path(row["n"])):
                continue
            P.process(DATA, row["n"], row["id"])
            done_new += 1  # transcript saved; precise timing may finish on a later day
            ep = json.load(open(ep_path(row["n"])))
            P.log(f"ep {row['n']}: precise timing pass")
            R.retime_episode(ep)
            if FG.fill_episode(ep) >= 8:
                FG.refresh_summary(ep)
            json.dump(ep, open(ep_path(row["n"]), "w"), ensure_ascii=False, indent=1)
    except P.QuotaExhausted as e:
        P.log(f"STOPPED for today: {e}")
    finally:
        # 3. review flashcards for ready episodes that lack them (text-only; light models usually still have quota)
        try:
            for row in playlist():
                p = ep_path(row["n"])
                if os.path.exists(p):
                    ep = json.load(open(p))
                    if not ep.get("review") and not ep.get("needs_redo"):
                        ep["review"] = RV.make_cards(ep)
                        json.dump(ep, open(p, "w"), ensure_ascii=False, indent=1)
        except Exception as e:  # never let flashcards block the daily save
            P.log(f"flashcards skipped: {e!r:.150}")
        idx = build_index()
        ready = sum(1 for e in idx if e["ready"])
        synced = sum(1 for e in idx if e.get("synced"))
        P.log(f"SUMMARY: {ready}/{len(idx)} episodes ready, {synced} precisely synced, {done_new} new today")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
