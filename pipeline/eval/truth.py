"""Ground-truth start times for ep1 segments from the sparse caption track (continuous parts only)."""
import json, os, sys, difflib, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import align
caps = []
for ms, txt in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'caps_ep1.json'))):
    ws = align.norm_words(txt)
    for k, w in enumerate(ws):
        caps.append((ms / 1000 + k * 0.33, w))
def truth(segs):
    words, owner = [], []
    for i, s in enumerate(segs):
        if s['kind'] != 'speech': continue
        for w in align.norm_words(s['text']): words.append(w); owner.append(i)
    sm = difflib.SequenceMatcher(None, words, [w for _, w in caps], autojunk=False)
    out = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            i = owner[a + k]
            if owner.index(i) == a + k and n >= 4:  # segment's first word, inside a solid match
                out[i] = caps[b + k][0]
    return out
def report(segs, label):
    t = truth(segs)
    errs = sorted(abs(segs[i]['t'] - v) for i, v in t.items())
    if not errs: print(label, 'no truth'); return
    print(f"{label}: n={len(errs)} median={errs[len(errs)//2]:.1f}s p90={errs[int(len(errs)*.9)]:.1f}s max={errs[-1]:.1f}s")
if __name__ == '__main__':
    report(json.load(open(sys.argv[1]))['segments'], sys.argv[1])
