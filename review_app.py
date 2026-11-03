"""Review tool: listen to each clip, correct Kotib's transcript, save.

    python review_app.py out_dataset/                  # this computer only
    python review_app.py out_dataset/ --share-lan      # reviewers on the office network open
                                                      # http://<this-PC's-IP>:7861

Several reviewers can work at the same time: each browser gets a different clip.
Every save is appended to out_dataset/reviewed.jsonl (never overwrites), so it is
safe to stop and restart. The latest save for a clip wins.
"""
import argparse
import json
import os
import random
import threading
import time

import gradio as gr

p = argparse.ArgumentParser()
p.add_argument("dataset_dir")
p.add_argument("--order", choices=["by-call", "random", "worst-first", "in-order"], default="by-call",
               help="by-call = whole calls in random order, clips in time order (best for the "
                    "test set: no call is split between test and training); "
                    "worst-first = least confident clips first")
p.add_argument("--share-lan", action="store_true", help="listen on all network interfaces")
p.add_argument("--port", type=int, default=7861)
args = p.parse_args()

DS = args.dataset_dir
MANIFEST = os.path.join(DS, "manifest.jsonl")
REVIEWED = os.path.join(DS, "reviewed.jsonl")
CLAIM_TIMEOUT = 15 * 60  # a clip opened by one reviewer is hidden from others for 15 min

lock = threading.Lock()
items = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
if args.order == "by-call":
    calls = sorted({x["call"] for x in items})
    random.Random(0).shuffle(calls)
    rank = {c: i for i, c in enumerate(calls)}
    items.sort(key=lambda x: (rank[x["call"]], x["start"], x["channel"]))
elif args.order == "random":
    random.Random(0).shuffle(items)
elif args.order == "worst-first":
    items.sort(key=lambda x: x["avg_logprob"])
by_audio = {x["audio"]: x for x in items}

reviewed = {}
if os.path.exists(REVIEWED):
    for l in open(REVIEWED, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            reviewed[r["audio"]] = r
claims = {}  # audio -> (session id, time)


def total_hours(status_ok=True):
    sec = sum(by_audio[a]["end"] - by_audio[a]["start"] for a, r in reviewed.items()
              if a in by_audio and (r["status"] != "bad_audio") == status_ok)
    return sec / 3600


def progress():
    n = len(reviewed)
    corrected = sum(r["status"] == "corrected" for r in reviewed.values())
    bad = sum(r["status"] == "bad_audio" for r in reviewed.values())
    h = total_hours()
    usable = f"{h:.1f} h" if h >= 1 else f"{h * 60:.0f} min"
    return (f"**Reviewed {n} / {len(items)} clips** · {usable} usable · "
            f"{corrected} corrected · {n - corrected - bad} already correct · {bad} bad audio")


def next_item(session):
    now = time.time()
    with lock:
        for x in items:
            a = x["audio"]
            if a in reviewed:
                continue
            owner = claims.get(a)
            if owner and owner[0] != session and now - owner[1] < CLAIM_TIMEOUT:
                continue
            claims[a] = (session, now)
            return x
    return None


def show(session):
    x = next_item(session)
    if x is None:
        return None, "", "🎉 Everything is reviewed.", None, progress(), time.time()
    info = (f"`{x['audio']}` · call `{x['call']}` ch{x['channel']} · "
            f"{x['end'] - x['start']:.1f}s · model confidence {x['avg_logprob']:.2f}")
    return os.path.join(DS, x["audio"]), x["text"], info, x["audio"], progress(), time.time()


def save(session, audio_key, text, reviewer, started, status):
    if audio_key:
        orig = by_audio[audio_key]["text"]
        text = " ".join(text.split())
        if status == "ok" and text != " ".join(orig.split()):
            status = "corrected"
        rec = {"audio": audio_key, "text": text if status != "bad_audio" else "",
               "model_text": orig, "status": status, "reviewer": reviewer or "anonymous",
               "seconds_spent": round(time.time() - (started or time.time()), 1),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with lock:
            with open(REVIEWED, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            reviewed[audio_key] = rec
            claims.pop(audio_key, None)
    return show(session)


def skip(session, audio_key):
    # leave it unreviewed for someone else; this browser moves on
    with lock:
        if audio_key:
            claims[audio_key] = ("skipped", time.time())
    return show(session)


GUIDE = """**Qoidalar / Rules** — write exactly what is said, in Latin script.
Use oʻ / gʻ (or o' / g' — be consistent across the team) · numbers as words (*yuz*, not *100*) ·
Russian words as heard, in Latin · unclear word → `[?]` · only noise/music/silence or
unintelligible → **Bad audio**. Do not fix the speaker's grammar."""

with gr.Blocks(title="Transcript review") as demo:
    session = gr.State(lambda: os.urandom(8).hex())
    key = gr.State(None)
    started = gr.State(None)
    gr.Markdown("# Transcript review\n" + GUIDE)
    stats = gr.Markdown()
    reviewer = gr.Textbox(label="Your name", max_lines=1)
    info = gr.Markdown()
    audio = gr.Audio(type="filepath", label="Clip", interactive=False, autoplay=True)
    text = gr.Textbox(label="Transcript — correct it to exactly what is said", lines=4)
    with gr.Row():
        b_ok = gr.Button("✅ Save (Ctrl+Enter)", variant="primary")
        b_bad = gr.Button("🗑 Bad audio")
        b_skip = gr.Button("⏭ Skip")
    outs = [audio, text, info, key, stats, started]
    b_ok.click(lambda s, k, t, r, st: save(s, k, t, r, st, "ok"),
               [session, key, text, reviewer, started], outs)
    b_bad.click(lambda s, k, t, r, st: save(s, k, t, r, st, "bad_audio"),
                [session, key, text, reviewer, started], outs)
    b_skip.click(skip, [session, key], outs)
    demo.load(show, session, outs)
    demo.load(None, None, None, js="""() => { document.addEventListener('keydown', e => {
        if (e.ctrlKey && e.key === 'Enter') { [...document.querySelectorAll('button')]
            .find(b => b.innerText.includes('Save'))?.click(); } }); }""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0" if args.share_lan else "127.0.0.1", server_port=args.port,
                allowed_paths=[os.path.abspath(DS)])
