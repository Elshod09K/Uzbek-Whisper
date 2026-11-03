"""Turn long call recordings into a training dataset of short clips with pseudo-labels.

For each .wav call:
  1. resample to 16 kHz; split stereo into separate channels (agent / customer)
  2. find speech with VAD, transcribe each chunk with a faster-whisper model (clips under 30 s)
  3. save each clip as a 16 kHz mono WAV
  4. write one JSON line per clip with the text and confidence scores, and a "keep"
     flag that is true only for transcripts the model was confident about

    python prepare_dataset.py /path/to/calls out_dataset/ --limit 20      # try 20 calls first
    python prepare_dataset.py /path/to/calls out_dataset/                 # everything (resumable)

Output: out_dataset/clips/*.wav and out_dataset/manifest.jsonl
Re-running skips calls already in out_dataset/done.txt, so it is safe to stop and restart.
(For the multi-user web workflow, see webapp/ instead.)
"""
import argparse
import json
import os
import sys
import time

from pipeline import DEFAULT_MODEL, load_model, transcribe_call, wav_bytes


def find_wavs(folder):
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".wav"):
                yield os.path.join(root, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calls_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="faster-whisper / CT2 model (default: Kotib uzbek_stt_v1 medium)")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--limit", type=int, default=0, help="process at most N calls (0 = all)")
    a = p.parse_args()

    clips_dir = os.path.join(a.out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    done_path = os.path.join(a.out_dir, "done.txt")
    done = set(open(done_path).read().split("\n")) if os.path.exists(done_path) else set()
    model = load_model(a.model, a.device)

    manifest = open(os.path.join(a.out_dir, "manifest.jsonl"), "a", encoding="utf-8")
    done_f = open(done_path, "a")
    n_calls = n_kept = n_clips = 0
    kept_sec = 0.0
    t_start = time.time()

    for path in find_wavs(a.calls_dir):
        rel = os.path.relpath(path, a.calls_dir)
        if rel in done:
            continue
        if a.limit and n_calls >= a.limit:
            break
        base = rel.replace(os.sep, "__").rsplit(".", 1)[0]
        n_ch = 0
        try:
            for rec, piece in transcribe_call(model, path):
                n_ch = rec.pop("n_channels")
                clip_name = f"{base}__ch{rec['channel']}__{rec.pop('index'):04d}.wav"
                with open(os.path.join(clips_dir, clip_name), "wb") as f:
                    f.write(wav_bytes(piece))
                manifest.write(json.dumps({"audio": f"clips/{clip_name}", "call": rel, **rec},
                                          ensure_ascii=False) + "\n")
                n_clips += 1
                if rec["keep"]:
                    n_kept += 1
                    kept_sec += rec["end"] - rec["start"]
        except Exception as e:
            print(f"skip {rel}: {e}", file=sys.stderr)
            continue
        manifest.flush()
        done_f.write(rel + "\n")
        done_f.flush()
        n_calls += 1
        el = time.time() - t_start
        print(f"[{n_calls}] {rel}: {n_ch} ch | total clips {n_clips}, kept {n_kept} "
              f"({kept_sec / 3600:.2f} h) | {el / n_calls:.0f}s per call", flush=True)

    print(f"\nDone: {n_calls} calls, {n_clips} clips, {n_kept} kept ({kept_sec / 3600:.2f} h of training audio)")


if __name__ == "__main__":
    main()
