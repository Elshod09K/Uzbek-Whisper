"""Turn long call recordings into a training dataset of short clips with pseudo-labels.

For each .wav call:
  1. resample to 16 kHz; split stereo into separate channels (agent / customer)
  2. find speech with VAD, transcribe with a faster-whisper model -> segments under 30 s
  3. save each segment as a 16 kHz mono WAV clip
  4. write one JSON line per clip with the text and confidence scores, and a "keep"
     flag that is true only for transcripts the model was confident about

    python prepare_dataset.py /path/to/calls out_dataset/ --limit 20      # try 20 calls first
    python prepare_dataset.py /path/to/calls out_dataset/                 # everything (resumable)

Output: out_dataset/clips/*.wav and out_dataset/manifest.jsonl
Re-running skips calls already in out_dataset/done.txt, so it is safe to stop and restart.
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

SR = 16000
Score = __import__('collections').namedtuple('Score', 'avg_logprob no_speech_prob compression_ratio')

# Confidence filters for keeping a pseudo-label (tune after checking some by ear).
MIN_AVG_LOGPROB = -0.5     # model's average token log-probability; higher = more confident
MAX_NO_SPEECH = 0.4        # probability the segment is not speech
MAX_COMPRESSION = 2.2      # high = repetitive / hallucinated text
MIN_DUR, MAX_DUR = 1.0, 30.0
MIN_CHARS_PER_SEC, MAX_CHARS_PER_SEC = 3, 30


def find_wavs(folder):
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".wav"):
                yield os.path.join(root, name)


def save_clip(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def keep_reason(seg, dur, text):
    if not text:
        return "empty"
    if not (MIN_DUR <= dur <= MAX_DUR):
        return "duration"
    if seg.avg_logprob < MIN_AVG_LOGPROB:
        return "low_confidence"
    if seg.no_speech_prob > MAX_NO_SPEECH:
        return "no_speech"
    if seg.compression_ratio > MAX_COMPRESSION:
        return "repetitive"
    cps = len(text) / dur
    if not (MIN_CHARS_PER_SEC <= cps <= MAX_CHARS_PER_SEC):
        return "speed"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calls_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default="yuldashev222/uzbek_stt_v1-ct2",
                   help="faster-whisper / CT2 model (default: Kotib uzbek_stt_v1 medium)")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--limit", type=int, default=0, help="process at most N calls (0 = all)")
    a = p.parse_args()

    clips_dir = os.path.join(a.out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    done_path = os.path.join(a.out_dir, "done.txt")
    done = set(open(done_path).read().split("\n")) if os.path.exists(done_path) else set()

    compute = "int8_float16" if a.device in ("cuda", "auto") else "int8"
    try:
        model = WhisperModel(a.model, device=a.device, compute_type=compute)
    except ValueError:  # e.g. auto -> cpu without float16 support
        model = WhisperModel(a.model, device=a.device, compute_type="int8")

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
        try:
            audio = decode_audio(path, sampling_rate=SR, split_stereo=True)
            channels = list(audio)  # (left, right)
            if np.allclose(channels[0][:SR * 30], channels[1][:SR * 30], atol=1e-3):
                channels = [channels[0]]  # mono file (or identical channels)
        except Exception as e:
            print(f"skip {rel}: {e}", file=sys.stderr)
            continue

        base = rel.replace(os.sep, "__").rsplit(".", 1)[0]
        for ch, samples in enumerate(channels):
            # Our own VAD so each clip is exactly one speech chunk under MAX_DUR seconds.
            chunks = get_speech_timestamps(samples, VadOptions(
                min_silence_duration_ms=500, max_speech_duration_s=MAX_DUR - 2, speech_pad_ms=200))
            for i, c in enumerate(chunks):
                piece = samples[c["start"]:c["end"]]
                dur = len(piece) / SR
                if dur < MIN_DUR:
                    continue
                segs, _ = model.transcribe(piece, language="uz", beam_size=5, vad_filter=False,
                                           condition_on_previous_text=False, without_timestamps=True)
                segs = list(segs)
                if not segs:
                    continue
                text = " ".join(x.text.strip() for x in segs).strip()
                seg = Score(
                    avg_logprob=min(x.avg_logprob for x in segs),
                    no_speech_prob=max(x.no_speech_prob for x in segs),
                    compression_ratio=max(x.compression_ratio for x in segs))
                reason = keep_reason(seg, dur, text)
                clip_name = f"{base}__ch{ch}__{i:04d}.wav"
                save_clip(os.path.join(clips_dir, clip_name), piece)
                manifest.write(json.dumps({
                    "audio": f"clips/{clip_name}", "text": text, "call": rel, "channel": ch,
                    "start": round(c["start"] / SR, 2), "end": round(c["end"] / SR, 2),
                    "avg_logprob": round(seg.avg_logprob, 3),
                    "no_speech_prob": round(seg.no_speech_prob, 3),
                    "compression_ratio": round(seg.compression_ratio, 2),
                    "keep": not reason, "drop_reason": reason,
                }, ensure_ascii=False) + "\n")
                n_clips += 1
                if not reason:
                    n_kept += 1
                    kept_sec += dur
        manifest.flush()
        done_f.write(rel + "\n")
        done_f.flush()
        n_calls += 1
        el = time.time() - t_start
        print(f"[{n_calls}] {rel}: {len(channels)} ch | total clips {n_clips}, kept {n_kept} "
              f"({kept_sec / 3600:.2f} h) | {el / n_calls:.0f}s per call", flush=True)

    print(f"\nDone: {n_calls} calls, {n_clips} clips, {n_kept} kept ({kept_sec / 3600:.2f} h of training audio)")


if __name__ == "__main__":
    main()
