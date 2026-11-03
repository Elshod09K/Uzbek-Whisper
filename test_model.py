"""Test hostmepanda/whisper-large-v3-turbo-uzbek-ct2 with faster-whisper.

Usage:
    python test_model.py audio.wav [more.wav ...]
    python test_model.py audio_dir/ --refs dev.tsv   # also computes WER (FLEURS .tsv format)

Options:
    --device cpu|cuda   (default: cuda if available, else cpu)
"""
import argparse
import csv
import os
import sys
import time

from faster_whisper import WhisperModel

MODEL_ID = "hostmepanda/whisper-large-v3-turbo-uzbek-ct2"


def load_refs(tsv_path):
    """FLEURS tsv: id, file_name, raw_transcription, transcription, ..."""
    refs = {}
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 4:
                refs[row[1]] = row[3]
    return refs


def normalize(text):
    import re
    text = text.lower().replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="audio files or directories")
    p.add_argument("--refs", help="FLEURS-style .tsv with reference transcripts (for WER)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading {MODEL_ID} on {device} ({compute_type})...")
    model = WhisperModel(MODEL_ID, device=device, compute_type=compute_type)

    files = []
    for path in args.inputs:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, f) for f in os.listdir(path)
                            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")))
        else:
            files.append(path)
    if not files:
        sys.exit("No audio files found.")

    refs = load_refs(args.refs) if args.refs else {}
    hyps, gold = [], []
    for f in files:
        t0 = time.time()
        segments, info = model.transcribe(f, language="uz", beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        dt = time.time() - t0
        print(f"\n[{os.path.basename(f)}] {info.duration:.1f}s audio, {dt:.1f}s to transcribe")
        print(f"  HYP: {text}")
        ref = refs.get(os.path.basename(f))
        if ref is not None:
            print(f"  REF: {ref}")
            hyps.append(normalize(text))
            gold.append(normalize(ref))

    if gold:
        import jiwer
        print(f"\nWER: {jiwer.wer(gold, hyps):.2%}   CER: {jiwer.cer(gold, hyps):.2%}   ({len(gold)} clips)")


if __name__ == "__main__":
    main()
