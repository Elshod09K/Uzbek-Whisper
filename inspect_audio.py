"""Summarize a folder of .wav files: channels, sample rate, encoding, durations.

    python inspect_audio.py /path/to/calls            # scans up to 2000 files
    python inspect_audio.py /path/to/calls --max 0    # scan everything

Reads only the WAV headers, so it is fast even on terabytes. Prints nothing
about the audio content, so the output is safe to share.
"""
import argparse
import collections
import os
import struct

ENCODINGS = {1: "PCM", 3: "float", 6: "A-law", 7: "mu-law", 0xFFFE: "extensible"}


def wav_info(path):
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None
        f.read(4)
        if f.read(4) != b"WAVE":
            return None
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16 + (size & 1), 1)
            elif cid == b"data":
                if fmt is None:
                    return None
                code, ch, rate, byte_rate, _, bits = fmt
                dur = size / byte_rate if byte_rate else 0
                return ENCODINGS.get(code, f"code {code}"), ch, rate, bits, dur
            else:
                f.seek(size + (size & 1), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--max", type=int, default=2000, help="max files to scan (0 = all)")
    a = p.parse_args()

    formats, durs, bad, total_bytes, n = collections.Counter(), [], 0, 0, 0
    for root, _, files in os.walk(a.folder):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(root, name)
            info = wav_info(path)
            n += 1
            total_bytes += os.path.getsize(path)
            if info is None:
                bad += 1
            else:
                enc, ch, rate, bits, dur = info
                formats[(enc, "stereo" if ch == 2 else f"{ch}ch" if ch != 1 else "mono", rate, bits)] += 1
                durs.append(dur)
            if a.max and n >= a.max:
                break
        if a.max and n >= a.max:
            break

    print(f"Scanned {n} .wav files ({total_bytes / 1e9:.2f} GB){' — stopped at --max' if a.max and n >= a.max else ''}")
    print("\nFormat (encoding, channels, sample rate, bits) -> files")
    for (enc, ch, rate, bits), c in formats.most_common():
        print(f"  {enc:8s} {ch:7s} {rate:6d} Hz {bits:2d}-bit  {c}")
    if bad:
        print(f"  unreadable header: {bad}")
    if durs:
        durs.sort()
        hours = sum(durs) / 3600
        print(f"\nDuration: total {hours:.1f} h, median {durs[len(durs)//2]:.0f}s, "
              f"min {durs[0]:.0f}s, max {durs[-1]:.0f}s")
        if total_bytes:
            print(f"Estimated hours for the whole archive: ~{hours / total_bytes * 1e12:.0f} h per TB")


if __name__ == "__main__":
    main()
