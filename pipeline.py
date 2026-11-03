"""Shared call → clips + transcripts logic (used by prepare_dataset.py and webapp/worker)."""
import collections
import io
import wave

import numpy as np

SR = 16000
Score = collections.namedtuple("Score", "avg_logprob no_speech_prob compression_ratio")

# Confidence filters for keeping a pseudo-label (tune after checking some by ear).
MIN_AVG_LOGPROB = -0.5     # model's average token log-probability; higher = more confident
MAX_NO_SPEECH = 0.4        # probability the segment is not speech
MAX_COMPRESSION = 2.2      # high = repetitive / hallucinated text
MIN_DUR, MAX_DUR = 1.0, 30.0
MIN_CHARS_PER_SEC, MAX_CHARS_PER_SEC = 3, 30

DEFAULT_MODEL = "yuldashev222/uzbek_stt_v1-ct2"  # Kotib uzbek_stt_v1 medium, CT2


def load_model(name_or_path=DEFAULT_MODEL, device="auto"):
    from faster_whisper import WhisperModel
    for compute in (["int8_float16", "int8"] if device in ("cuda", "auto") else ["int8"]):
        try:
            return WhisperModel(name_or_path, device=device, compute_type=compute)
        except ValueError:  # e.g. no float16 support on this device
            continue
    raise RuntimeError(f"could not load {name_or_path}")


def wav_bytes(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


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


def load_channels(path):
    """16 kHz float32 channels; stereo split into [left, right], mono/identical -> [mono]."""
    from faster_whisper.audio import decode_audio
    left, right = decode_audio(path, sampling_rate=SR, split_stereo=True)
    if np.allclose(left[:SR * 30], right[:SR * 30], atol=1e-3):
        return [left]
    return [left, right]


def transcribe_call(model, path):
    """Yield (record, clip_audio) for each speech chunk of each channel of one call."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    channels = load_channels(path)
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
            seg = Score(avg_logprob=min(x.avg_logprob for x in segs),
                        no_speech_prob=max(x.no_speech_prob for x in segs),
                        compression_ratio=max(x.compression_ratio for x in segs))
            reason = keep_reason(seg, dur, text)
            yield {
                "index": i, "text": text, "channel": ch, "n_channels": len(channels),
                "start": round(c["start"] / SR, 2), "end": round(c["end"] / SR, 2),
                "avg_logprob": round(seg.avg_logprob, 3),
                "no_speech_prob": round(seg.no_speech_prob, 3),
                "compression_ratio": round(seg.compression_ratio, 2),
                "keep": not reason, "drop_reason": reason,
            }, piece
