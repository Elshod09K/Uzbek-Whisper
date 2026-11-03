"""Uzbek speech-to-text models behind one interface: load_model(name)(audio_path) -> text."""
import functools

# name -> (backend, repo id)
MODELS = {
    "hostmepanda turbo (CT2)":            ("ct2", "hostmepanda/whisper-large-v3-turbo-uzbek-ct2"),
    "Kotib uzbek_stt_v1 medium (CT2)":    ("ct2", "yuldashev222/uzbek_stt_v1-ct2"),
    "navai-uz whisper-medium-uzbek":      ("hf-whisper", "navai-uz/whisper-medium-uzbek"),
    "sarahai uzbek-stt-3 (wav2vec2)":     ("hf-ctc", "sarahai/uzbek-stt-3"),
    "BuzzASR uzbek (large-v3)":           ("hf-whisper", "BuzzASR/uzbek"),
    "Abduqayum call-center medium":       ("hf-whisper", "Abduqayum/whisper-uzbek-medium-callcenter"),
    "OpenAI whisper-large-v3-turbo (base)": ("ct2", "large-v3-turbo"),
}

# Per-model decoding settings (from the model cards).
GENERATE_KWARGS = {
    # Card: language/task prompt is baked into generation_config; greedy + anti-repetition.
    "BuzzASR/uzbek": {"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
}
DEFAULT_GENERATE_KWARGS = {"language": "uz", "task": "transcribe", "num_beams": 5}


def _device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


@functools.lru_cache(maxsize=2)
def load_model(name):
    backend, repo = MODELS[name]

    if backend == "ct2":
        from faster_whisper import WhisperModel
        dev = _device()
        m = WhisperModel(repo, device=dev, compute_type="float16" if dev == "cuda" else "int8")

        def run(path):
            segs, _ = m.transcribe(path, language="uz", beam_size=5)
            return " ".join(s.text.strip() for s in segs)
        return run

    import torch
    from transformers import pipeline
    from faster_whisper.audio import decode_audio  # 16 kHz mono float32, any format
    if torch.cuda.is_available():
        dev, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        dev, dtype = "mps", torch.float16
    else:
        dev, dtype = "cpu", torch.float32

    if backend == "hf-whisper":
        pipe = pipeline("automatic-speech-recognition", model=repo, device=dev,
                        torch_dtype=dtype, chunk_length_s=30)
        gen = GENERATE_KWARGS.get(repo, DEFAULT_GENERATE_KWARGS)

        def run(path):
            audio = decode_audio(path)
            out = pipe({"raw": audio, "sampling_rate": 16000}, generate_kwargs=dict(gen))
            return out["text"].strip()
        return run

    if backend == "hf-ctc":
        pipe = pipeline("automatic-speech-recognition", model=repo, device=dev,
                        torch_dtype=dtype, chunk_length_s=20)

        def run(path):
            audio = decode_audio(path)
            text = pipe({"raw": audio, "sampling_rate": 16000})["text"].strip()
            return text.replace("[UNK]", "ʻ")  # its vocab lacks the oʻ/gʻ apostrophe
        return run

    raise ValueError(backend)
