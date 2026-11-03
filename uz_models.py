"""Uzbek speech-to-text models behind one interface: load_model(name)(audio_path) -> text."""
import functools

# name -> (backend, repo id)
MODELS = {
    "hostmepanda turbo (CT2)":            ("ct2", "hostmepanda/whisper-large-v3-turbo-uzbek-ct2"),
    "Kotib uzbek_stt_v1 medium (CT2)":    ("ct2", "yuldashev222/uzbek_stt_v1-ct2"),
    "navai-uz whisper-medium-uzbek":      ("hf-whisper", "navai-uz/whisper-medium-uzbek"),
    "sarahai uzbek-stt-3 (wav2vec2)":     ("hf-ctc", "sarahai/uzbek-stt-3"),
    "OpenAI whisper-large-v3-turbo (base)": ("ct2", "large-v3-turbo"),
}


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
    dev = 0 if torch.cuda.is_available() else -1

    if backend == "hf-whisper":
        pipe = pipeline("automatic-speech-recognition", model=repo, device=dev, chunk_length_s=30)

        def run(path):
            audio = decode_audio(path)
            out = pipe({"raw": audio, "sampling_rate": 16000},
                       generate_kwargs={"language": "uz", "task": "transcribe", "num_beams": 5})
            return out["text"].strip()
        return run

    if backend == "hf-ctc":
        pipe = pipeline("automatic-speech-recognition", model=repo, device=dev, chunk_length_s=20)

        def run(path):
            audio = decode_audio(path)
            return pipe({"raw": audio, "sampling_rate": 16000})["text"].strip()
        return run

    raise ValueError(backend)
