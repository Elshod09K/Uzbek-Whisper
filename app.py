"""Record your voice (or upload a file) in the browser and transcribe it in Uzbek.

    pip install -r requirements.txt
    python app.py          # then open http://127.0.0.1:7860
"""
import time

import gradio as gr
from faster_whisper import WhisperModel

MODEL_ID = "hostmepanda/whisper-large-v3-turbo-uzbek-ct2"

try:
    import ctranslate2
    DEVICE = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
except Exception:
    DEVICE = "cpu"
COMPUTE = "float16" if DEVICE == "cuda" else "int8"

print(f"Loading {MODEL_ID} on {DEVICE} ({COMPUTE}) — first run downloads ~1.6 GB...")
model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE)


def transcribe(audio_path):
    if not audio_path:
        return "Avval ovoz yozing yoki fayl yuklang. (Record or upload audio first.)"
    t0 = time.time()
    segments, info = model.transcribe(audio_path, language="uz", beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments)
    return f"{text}\n\n— {info.duration:.1f}s audio, {time.time() - t0:.1f}s on {DEVICE}"


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ovoz / Audio"),
    outputs=gr.Textbox(label="Matn / Transcript", lines=6),
    title="Uzbek Whisper (large-v3-turbo, CT2)",
    description="Mikrofon tugmasini bosing, o'zbek tilida gapiring, so'ng Submit tugmasini bosing.",
)

if __name__ == "__main__":
    demo.launch()
