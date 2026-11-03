"""Record your voice (or upload a file) and compare Uzbek speech-to-text models.

    pip install -r requirements.txt
    python app.py          # then open http://127.0.0.1:7860

Each model downloads on first use (0.3-1.6 GB each).
"""
import time

import gradio as gr

from uz_models import MODELS, load_model

NAMES = list(MODELS)


def transcribe(audio_path, model_name):
    if not audio_path:
        return "Avval ovoz yozing yoki fayl yuklang. (Record or upload audio first.)"
    t0 = time.time()
    run = load_model(model_name)
    t1 = time.time()
    text = run(audio_path)
    load_note = f", {t1 - t0:.0f}s loading" if t1 - t0 > 1 else ""
    return f"{text}\n\n— {model_name}: {time.time() - t1:.1f}s{load_note}"


def compare_all(audio_path):
    if not audio_path:
        return "Avval ovoz yozing yoki fayl yuklang. (Record or upload audio first.)"
    out = []
    for name in NAMES:
        try:
            out.append(f"### {name}\n{load_model(name)(audio_path)}")
        except Exception as e:
            out.append(f"### {name}\n(error: {e})")
        load_model.cache_clear()  # free memory before the next model
    return "\n\n".join(out)


with gr.Blocks(title="Uzbek STT") as demo:
    gr.Markdown("# Uzbek speech-to-text\nMikrofon tugmasini bosing, o'zbek tilida gapiring, so'ng tugmani bosing.")
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ovoz / Audio")
    model = gr.Dropdown(NAMES, value=NAMES[0], label="Model")
    with gr.Row():
        btn = gr.Button("Transcribe", variant="primary")
        btn_all = gr.Button("Compare all models (slow)")
    out = gr.Textbox(label="Matn / Transcript", lines=6)
    out_all = gr.Markdown()
    btn.click(transcribe, [audio, model], out)
    btn_all.click(compare_all, audio, out_all)

if __name__ == "__main__":
    demo.launch()
