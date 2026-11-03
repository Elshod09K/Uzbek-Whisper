# Uzbek Whisper test

Tests [hostmepanda/whisper-large-v3-turbo-uzbek-ct2](https://huggingface.co/hostmepanda/whisper-large-v3-turbo-uzbek-ct2) with faster-whisper.

```bash
pip install -r requirements.txt
```

**Try your own voice** (browser UI with microphone + file upload):

```bash
python app.py   # open http://127.0.0.1:7860
```

**Transcribe files / measure WER:**

```bash
python test_model.py my_audio.wav
python test_model.py clips_dir/ --refs dev.tsv   # FLEURS-format references
```

Runs on CPU (int8) or NVIDIA GPU (float16) automatically. First run downloads ~1.6 GB.
Sanity check on 5 FLEURS uz_uz dev clips (CPU): WER 16.7%, CER 5.8%.
