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

## Benchmark (20 FLEURS uz_uz dev clips, clean read speech; lower is better)

| Model | WER | CER | Licence |
|---|---|---|---|
| navai-uz/whisper-medium-uzbek | 17.3% | 6.6% | Apache-2.0 |
| Kotib/uzbek_stt_v1 (medium, CT2) | 19.8% | 7.5% | Apache-2.0 |
| Abduqayum/whisper-uzbek-medium-callcenter | 22.2% | 8.9% | Apache-2.0 |
| BuzzASR/uzbek (large-v3) | 25.5% | 6.4% | MIT |
| hostmepanda turbo CT2 | 27.4% | 8.6% | MIT |
| sarahai/uzbek-stt-3 (wav2vec2) | 78.0% | 19.4% | Apache-2.0 |
| openai whisper-large-v3-turbo (no fine-tune) | 114.1% | 81.1% | MIT |

FLEURS is clean read speech; phone-call accuracy will differ. Choose a model on a held-out set of real calls.
Reproduce: `python benchmark.py clips_dir/ dev.tsv` (see `inspect_audio.py` for checking your own WAV archive).

## Building your own training data from calls

1. **Auto-transcribe** calls into short clips (runs on the RTX 4060 desktop; resumable):
   ```bash
   python prepare_dataset.py /path/to/calls dataset/ --limit 50
   ```
   Splits stereo into agent/customer channels, cuts speech into clips under 30 s, transcribes them with
   Kotib uzbek_stt_v1, and writes `dataset/manifest.jsonl` with a confidence-based `keep` flag.
2. **Human review**: listen, correct, save:
   ```bash
   python review_app.py dataset/ --share-lan
   ```
   Reviewers on the office network open `http://<PC-IP>:7861`. Saves go to `dataset/reviewed.jsonl`.
3. **Fine-tune** Kotib on the reviewed clips (next step).

For the multi-user web workflow (server + remote GPU worker + reviewer accounts), see [`webapp/README.md`](webapp/README.md).
