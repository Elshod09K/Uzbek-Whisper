# Pseudo-label filter spec (`agreement_filter.py`, not implemented yet)

Second pass over `manifest.jsonl` from `prepare_dataset.py`. It decides which machine-transcribed clips are good enough to train on without a human check. All thresholds are **starting points**. Tune them with the calibration step at the end.

## Inputs per clip
- `text_a`, `avg_logprob`, `no_speech_prob`, `compression_ratio`: from model A (Kotib, already in the manifest)
- `text_b`, `avg_logprob_b`: from model B, a second pass with a *different* model. Prefer `Abduqayum/whisper-uzbek-medium-callcenter` (different training data, call-domain); otherwise `navai-uz/whisper-medium-uzbek`. Use the same decoding settings: `language="uz"`, `beam_size=5`, `condition_on_previous_text=False`, no VAD (clips are already segmented).
- `duration` in seconds

Normalise both texts with the **shared normaliser** used in training and evaluation before any comparison. Rules 3, 7 and 8 run on the **raw** text; rules 6 and 9 on normalised text.

## Rules, applied in order (first failing rule = drop, record the reason)

| # | Rule | Drop if | Reason code |
|---|---|---|---|
| 1 | Duration | `duration < 1.0` or `> 28` | `duration` |
| 2 | Empty | either normalised text is empty | `empty` |
| 3 | Silence | `no_speech_prob > 0.6` **and** `avg_logprob < -0.7` | `no_speech` |
| 4 | Confidence | `avg_logprob < -0.8` (model A) | `low_conf` |
| 5 | Looping | `compression_ratio > 2.2`, **or** any 1–4-gram repeated ≥ 4 times in a row | `repeat` |
| 6 | Speaking rate | for `duration ≥ 2 s`: chars/second (normalised, no spaces) `< 4` or `> 22` | `rate` |
| 7 | Script | any Cyrillic character, or Kazakh-specific letters `қ ғ ү ұ ә ө і һ` | `script` |
| 8 | Hallucination strings | text matches the list below (case-insensitive substring) | `halluc` |
| 9 | Agreement | `max(WER(A,B), WER(B,A)) > 0.15` **or** `CER(A,B) > 0.08` | `disagree` |

**Kept text:** `text_a` (the model being fine-tuned), unless `avg_logprob_b > avg_logprob + 0.2`; then use `text_b`.

**Tiers:** `tier = "high"` if `WER(A,B) ≤ 0.05`, else `"mid"`. The first training runs can use only `high`, then add `mid` and compare on the test set.

## Hallucination string list (substrings, case-insensitive)
```
subtitr          amara org        tahrirlovchi     obuna bo'ling
kanalimizga      ko'rganingiz uchun rahmat         tomosha qilganingiz
musiqa           www              http             thank you for watching
продолжение      субтитр          редактор         ♪
```
Don't add common call phrases such as *rahmat* on their own. They really occur in calls. Rule 5 catches looped *rahmat rahmat rahmat*.

## Output
`manifest.filtered.jsonl`: all input fields plus `text_b`, `wer_ab`, `cer_ab`, `keep`, `reason`, `tier`, `final_text`. Print the keep rate overall and per reason, and kept hours by tier.

## Calibration (once, before trusting the filter)
1. Take ~200 human-reviewed clips and run the filter on them.
2. Among clips the filter **keeps**, measure WER of `final_text` against the human text. Target: **≤ 10%** for `mid`, **≤ 5%** for `high`.
3. Too high: tighten rules 9 and 4. Keep rate < 20% with WER well under target: loosen them.
4. Re-check after each fine-tuning round (the fine-tuned model becomes model A).
