"""Compare all models in uz_models.py on a folder of clips with FLEURS-style references.

    python benchmark.py clips_dir/ dev.tsv                 # all models
    python benchmark.py clips_dir/ dev.tsv BuzzASR Kotib   # only names containing these words
"""
import os
import sys
import time

import jiwer

from test_model import load_refs, normalize
from uz_models import MODELS, load_model

clip_dir, tsv = sys.argv[1], sys.argv[2]
refs = load_refs(tsv)
files = sorted(f for f in os.listdir(clip_dir) if f in refs)
gold = [normalize(refs[f]) for f in files]

wanted = sys.argv[3:]
for name in MODELS:
    if wanted and not any(w.lower() in name.lower() for w in wanted):
        continue
    t0 = time.time()
    run = load_model(name)
    hyps = [normalize(run(os.path.join(clip_dir, f))) for f in files]
    load_model.cache_clear()
    print(f"{name:40s} WER {jiwer.wer(gold, hyps):6.1%}  CER {jiwer.cer(gold, hyps):6.1%}  "
          f"({time.time() - t0:.0f}s, {len(files)} clips)", flush=True)
