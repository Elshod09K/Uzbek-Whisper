"""GPU worker: runs on the teammate's PC. Polls the server for batches, transcribes, uploads clips.

    python webapp/worker/worker.py --server https://stt.example.uz --token <WORKER_TOKEN> --name home-4060

Only outgoing HTTPS connections, so no port forwarding is needed. Safe to stop with Ctrl+C
and start again: the server remembers which calls are done.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for pipeline.py
from pipeline import load_model, transcribe_call, wav_bytes  # noqa: E402

CACHE = Path(os.environ.get("UZSTT_WORKER_CACHE", Path.home() / ".uzstt_worker"))


class Server:
    def __init__(self, url, token, name):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "X-Worker-Name": name})

    def get(self, path, **kw):
        r = self.s.get(self.url + path, timeout=kw.pop("timeout", 60), **kw)
        r.raise_for_status()
        return r

    def post(self, path, **kw):
        r = self.s.post(self.url + path, timeout=kw.pop("timeout", 600), **kw)
        r.raise_for_status()
        return r


def model_path(server, m):
    """Return something faster-whisper can load: a HF repo id or a local CT2 folder."""
    if m["source"].startswith("hf:"):
        return m["source"][3:]
    target = CACHE / "models" / str(m["id"])
    if not (target / "model.bin").exists():
        print(f"Downloading model {m['name']} ...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        zpath = target.with_suffix(".zip")
        with server.get(f"/api/worker/model/{m['id']}.zip", stream=True, timeout=600) as r, open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target)
        zpath.unlink()
        # allow zips that contain a single top-level folder
        inner = [p for p in target.iterdir() if p.is_dir()]
        if not (target / "model.bin").exists() and len(inner) == 1 and (inner[0] / "model.bin").exists():
            for p in inner[0].iterdir():
                p.rename(target / p.name)
    return str(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", default=os.environ.get("UZSTT_WORKER_TOKEN"))
    ap.add_argument("--name", default=os.environ.get("UZSTT_WORKER_NAME", "worker"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks when idle")
    ap.add_argument("--once", action="store_true", help="exit when there is no work (for testing)")
    a = ap.parse_args()
    if not a.token:
        sys.exit("--token is required (find it on the admin dashboard)")

    server = Server(a.server, a.token, a.name)
    loaded = (None, None)  # (model id, WhisperModel)
    print(f"Worker '{a.name}' polling {a.server}", flush=True)

    while True:
        try:
            job = server.get("/api/worker/job").json()
        except Exception as e:
            print(f"server unreachable ({e}); retrying in {a.poll}s", flush=True)
            time.sleep(a.poll)
            continue
        if not job["batch_id"]:
            if a.once:
                print("No work. Exiting (--once).")
                return
            time.sleep(a.poll)
            continue

        m = job["model"]
        if loaded[0] != m["id"]:
            print(f"Loading model: {m['name']}", flush=True)
            loaded = (m["id"], load_model(model_path(server, m), a.device))
        model = loaded[1]

        calls = job["calls"]
        print(f"Batch {job['batch_id']}: {len(calls)} calls to transcribe", flush=True)
        for i, c in enumerate(calls, 1):
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / Path(c["filename"]).name
                try:
                    with server.get(f"/api/worker/call/{c['id']}/audio", stream=True, timeout=600) as r, \
                            open(local, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    meta, files = [], {}
                    for j, (rec, piece) in enumerate(transcribe_call(model, str(local))):
                        meta.append({k: rec[k] for k in ("channel", "start", "end", "text", "avg_logprob",
                                                          "no_speech_prob", "compression_ratio")})
                        files[f"clip_{j}"] = (f"{j}.wav", wav_bytes(piece), "audio/wav")
                    server.post(f"/api/worker/call/{c['id']}/result",
                                data={"meta": json.dumps(meta, ensure_ascii=False)}, files=files)
                    print(f"  [{i}/{len(calls)}] {c['filename']}: {len(meta)} clips, "
                          f"{time.time() - t0:.0f}s", flush=True)
                except requests.HTTPError as e:
                    print(f"  [{i}/{len(calls)}] {c['filename']}: upload failed ({e}); will retry", flush=True)
                    time.sleep(10)
                    break
                except Exception as e:  # bad/unreadable audio: report and move on
                    print(f"  [{i}/{len(calls)}] {c['filename']}: FAILED {e}", flush=True)
                    try:
                        server.post(f"/api/worker/call/{c['id']}/result", data={"error": repr(e)})
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
