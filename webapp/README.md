# Uzbek STT data platform

Calls → **worker** (GPU PC) transcribes with the active model → **reviewers** correct each clip on the
website → dashboard says **Ready to train** → export → train → upload the new model → activate → repeat.

```
admin  ──► /admin   inbox, batches, progress, ready-to-train, models, users, export, settings
reviewer ► /review  only the review page (reviewers cannot open /admin)
worker ──► /api/worker/*   (token auth; the worker connects out, no ports open at the worker's home)
```

## The cycle
1. Put call files in the inbox (upload on the dashboard, or copy into `$UZSTT_DATA/inbox/`).
2. **First batch: tick "test set"** (~30–50 calls). It's never used for training, only for measuring models.
3. The worker picks up the batch, transcribes with the **active** model, and uploads clips.
4. Reviewers correct clips on `/review` (test-set clips come first).
5. When the batches in review are **75% done** (setting), the next batch is queued automatically,
   so the worker transcribes while reviewers keep working.
6. When there are enough new reviewed hours (setting, default 20 h) plus a test set, the dashboard shows
   **Ready to train** → *Start training round* → download the **Training set** and **Test set** exports.
7. Train. Upload the new CT2 model as a zip on the dashboard and **Activate** it. The next batches are
   transcribed with the better model.

## Server setup (Ubuntu, with your domain)
```bash
sudo useradd -m -s /bin/bash uzstt && sudo mkdir -p /srv/uzstt && sudo chown uzstt /srv/uzstt
sudo git clone -b claude/intelligent-sagan-8tzjhs https://github.com/elshod09k/uzbek-whisper.git /opt/uzbek-whisper
sudo chown -R uzstt /opt/uzbek-whisper && cd /opt/uzbek-whisper
sudo -u uzstt python3 -m venv .venv && sudo -u uzstt .venv/bin/pip install -r webapp/requirements-server.txt
sudo -u uzstt env UZSTT_DATA=/srv/uzstt .venv/bin/python -m webapp.server.manage add-user YOURNAME --role admin
sudo cp webapp/deploy/uzstt.service /etc/systemd/system/ && sudo systemctl enable --now uzstt
sudo apt install caddy   # then put webapp/deploy/Caddyfile (with your domain) in /etc/caddy/Caddyfile
sudo systemctl reload caddy
```
Point the domain's DNS A record (e.g. `stt.yourdomain.uz`) to the server first. Caddy issues HTTPS
automatically. The worker token is on the dashboard (Workers → Worker token).

Disk: the server stores calls + clips. Roughly 2× the size of the calls you put through it.

## Worker setup (the RTX 4060 PC)
Install Python 3.10+, git, and an NVIDIA driver. Then:
```bash
git clone -b claude/intelligent-sagan-8tzjhs https://github.com/elshod09k/uzbek-whisper.git
cd uzbek-whisper
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r webapp/requirements-worker.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*   # GPU libraries for faster-whisper on Windows/Linux
python webapp/worker/worker.py --server https://stt.yourdomain.uz --token TOKEN --name home-4060
```
On Windows you can edit and double-click `webapp/deploy/uzstt-worker.bat` instead (it restarts automatically).
If the PC is off, batches simply wait. When it comes back, the worker continues where it stopped.

## Local test
```bash
pip install -r webapp/requirements-server.txt
UZSTT_DATA=./data python -m webapp.server.manage add-user admin --role admin
UZSTT_DATA=./data python -m uvicorn webapp.server.app:app --port 8000   # http://127.0.0.1:8000
```
