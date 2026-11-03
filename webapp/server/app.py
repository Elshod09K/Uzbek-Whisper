"""Uzbek STT data platform: batches of calls → worker transcription → human review → training rounds.

Run (from repo root):
    UZSTT_DATA=/srv/uzstt python -m uvicorn webapp.server.app:app --host 127.0.0.1 --port 8000
Create the first admin:
    UZSTT_DATA=/srv/uzstt python -m webapp.server.manage add-user admin --role admin
"""
import json
import os
import secrets
import shutil
import tarfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import db

HERE = Path(__file__).parent
REPO = HERE.parent.parent
CLAIM_TIMEOUT = 15 * 60
WORKER_STALE = 60 * 60
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

db.init()
D = Path(db.DATA_DIR)


def _secret(name):
    p = D / name
    if not p.exists():
        p.write_text(secrets.token_urlsafe(32))
        p.chmod(0o600)
    return p.read_text().strip()


SECRET_KEY = _secret("secret_key")
WORKER_TOKEN = os.environ.get("UZSTT_WORKER_TOKEN") or _secret("worker_token")

app = FastAPI(title="Uzbek STT platform", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax",
                   https_only=os.environ.get("UZSTT_HTTPS", "0") == "1", max_age=14 * 86400)
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.filters["hours"] = lambda s: f"{(s or 0) / 3600:.1f} h" if (s or 0) >= 3600 else f"{(s or 0) / 60:.0f} min"
templates.env.filters["ago"] = lambda t: "never" if not t else (
    f"{int(time.time() - t)}s ago" if time.time() - t < 120 else
    f"{int((time.time() - t) / 60)} min ago" if time.time() - t < 7200 else
    time.strftime("%Y-%m-%d %H:%M", time.localtime(t)))


# ---------------------------------------------------------------- auth
class Redirect(Exception):
    def __init__(self, to):
        self.to = to


@app.exception_handler(Redirect)
async def _redirect(request, exc):
    return RedirectResponse(exc.to, status_code=303)


def current_user(request: Request):
    uid = request.session.get("uid")
    u = db.one("SELECT * FROM users WHERE id=?", uid) if uid else None
    if not u:
        if request.url.path.startswith("/api/"):
            raise HTTPException(401, "login required")
        raise Redirect("/login")
    return u


def admin_user(u=Depends(current_user)):
    if u["role"] != "admin":
        raise HTTPException(403, "admins only")
    return u


def worker_auth(request: Request):
    tok = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(tok, WORKER_TOKEN):
        raise HTTPException(401, "bad worker token")
    return request.headers.get("x-worker-name", "worker")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    u = db.one("SELECT * FROM users WHERE username=?", username.strip())
    if not u or not db.check_pw(password, u["pw_hash"]):
        time.sleep(1)
        return RedirectResponse("/login?error=1", status_code=303)
    request.session.clear()
    request.session["uid"] = u["id"]
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def home(u=Depends(current_user)):
    return RedirectResponse("/admin" if u["role"] == "admin" else "/review", status_code=303)


# ---------------------------------------------------------------- pipeline logic
def inbox_files():
    out = []
    for root, _, files in os.walk(D / "inbox"):
        out += [Path(root) / f for f in sorted(files) if f.lower().endswith(AUDIO_EXT)]
    return sorted(out)


def create_batch(n, is_test=False):
    files = inbox_files()[:n]
    if not files:
        return None
    cur = db.run("INSERT INTO batches (status, is_test, created) VALUES ('queued', ?, ?)",
                 int(is_test), time.time())
    bid = cur.lastrowid
    dest = D / "calls" / f"batch_{bid}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        rel = f.relative_to(D / "inbox")
        target = dest / str(rel).replace(os.sep, "__")
        shutil.move(str(f), target)
        db.run("INSERT INTO calls (batch_id, filename, path) VALUES (?,?,?)",
               bid, str(rel), str(target.relative_to(D)))
    return bid


def batch_stats():
    rows = db.q("""
      SELECT b.*, m.name AS model_name,
        (SELECT COUNT(*) FROM calls c WHERE c.batch_id=b.id) AS n_calls,
        (SELECT COUNT(*) FROM calls c WHERE c.batch_id=b.id AND c.status!='pending') AS calls_done,
        (SELECT COUNT(*) FROM clips k WHERE k.batch_id=b.id) AS n_clips,
        (SELECT COUNT(*) FROM clips k WHERE k.batch_id=b.id AND k.status!='pending') AS clips_done,
        (SELECT COALESCE(SUM(k."end"-k.start),0) FROM clips k WHERE k.batch_id=b.id) AS secs
      FROM batches b LEFT JOIN models m ON m.id=b.model_id ORDER BY b.id DESC""")
    out = []
    for r in rows:
        d = dict(r)
        d["frac"] = d["clips_done"] / d["n_clips"] if d["n_clips"] else 0
        out.append(d)
    return out


def refresh_batch_status(bid):
    b = db.one("SELECT status FROM batches WHERE id=?", bid)
    if b and b["status"] == "transcribing":
        if not db.one("SELECT 1 FROM calls WHERE batch_id=? AND status='pending'", bid):
            db.run("UPDATE batches SET status='review', transcribed=? WHERE id=?", time.time(), bid)
    if b and b["status"] in ("review", "transcribing"):
        if (db.one("SELECT 1 FROM clips WHERE batch_id=?", bid)
                and not db.one("SELECT 1 FROM clips WHERE batch_id=? AND status='pending'", bid)
                and not db.one("SELECT 1 FROM calls WHERE batch_id=? AND status='pending'", bid)):
            db.run("UPDATE batches SET status='reviewed' WHERE id=?", bid)


def maybe_queue_next():
    """Keep the worker busy: when the reviewers are close to running out, queue the next batch."""
    if not db.setting("auto_next", int):
        return None
    if not db.one("SELECT 1 FROM batches"):  # the admin starts the pipeline (and the test set) by hand
        return None
    if db.one("SELECT 1 FROM batches WHERE status IN ('queued','transcribing')"):
        return None
    threshold = db.setting("auto_next_at", float)
    in_review = [b for b in batch_stats() if b["status"] == "review"]
    if all(b["frac"] >= threshold for b in in_review):
        return create_batch(db.setting("batch_size", int))
    return None


USABLE = "k.status IN ('ok','corrected') AND k.text NOT LIKE '%[?]%' AND TRIM(k.text)!=''"


def training_status():
    last = db.one("SELECT * FROM training_rounds ORDER BY id DESC LIMIT 1")
    upto = last["clip_upto"] if last else 0
    new_train = db.one(f"""SELECT COALESCE(SUM(k."end"-k.start),0) AS s, COUNT(*) AS n FROM clips k
        JOIN batches b ON b.id=k.batch_id WHERE b.is_test=0 AND {USABLE} AND k.id>?""", upto)
    all_train = db.one(f"""SELECT COALESCE(SUM(k."end"-k.start),0) AS s FROM clips k
        JOIN batches b ON b.id=k.batch_id WHERE b.is_test=0 AND {USABLE}""")["s"]
    test = db.one(f"""SELECT COALESCE(SUM(k."end"-k.start),0) AS s FROM clips k
        JOIN batches b ON b.id=k.batch_id WHERE b.is_test=1 AND {USABLE}""")["s"]
    need_new = db.setting("ready_hours", float) * 3600
    need_test = db.setting("min_test_hours", float) * 3600
    return {"new_train_s": new_train["s"], "new_train_n": new_train["n"], "all_train_s": all_train,
            "test_s": test, "need_new_s": need_new, "need_test_s": need_test,
            "ready": new_train["s"] >= need_new and test >= need_test, "last": last}


# ---------------------------------------------------------------- admin
@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, u=Depends(admin_user), msg: str = ""):
    maybe_queue_next()
    workers = db.q("SELECT worker, MAX(worker_seen) AS seen FROM batches WHERE worker IS NOT NULL GROUP BY worker")
    return templates.TemplateResponse(request, "admin.html", {
        "u": u, "msg": msg, "batches": batch_stats(), "inbox": len(inbox_files()),
        "models": db.q("SELECT * FROM models ORDER BY id DESC"),
        "users": db.q("SELECT id, username, role, created FROM users ORDER BY role, username"),
        "settings": {r["key"]: r["value"] for r in db.q("SELECT * FROM settings")},
        "train": training_status(), "rounds": db.q("SELECT * FROM training_rounds ORDER BY id DESC"),
        "workers": workers, "worker_token": WORKER_TOKEN,
        "reviewers": db.q("""SELECT reviewer, COUNT(*) n, SUM("end"-start) secs, SUM(seconds_spent) spent,
            SUM(status='corrected') corrected, MAX(reviewed) last FROM clips
            WHERE reviewer IS NOT NULL GROUP BY reviewer ORDER BY n DESC"""),
    })


def back(msg):
    return RedirectResponse(f"/admin?msg={msg}", status_code=303)


@app.post("/admin/inbox/upload")
async def upload_calls(files: list[UploadFile] = File(...), u=Depends(admin_user)):
    n = 0
    for f in files:
        name = Path(f.filename or "").name
        if not name.lower().endswith(AUDIO_EXT):
            continue
        dest = D / "inbox" / name
        i = 1
        while dest.exists():
            dest = D / "inbox" / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out, 1 << 20)
        n += 1
    return back(f"Uploaded {n} files to the inbox")


@app.post("/admin/batches/new")
def new_batch(n: int = Form(...), is_test: str = Form(""), u=Depends(admin_user)):
    bid = create_batch(max(1, n), is_test=bool(is_test))
    return back(f"Batch {bid} queued" if bid else "Inbox is empty")


@app.post("/admin/batches/{bid}/delete")
def delete_batch(bid: int, u=Depends(admin_user)):
    b = db.one("SELECT * FROM batches WHERE id=?", bid)
    if b and b["status"] == "queued":  # only untouched batches: move calls back to the inbox
        for c in db.q("SELECT * FROM calls WHERE batch_id=?", bid):
            src = D / c["path"]
            if src.exists():
                shutil.move(str(src), D / "inbox" / src.name)
        db.run("DELETE FROM calls WHERE batch_id=?", bid)
        db.run("DELETE FROM batches WHERE id=?", bid)
        return back(f"Batch {bid} removed, calls back in inbox")
    return back("Only queued batches can be removed")


@app.post("/admin/settings")
async def save_settings(request: Request, u=Depends(admin_user)):
    form = await request.form()
    for k in db.DEFAULT_SETTINGS:
        if k == "auto_next":
            db.run("UPDATE settings SET value=? WHERE key=?", "1" if form.get(k) else "0", k)
        elif k in form:
            float(form[k])  # validate
            db.run("UPDATE settings SET value=? WHERE key=?", str(form[k]), k)
    return back("Settings saved")


@app.post("/admin/models/add")
def add_model(name: str = Form(...), repo: str = Form(...), u=Depends(admin_user)):
    db.run("INSERT INTO models (name, source, created) VALUES (?,?,?)", name, "hf:" + repo.strip(), time.time())
    return back("Model added (not active yet)")


@app.post("/admin/models/upload")
async def upload_model(name: str = Form(...), file: UploadFile = File(...), notes: str = Form(""),
                       u=Depends(admin_user)):
    cur = db.run("INSERT INTO models (name, source, notes, created) VALUES (?,?,?,?)",
                 name, "upload", notes, time.time())
    with open(D / "models" / f"{cur.lastrowid}.zip", "wb") as out:
        shutil.copyfileobj(file.file, out, 1 << 20)
    return back("Model uploaded (not active yet)")


@app.post("/admin/models/{mid}/activate")
def activate_model(mid: int, u=Depends(admin_user)):
    db.run("UPDATE models SET active=(id=?)", mid)
    return back("Active model changed; the next batch will be transcribed with it")


@app.post("/admin/users/add")
def add_user(username: str = Form(...), password: str = Form(...), role: str = Form("reviewer"),
             u=Depends(admin_user)):
    if role not in ("admin", "reviewer") or len(password) < 8:
        return back("Password must be at least 8 characters")
    try:
        db.run("INSERT INTO users (username, pw_hash, role, created) VALUES (?,?,?,?)",
               username.strip(), db.hash_pw(password), role, time.time())
    except Exception:
        return back("Username already exists")
    return back(f"User {username} created")


@app.post("/admin/users/{uid}/delete")
def delete_user(uid: int, u=Depends(admin_user)):
    if uid == u["id"]:
        return back("You cannot delete yourself")
    db.run("DELETE FROM users WHERE id=?", uid)
    return back("User deleted")


@app.post("/admin/rounds/start")
def start_round(notes: str = Form(""), u=Depends(admin_user)):
    t = training_status()
    upto = db.one("SELECT COALESCE(MAX(id),0) m FROM clips WHERE status!='pending'")["m"]
    db.run("INSERT INTO training_rounds (created, clip_upto, train_hours, test_hours, notes) VALUES (?,?,?,?,?)",
           time.time(), upto, t["all_train_s"] / 3600, t["test_s"] / 3600, notes)
    return back("Training round recorded. Download the train and test exports below.")


@app.get("/admin/export")
def export(kind: str = "train", u=Depends(admin_user)):
    """tar.gz with manifest.jsonl + clips. kind: train | test | all-reviewed."""
    cond = {"train": f"b.is_test=0 AND {USABLE}", "test": f"b.is_test=1 AND {USABLE}",
            "all-reviewed": "k.status!='pending'"}.get(kind)
    if not cond:
        raise HTTPException(400, "kind must be train, test or all-reviewed")
    rows = db.q(f"""SELECT k.*, b.is_test, c.filename AS call FROM clips k JOIN batches b ON b.id=k.batch_id
                   JOIN calls c ON c.id=k.call_id WHERE {cond} ORDER BY k.id""")
    out = D / "exports" / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        lines = []
        for r in rows:
            arc = f"clips/{r['id']}.wav"
            tar.add(D / r["path"], arcname=arc)
            lines.append(json.dumps({
                "audio": arc, "text": r["text"], "model_text": r["model_text"], "status": r["status"],
                "duration": round(r["end"] - r["start"], 2), "call": r["call"], "channel": r["channel"],
                "batch": r["batch_id"], "is_test": bool(r["is_test"]), "reviewer": r["reviewer"],
            }, ensure_ascii=False))
        data = ("\n".join(lines) + "\n").encode()
        info = tarfile.TarInfo("manifest.jsonl")
        info.size = len(data)
        info.mtime = int(time.time())
        import io
        tar.addfile(info, io.BytesIO(data))
    return FileResponse(out, filename=out.name, media_type="application/gzip")


# ---------------------------------------------------------------- review
@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request, u=Depends(current_user)):
    return templates.TemplateResponse(request, "review.html", {"u": u})


@app.get("/guide", response_class=HTMLResponse)
def guide(request: Request, u=Depends(current_user)):
    text = (REPO / "ANNOTATION_GUIDE.md").read_text(encoding="utf-8")
    return templates.TemplateResponse(request, "guide.html", {"u": u, "text": text})


def _claim_next(username):
    now = time.time()
    with_lock = db.conn()
    with_lock.execute("BEGIN IMMEDIATE")
    try:
        # keep a clip this reviewer already holds (e.g. after a page reload)
        r = with_lock.execute("""SELECT k.id FROM clips k WHERE k.status='pending' AND k.claimed_by=?
            AND k.claimed_at>? LIMIT 1""", (username, now - CLAIM_TIMEOUT)).fetchone()
        if not r:
            r = with_lock.execute("""SELECT k.id FROM clips k JOIN batches b ON b.id=k.batch_id
                WHERE k.status='pending' AND b.status IN ('review','transcribing')
                  AND (k.claimed_by IS NULL OR k.claimed_at<?)
                ORDER BY b.is_test DESC, b.id, k.call_id, k.start, k.channel LIMIT 1""",
                (now - CLAIM_TIMEOUT,)).fetchone()
        if r:
            with_lock.execute("UPDATE clips SET claimed_by=?, claimed_at=? WHERE id=?", (username, now, r["id"]))
        with_lock.execute("COMMIT")
    except Exception:
        with_lock.execute("ROLLBACK")
        raise
    return r["id"] if r else None


def _my_stats(username):
    today = time.time() - (time.time() % 86400)
    r = db.one("""SELECT COUNT(*) n, COALESCE(SUM("end"-start),0) secs FROM clips
                  WHERE reviewer=? AND reviewed>?""", username, today)
    left = db.one("""SELECT COUNT(*) n FROM clips k JOIN batches b ON b.id=k.batch_id
                     WHERE k.status='pending' AND b.status IN ('review','transcribing')""")["n"]
    return {"today_n": r["n"], "today_min": round(r["secs"] / 60, 1), "left": left}


@app.get("/api/review/next")
def review_next(u=Depends(current_user)):
    cid = _claim_next(u["username"])
    if cid is None:
        return {"clip": None, "stats": _my_stats(u["username"])}
    k = db.one("""SELECT k.*, c.filename AS call, b.is_test FROM clips k JOIN calls c ON c.id=k.call_id
                  JOIN batches b ON b.id=k.batch_id WHERE k.id=?""", cid)
    return {"clip": {"id": k["id"], "text": k["model_text"], "call": k["call"], "channel": k["channel"],
                     "start": k["start"], "duration": round(k["end"] - k["start"], 1),
                     "confidence": k["avg_logprob"], "is_test": bool(k["is_test"]),
                     "audio": f"/clip/{k['id']}.wav"},
            "stats": _my_stats(u["username"])}


@app.post("/api/review/{cid}")
async def review_save(cid: int, request: Request, u=Depends(current_user)):
    body = await request.json()
    action = body.get("action")
    k = db.one("SELECT * FROM clips WHERE id=?", cid)
    if not k:
        raise HTTPException(404)
    if action == "skip":
        db.run("UPDATE clips SET claimed_by='~skipped', claimed_at=? WHERE id=? AND status='pending'",
               time.time(), cid)
        return {"ok": True}
    if action not in ("save", "bad_audio"):
        raise HTTPException(400, "bad action")
    text = " ".join(str(body.get("text", "")).split())
    status = "bad_audio" if action == "bad_audio" else (
        "ok" if text == " ".join((k["model_text"] or "").split()) else "corrected")
    if status != "bad_audio" and not text:
        raise HTTPException(400, "empty text: use Bad audio instead")
    db.run("""UPDATE clips SET status=?, text=?, reviewer=?, reviewed=?, seconds_spent=?, claimed_by=NULL
              WHERE id=?""", status, "" if status == "bad_audio" else text, u["username"], time.time(),
           float(body.get("seconds") or 0), cid)
    refresh_batch_status(k["batch_id"])
    maybe_queue_next()
    return {"ok": True, "status": status}


@app.get("/clip/{cid}.wav")
def clip_audio(cid: int, u=Depends(current_user)):
    k = db.one("SELECT path FROM clips WHERE id=?", cid)
    if not k:
        raise HTTPException(404)
    return FileResponse(D / k["path"], media_type="audio/wav")


# ---------------------------------------------------------------- worker API
@app.get("/api/worker/job")
def worker_job(name=Depends(worker_auth)):
    now = time.time()
    b = db.one("SELECT * FROM batches WHERE status='transcribing' AND worker=? ORDER BY id LIMIT 1", name)
    if not b:  # take over a batch whose worker disappeared, else claim the next queued one
        b = db.one("""SELECT * FROM batches WHERE status='transcribing' AND worker_seen<? ORDER BY id LIMIT 1""",
                   now - WORKER_STALE) or db.one("SELECT * FROM batches WHERE status='queued' ORDER BY id LIMIT 1")
        if not b:
            db.run("UPDATE batches SET worker_seen=? WHERE worker=?", now, name)
            return {"batch_id": None}
        model_id = b["model_id"] or db.one("SELECT id FROM models WHERE active=1")["id"]
        db.run("UPDATE batches SET status='transcribing', worker=?, worker_seen=?, model_id=? WHERE id=?",
               name, now, model_id, b["id"])
        b = db.one("SELECT * FROM batches WHERE id=?", b["id"])
    db.run("UPDATE batches SET worker_seen=? WHERE id=?", now, b["id"])
    m = db.one("SELECT * FROM models WHERE id=?", b["model_id"])
    calls = db.q("SELECT id, filename FROM calls WHERE batch_id=? AND status='pending' ORDER BY id", b["id"])
    return {"batch_id": b["id"], "model": {"id": m["id"], "name": m["name"], "source": m["source"]},
            "calls": [dict(c) for c in calls]}


@app.get("/api/worker/call/{call_id}/audio")
def worker_call_audio(call_id: int, name=Depends(worker_auth)):
    c = db.one("SELECT path FROM calls WHERE id=?", call_id)
    if not c:
        raise HTTPException(404)
    return FileResponse(D / c["path"])


@app.post("/api/worker/call/{call_id}/result")
async def worker_call_result(call_id: int, request: Request, name=Depends(worker_auth)):
    c = db.one("SELECT * FROM calls WHERE id=?", call_id)
    if not c:
        raise HTTPException(404)
    form = await request.form()
    if form.get("error"):
        db.run("UPDATE calls SET status='failed', error=? WHERE id=?", str(form["error"])[:2000], call_id)
    else:
        records = json.loads(form["meta"])
        out_dir = D / "clips" / f"batch_{c['batch_id']}" / f"call_{call_id}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        db.run("DELETE FROM clips WHERE call_id=? AND status='pending'", call_id)
        for i, r in enumerate(records):
            path = out_dir / f"{i:04d}_ch{r['channel']}.wav"
            data = await form[f"clip_{i}"].read()
            path.write_bytes(data)
            db.run("""INSERT INTO clips (batch_id, call_id, channel, start, "end", path, model_text,
                      avg_logprob, no_speech_prob, compression_ratio) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                   c["batch_id"], call_id, r["channel"], r["start"], r["end"], str(path.relative_to(D)),
                   r["text"], r["avg_logprob"], r["no_speech_prob"], r["compression_ratio"])
        db.run("UPDATE calls SET status='done', error=NULL WHERE id=?", call_id)
    db.run("UPDATE batches SET worker_seen=? WHERE id=?", time.time(), c["batch_id"])
    refresh_batch_status(c["batch_id"])
    return {"ok": True}


@app.get("/api/worker/model/{mid}.zip")
def worker_model(mid: int, name=Depends(worker_auth)):
    p = D / "models" / f"{mid}.zip"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/zip")


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True})
