"""SQLite storage for the review platform."""
import hashlib
import hmac
import os
import sqlite3
import threading
import time

DATA_DIR = os.path.abspath(os.environ.get("UZSTT_DATA", "data"))
DB_PATH = os.path.join(DATA_DIR, "platform.db")
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin','reviewer')), created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,  -- 'hf:<repo>' or 'upload'
  active INTEGER NOT NULL DEFAULT 0, notes TEXT DEFAULT '', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
  id INTEGER PRIMARY KEY, status TEXT NOT NULL,  -- queued, transcribing, review, reviewed
  is_test INTEGER NOT NULL DEFAULT 0, model_id INTEGER, created REAL NOT NULL,
  worker TEXT, worker_seen REAL, transcribed REAL);
CREATE TABLE IF NOT EXISTS calls (
  id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL, filename TEXT NOT NULL, path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', error TEXT);  -- pending, done, failed
CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL, call_id INTEGER NOT NULL,
  channel INTEGER, start REAL, "end" REAL, path TEXT NOT NULL,
  model_text TEXT, avg_logprob REAL, no_speech_prob REAL, compression_ratio REAL,
  status TEXT NOT NULL DEFAULT 'pending',  -- pending, ok, corrected, bad_audio
  text TEXT, reviewer TEXT, reviewed REAL, seconds_spent REAL,
  claimed_by TEXT, claimed_at REAL);
CREATE INDEX IF NOT EXISTS clips_queue ON clips(status, batch_id, call_id, channel, start);
CREATE TABLE IF NOT EXISTS training_rounds (
  id INTEGER PRIMARY KEY, created REAL NOT NULL, clip_upto INTEGER NOT NULL,
  train_hours REAL, test_hours REAL, notes TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

DEFAULT_SETTINGS = {
    "batch_size": "50",          # calls per batch
    "auto_next_at": "0.75",      # queue next batch when current one is this fraction reviewed
    "auto_next": "1",
    "ready_hours": "20",         # reviewed training hours needed for a training round
    "min_test_hours": "3",       # reviewed test hours needed before the first round
}


def conn():
    c = getattr(_local, "conn", None)
    if c is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


def init():
    c = conn()
    c.executescript(SCHEMA)
    for k, v in DEFAULT_SETTINGS.items():
        c.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (k, v))
    if not c.execute("SELECT 1 FROM models").fetchone():
        c.execute("INSERT INTO models (name, source, active, notes, created) VALUES (?,?,?,?,?)",
                  ("Kotib uzbek_stt_v1 (baseline)", "hf:yuldashev222/uzbek_stt_v1-ct2", 1,
                   "Starting model", time.time()))
    for d in ("inbox", "calls", "clips", "models", "exports"):
        os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)


def q(sql, *args):
    return conn().execute(sql, args).fetchall()


def one(sql, *args):
    return conn().execute(sql, args).fetchone()


def run(sql, *args):
    return conn().execute(sql, args)


def setting(key, cast=str):
    r = one("SELECT value FROM settings WHERE key=?", key)
    return cast(r["value"] if r else DEFAULT_SETTINGS[key])


# --- passwords (scrypt, stdlib) ---
def hash_pw(pw):
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + ":" + h.hex()


def check_pw(pw, stored):
    salt, h = stored.split(":")
    calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
    return hmac.compare_digest(calc.hex(), h)
