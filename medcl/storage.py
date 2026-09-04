"""SQLite queue plus owner-only files; one immutable protocol snapshot per job."""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from medcl.benchmarks import state_path
from medcl.reports import submitter_result

DB_SCHEMA_VERSION = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _secure_sqlite(root: Path) -> None:
    for name in ("medcl.sqlite3", "medcl.sqlite3-wal", "medcl.sqlite3-shm"):
        path = root / name
        if path.exists():
            os.chmod(path, 0o600)


def _replace_private_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    temporary.write_text(encode(value), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def initialize(root: Path | None = None) -> Path:
    root = Path(root or state_path())
    jobs_root = root / "jobs"
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        jobs_root.mkdir(exist_ok=True, mode=0o700)
        os.chmod(jobs_root, 0o700)
    except OSError:
        raise RuntimeError("私有状态目录无法创建或收紧为仅所有者访问") from None
    with closing(sqlite3.connect(root / "medcl.sqlite3", timeout=10)) as db:
        with db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                config TEXT NOT NULL, message TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                result TEXT
            );
            CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
            CREATE TRIGGER IF NOT EXISTS immutable_job_config BEFORE UPDATE OF config ON jobs
            BEGIN SELECT RAISE(ABORT, 'evaluation configuration is immutable'); END;
            CREATE TABLE IF NOT EXISTS worker_state (id INTEGER PRIMARY KEY CHECK(id=1), heartbeat TEXT NOT NULL);
            """)
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version < DB_SCHEMA_VERSION:
                for job_id, encoded in db.execute("SELECT id,result FROM jobs WHERE result IS NOT NULL").fetchall():
                    result = submitter_result(json.loads(encoded))
                    db.execute("UPDATE jobs SET result=? WHERE id=?", (encode(result), job_id))
                    result_file = jobs_root / job_id / "result.json"
                    if result_file.is_file():
                        _replace_private_json(result_file, submitter_result(json.loads(result_file.read_text(encoding="utf-8"))))
                for folder in jobs_root.iterdir():
                    if not folder.is_dir():
                        continue
                    os.chmod(folder, 0o700)
                    for directory in (folder / "uploads", folder / "visuals"):
                        if directory.is_dir():
                            os.chmod(directory, 0o700)
                    for name in ("private.json", "result.json", "result.pending.json", "worker.private.log", "inference-error.txt"):
                        path = folder / name
                        if path.is_file():
                            os.chmod(path, 0o600)
                    uploads = folder / "uploads"
                    if uploads.is_dir():
                        for path in uploads.iterdir():
                            if path.is_file():
                                os.chmod(path, 0o600)
                    visuals = folder / "visuals"
                    if visuals.is_dir():
                        for path in visuals.iterdir():
                            if path.is_file():
                                os.chmod(path, 0o600)
                db.execute(f"PRAGMA user_version={DB_SCHEMA_VERSION}")
    _secure_sqlite(root)
    return root


@contextmanager
def connection(root: Path | None = None):
    root = Path(root or state_path())
    db = sqlite3.connect(root / "medcl.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()
        _secure_sqlite(root)


def job_dir(job_id: str, root: Path | None = None) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("无效评测编号")
    return Path(root or state_path()) / "jobs" / job_id


def create_job(config: dict, private: dict, uploads: list[dict], root: Path | None = None) -> str:
    root = initialize(root)
    # A browser rerun never mutates an existing evaluation or overwrites another upload.
    job_id = uuid.uuid4().hex
    folder = job_dir(job_id, root)
    folder.mkdir(mode=0o700)
    os.chmod(folder, 0o700)
    uploads_dir = folder / "uploads"
    uploads_dir.mkdir(mode=0o700)
    os.chmod(uploads_dir, 0o700)
    saved = []
    try:
        for item in uploads:
            suffix = Path(item["name"]).suffix.lower()
            filename = f"stage-{item['stage']:02d}{suffix}"
            destination = uploads_dir / filename
            with destination.open("xb") as handle:
                handle.write(item["data"])
            destination.chmod(0o600)
            saved.append({"stage": item["stage"], "filename": filename})
        private = {**private, "uploads": saved}
        private_path = folder / "private.json"
        private_path.write_text(encode(private), encoding="utf-8")
        private_path.chmod(0o600)
        with connection(root) as db:
            db.execute("INSERT INTO jobs(id,created_at,updated_at,status,config,message) VALUES(?,?,?,'queued',?,?)",
                       (job_id, now(), now(), encode(config), "等待独立评分 worker"))
    except BaseException:
        # Keep any partially written payload as a private, unqueued diagnostic; never erase uploads silently.
        raise
    return job_id


def _decode(row) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["config"] = json.loads(result["config"])
    result["result"] = submitter_result(json.loads(result["result"])) if result["result"] else None
    return result


def get_job(job_id: str, root: Path | None = None) -> dict | None:
    job_dir(job_id, root)
    with connection(root) as db:
        return _decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def list_jobs(root: Path | None = None, limit: int = 100) -> list[dict]:
    with connection(root) as db:
        return [_decode(row) for row in db.execute("SELECT id,created_at,updated_at,status,config,message,progress,NULL AS result FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))]


def claim_job(root: Path | None = None) -> dict | None:
    with connection(root) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
        if row is None:
            return None
        db.execute("UPDATE jobs SET status='running',updated_at=?,message=? WHERE id=?",
                   (now(), "校验输入与冻结测试协议", row["id"]))
        return _decode(row)


def update_job(job_id: str, *, message: str, progress: float | None = None,
               status: str | None = None, result: dict | None = None, root: Path | None = None) -> None:
    fields, args = ["updated_at=?", "message=?"], [now(), message]
    if progress is not None:
        fields.append("progress=?")
        args.append(max(0.0, min(1.0, float(progress))))
    if status is not None:
        if status not in ("running", "completed", "failed"):
            raise ValueError("无效状态")
        fields.append("status=?")
        args.append(status)
    if result is not None:
        fields.append("result=?")
        args.append(encode(submitter_result(result)))
    with connection(root) as db:
        db.execute(f"UPDATE jobs SET {','.join(fields)} WHERE id=?", (*args, job_id))


def heartbeat(root: Path | None = None) -> None:
    with connection(root) as db:
        db.execute("INSERT OR REPLACE INTO worker_state VALUES(1,?)", (now(),))


def worker_alive(root: Path | None = None) -> bool:
    with connection(root) as db:
        row = db.execute("SELECT heartbeat FROM worker_state WHERE id=1").fetchone()
    return bool(row and (datetime.now(timezone.utc) - datetime.fromisoformat(row[0])).total_seconds() < 20)
