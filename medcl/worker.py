"""Independent single worker. SQLite survives UI reloads; flock prevents duplicate workers."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from medcl.benchmarks import PROJECT
from medcl.storage import claim_job, connection, heartbeat, initialize, job_dir, now, update_job


def run_worker(root: Path | None = None, once: bool = False) -> None:
    root = initialize(root)
    os.umask(0o077)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    lock_path = root / "worker.lock"
    with lock_path.open("a+") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("MedCL worker already running") from None
        with connection(root) as db:
            db.execute("UPDATE jobs SET status='failed',updated_at=?,message=? WHERE status='running'",
                       (now(), "worker 曾中断；原配置与上传已保留，请新建评测"))
        print("MedCL worker ready (one worker, local evaluation only)", flush=True)
        while True:
            heartbeat(root)
            job = claim_job(root)
            if job is None:
                if once:
                    return
                time.sleep(2)
                continue
            log = job_dir(job["id"], root) / "worker.private.log"
            with log.open("ab") as handle:
                log.chmod(0o600)
                # The child inherits the lock: a killed parent cannot permit a second concurrent scorer.
                process = subprocess.Popen([sys.executable, "-B", "-m", "medcl.runner", job["id"], "--state", str(root)],
                                           cwd=PROJECT, stdout=handle, stderr=handle, stdin=subprocess.DEVNULL,
                                           pass_fds=(lock.fileno(),), start_new_session=True,
                                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "en_US.UTF-8",
                                                "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                start = time.monotonic()
                failure = None
                try:
                    while process.poll() is None:
                        heartbeat(root)
                        if time.monotonic() - start > 600:
                            failure = "评测超过 10 分钟上限；已停止，上传与配置已保留"
                            break
                        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=3)
                        if rss.stdout.strip().isdigit() and int(rss.stdout.strip()) > 4 * 1024 * 1024:
                            failure = "评测超过 4 GiB 内存上限；已停止"
                            break
                        time.sleep(2)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                if failure or process.returncode:
                    with connection(root) as db:
                        status = db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
                    if status != "failed":
                        update_job(job["id"], status="failed", message=failure or "评分进程异常退出；请检查提交格式或联系管理员", root=root)
            if once:
                heartbeat(root)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedCL single scoring worker")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--once", action="store_true")
    options = parser.parse_args()
    run_worker(options.state, options.once)
