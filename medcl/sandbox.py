"""Fail-closed macOS sandbox for vetted numerical models; other OSes are prediction-only."""

from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

INFER = Path(__file__).with_name("infer.py").resolve()
EXE = Path(sys.executable).resolve()


def _literal(path) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def profile(read_paths: list[Path], output: Path) -> str:
    # Deny all user-data reads, all filesystem writes and all network traffic by default.
    # Only Python's runtime, this audited entrypoint and per-task label-free files are readable.
    allowed = " ".join(f"(literal {_literal(p.resolve())})" for p in [INFER, *read_paths])
    prefix = Path(sys.prefix).resolve()
    return f'''(version 1)
      (allow default)
      (deny network*)
      (deny file-read*)
      (deny file-write*)
      (deny process-fork)
      (deny process-exec)
      (allow process-exec (literal {_literal(EXE)}))
      (allow file-read* (literal "/") (subpath "/System") (subpath "/usr/lib")
        (subpath {_literal(prefix / "lib")}) (subpath {_literal(prefix / "bin")})
        (literal "/dev/urandom") (literal "/dev/null") {allowed})
      (allow file-write* (literal {_literal(output.resolve())}) (literal "/dev/null"))'''


def clean_env() -> dict:
    return {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "LANG": "en_US.UTF-8"}


@lru_cache(maxsize=1)
def sandbox_available() -> bool:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return False
    with tempfile.TemporaryDirectory(prefix="medcl-isolation-check-") as folder:
        root = Path(folder).resolve()
        denied = root / "must-not-read"
        denied.write_text("isolation-test-only", encoding="utf-8")
        out = root / "permitted-output"
        code = """import json,os,socket,sys
assert dict(os.environ) == json.loads(sys.argv[3])
try:
    open(sys.argv[1]).read()
except PermissionError:
    pass
else:
    raise SystemExit(11)
try:
    socket.socket().connect(('127.0.0.1',9))
except PermissionError:
    pass
else:
    raise SystemExit(12)
try:
    open(sys.argv[1]+'-write','w').write('not-permitted')
except PermissionError:
    pass
else:
    raise SystemExit(13)
try:
    pid = os.fork()
except PermissionError:
    pass
else:
    if pid == 0:
        os._exit(0)
    os.waitpid(pid,0)
    raise SystemExit(14)
open(sys.argv[2],'w').write('ok')
import numpy,safetensors.numpy
"""
        try:
            result = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile([], out), str(EXE), "-I", "-B",
                                     "-c", code, str(denied), str(out), json.dumps(clean_env())], env=clean_env(), cwd="/", capture_output=True, timeout=15)
            return result.returncode == 0 and out.read_text() == "ok"
        except (OSError, subprocess.SubprocessError):
            return False


def model_predictions(architecture: str, weights: Path, images: np.ndarray, active_classes: list[int],
                      workdir: Path, all_classes: list[int] | None = None) -> np.ndarray:
    if not sandbox_available():
        raise ValueError("模型隔离不可用：仅接受预测，不降级执行上传模型")
    with tempfile.TemporaryDirectory(prefix="inference-", dir=workdir) as folder:
        root = Path(folder).resolve()
        inputs, manifest, output = root / "images.npz", root / "protocol.json", root / "prediction.npy"
        np.savez(inputs, images=images)
        manifest.write_text(json.dumps({"architecture": architecture, "active_classes": active_classes,
                                        "all_classes": all_classes or active_classes}), encoding="utf-8")
        command = ["/usr/bin/sandbox-exec", "-p", profile([weights, inputs, manifest], output),
                   str(EXE), "-I", "-B", str(INFER), str(manifest), str(weights.resolve()), str(inputs), str(output)]
        log = root / "private-stderr.txt"
        with log.open("wb") as errors:
            process = subprocess.Popen(command, cwd="/", env=clean_env(), stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=errors)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > 120:
                        raise ValueError("模型推理超过 120 秒限制")
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=3)
                    if rss.stdout.strip().isdigit() and int(rss.stdout.strip()) > 2 * 1024 * 1024:
                        raise ValueError("模型推理超过 2 GiB 内存限制")
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if process.returncode or not output.is_file():
            (workdir / "inference-error.txt").write_bytes(log.read_bytes())
            raise ValueError("隔离推理失败：请核对结构、输入维度及 float32 权重；服务器保留失败状态")
        return np.load(output, allow_pickle=False)
