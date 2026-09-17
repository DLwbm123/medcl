"""Fail-closed model inference: macOS Seatbelt or Linux Landlock + seccomp."""

from functools import lru_cache
from contextlib import contextmanager
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
RUNTIME = INFER.with_name("model_runtime.py")
ISOLATION = INFER.with_name("linux_isolation.py")


def _literal(path) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def profile(read_paths: list[Path], output: Path, scratch: Path | None = None) -> str:
    # Deny all user-data reads, all filesystem writes and all network traffic by default.
    # Only Python's runtime, this audited entrypoint and per-task label-free files are readable.
    allowed = " ".join(f"(literal {_literal(p.resolve())})" for p in [INFER, RUNTIME, *read_paths])
    temporary = f"(subpath {_literal(scratch.resolve())})" if scratch else ""
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
        (literal "/dev/urandom") (literal "/dev/null") {allowed} {temporary})
      (allow file-write* (literal {_literal(output.resolve())}) (literal "/dev/null") {temporary})'''


def clean_env() -> dict:
    return {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL", "LANG": "C.UTF-8" if platform.system() == "Linux" else "en_US.UTF-8"}


@contextmanager
def command_for(code, read_paths, output):
    """Return a neutral command; private paths and bootstrap code travel on stdin."""
    with tempfile.TemporaryDirectory(prefix="inference-scratch-", dir=output.parent) as temporary:
        scratch = Path(temporary).resolve()
        code = f"import tempfile\ntempfile.tempdir = {str(scratch)!r}\n" + code
        if platform.system() == "Darwin":
            with tempfile.NamedTemporaryFile(mode="w", prefix="inference-policy-", suffix=".sb", dir="/tmp") as policy:
                policy.write(profile(read_paths, output, scratch))
                policy.flush()
                yield ["/usr/bin/sandbox-exec", "-f", policy.name, str(EXE), "-I", "-B", "-"], code
            return
        if platform.system() != "Linux":
            raise RuntimeError("unsupported model sandbox")
        prefix = Path(sys.base_prefix).resolve()
        runtime = [prefix / "lib", prefix / "bin", Path("/usr/lib"), Path("/lib"),
                   Path("/etc/ld.so.cache"), Path("/dev/urandom"), Path("/dev/null"),
                   Path("/proc/cpuinfo"), Path("/proc/meminfo"), INFER, RUNTIME]
        reads = [str(p.resolve()) for p in [*runtime, *read_paths] if p.exists()]
        bootstrap = ("import runpy\n" + f"runpy.run_path({str(ISOLATION)!r})['restrict']({reads!r}, {str(output)!r}, {str(scratch)!r})\n")
        yield [str(EXE), "-I", "-B", "-"], bootstrap + code


@lru_cache(maxsize=1)
def sandbox_available() -> bool:
    if platform.system() not in ("Darwin", "Linux"):
        return False
    with tempfile.TemporaryDirectory(prefix="medcl-isolation-check-") as folder:
        root = Path(folder).resolve()
        denied = root / "must-not-read"
        denied.write_text("isolation-test-only", encoding="utf-8")
        out = root / "permitted-output"
        out.touch(mode=0o600)
        code = f"import sys\nsys.argv = { ['check', str(denied), str(out), json.dumps(clean_env())]!r}\n" + """import json,os,socket,sys
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
            with command_for(code, [], out) as (command, script):
                result = subprocess.run(command, input=script.encode(), env=clean_env(), cwd="/", capture_output=True, timeout=20)
            return result.returncode == 0 and out.read_text() == "ok"
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return False


def model_predictions(architecture: str, weights: Path, images: np.ndarray, active_classes: list[int],
                      workdir: Path, all_classes: list[int] | None = None, model_options: dict | None = None) -> np.ndarray:
    if not sandbox_available():
        raise ValueError("模型隔离不可用：仅接受预测，不降级执行上传模型")
    with tempfile.TemporaryDirectory(prefix="inference-", dir=workdir) as folder:
        root = Path(folder).resolve()
        inputs, manifest, output = root / "images.npz", root / "protocol.json", root / "prediction.npy"
        np.savez(inputs, images=images)
        inputs.chmod(0o600)
        manifest.write_text(json.dumps({"architecture": architecture, "active_classes": active_classes,
                                        "all_classes": all_classes or active_classes, "model_options": model_options or {}}), encoding="utf-8")
        manifest.chmod(0o600)
        output.touch(mode=0o600)
        code = ("import runpy,sys\n" +
                f"sys.argv = {[str(INFER), str(manifest), str(weights.resolve()), str(inputs), str(output)]!r}\n" +
                f"runpy.run_path({str(INFER)!r}, run_name='__main__')\n")
        log = root / "private-stderr.txt"
        with command_for(code, [weights, inputs, manifest], output) as (command, script), log.open("wb") as errors:
            log.chmod(0o600)
            process = subprocess.Popen(command, cwd="/", env=clean_env(), stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=errors)
            process.stdin.write(script.encode())
            process.stdin.close()
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > 300:
                        raise ValueError("模型推理超过 300 秒限制")
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=3)
                    if rss.stdout.strip().isdigit() and int(rss.stdout.strip()) > 2 * 1024 * 1024:
                        raise ValueError("模型推理超过 2 GiB 内存限制")
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if process.returncode or not output.is_file():
            error_log = workdir / "inference-error.txt"
            error_log.write_bytes(log.read_bytes())
            error_log.chmod(0o600)
            raise ValueError("隔离推理失败：请核对结构、输入维度及权重格式；服务器保留失败状态")
        output.chmod(0o600)
        return np.load(output, allow_pickle=False)
