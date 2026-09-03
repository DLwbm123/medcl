"""Start the UI and one worker with the same existing Python environment."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    children = []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "medcl.worker"], cwd=PROJECT))
        children.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.address=127.0.0.1", "--server.port=8501"], cwd=PROJECT))
        print("MedCL local URL: http://127.0.0.1:8501", flush=True)
        if not args.no_browser:
            for _ in range(30):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open("http://127.0.0.1:8501")
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.5)
        while all(child.poll() is None for child in children):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
