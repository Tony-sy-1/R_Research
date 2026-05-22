import atexit
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_free_port(start: int = 8501, end: int = 8999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("无法找到可用端口，请关闭占用 8501-8999 的程序后重试。")


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def run() -> None:
    root = app_root()
    app_file = root / "app" / "streamlit_app.py"
    if not app_file.exists():
        raise FileNotFoundError(f"未找到应用入口: {app_file}")

    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    process = subprocess.Popen(cmd, cwd=str(root), env=env)
    atexit.register(lambda: process.poll() is None and process.terminate())

    url = f"http://127.0.0.1:{port}"
    timeout_seconds = 40
    start = time.time()
    while time.time() - start < timeout_seconds:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                process.wait()
                return
        time.sleep(0.3)

    process.terminate()
    raise TimeoutError("启动超时：Streamlit 服务未在 40 秒内就绪。")


if __name__ == "__main__":
    run()
