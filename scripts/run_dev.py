from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_config import load_config
from runtime_logging import configure_logging

import logging

logger = logging.getLogger(__name__)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()


def main() -> int:
    configure_logging()
    config = load_config()
    env = os.environ.copy()
    env.setdefault("DEMO_API_URL", config.effective_frontend_api_url)
    env.setdefault("PYTHONUNBUFFERED", "1")

    if config.demo.api_port == 8000:
        logger.warning(
            "API_PORT is set to 8000. Reflex commonly uses port 8000 internally; "
            "prefer API_PORT=8001 to avoid routing collisions."
        )

    backend_cmd = [
        "uvicorn",
        "demo_app.main:create_app",
        "--factory",
        "--host",
        config.demo.api_host,
        "--port",
        str(config.demo.api_port),
        "--reload",
    ]
    frontend_cmd = ["reflex", "run"]

    logger.info("Starting backend with API URL %s", config.effective_frontend_api_url)
    logger.info("Backend command: %s", " ".join(backend_cmd))
    logger.info("Frontend command: %s", " ".join(frontend_cmd))
    logger.info("Frontend DEMO_API_URL=%s", env["DEMO_API_URL"])

    backend = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT, env=env)
    frontend = subprocess.Popen(frontend_cmd, cwd=PROJECT_ROOT, env=env)
    processes = [backend, frontend]

    def _handle_signal(signum, _frame) -> None:
        for process in processes:
            _terminate(process)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0
    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    logger.info("Process exited pid=%s code=%s", process.pid, code)
                    exit_code = code
                    raise SystemExit(code)
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping dev processes")
        for process in processes:
            _terminate(process)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill(process)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
