"""Frozen entry for YOLO Dataset Editor (Flask + bundled Vite UI)."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 3777


def main() -> int:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    def _open_browser() -> None:
        time.sleep(1.2)
        webbrowser.open(f"http://127.0.0.1:{PORT}/")

    threading.Thread(target=_open_browser, daemon=True).start()

    import yolo_server

    yolo_server.run_server(host="0.0.0.0", port=PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
