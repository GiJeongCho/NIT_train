from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NIT_train"
FRONT_DIR = APP_DIR / "NIT_train_front"
LOG_DIR = APP_DIR / "logs"
SETTINGS_DIR = Path(os.environ.get("APPDATA", Path.home())) / "NIT_train"
PYTHON_SETTING = SETTINGS_DIR / "python_path.txt"
ULTRALYTICS_DIR = APP_DIR / "workspace" / ".ultralytics"
UI_URL = "http://127.0.0.1:8890"
API_URL = "http://127.0.0.1:8888/healthz"


def python_candidates() -> list[Path]:
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")
    user = Path.home()
    for base in (user / "anaconda3", user / "miniconda3", Path(os.environ.get("LOCALAPPDATA", user)) / "anaconda3", Path("C:/ProgramData/anaconda3")):
        candidates.append(base / "envs" / "NIT" / "python.exe")
    if PYTHON_SETTING.is_file():
        try:
            candidates.insert(0, Path(PYTHON_SETTING.read_text(encoding="utf-8").strip()))
        except OSError:
            pass
    path_python = shutil.which("python")
    if path_python:
        candidates.append(Path(path_python))
    # Preserve order while avoiding duplicate probes.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


class Launcher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NIT 학습 도구")
        self.root.geometry("520x290")
        self.root.resizable(False, False)
        self.busy = False
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files = []

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="NIT 학습 도구", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, text="NIT Python 환경으로 API와 프론트를 함께 실행합니다. Docker는 사용하지 않습니다.", wraplength=470).pack(anchor="w", pady=(8, 14))
        self.status = tk.StringVar(value="시작 버튼을 누르면 Python 환경을 확인합니다.")
        ttk.Label(frame, textvariable=self.status, wraplength=470).pack(anchor="w", pady=(0, 12))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 14))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="시작 / 브라우저 열기", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="종료", command=self.stop)
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="데이터 폴더", command=self.open_folder).pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def set_status(self, value: str) -> None:
        self.root.after(0, self.status.set, value)

    def python_executable(self) -> Path:
        for candidate in [APP_DIR / "runtime" / "python.exe"]:
            if not candidate.is_file():
                continue
            try:
                probe_env = os.environ.copy()
                probe_env["PATH"] = os.pathsep.join([str(candidate.parent), str(candidate.parent / "Library/bin"), str(candidate.parent / "Scripts"), probe_env["PATH"]])
                probe_env.pop("PYTHONHOME", None)
                probe_env.pop("PYTHONPATH", None)
                probe_env["PYTHONNOUSERSITE"] = "1"
                self.child_env = probe_env
                probe_env["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_DIR)
                probe = subprocess.run(
                    [str(candidate), "-c", "import torch; print('0' if torch.cuda.is_available() else 'cpu')"],
                    cwd=APP_DIR, stdout=subprocess.PIPE, stderr=self._open_log("api"),
                    timeout=180, check=False, env=probe_env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if probe.returncode == 0:
                    self.child_env["NIT_TRAIN_DEVICE"] = probe.stdout.decode().strip()
                    return candidate
            except (OSError, subprocess.TimeoutExpired):
                continue
        raise RuntimeError(
            "내장 Python을 시작할 수 없습니다. logs/api.log를 확인하세요."
        )

    def select_python(self) -> None:
        chosen = filedialog.askopenfilename(
            title="NIT 환경의 python.exe 선택",
            filetypes=[("Python", "python.exe"), ("모든 파일", "*.*")],
        )
        if not chosen:
            return
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        PYTHON_SETTING.write_text(chosen, encoding="utf-8")
        self.status.set(f"Python 환경 저장됨: {chosen}")

    def _open_log(self, name: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handle = (LOG_DIR / f"{name}.log").open("ab", buffering=0)
        self.log_files.append(handle)
        return handle

    def _spawn(self, name: str, args: list[str], cwd: Path) -> subprocess.Popen[bytes]:
        child_env = self.child_env.copy()
        child_env["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_DIR)
        return subprocess.Popen(
            args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=self._open_log(name), stderr=subprocess.STDOUT,
            env=child_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    @staticmethod
    def is_ready(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status < 500
        except (OSError, urllib.error.URLError):
            return False

    def start(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.start_button.configure(state="disabled")
        self.progress.start(12)
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        try:
            if self.is_ready(API_URL) and self.is_ready(UI_URL):
                self.root.after(0, lambda: webbrowser.open(UI_URL))
                self.set_status(f"이미 실행 중입니다: {UI_URL}")
                return
            self.set_status("NIT Python 환경의 패키지와 GPU를 확인 중…")
            python = self.python_executable()
            if not (APP_DIR / "src" / "api.py").is_file() or not (FRONT_DIR / "server.py").is_file():
                raise FileNotFoundError("설치된 API 또는 프론트 파일을 찾지 못했습니다. 설치 EXE를 다시 실행하세요.")

            if not self.is_ready(API_URL):
                self.set_status("학습 API를 시작하는 중…")
                self.processes["api"] = self._spawn("api", [str(python), "src/api.py"], APP_DIR)
            if not self.is_ready(UI_URL):
                self.set_status("프론트 웹 서버를 시작하는 중…")
                self.processes["front"] = self._spawn(
                    "front", [str(python), "server.py", "--host", "127.0.0.1", "--port", "8890", "--api", "http://127.0.0.1:8888"], FRONT_DIR
                )

            self.set_status("API와 프론트가 준비되기를 기다리는 중…")
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                dead = [name for name, process in self.processes.items() if process.poll() is not None]
                if dead:
                    raise RuntimeError(f"{', '.join(dead)} 프로세스가 종료됐습니다. logs 폴더의 로그를 확인하세요.")
                if self.is_ready(API_URL) and self.is_ready(UI_URL):
                    self.root.after(0, lambda: webbrowser.open(UI_URL))
                    self.set_status(f"실행 중입니다: {UI_URL}")
                    return
                time.sleep(1)
            raise RuntimeError("서비스가 제한 시간 안에 준비되지 않았습니다. logs 폴더의 api.log와 front.log를 확인하세요.")
        except Exception as exc:
            message = str(exc)
            self.set_status("실행하지 못했습니다.")
            self.root.after(0, lambda: messagebox.showerror("실행 오류", message))
            self._stop_processes()
        finally:
            self.busy = False
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.start_button.configure(state="normal"))

    def _stop_processes(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()
        for handle in self.log_files:
            try:
                handle.close()
            except OSError:
                pass
        self.log_files.clear()

    def stop(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.start_button.configure(state="disabled")
        self.set_status("이 실행기에서 시작한 프론트와 API를 종료하는 중…")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        self._stop_processes()
        self.set_status("종료했습니다. workspace 데이터는 보존됩니다.")
        self.busy = False
        self.root.after(0, lambda: self.start_button.configure(state="normal"))

    def open_folder(self) -> None:
        (APP_DIR / "workspace").mkdir(parents=True, exist_ok=True)
        os.startfile(APP_DIR / "workspace")

    def close(self) -> None:
        self._stop_processes()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(None)
    Launcher().run()
