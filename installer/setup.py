from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import zipfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


def payload_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "payload"


class Setup:
    def __init__(self) -> None:
        self.target = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NIT_train"
        self.root = tk.Tk()
        self.root.title("NIT 학습 도구 설치")
        self.root.geometry("500x270")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="NIT 학습 도구 설치", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"설치 위치: {self.target}", wraplength=450).pack(anchor="w", pady=(12, 8))
        ttk.Label(frame, text="프론트, API와 Python 환경을 함께 설치합니다. GPU가 없으면 CPU로 실행합니다.", wraplength=450).pack(anchor="w", pady=(0, 16))
        self.status = tk.StringVar(value="설치를 시작할 준비가 되었습니다.")
        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(0, 12))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 16))
        self.button = ttk.Button(frame, text="설치하고 실행", command=self.install)
        self.button.pack(anchor="e")

    def install(self) -> None:
        self.button.configure(state="disabled")
        self.progress.start(12)
        threading.Thread(target=self._install, daemon=True).start()

    def _install(self) -> None:
        try:
            source_root = payload_dir()
            if not source_root.is_dir():
                raise FileNotFoundError("설치 파일 안에서 앱 데이터를 찾지 못했습니다.")
            self.target.mkdir(parents=True, exist_ok=True)
            archive = Path(sys.executable).with_name("runtime.zip")
            if not archive.is_file():
                raise FileNotFoundError("배포 ZIP을 모두 풀고 runtime.zip 옆의 설치기를 실행하세요.")
            runtime = self.target / "runtime"
            self.root.after(0, self.status.set, "Python과 모델 라이브러리를 설치하는 중…")
            with zipfile.ZipFile(archive) as z:
                z.extractall(runtime)
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
            env = os.environ.copy()
            env["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Library/bin"), env["PATH"]])
            subprocess.run([str(runtime / "python.exe"), "-I", str(runtime / "Scripts/conda-unpack-script.py")],
                           env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            for source in source_root.iterdir():
                destination = self.target / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
            (self.target / "workspace").mkdir(exist_ok=True)
            (self.target / "external_datasets").mkdir(exist_ok=True)
            launcher = self.target / "NIT_train.exe"
            if not launcher.is_file():
                raise FileNotFoundError("실행 프로그램이 설치되지 않았습니다.")
            self.root.after(0, lambda: self.status.set("설치 완료. 앱을 여는 중…"))
            subprocess.Popen([str(launcher)], cwd=self.target)
            self.root.after(0, self.root.destroy)
        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.button.configure(state="normal"))
            self.root.after(0, lambda: messagebox.showerror("설치 오류", error_message))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Setup().run()
