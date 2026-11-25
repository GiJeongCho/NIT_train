"""Build the distributable ZIP after building the setup executable."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

base = Path(__file__).resolve().parent
archive = base / "runtime.zip"
if "--reuse-runtime" not in sys.argv:
    prefix = os.environ.get("NIT_RUNTIME_PREFIX", Path.home() / "anaconda3/envs/NIT")
    subprocess.run(["conda-pack", "-p", str(prefix), "-o", str(archive), "--force",
                    "--ignore-missing-files", "--compress-level", "1", "--exclude", "*.pyc"], check=True)
with zipfile.ZipFile(base / "release/NIT_train_Windows.zip", "w") as out:
    for file in (base / "release/NIT_train_Setup.exe", archive):
        out.write(file, file.name)
