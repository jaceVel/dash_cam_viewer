"""
Runtime hook: fix sys.executable for subprocess calls in frozen build.

The main script calls:  subprocess.Popen([sys.executable, script_file, params_file])
In a PyInstaller bundle sys.executable == the .exe, not Python.
We redirect it to a real Python interpreter so those YOLO subprocess scripts work.
"""
import sys
import os
import shutil

if getattr(sys, 'frozen', False):
    sys._frozen_executable = sys.executable  # save the real exe path

    # 1. Look for python.exe next to the bundle (user can drop one there)
    bundle_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(bundle_dir, 'python', 'python.exe')
    if os.path.isfile(candidate):
        sys.executable = candidate
    else:
        # 2. Fall back to first python.exe on PATH
        found = shutil.which('python')
        if found:
            sys.executable = found
