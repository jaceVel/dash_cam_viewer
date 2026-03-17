# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for dashcam_object_detector_V2.py
Build with:  pyinstaller dashcam_detector.spec
Output:      dist\dashcam_detector\  (~7-8 GB due to PyTorch CUDA)
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
import os, glob

# ── Data files ────────────────────────────────────────────────────────────────
datas = []
datas += collect_data_files('folium')
datas += collect_data_files('ultralytics')
datas += collect_data_files('PyQt5')          # includes Qt5 bins, WebEngine resources

# exiftool binary (if present in project folder)
_exiftool = 'exiftool(-k).exe'
if os.path.exists(_exiftool):
    datas.append((_exiftool, '.'))

# ── Hidden imports ─────────────────────────────────────────────────────────────
hiddenimports = [
    # PyQt5 web stack
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.QtWebEngineCore',
    'PyQt5.QtWebChannel',
    'PyQt5.QtNetwork',
    'PyQt5.QtPrintSupport',
    # ML
    'torch',
    'torchvision',
    'ultralytics',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.nn',
    'ultralytics.utils',
    # Vision / data
    'cv2',
    'PIL',
    'PIL.Image',
    'sklearn',
    'scipy',
    # Std extras that sometimes get missed
    'statistics',
    'pathlib',
    'csv',
    'json',
    'queue',
    'tempfile',
]
# Collect all ultralytics submodules (it uses dynamic imports internally)
hiddenimports += collect_submodules('ultralytics')

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ['dashcam_object_detector_V2.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_subprocess_python.py'],
    excludes=[
        # Exclude Jupyter / IPython bloat
        'IPython', 'jupyter', 'notebook',
        'matplotlib',   # not used; remove if you add plots later
        'tkinter',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dashcam_detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX + PyTorch DLLs = bad results; leave off
    console=False,      # no console window  (change to True to debug crashes)
    disable_windowed_traceback=False,
    icon=None,          # set to 'your_icon.ico' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='dashcam_detector',
)
