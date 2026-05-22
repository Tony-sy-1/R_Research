# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

root = Path.cwd()

hiddenimports = collect_submodules("streamlit")


a = Analysis(
    ['desktop_launcher.py'],
    pathex=[str(root)],
    binaries=[],
    datas=[
        ('app/streamlit_app.py', 'app'),
        ('src/genomics_pipeline.py', 'src'),
        ('README.md', '.'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PoultryBreedingDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PoultryBreedingDesktop',
)
