@echo off
setlocal
cd /d %~dp0\..
python -m pip install -r requirements-desktop.txt
python -m PyInstaller --clean --noconfirm build_windows_exe.spec
echo.
echo Build complete. Executable folder: dist\PoultryBreedingDesktop
pause
