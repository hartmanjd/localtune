@echo off
rem Double-click this to set up localtune and benchmark this PC. It starts files\friend_setup.ps1,
rem which does the work, and keeps the window open at the end so the last message can be read.
if not exist "%~dp0files\friend_setup.ps1" (
  echo The files folder isn't next to this file. Unzip everything Justin sent into one folder
  echo first ^(right-click the zip, Extract All^), then double-click "1 Start here" in there.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0files\friend_setup.ps1"
pause
