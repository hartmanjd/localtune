@echo off
rem Double-click this to remove what "1 Start here" added to this PC: the models it downloaded,
rem localtune and llama.cpp, and uv if the setup installed it. It asks before removing anything.
if not exist "%~dp0files\friend_cleanup.ps1" (
  echo The files folder isn't next to this file. Unzip everything Justin sent into one folder
  echo first ^(right-click the zip, Extract All^), then double-click "2 Remove when done" in there.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0files\friend_cleanup.ps1"
pause
