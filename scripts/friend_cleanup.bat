@echo off
rem Double-click this to remove what friend_setup.bat added to this PC: the models it downloaded,
rem localtune and llama.cpp, and uv if the setup installed it. It asks before removing anything.
if not exist "%~dp0friend_cleanup.ps1" (
  echo friend_cleanup.ps1 isn't next to this file. Unzip everything Justin sent into one folder
  echo first ^(right-click the zip, Extract All^), then double-click friend_cleanup.bat in there.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0friend_cleanup.ps1"
pause
