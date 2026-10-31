@echo off
rem Double-click this to set up localtune and benchmark this PC. It starts friend_setup.ps1,
rem which does the work, and keeps the window open at the end so the last message can be read.
if not exist "%~dp0friend_setup.ps1" (
  echo friend_setup.ps1 isn't next to this file. Unzip everything Justin sent into one folder
  echo first ^(right-click the zip, Extract All^), then double-click friend_setup.bat in there.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0friend_setup.ps1"
pause
