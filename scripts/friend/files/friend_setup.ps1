# Sets up localtune on a friend's Windows PC, benchmarks the same models as the rented GPUs and
# leaves the results on their Desktop to send back. It does every step of the friend guide in
# one go: install uv, install localtune from the wheel next to this script, check for an NVIDIA
# card and Ollama, download the models, install llama.cpp, run the quick sweep. Started by
# double-clicking "1 Start here.bat" in the folder above this one. Safe to run again: finished steps are
# skipped and the sweep carries on where it stopped.

$ErrorActionPreference = "Stop"

# The 7 models of the quick sweep on the rented GPUs (rental_setup.sh), so every computer is
# measured on the same work, and 4 the rented GPUs never ran, to check the speed guesses on
# models they weren't trained on. The sizes are each download in GB. Only the models that fit
# on this PC's graphics card are run (see step 1), so a small card never has to push a model
# through the PC's RAM, which can make the whole PC slow for hours.
$AllModels = [ordered]@{
    "qwen3:0.6b"        = 0.5
    "llama3.2:3b"       = 2.0
    "llama3.1:8b"       = 4.9
    "qwen3:8b"          = 5.2
    "qwen3:14b"         = 9.3
    "gpt-oss:20b"       = 13.8
    "mistral-small:24b" = 14.3
    "qwen3:14b-q8_0"    = 15.7
    "gemma4:26b"        = 18.0
    "qwen3:30b"         = 18.6
    "qwen3.8:27b"       = 17.3
}
$KeepFreeGB = 1.5

function Step($text) { Write-Host ""; Write-Host "== $text ==" -ForegroundColor Magenta }
function Say($text) { Write-Host $text -ForegroundColor Yellow }
function Stop-Here($text) {
    Write-Host ""
    Write-Host $text -ForegroundColor Red
    Write-Host "When that's done, double-click '1 Start here' again. It picks up where it stopped."
    exit 1
}

# Windows puts an idle computer to sleep after a while, which would stop a download or the
# sweep halfway. This tells Windows the computer is in use until the window closes, the way
# video players do. The screen may still turn off; that's fine.
Add-Type -Namespace LocalTune -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
"@
[void][LocalTune.Power]::SetThreadExecutionState([uint32]2147483649)

Write-Host "localtune friend setup" -ForegroundColor Cyan
Write-Host "This measures how fast AI models run on this PC. It takes a few hours, most of it"
Write-Host "downloading up to about 120 GB of models and then running them one by one."
Write-Host "Please close games and other programs that use the graphics card, and leave the PC on."
Write-Host "You can stop at any time by closing this window; run it again to carry on."
Read-Host "Press Enter to start"

# Step 1: the graphics card. localtune measures NVIDIA cards only; nvidia-smi comes with
# NVIDIA's driver, so finding it means both the card and the driver are there.
Step "1/7 Check for an NVIDIA graphics card"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    $CardGB = [double](@(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)[0]) / 1024
} elseif (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }) {
    Start-Process "https://www.nvidia.com/en-us/drivers/"
    Stop-Here "This PC has an NVIDIA card, but its driver isn't installed. Install the driver from the page that just opened (or the NVIDIA App)."
} else {
    Write-Host ""
    Write-Host "This PC doesn't have an NVIDIA graphics card, which localtune needs for now." -ForegroundColor Red
    Write-Host "Nothing was installed. Thanks for offering to help anyway!"
    exit 1
}

# The models that fit on the card with room to spare (the same 1.5 GB localtune's estimate
# keeps free for the screen and the conversation).
$Models = [ordered]@{}
foreach ($name in $AllModels.Keys) {
    if ($AllModels[$name] -le $CardGB - $KeepFreeGB) { $Models[$name] = $AllModels[$name] }
}
Write-Host "This card has $([math]::Round($CardGB)) GB, so it runs $($Models.Count) of the $($AllModels.Count) models."

# What this script adds to the PC is noted in localtune's folder as it goes: whether it installed
# uv, and which models it downloaded (models that were already on the PC aren't listed). The
# clean-up at the end (friend_cleanup.ps1) removes only what is listed there.
$LocaltuneHome = if ($env:LOCALTUNE_HOME) { $env:LOCALTUNE_HOME } else { Join-Path $env:USERPROFILE ".localtune" }
New-Item -ItemType Directory -Force $LocaltuneHome | Out-Null
$Downloaded = Join-Path $LocaltuneHome "friend-downloaded-models.txt"
$InstalledUv = Join-Path $LocaltuneHome "friend-installed-uv.txt"

# Step 2: uv, the tool that installs localtune and the Python it runs on. It lands in the
# user's own folder, so no admin rights are needed.
Step "2/7 Install uv"
$UvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:Path = "$UvBin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Set-Content $InstalledUv "yes"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$UvBin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Stop-Here "uv didn't install. Check the internet connection."
}

# Step 3: localtune itself, from the wheel file Justin sent along with this script. Nothing is
# downloaded from PyPI. The newest wheel wins if there are several.
Step "3/7 Install localtune"
$Wheel = Get-ChildItem -Path $PSScriptRoot, (Split-Path $PSScriptRoot) -Filter "localtune-*.whl" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Wheel) {
    Stop-Here "The localtune file (localtune-...whl) isn't in this folder. Unzip everything Justin sent into one folder, and double-click '1 Start here' from there."
}
uv tool install --force $Wheel.FullName
if ($LASTEXITCODE -ne 0) { Stop-Here "localtune didn't install. Send Justin a screenshot of this window." }
$env:Path = "$(uv tool dir --bin);$env:Path"

# Step 4: Ollama downloads the models. It has to be installed by hand, and running so that it
# can download; if it's only closed, it's started here in the background.
Step "4/7 Check for Ollama"
$Ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $Ollama) {
    $default = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $default) { $Ollama = Get-Item $default }
}
if (-not $Ollama) {
    Start-Process "https://ollama.com/download/windows"
    Stop-Here "Ollama isn't installed. Download and install it from the page that just opened."
}
$OllamaExe = if ($Ollama.Source) { $Ollama.Source } else { $Ollama.FullName }
function Test-Ollama {
    try { Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 2 | Out-Null; $true } catch { $false }
}
if (-not (Test-Ollama)) {
    Start-Process $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    foreach ($i in 1..30) { if (Test-Ollama) { break }; Start-Sleep 1 }
}
if (-not (Test-Ollama)) { Stop-Here "Ollama didn't start. Open the Ollama app from the Start menu." }
Write-Host "Ollama is running."

# Step 5: the models. Only the ones not downloaded yet count against the free disk space, on
# the drive where Ollama keeps its models.
Step "5/7 Download the models (the first time this takes a while)"
$have = (& $OllamaExe list) -join "`n"
$missing = @($Models.Keys | Where-Object { $have -notmatch "(?m)^$([regex]::Escape($_))\s" })
$needGB = ($missing | ForEach-Object { $Models[$_] } | Measure-Object -Sum).Sum
$store = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($store).Substring(0, 1))
$freeGB = [math]::Round($drive.Free / 1GB)
if ($needGB -and $freeGB -lt $needGB + 10) {
    Stop-Here "The models need about $([math]::Ceiling($needGB + 10)) GB free on drive $($drive.Name):, and it has $freeGB GB. Free up some space (empty the Recycle Bin, uninstall big games you don't play)."
}
foreach ($model in $Models.Keys) {
    Write-Host "-- $model"
    if ($missing -contains $model) { Add-Content $Downloaded $model }
    & $OllamaExe pull $model
    if ($LASTEXITCODE -ne 0) { Stop-Here "Downloading $model failed. Check the internet connection." }
}

# Step 6: llama.cpp, the engine that runs the benchmarks, installed by localtune into its own
# folder, then a check that everything is ready.
Step "6/7 Install llama.cpp and check this PC"
localtune doctor --install
if ($LASTEXITCODE -ne 0) { Stop-Here "This PC isn't ready yet: see the items marked with a cross above." }

# Step 7: the quick sweep, each model for chat, code and documents once with llama.cpp's usual
# settings, as on the rented GPUs. --resume skips what an earlier, stopped run of this script
# already finished.
Step "7/7 Benchmark every model (about 1 to 3 hours)"
localtune sweep @($Models.Keys) --resume

# The results: copied to the Desktop under this PC's name, so Justin can tell the files apart.
$results = Join-Path $LocaltuneHome "results.jsonl"
if (-not (Test-Path $results)) { Stop-Here "No results were saved. Send Justin a screenshot of this window." }
$desktop = [Environment]::GetFolderPath("Desktop")
$copy = Join-Path $desktop "localtune-results-$env:COMPUTERNAME.jsonl"
Copy-Item $results $copy -Force
Write-Host ""
Write-Host "All done, thank you!" -ForegroundColor Green
Write-Host "Please send Justin this file from your Desktop:"
Write-Host "  $copy" -ForegroundColor Cyan
Write-Host "It holds the speed measurements and this PC's hardware (graphics card, CPU, RAM)."
Start-Process explorer.exe "/select,`"$copy`""

# Clean-up: the models take up to about 120 GB, so the last step offers to remove them and localtune
# again. The results file on the Desktop stays. Saying no leaves everything in place, and
# "2 Remove when done.bat" does the same clean-up any time later.
Write-Host ""
$answer = Read-Host "Remove the downloaded models and localtune from this PC now? Your results file stays on the Desktop. (Y/N)"
if ($answer -match "^\s*y") {
    & (Join-Path $PSScriptRoot "friend_cleanup.ps1") -Yes
} else {
    Write-Host "Everything is left in place. To remove it later, double-click '2 Remove when done'."
}
