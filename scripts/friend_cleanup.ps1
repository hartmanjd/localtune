# Removes what friend_setup.ps1 added to this PC, once the results are sent: the models it
# downloaded, localtune with llama.cpp and its saved data, and uv if the setup installed it.
# Started by double-clicking friend_cleanup.bat, or by friend_setup.ps1 at the end when the
# friend says yes. It only removes what the setup noted it added (in localtune's folder), so
# models and tools that were on the PC before stay. Ollama stays too: the friend installed it
# themselves, and it can be uninstalled like any app (Settings, Apps). The results file on the
# Desktop is never touched.

param([switch]$Yes)

$ErrorActionPreference = "Continue"

$LocaltuneHome = if ($env:LOCALTUNE_HOME) { $env:LOCALTUNE_HOME } else { Join-Path $env:USERPROFILE ".localtune" }
$Downloaded = Join-Path $LocaltuneHome "friend-downloaded-models.txt"
$InstalledUv = Join-Path $LocaltuneHome "friend-installed-uv.txt"
$UvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:Path = "$UvBin;$env:Path"

if (-not (Test-Path $LocaltuneHome)) {
    Write-Host "localtune isn't set up on this PC, so there's nothing to remove."
    exit 0
}

$models = @()
if (Test-Path $Downloaded) { $models = @(Get-Content $Downloaded | Where-Object { $_.Trim() } | Sort-Object -Unique) }

if (-not $Yes) {
    Write-Host "localtune clean-up" -ForegroundColor Cyan
    Write-Host "This removes from this PC:"
    if ($models) { Write-Host "  - the models the setup downloaded: $($models -join ', ')" }
    Write-Host "  - localtune, llama.cpp and localtune's saved data"
    if (Test-Path $InstalledUv) { Write-Host "  - uv, which the setup installed" }
    Write-Host "Your results file on the Desktop and Ollama itself stay."
    $answer = Read-Host "Go ahead? (Y/N)"
    if ($answer -notmatch "^\s*y") { Write-Host "Nothing was removed."; exit 0 }
}

# Keeps a copy of the results on the Desktop first, in case the setup was stopped before it
# made one, since localtune's folder is about to go.
$results = Join-Path $LocaltuneHome "results.jsonl"
$copy = Join-Path ([Environment]::GetFolderPath("Desktop")) "localtune-results-$env:COMPUTERNAME.jsonl"
if ((Test-Path $results) -and -not (Test-Path $copy)) { Copy-Item $results $copy }

# The models: Ollama removes them, so it has to be running.
if ($models) {
    Write-Host ""
    Write-Host "== Removing the models ==" -ForegroundColor Magenta
    $Ollama = Get-Command ollama -ErrorAction SilentlyContinue
    $OllamaExe = if ($Ollama) { $Ollama.Source } else { Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
    if (Test-Path $OllamaExe) {
        function Test-Ollama {
            try { Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 2 | Out-Null; $true } catch { $false }
        }
        if (-not (Test-Ollama)) {
            Start-Process $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
            foreach ($i in 1..30) { if (Test-Ollama) { break }; Start-Sleep 1 }
        }
        foreach ($model in $models) {
            Write-Host "-- $model"
            & $OllamaExe rm $model
        }
    } else {
        Write-Host "Ollama isn't on this PC anymore, so its models went with it."
    }
}

# localtune, and then uv if the setup installed it: uv's own uninstall steps (its cache, the
# Python it downloaded, its tools, then the program itself).
Write-Host ""
Write-Host "== Removing localtune ==" -ForegroundColor Magenta
$removeUv = Test-Path $InstalledUv
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv tool uninstall localtune
    if ($removeUv) {
        Write-Host ""
        Write-Host "== Removing uv ==" -ForegroundColor Magenta
        uv cache clean
        foreach ($folder in @((uv python dir), (uv tool dir))) {
            if ($folder -and (Test-Path $folder)) { Remove-Item -Recurse -Force $folder -ErrorAction SilentlyContinue }
        }
        Remove-Item -Force (Join-Path $UvBin "uv.exe"), (Join-Path $UvBin "uvx.exe"), (Join-Path $UvBin "uvw.exe") -ErrorAction SilentlyContinue
    }
}
Remove-Item -Recurse -Force $LocaltuneHome -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "All cleaned up. Thanks again for helping!" -ForegroundColor Green
if (Test-Path $copy) { Write-Host "Your results file is still on the Desktop: $copy" }
