$ErrorActionPreference = "Stop"
$InstallerDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $InstallerDir
$FrontRoot = Join-Path (Split-Path -Parent $ProjectRoot) "NIT_train_front"
$Payload = Join-Path $InstallerDir "payload"
$Build = Join-Path $InstallerDir "build"
$Dist = Join-Path $InstallerDir "release"

foreach ($Path in @($Payload, $Build, $Dist)) {
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $Resolved.StartsWith([System.IO.Path]::GetFullPath($InstallerDir), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer output path escaped installer folder: $Resolved"
    }
    if (Test-Path -LiteralPath $Resolved) { Remove-Item -LiteralPath $Resolved -Recurse -Force }
    New-Item -ItemType Directory -Path $Resolved | Out-Null
}

function Copy-ItemSet([string]$SourceRoot, [string]$TargetRoot, [string[]]$Items) {
    foreach ($Item in $Items) {
        $Source = Join-Path $SourceRoot $Item
        if (-not (Test-Path -LiteralPath $Source)) { throw "Required packaging file missing: $Source" }
        $Destination = Join-Path $TargetRoot $Item
        $Parent = Split-Path -Parent $Destination
        if (-not (Test-Path -LiteralPath $Parent)) { New-Item -ItemType Directory -Path $Parent -Force | Out-Null }
        Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    }
}

# The native launcher runs both Python entry points directly; Docker assets are not packaged.
Copy-ItemSet $ProjectRoot $Payload @("requirements.txt", "VERSION", "src", "test_model")
$Front = Join-Path $Payload "NIT_train_front"
New-Item -ItemType Directory -Path $Front | Out-Null
Copy-ItemSet $FrontRoot $Front @("server.py", "index.html", "package.json", "assets")

Push-Location $InstallerDir
try {
    pyinstaller --noconfirm --clean --onefile --windowed --name NIT_train --distpath $Dist --workpath (Join-Path $Build "launcher") launcher.py
    Copy-Item -LiteralPath (Join-Path $Dist "NIT_train.exe") -Destination (Join-Path $Payload "NIT_train.exe") -Force
    pyinstaller --noconfirm --clean --onefile --windowed --name NIT_train_Setup --distpath $Dist --workpath (Join-Path $Build "setup") --add-data "$Payload;payload" setup.py
} finally { Pop-Location }

Write-Host "Installer created: $(Join-Path $Dist 'NIT_train_Setup.exe')"
