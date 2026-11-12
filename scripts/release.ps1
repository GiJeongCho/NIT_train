# NIT_train 릴리스 빌드 (Windows / PowerShell)
#
# 버전(VERSION 파일 또는 -Version), 깃 커밋, 빌드 시각을 이미지에 박아 태깅한다.
#   .\scripts\release.ps1                 # VERSION 파일 버전으로 빌드
#   .\scripts\release.ps1 -Version 1.1.0  # 특정 버전으로 빌드
#   .\scripts\release.ps1 -Up             # 빌드 후 기동(docker compose up -d)
#   .\scripts\release.ps1 -Cpu -Up        # GPU 없는 호스트
[CmdletBinding()]
param(
  [string]$Version,
  [switch]$Up,
  [switch]$Cpu,
  [switch]$NoLatest
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Version) { $Version = (Get-Content "$root\VERSION" -Raw).Trim() }
try { $sha = (git rev-parse --short=12 HEAD 2>$null).Trim() } catch { $sha = "" }
$date = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

$env:VERSION = $Version
$env:GIT_SHA = $sha
$env:BUILD_DATE = $date
$env:NIT_TRAIN_TAG = $Version

$files = @("-f", "docker-compose.yml")
if ($Cpu) { $files += @("-f", "docker-compose.cpu.yml") }

Write-Host "[release] nit-train $Version (git $sha, $date)" -ForegroundColor Cyan
docker compose @files build

if (-not $NoLatest) {
  docker tag "nit-train:$Version"       "nit-train:latest"
  docker tag "nit-train-front:$Version" "nit-train-front:latest"
}

if ($Up) { docker compose @files up -d }

Write-Host "[release] 완료. 화면 http://localhost:8890  API http://localhost:8888/docs" -ForegroundColor Green
