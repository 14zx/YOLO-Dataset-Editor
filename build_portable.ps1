# Builds portable YOLO Dataset Editor (Flask + Ultralytics + bundled Vite UI).
# Requires: Node.js (npm), Python venv with ultralytics, flask, flask-cors, opencv, pyinstaller.
# Default Python: %USERPROFILE%\yolo-editor-venv (same as run_yolo_editor.bat).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== Sync editor component ===" -ForegroundColor Cyan
Copy-Item -Force (Join-Path $Root "yolo-dataset-editor.jsx") (Join-Path $Root "web\src\YoloEditor.jsx")

Write-Host "=== Vite build ===" -ForegroundColor Cyan
Push-Location (Join-Path $Root "web")
if (-not (Test-Path "node_modules")) {
  npm install
}
npm run build
Pop-Location

$PyDefault = Join-Path $env:USERPROFILE "yolo-editor-venv\Scripts\python.exe"
$Py = if ($env:YOLO_EDITOR_PYTHON) { $env:YOLO_EDITOR_PYTHON } else { $PyDefault }
if (-not (Test-Path $Py)) {
  Write-Error "Python not found: $Py (set YOLO_EDITOR_PYTHON or create yolo-editor-venv)"
}

Write-Host "=== PyInstaller ($Py) ===" -ForegroundColor Cyan
& $Py -m pip install -q pyinstaller

$OutDir = Join-Path $Root "dist\YOLO-Dataset-Editor"
if (Test-Path $OutDir) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $bak = Join-Path $Root "dist\YOLO-Dataset-Editor.$stamp.bak"
  Write-Host "Rename locked dist -> $bak" -ForegroundColor Yellow
  Rename-Item -Path $OutDir -NewName (Split-Path $bak -Leaf) -ErrorAction SilentlyContinue
}

$Dist = Join-Path $Root "web\dist"
& $Py -m PyInstaller --noconfirm --onedir --name YOLO-Dataset-Editor `
  --add-data "$Dist;dist" `
  --collect-data ultralytics `
  --collect-data cv2 `
  --hidden-import ultralytics `
  --hidden-import ultralytics.models.yolo.detect `
  --hidden-import ultralytics.models.yolo.segment `
  --hidden-import ultralytics.nn.tasks `
  --hidden-import lap `
  --hidden-import flask `
  --hidden-import flask_cors `
  --exclude-module datasets `
  --exclude-module huggingface_hub `
  --exclude-module torchaudio `
  --exclude-module tensorboard `
  (Join-Path $Root "yolo_editor_frozen.py")

$OutDir = Join-Path $Root "dist\YOLO-Dataset-Editor"
$ZipDir = Join-Path $Root "portable_dist"
New-Item -ItemType Directory -Force -Path $ZipDir | Out-Null
$Zip = Join-Path $ZipDir "YOLO-Dataset-Editor-Portable.zip"

Write-Host "=== Archive (tar zip) ===" -ForegroundColor Cyan
Push-Location (Join-Path $Root "dist")
if (Test-Path $Zip) { Remove-Item $Zip -Force }
& tar.exe -a -c -f $Zip "YOLO-Dataset-Editor"
Pop-Location

Write-Host ""
Write-Host "Done: $OutDir" -ForegroundColor Green
Write-Host "Zip:  $Zip" -ForegroundColor Green
