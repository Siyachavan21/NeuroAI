Write-Host "Starting NeuroAI Healthcare Application..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Ensure backend venv is used and install deps only if missing
$backendPath = Join-Path $PSScriptRoot 'backend'
$venvPython = Join-Path (Join-Path $PSScriptRoot '..\venv\Scripts') 'python.exe'
if (-Not (Test-Path $venvPython)) { $venvPython = 'python' }

Write-Host "`nChecking backend dependencies..." -ForegroundColor Yellow
Push-Location $backendPath
& $venvPython start_server.py | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Attempting one-time dependency install..." -ForegroundColor Yellow
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install -r requirements.txt
}
Pop-Location

Write-Host "`nStarting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; & '$venvPython' start_server.py" -WindowStyle Normal

Write-Host "`nStarting Frontend Application..." -ForegroundColor Yellow
$frontendPath = Join-Path $PSScriptRoot 'frontendd'
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; if (-not (Test-Path node_modules)) { npm ci --no-fund --no-audit }; npm start" -WindowStyle Normal

Write-Host "`nBoth servers are starting..." -ForegroundColor Green
Write-Host "Backend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "`nPress any key to close this window..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
